"""文档上传与入库接口。

协议：
- ``POST /api/docs/upload``，``multipart/form-data``
  - ``file``: PDF 文件（必填）
  - ``dept_id``: 部门 ID（必填，作为多租户隔离字段）
  - ``doc_id``: 文档稳定标识（可选，默认使用原始文件名）
- 返回 JSON，包含切块数量、版本状态、写入/停用点数等。
- ``GET /api/docs/departments``：按 dept_id 聚合当前有效版本的分块数量。

处理流程：
    保存临时文件 -> PDF 解析为 Markdown -> Parent-Child 切块
    -> Embedding + 写入 Qdrant（带 doc_hash 版本控制）-> 删除临时文件

状态码：
- 200: 入库成功（含 ``unchanged``，表示内容未修改）
- 400: 文件为空、类型不支持、解析失败、切块为空
- 413: 文件超过大小上限
- 502: Embedding 服务调用失败
- 503: Embedding 未配置
- 500: 其他未预期错误

说明：
- ``doc_hash`` 使用原始文件字节的 SHA-256，调整切块参数不会误判为“内容已修改”；
- 同一 ``doc_id`` 重复上传且内容一致时返回 ``unchanged``，不会重复写入；
- 内容变化时会先停用旧版本中真正发生变化的分块，再写入新版本；
- 并发上传同一 ``doc_id`` 目前未做进程内串行化，生产环境建议由上层按文档加锁。
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import uuid
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from qdrant_client.http.exceptions import ResponseHandlingException

from app.core.chunking import process_document
from app.core.embedding import EmbeddingConfigurationError, EmbeddingServiceError
from app.core.pdf_parser import PDFParseError, parse_pdf_to_markdown
from app.core.retriever import get_retriever

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/docs", tags=["文档管理"])

# 临时文件目录（相对于服务启动目录，即项目根目录）
TMP_DIR = Path("tmp")

# 单个文件大小上限
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_UPLOAD_MB = MAX_UPLOAD_BYTES // (1024 * 1024)

# 分块读取大小，避免一次把大文件读进内存
READ_CHUNK_SIZE = 1024 * 1024

# 部门 ID / 文档 ID 长度上限，与 QdrantRetriever 的校验保持一致
MAX_DEPT_ID_LENGTH = 128
MAX_DOC_ID_LENGTH = 256


class UploadResponse(BaseModel):
    """文档入库结果。"""

    doc_id: str = Field(description="文档稳定标识")
    dept_id: str = Field(description="所属部门")
    filename: str = Field(description="上传时的原始文件名")
    status: Literal["created", "updated", "unchanged"] = Field(
        description="created=首次入库，updated=新版本已生效，unchanged=内容未修改"
    )
    message: str = Field(description="可直接展示给用户的结果说明")
    chunk_count: int = Field(description="本次切分得到的子块数量")
    parent_count: int = Field(description="本次切分得到的父块数量")
    written_points: int = Field(description="本次写入的向量点数")
    deactivated_points: int = Field(description="本次停用的旧版本点数")
    doc_hash: str = Field(description="原始文件字节的 SHA-256")
    size_bytes: int = Field(description="文件大小（字节）")
    elapsed_seconds: float = Field(description="服务端处理耗时（秒）")


class DepartmentSummary(BaseModel):
    """部门聚合结果。"""

    dept_id: str = Field(description="部门 ID")
    count: int = Field(
        description="该部门下当前有效版本的分块数量（is_active=True 的向量点）"
    )


@router.get(
    "/departments",
    response_model=list[DepartmentSummary],
    summary="查询已入库部门",
)
async def get_departments() -> list[DepartmentSummary]:
    """按 ``dept_id`` 聚合当前有效版本的分块数量。

    ``count`` 统计的是向量点（子块）数量，不是文档数量。
    """
    try:
        rows = await asyncio.to_thread(_list_departments_sync)
    except ResponseHandlingException as exc:
        logger.exception("查询部门列表失败：Qdrant 不可用")
        raise HTTPException(
            status_code=503,
            detail="向量数据库不可用，请稍后重试",
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("查询部门列表失败")
        raise HTTPException(
            status_code=500,
            detail="查询部门列表失败",
        ) from exc

    return [DepartmentSummary(**row) for row in rows]


@router.post(
    "/upload",
    response_model=UploadResponse,
    summary="上传 PDF 并入库",
)
async def upload_document(
    file: UploadFile = File(..., description="待入库的 PDF 文件"),
    dept_id: str = Form(..., description="部门 ID，用于多租户隔离"),
    doc_id: str | None = Form(
        None,
        description="文档稳定标识；同一文档的多个版本必须保持一致，默认使用文件名",
    ),
) -> UploadResponse:
    """接收 PDF，解析、切块、向量化并写入 Qdrant。"""
    started = time.perf_counter()

    normalized_dept = _normalize_dept_id(dept_id)
    filename = _normalize_filename(file.filename)
    _validate_pdf_suffix(filename)
    stable_doc_id = _normalize_doc_id(doc_id, filename)

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    # 使用随机文件名，避免原始文件名中的路径分隔符或特殊字符造成路径穿越
    temp_path = TMP_DIR / f"{uuid.uuid4().hex}.pdf"

    try:
        size_bytes, doc_hash = await _save_upload(file, temp_path)
        markdown = await _parse_pdf(temp_path, filename)
        docs = await _chunk_markdown(markdown, normalized_dept)
        result = await _ingest(docs, stable_doc_id, doc_hash)

        elapsed = round(time.perf_counter() - started, 3)
        logger.info(
            "文档入库完成：dept_id=%s, doc_id=%s, 状态=%s, 子块=%d, "
            "写入=%d, 停用=%d, 耗时=%.2fs",
            normalized_dept,
            stable_doc_id,
            result.status,
            len(docs),
            result.written_points,
            result.deactivated_points,
            elapsed,
        )

        return UploadResponse(
            doc_id=result.doc_id,
            dept_id=normalized_dept,
            filename=filename,
            status=result.status,
            message=result.message,
            chunk_count=len(docs),
            parent_count=len(
                {
                    doc["parent_id"]
                    for doc in docs
                    if isinstance(doc.get("parent_id"), str)
                }
            ),
            written_points=result.written_points,
            deactivated_points=result.deactivated_points,
            doc_hash=result.doc_hash,
            size_bytes=size_bytes,
            elapsed_seconds=elapsed,
        )
    finally:
        await _remove_temp_file(temp_path)


# =====================================================================
# 内部步骤
# =====================================================================
async def _save_upload(
    file: UploadFile,
    destination: Path,
) -> tuple[int, str]:
    """流式保存上传文件，同时计算 SHA-256。

    Returns:
        (文件字节数, 十六进制 SHA-256)
    """
    hasher = hashlib.sha256()
    size = 0

    try:
        with destination.open("wb") as output:
            while True:
                chunk = await file.read(READ_CHUNK_SIZE)
                if not chunk:
                    break

                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"文件超过大小上限（{MAX_UPLOAD_MB} MB）",
                    )

                hasher.update(chunk)
                output.write(chunk)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("保存上传文件失败：%s", destination)
        raise HTTPException(status_code=500, detail="保存上传文件失败") from exc

    if size == 0:
        raise HTTPException(status_code=400, detail="上传文件为空")

    return size, hasher.hexdigest()


async def _parse_pdf(path: Path, filename: str) -> str:
    """在线程池中执行 PDF 解析，避免阻塞事件循环。"""
    try:
        return await asyncio.to_thread(parse_pdf_to_markdown, path)
    except PDFParseError as exc:
        logger.warning("PDF 解析失败：file=%s, error=%s", filename, exc)
        raise HTTPException(
            status_code=400,
            detail=f"PDF 解析失败：{exc}",
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("PDF 解析出现未预期异常：file=%s", filename)
        raise HTTPException(status_code=400, detail="PDF 解析失败") from exc


async def _chunk_markdown(markdown: str, dept_id: str) -> list[dict]:
    """在线程池中执行 Parent-Child 切块。"""
    try:
        docs = await asyncio.to_thread(process_document, markdown, dept_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("文档切块失败：dept_id=%s", dept_id)
        raise HTTPException(status_code=400, detail=f"文档切分失败：{exc}") from exc

    if not docs:
        raise HTTPException(
            status_code=400,
            detail="未能从文档中提取有效内容（文档可能为空或只有图片）",
        )
    return docs


def _list_departments_sync() -> list[dict]:
    """同步执行 Qdrant facet 聚合（在线程池中调用）。"""
    retriever = get_retriever()
    return retriever.list_departments()


def _ingest_sync(
    docs: list[dict],
    doc_id: str,
    doc_hash: str,
):
    """同步执行 Embedding + Qdrant 写入（在线程池中调用）。"""
    retriever = get_retriever()
    return retriever.add_documents(
        docs,
        doc_id=doc_id,
        doc_hash=doc_hash,
    )


async def _ingest(
    docs: list[dict],
    doc_id: str,
    doc_hash: str,
):
    """在线程池中执行向量化与入库。"""
    try:
        return await asyncio.to_thread(_ingest_sync, docs, doc_id, doc_hash)
    except EmbeddingConfigurationError as exc:
        logger.exception("Embedding 未配置：%s", exc)
        raise HTTPException(
            status_code=503,
            detail="Embedding 服务未配置，请联系管理员",
        ) from exc
    except EmbeddingServiceError as exc:
        logger.exception("Embedding 调用失败：%s", exc)
        raise HTTPException(
            status_code=502,
            detail="向量化服务调用失败，请稍后重试",
        ) from exc
    except ResponseHandlingException as exc:
        logger.exception("向量数据库不可用：doc_id=%s", doc_id)
        raise HTTPException(
            status_code=503,
            detail="向量数据库不可用，请稍后重试",
        ) from exc
    except (TypeError, ValueError) as exc:
        logger.warning("入库参数不合法：doc_id=%s, error=%s", doc_id, exc)
        raise HTTPException(status_code=400, detail=f"入库参数不合法：{exc}") from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("文档入库失败：doc_id=%s", doc_id)
        raise HTTPException(status_code=500, detail="文档入库失败") from exc


# =====================================================================
# 参数校验与临时文件清理
# =====================================================================
def _normalize_dept_id(value: str) -> str:
    normalized = (value or "").strip()
    if not normalized:
        raise HTTPException(status_code=400, detail="dept_id 不能为空")
    if len(normalized) > MAX_DEPT_ID_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"dept_id 长度不能超过 {MAX_DEPT_ID_LENGTH} 个字符",
        )
    return normalized


def _normalize_filename(value: str | None) -> str:
    filename = Path((value or "").strip()).name
    if not filename:
        raise HTTPException(status_code=400, detail="缺少文件名")
    return filename


def _validate_pdf_suffix(filename: str) -> None:
    if Path(filename).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=400,
            detail="仅支持 PDF 文件（后缀 .pdf）",
        )


def _normalize_doc_id(value: str | None, filename: str) -> str:
    """文档稳定标识：优先使用显式传入值，否则回退到原始文件名。"""
    candidate = (value or "").strip() or filename
    if len(candidate) > MAX_DOC_ID_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"doc_id 长度不能超过 {MAX_DOC_ID_LENGTH} 个字符",
        )
    return candidate


async def _remove_temp_file(path: Path) -> None:
    """删除临时文件。

    Windows 上杀毒软件 / 索引服务可能短暂占用刚写入的文件，
    因此失败后做少量重试；仍失败时记录完整原因，便于排查。
    使用 ``asyncio.sleep`` 而不是 ``time.sleep``，避免阻塞事件循环。
    """
    last_error: OSError | None = None

    for attempt in range(3):
        try:
            path.unlink(missing_ok=True)
            return
        except OSError as exc:
            last_error = exc
            await asyncio.sleep(0.1 * (attempt + 1))

    logger.warning(
        "临时文件删除失败，请手动清理：%s（%s: %s）",
        path,
        type(last_error).__name__ if last_error else "OSError",
        last_error,
    )


__all__ = ["DepartmentSummary", "UploadResponse", "router"]
