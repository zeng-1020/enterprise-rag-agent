"""PDF 转 Markdown。

实现说明：
1. 一次性读入文件字节，再以内存流方式打开（``pymupdf.open(stream=...)``），
   先做加密、页数等基础校验；
2. 使用 ``pymupdf4llm`` 将版面转换为 Markdown，保留 ``#`` / ``##``
   标题层级，供 :func:`app.core.chunking.process_document` 切分父块；
3. 所有异常统一转换为 :class:`PDFParseError`，由 API 层转换成 400 响应。

注意：
- 以内存流而不是文件路径打开，是为了不在 Windows 上持有文件句柄：
  上传接口在解析结束后会立即删除临时文件，按路径打开时 PyMuPDF 可能
  短暂锁定文件，导致 ``unlink`` 失败并残留垃圾文件；
- 因此整个 PDF 会被读入内存，调用方应限制文件大小（上传接口限制 50 MB）；
- 扫描件（纯图片、无文本层）需要 OCR，默认不启用；此类文件会得到空文本
  并抛出 ``PDFParseError``，提示调用方改用 OCR 方案；
- ``pymupdf4llm`` 首次调用会初始化版面分析组件，耗时可能略高于后续调用。
"""

from __future__ import annotations

import logging
from pathlib import Path

import pymupdf
import pymupdf4llm

logger = logging.getLogger(__name__)


class PDFParseError(ValueError):
    """PDF 无法解析时抛出的业务异常。"""


def parse_pdf_to_markdown(file_path: str | Path) -> str:
    """把 PDF 文件解析为 Markdown 文本。

    Args:
        file_path: PDF 文件路径。

    Returns:
        解析后的 Markdown 文本，已去除首尾空白。

    Raises:
        PDFParseError: 文件不存在、不是 PDF、已加密、损坏、
            没有页面或未能提取到任何文本。
    """
    path = Path(file_path)

    if not path.is_file():
        raise PDFParseError(f"文件不存在：{path}")
    if path.suffix.lower() != ".pdf":
        raise PDFParseError(f"仅支持 PDF 文件，实际后缀：{path.suffix or '无'}")

    try:
        # read_bytes 内部 with open(...) 会立即关闭句柄，不会锁定文件
        data = path.read_bytes()
    except OSError as exc:
        logger.warning("无法读取 PDF：%s（%s: %s）", path, type(exc).__name__, exc)
        raise PDFParseError("无法读取 PDF 文件") from exc

    try:
        document = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:  # noqa: BLE001 - 不同损坏类型异常不一致
        # 不把底层异常拼进用户可见信息：其中通常包含临时文件路径
        logger.warning("无法打开 PDF：%s（%s: %s）", path, type(exc).__name__, exc)
        raise PDFParseError(
            "无法打开 PDF，文件可能已损坏、被加密或不是有效的 PDF"
        ) from exc

    try:
        if document.needs_pass:
            raise PDFParseError("PDF 已加密，请先解除密码后重新上传")
        if document.page_count == 0:
            raise PDFParseError("PDF 不包含任何页面")

        # pymupdf4llm 2.x 直接接受已打开的 Document，避免再次按路径打开
        markdown = pymupdf4llm.to_markdown(document)
    except PDFParseError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("PDF 解析异常：%s", path)
        raise PDFParseError("PDF 内容解析失败，文件可能已损坏") from exc
    finally:
        if not document.is_closed:
            document.close()

    if not isinstance(markdown, str) or not markdown.strip():
        raise PDFParseError(
            "未能从 PDF 中提取到文本；如果这是扫描件，请先做 OCR 再上传"
        )

    return markdown.strip()


__all__ = ["PDFParseError", "parse_pdf_to_markdown"]
