"""查询已入库的文档。

原理：
- 上传接口不会保留原始 PDF 文件（解析完成后立即删除临时文件），
  Qdrant 中只保存切块文本、向量和元数据；
- 因此“已上传的文档”等价于“Qdrant 中存在的 doc_id 版本”，
  本脚本按 ``dept_id + doc_id + doc_hash`` 聚合展示。

用法：
    # 默认读取 QDRANT_COLLECTION_PREFIX（未设置时为 rag），列出所有部门
    .venv\\Scripts\\python.exe scripts\\list_documents.py

    # 当前开发服务使用 rag_smoke 前缀
    .venv\\Scripts\\python.exe scripts\\list_documents.py --prefix rag_smoke

    # 只看某个部门 / 某份文档，并显示分块内容
    .venv\\Scripts\\python.exe scripts\\list_documents.py --prefix rag_smoke --dept-id dept_A
    .venv\\Scripts\\python.exe scripts\\list_documents.py --prefix rag_smoke --doc-id manual_handbook.pdf --show-chunks

    # 包含已被软删除的历史版本
    .venv\\Scripts\\python.exe scripts\\list_documents.py --prefix rag_smoke --include-inactive
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

# 允许直接运行本脚本：把项目根目录加入 sys.path，便于导入 app.config
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings  # noqa: E402

BATCH_SIZE = 256

PAYLOAD_FIELDS = ["dept_id", "doc_id", "doc_hash", "is_active", "indexed_at"]


def build_filter(
    *,
    dept_id: str | None,
    doc_id: str | None,
    include_inactive: bool,
) -> Filter | None:
    """构造部门 / 文档 / 版本过滤条件。"""
    must = []
    if dept_id:
        must.append(
            FieldCondition(key="dept_id", match=MatchValue(value=dept_id.strip()))
        )
    if doc_id:
        must.append(
            FieldCondition(key="doc_id", match=MatchValue(value=doc_id.strip()))
        )
    if not include_inactive:
        must.append(
            FieldCondition(key="is_active", match=MatchValue(value=True))
        )
    return Filter(must=must) if must else None


def iter_points(
    client: QdrantClient,
    collection: str,
    scroll_filter: Filter | None,
    *,
    with_payload: bool | list[str],
) -> Iterator[Any]:
    """按批次遍历所有匹配的点，避免一次拉取过多数据。"""
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            scroll_filter=scroll_filter,
            limit=BATCH_SIZE,
            offset=offset,
            with_payload=with_payload,
            with_vectors=False,
        )
        yield from points
        if offset is None:
            break


def collect_documents(
    client: QdrantClient,
    collection: str,
    scroll_filter: Filter | None,
    *,
    show_chunks: bool,
) -> list[dict]:
    """聚合文档版本信息。"""
    groups: dict[tuple[str, str, str, bool], dict] = defaultdict(
        lambda: {
            "chunk_count": 0,
            "indexed_at": None,
            "texts": [],
        }
    )

    payload_fields: bool | list[str] = True if show_chunks else PAYLOAD_FIELDS
    for point in iter_points(
        client, collection, scroll_filter, with_payload=payload_fields
    ):
        payload = point.payload or {}
        dept_id = payload.get("dept_id", "")
        doc_id = payload.get("doc_id", "")
        doc_hash = payload.get("doc_hash", "")
        is_active = bool(payload.get("is_active", False))

        key = (dept_id, doc_id, doc_hash, is_active)
        group = groups[key]
        group["chunk_count"] += 1

        indexed_at = payload.get("indexed_at")
        if indexed_at and (group["indexed_at"] is None or indexed_at > group["indexed_at"]):
            group["indexed_at"] = indexed_at

        if show_chunks:
            text = str(payload.get("text", "")).replace("\n", " ").strip()
            group["texts"].append(text)

    documents = [
        {
            "dept_id": dept_id,
            "doc_id": doc_id,
            "doc_hash": doc_hash,
            "is_active": is_active,
            "chunk_count": group["chunk_count"],
            "indexed_at": group["indexed_at"],
            "chunks": group["texts"],
        }
        for (dept_id, doc_id, doc_hash, is_active), group in groups.items()
    ]

    # 部门 -> 文档 -> 当前有效版本优先 -> 入库时间倒序
    documents.sort(
        key=lambda item: (
            item["dept_id"],
            item["doc_id"],
            not item["is_active"],
            item["indexed_at"] or "",
        )
    )
    return documents


def print_documents(documents: list[dict]) -> None:
    if not documents:
        print("没有查询到匹配的文档。")
        print("提示：确认 QDRANT_COLLECTION_PREFIX 与服务启动时一致（当前为 "
              f"{settings.qdrant_collection_prefix!r}）。")
        return

    current_dept = None
    for item in documents:
        if item["dept_id"] != current_dept:
            current_dept = item["dept_id"]
            print(f"\ndept_id={current_dept}")

        status = "当前有效" if item["is_active"] else "历史版本"
        print(
            f"  doc_id={item['doc_id']}\n"
            f"    [{status}] 分块={item['chunk_count']}  "
            f"hash={item['doc_hash'][:12]}…  "
            f"indexed_at={item['indexed_at'] or '未知'}"
        )

        for index, text in enumerate(item["chunks"], start=1):
            preview = text[:120]
            print(f"      子块{index}: {preview}{'…' if len(text) > 120 else ''}")

    total_chunks = sum(item["chunk_count"] for item in documents)
    total_active = sum(1 for item in documents if item["is_active"])
    print(
        f"\n共 {len(documents)} 个文档版本（当前有效 {total_active} 个），"
        f"{total_chunks} 个分块。"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="查询 Qdrant 中已入库的文档")
    parser.add_argument(
        "--prefix",
        default=settings.qdrant_collection_prefix,
        help="集合前缀，集合名为 {prefix}_chunks（默认读取配置）",
    )
    parser.add_argument("--collection", help="直接指定集合名，优先级高于 --prefix")
    parser.add_argument("--dept-id", help="只查询指定部门")
    parser.add_argument("--doc-id", help="只查询指定文档")
    parser.add_argument(
        "--include-inactive",
        action="store_true",
        help="同时显示被软删除的历史版本（默认只看当前有效版本）",
    )
    parser.add_argument(
        "--show-chunks",
        action="store_true",
        help="显示每个分块的前 120 个字符",
    )
    parser.add_argument("--json", action="store_true", help="以 JSON 输出，便于脚本处理")
    args = parser.parse_args()

    collection = args.collection or f"{args.prefix}_chunks"

    api_key = (
        settings.qdrant_api_key.get_secret_value()
        if settings.qdrant_api_key
        else None
    )
    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=api_key,
        timeout=settings.qdrant_timeout,
        prefer_grpc=settings.qdrant_prefer_grpc,
    )
    try:
        if not client.collection_exists(collection):
            print(f"集合不存在：{collection}", file=sys.stderr)
            return 1

        scroll_filter = build_filter(
            dept_id=args.dept_id,
            doc_id=args.doc_id,
            include_inactive=args.include_inactive,
        )
        documents = collect_documents(
            client,
            collection,
            scroll_filter,
            show_chunks=args.show_chunks,
        )
    finally:
        client.close()

    if args.json:
        print(json.dumps(documents, ensure_ascii=False, indent=2))
        return 0

    print(f"集合：{collection}")
    print_documents(documents)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
