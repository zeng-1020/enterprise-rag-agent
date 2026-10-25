"""真实环境冒烟测试：Qdrant + 云端 Embedding + 云端 LLM。

覆盖范围：
1. 环境检查（密钥是否填写、Qdrant 是否可连接）；
2. 文档入库（首次 created）；
3. 重复上传同内容（unchanged，不重复写点）；
4. 多租户隔离（dept_B 检不到 dept_A 的年假内容）；
5. 文档版本更新（updated，旧版本 is_active=False 且检索不可见）；
6. Agent 全链路（检索 -> 评估 -> 生成，真实 LLM）。

隔离性：脚本启动时把 ``QDRANT_COLLECTION_PREFIX`` 覆盖为 ``rag_smoke``，
使用独立的 ``rag_smoke_chunks`` 集合，不会影响正式数据。

运行前必须：
1. 启动 Qdrant（默认 http://localhost:6333）；
2. 在 .env 中填写可用的 EMBEDDING_API_KEY / LLM_API_KEY。

用法（在项目根目录执行）：
    .\\.venv\\Scripts\\python.exe scripts\\smoke_test.py
    .\\.venv\\Scripts\\python.exe scripts\\smoke_test.py --full
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import sys
import time
from pathlib import Path

# 必须在导入 app 之前设置：环境变量优先级高于 .env
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ["QDRANT_COLLECTION_PREFIX"] = "rag_smoke"

from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.models import (  # noqa: E402
    FieldCondition,
    Filter,
    MatchValue,
)

from app.agent.graph import agent_graph_lifespan, build_thread_config  # noqa: E402
from app.agent.state import create_initial_state  # noqa: E402
from app.config import settings  # noqa: E402
from app.core.chunking import process_document  # noqa: E402
from app.core.retriever import QdrantRetriever  # noqa: E402

DEPT_A = "smoke_dept_a"
DEPT_B = "smoke_dept_b"
DOC_A_ID = "smoke-hr-handbook"
DOC_B_ID = "smoke-finance-policy"
CHECKPOINT_PATH = Path("data") / "smoke_checkpoints.sqlite"

DOC_A_V1 = """# 员工手册

## 年假制度
公司为全体正式员工提供带薪年假。入职满一年的员工，每年可享受 10 天带薪年假。
年假需要提前三个工作日通过 OA 系统申请，并经直属主管审批后生效。
未使用的年假可以结转至次年第一季度，逾期作废。

## 考勤管理
员工每日上下班需在考勤系统中各打卡一次。忘记打卡需在三个工作日内提交补卡申请。
"""

DOC_A_V2 = """# 员工手册

## 年假制度
公司为全体正式员工提供带薪年假。入职满一年的员工，每年可享受 15 天带薪年假；
入职满五年的员工，每年可享受 20 天带薪年假。
年假需要提前三个工作日通过 OA 系统申请，并经直属主管审批后生效。
未使用的年假可以结转至次年第一季度，逾期作废。

## 考勤管理
员工每日上下班需在考勤系统中各打卡一次。忘记打卡需在三个工作日内提交补卡申请。
"""

DOC_B = """# 财务报销制度

## 差旅报销
员工出差前需在 OA 系统提交出差申请，注明目的地、事由与预计费用。
差旅结束后十个工作日内提交报销单，并附上发票、行程单等凭证。
住宿标准按城市等级执行，超标部分需部门负责人特批。

## 发票要求
报销发票抬头必须为公司全称，发票内容需与实际业务一致。
"""


class SmokeError(RuntimeError):
    """冒烟测试断言失败。"""


def _configure_stdout() -> None:
    """Windows 控制台默认可能是 GBK，强制 UTF-8 避免中文乱码。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass


def _hash(text: str) -> str:
    """按文件内容计算文档哈希（真实上传时应对原始文件字节做哈希）。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _check(name: str, condition: bool, detail: str = "") -> None:
    mark = "PASS" if condition else "FAIL"
    print(f"[{mark}] {name}" + (f"：{detail}" if detail else ""))
    if not condition:
        raise SmokeError(f"{name} 未通过。{detail}")


def _prerequisites() -> QdrantClient:
    """检查密钥与 Qdrant 连通性，失败时给出可操作的提示。"""
    print("=" * 72)
    print("环境检查")
    print("=" * 72)
    print(f"Qdrant URL      : {settings.qdrant_url}")
    print(f"Embedding 模型  : {settings.embedding_model} ({settings.embedding_dim} 维)")
    print(f"Embedding 地址  : {settings.embedding_base_url}")
    print(f"LLM 模型        : {settings.llm_model}")

    if not settings.embedding_configured:
        print("\n[FAIL] EMBEDDING_API_KEY 未配置。请编辑 .env 后重新运行。")
        print("       DeepSeek 不提供 Embedding，可选 SiliconFlow 的 BAAI/bge-m3。")
        raise SystemExit(2)
    if not settings.llm_configured:
        print("\n[FAIL] LLM_API_KEY 未配置。请编辑 .env 后重新运行。")
        raise SystemExit(2)

    api_key = (
        settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
    )
    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=api_key or None,
        timeout=5,
    )
    try:
        client.get_collections()
    except Exception as exc:  # noqa: BLE001
        print(f"\n[FAIL] 无法连接 Qdrant（{type(exc).__name__}: {exc}）")
        print("       请先启动 Qdrant，例如：")
        print(
            "       docker run -d --name qdrant -p 6333:6333 -p 6334:6334 "
            '-v "${PWD}/qdrant_storage:/qdrant/storage" qdrant/qdrant'
        )
        raise SystemExit(2) from exc

    print("[PASS] 密钥配置与 Qdrant 连接正常")
    return client


def _count_points(
    client: QdrantClient,
    collection: str,
    *,
    dept_id: str,
    doc_id: str,
    is_active: bool,
) -> int:
    """统计某部门某文档指定版本状态的点数。"""
    result = client.count(
        collection_name=collection,
        count_filter=Filter(
            must=[
                FieldCondition(key="dept_id", match=MatchValue(value=dept_id)),
                FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
                FieldCondition(key="is_active", match=MatchValue(value=is_active)),
            ]
        ),
        exact=True,
    )
    return int(result.count)


def _ingest(
    retriever: QdrantRetriever,
    text: str,
    dept_id: str,
    doc_id: str,
):
    docs = process_document(text, dept_id)
    _check(
        f"切块完成（{doc_id}）",
        bool(docs),
        f"{len(docs)} 个子块，{len({d['parent_id'] for d in docs})} 个父块",
    )
    return retriever.add_documents(docs, doc_id=doc_id, doc_hash=_hash(text))


def _run_data_checks(client: QdrantClient, retriever: QdrantRetriever) -> None:
    collection = f"{settings.qdrant_collection_prefix}_chunks"

    print("\n" + "=" * 72)
    print("数据链路：入库 / 幂等 / 隔离 / 版本控制")
    print("=" * 72)

    # ---------- 首次入库 ----------
    result_a1 = _ingest(retriever, DOC_A_V1, DEPT_A, DOC_A_ID)
    _check("dept_A 首次入库", result_a1.status == "created", result_a1.message)

    result_b = _ingest(retriever, DOC_B, DEPT_B, DOC_B_ID)
    _check("dept_B 首次入库", result_b.status == "created", result_b.message)

    # ---------- 重复上传：内容未修改 ----------
    result_a2 = _ingest(retriever, DOC_A_V1, DEPT_A, DOC_A_ID)
    _check(
        "重复上传同内容返回 unchanged",
        result_a2.status == "unchanged" and result_a2.written_points == 0,
        result_a2.message,
    )

    # ---------- 多租户隔离 ----------
    hits_a = retriever.search("年假有几天", DEPT_A, top_k=5)
    _check("dept_A 能检索到年假资料", bool(hits_a), f"{len(hits_a)} 条")
    _check(
        "dept_A 检索结果全部属于 dept_A",
        all(hit.get("dept_id") == DEPT_A for hit in hits_a),
    )

    hits_b = retriever.search("年假有几天", DEPT_B, top_k=5)
    _check(
        "dept_B 检索不到 dept_A 的年假内容",
        all(hit.get("dept_id") == DEPT_B for hit in hits_b)
        and not any("年假" in (hit.get("text") or "") for hit in hits_b),
        f"dept_B 返回 {len(hits_b)} 条（来自本部门财务文档）",
    )

    # ---------- 版本更新 ----------
    active_before = _count_points(
        client, collection, dept_id=DEPT_A, doc_id=DOC_A_ID, is_active=True
    )
    result_a3 = _ingest(retriever, DOC_A_V2, DEPT_A, DOC_A_ID)
    _check(
        "修改内容后入库返回 updated",
        result_a3.status == "updated",
        result_a3.message,
    )
    _check(
        "只停用内容发生变化的旧分块",
        0 < result_a3.deactivated_points <= active_before,
        f"停用 {result_a3.deactivated_points} 条，更新前有效 {active_before} 条"
        "（版本间未改动的分块会复用同一点 ID）",
    )

    inactive_after = _count_points(
        client, collection, dept_id=DEPT_A, doc_id=DOC_A_ID, is_active=False
    )
    active_after = _count_points(
        client, collection, dept_id=DEPT_A, doc_id=DOC_A_ID, is_active=True
    )
    _check(
        "旧版本状态与返回值一致",
        inactive_after == result_a3.deactivated_points,
        f"is_active=False 共 {inactive_after} 条",
    )
    _check(
        "新版本有效点数正确",
        active_after == result_a3.written_points,
        f"is_active=True 共 {active_after} 条",
    )

    hits_a2 = retriever.search("年假有几天", DEPT_A, top_k=5)
    _check(
        "检索只返回新版本内容",
        bool(hits_a2)
        and any("15 天" in (hit.get("parent_text") or "") for hit in hits_a2)
        and all(hit.get("is_active") is True for hit in hits_a2),
        f"{len(hits_a2)} 条，均为 is_active=True",
    )

    # 历史版本仍可显式检索（审计用途）
    old_hits = retriever.search("年假有几天", DEPT_A, top_k=5, is_active=False)
    _check(
        "is_active=False 可检索到历史版本",
        bool(old_hits)
        and all(hit.get("is_active") is False for hit in old_hits)
        and any("10 天" in (hit.get("parent_text") or "") for hit in old_hits),
        f"{len(old_hits)} 条历史版本",
    )


async def _run_agent_checks(full: bool) -> None:
    print("\n" + "=" * 72)
    print("Agent 全链路：真实 LLM")
    print("=" * 72)

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.unlink(missing_ok=True)

    async with agent_graph_lifespan(CHECKPOINT_PATH) as graph:
        # ---------- YES 路径 ----------
        started = time.perf_counter()
        config = build_thread_config(f"smoke-yes-{int(time.time())}")
        state = await graph.ainvoke(
            create_initial_state("入职满一年的员工每年有多少天年假？", DEPT_A),
            config=config,
        )
        elapsed = time.perf_counter() - started

        answer = _last_answer(state)
        _check(
            "YES 路径判定正确",
            state.get("is_relevant") == "YES",
            f"is_relevant={state.get('is_relevant')}, retry={state.get('retry_count')}",
        )
        _check(
            "答案基于新版本资料生成",
            bool(answer) and ("15" in answer or "十五" in answer),
            f"耗时 {elapsed:.1f}s，答案长度 {len(answer)}",
        )
        print(f"       答案预览：{answer[:120].replace(chr(10), ' ')}")

        # ---------- NO 路径（可选，消耗更多 LLM 调用） ----------
        if full:
            started = time.perf_counter()
            config = build_thread_config(f"smoke-no-{int(time.time())}")
            state = await graph.ainvoke(
                create_initial_state("公司股票期权的行权价格是多少？", DEPT_A),
                config=config,
            )
            elapsed = time.perf_counter() - started
            _check(
                "NO 路径耗尽重试后结束",
                state.get("is_relevant") == "NO"
                and state.get("retry_count") == 2,
                f"is_relevant={state.get('is_relevant')}, "
                f"retry={state.get('retry_count')}, 耗时 {elapsed:.1f}s",
            )

    CHECKPOINT_PATH.unlink(missing_ok=True)


def _last_answer(state: dict) -> str:
    messages = state.get("messages") or []
    if not messages:
        return ""
    content = getattr(messages[-1], "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict)
        )
    return str(content)


def main() -> int:
    _configure_stdout()
    parser = argparse.ArgumentParser(description="企业级多租户 RAG 真实环境冒烟测试")
    parser.add_argument(
        "--full",
        action="store_true",
        help="额外执行 NO 路径（重试耗尽），会多消耗 4~5 次 LLM 调用",
    )
    parser.add_argument(
        "--skip-agent",
        action="store_true",
        help="只测数据链路，不调用 LLM",
    )
    args = parser.parse_args()

    client = _prerequisites()
    collection = f"{settings.qdrant_collection_prefix}_chunks"

    # 每次从干净集合开始，保证结果可重复
    if client.collection_exists(collection):
        print(f"\n清理上一次的冒烟集合：{collection}")
        client.delete_collection(collection)

    try:
        retriever = QdrantRetriever()
        _run_data_checks(client, retriever)
        if not args.skip_agent:
            asyncio.run(_run_agent_checks(args.full))
    except SmokeError as exc:
        print("\n" + "=" * 72)
        print(f"冒烟测试失败：{exc}")
        return 1
    finally:
        client.close()

    print("\n" + "=" * 72)
    print("冒烟测试全部通过")
    print(f"冒烟数据保留在集合 {collection}（前缀 rag_smoke），可随时安全删除")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
