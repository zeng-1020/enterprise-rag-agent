"""LangGraph 图组装与异步状态持久化。

图结构：
    START -> retrieve -> evaluate -> 条件路由
                                      ├─ YES                 -> generate -> END
                                      ├─ NO 且 retry < 2     -> rewrite -> retrieve
                                      └─ NO 且 retry >= 2    -> END

使用方式（由 FastAPI lifespan 托管连接）：
    from app.agent.graph import agent_graph_lifespan

    async with agent_graph_lifespan() as graph:
        config = build_thread_config("conversation-123")
        result = await graph.ainvoke(
            create_initial_state("用户问题", "dept_A"),
            config=config,
        )

注意：
- thread_id 通过运行配置传入（``configurable.thread_id``），
  同一个 thread_id 会复用并持久化同一份会话状态；
- 本图使用 ``AsyncSqliteSaver`` + 异步节点，支持 ``ainvoke`` / ``astream``；
  ``AsyncSqliteSaver`` 的连接必须由 ``async with`` 管理，因此不再提供全局单例，
  应由 FastAPI 的 lifespan 统一创建与关闭；
- 重试次数耗尽后图直接结束，不会生成答案；
  调用方应检查 ``is_relevant == "NO"`` 并返回兜底提示；
- SQLite 适合单实例部署；多实例生产环境建议替换为 PostgresSaver。
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agent.nodes import (
    evaluate_node,
    generate_node,
    retrieve_node,
    rewrite_node,
)
from app.agent.state import AgentState

logger = logging.getLogger(__name__)

# 最大重写/重试次数。达到该值后仍不相关，则结束图执行。
MAX_RETRIES = 2

# 默认 SQLite 检查点文件路径（相对于服务启动目录）
DEFAULT_CHECKPOINT_DB_PATH = Path("data") / "agent_checkpoints.sqlite"

__all__ = [
    "DEFAULT_CHECKPOINT_DB_PATH",
    "MAX_RETRIES",
    "agent_graph_lifespan",
    "build_agent_graph",
    "build_thread_config",
    "create_async_sqlite_checkpointer",
    "route_after_evaluate",
]


def route_after_evaluate(state: AgentState) -> str:
    """评估后的条件路由。

    Returns:
        - ``"generate"``: 资料相关，进入答案生成；
        - ``"rewrite"``: 不相关且仍有重试额度，重写查询后重新检索；
        - ``"end"``: 不相关且重试耗尽，结束图执行。
    """
    if state.get("is_relevant") == "YES":
        return "generate"

    if state.get("retry_count", 0) < MAX_RETRIES:
        return "rewrite"

    logger.info(
        "重试次数已耗尽（retry_count=%s），结束图执行",
        state.get("retry_count", 0),
    )
    return "end"


def build_agent_graph(checkpointer: BaseCheckpointSaver) -> CompiledStateGraph:
    """组装并编译异步 RAG Agent 图。

    Args:
        checkpointer: 异步检查点存储，通常由
            :func:`create_async_sqlite_checkpointer` 提供。
            必须是支持 async 方法的实现（如 ``AsyncSqliteSaver``、
            ``InMemorySaver``），同步 ``SqliteSaver`` 会导致 ``ainvoke`` 报错。

    Returns:
        编译后的 LangGraph 图。

    注意：
        使用 Checkpointer 后，调用 ``ainvoke`` / ``astream`` 时必须传入
        ``config=build_thread_config(thread_id)``，否则会因缺少 thread_id 报错。
    """
    if checkpointer is None:
        raise ValueError(
            "checkpointer 不能为空；请使用 create_async_sqlite_checkpointer "
            "创建异步检查点存储"
        )

    builder = StateGraph(AgentState)

    builder.add_node("retrieve", retrieve_node)
    builder.add_node("evaluate", evaluate_node)
    builder.add_node("rewrite", rewrite_node)
    builder.add_node("generate", generate_node)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "evaluate")

    # 条件路由：YES -> generate；NO -> rewrite 或结束
    builder.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {
            "generate": "generate",
            "rewrite": "rewrite",
            "end": END,
        },
    )

    # 重写查询后回到检索节点，形成有限重试循环
    builder.add_edge("rewrite", "retrieve")
    builder.add_edge("generate", END)

    graph = builder.compile(checkpointer=checkpointer)

    logger.info("RAG Agent 图编译完成：MAX_RETRIES=%d", MAX_RETRIES)
    return graph


@asynccontextmanager
async def create_async_sqlite_checkpointer(
    db_path: str | Path = DEFAULT_CHECKPOINT_DB_PATH,
) -> AsyncIterator[AsyncSqliteSaver]:
    """创建并托管异步 SQLite 检查点存储。

    - 自动创建父目录；
    - 使用 ``aiosqlite`` 异步连接，避免阻塞事件循环；
    - ``setup()`` 内部会开启 WAL 模式并创建 checkpoints / writes 表；
    - 连接的生命周期与 ``async with`` 代码块一致，退出时自动关闭。

    Args:
        db_path: SQLite 文件路径。

    Yields:
        已初始化表结构的 ``AsyncSqliteSaver``。
    """
    path = Path(db_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    async with AsyncSqliteSaver.from_conn_string(str(path)) as checkpointer:
        # setup() 幂等：已建表时直接返回，同时确保 WAL 模式生效
        await checkpointer.setup()
        logger.info("AsyncSqliteSaver 初始化完成：%s", path.resolve())
        try:
            yield checkpointer
        finally:
            logger.info("AsyncSqliteSaver 连接已释放：%s", path.resolve())


@asynccontextmanager
async def agent_graph_lifespan(
    db_path: str | Path | None = None,
) -> AsyncIterator[CompiledStateGraph]:
    """托管 Agent 图与检查点连接的异步上下文。

    专为 FastAPI ``lifespan`` 设计：进入时创建连接并编译图，
    退出时释放连接。并发请求共享同一个图实例与 SQLite 连接，
    ``AsyncSqliteSaver`` 内部通过锁串行化写操作。

    Args:
        db_path: SQLite 检查点路径；默认 ``data/agent_checkpoints.sqlite``。

    Yields:
        可直接 ``await graph.ainvoke(...)`` 的已编译图。
    """
    checkpoint_path = (
        Path(db_path).expanduser() if db_path is not None else DEFAULT_CHECKPOINT_DB_PATH
    )

    async with create_async_sqlite_checkpointer(checkpoint_path) as checkpointer:
        graph = build_agent_graph(checkpointer)
        logger.info("Agent 图已就绪，检查点文件：%s", checkpoint_path.resolve())
        yield graph


def build_thread_config(thread_id: str) -> dict[str, Any]:
    """构造 LangGraph 运行配置。

    Args:
        thread_id: 会话/线程 ID，用于隔离并持久化不同用户或会话的状态。

    Returns:
        可直接传给 ``graph.ainvoke(..., config=...)`` 的配置字典。

    Raises:
        ValueError: thread_id 为空或类型错误。
    """
    if not isinstance(thread_id, str):
        raise ValueError(
            f"thread_id 必须是字符串，实际类型：{type(thread_id).__name__}"
        )

    normalized = thread_id.strip()
    if not normalized:
        raise ValueError("thread_id 不能为空")

    return {"configurable": {"thread_id": normalized}}
