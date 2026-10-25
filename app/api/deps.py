"""API 层公共依赖。"""

from __future__ import annotations

from fastapi import HTTPException, Request
from langgraph.graph.state import CompiledStateGraph


def get_agent_graph(request: Request) -> CompiledStateGraph:
    """从应用状态中获取 lifespan 托管的 Agent 图。

    图与 AsyncSqliteSaver 连接由 ``app.main.lifespan`` 创建和关闭，
    这里只读取引用，避免在每个请求中重复建连或编译图。

    Raises:
        HTTPException: 服务尚未完成启动（或已关闭）时返回 503。
    """
    graph = getattr(request.app.state, "agent_graph", None)
    if graph is None:
        raise HTTPException(status_code=503, detail="Agent 图尚未就绪")
    return graph
