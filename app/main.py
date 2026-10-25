"""FastAPI 应用入口。

职责：
1. 通过 ``lifespan`` 托管 Agent 图与 AsyncSqliteSaver 的异步数据库连接：
   服务启动时创建连接、初始化表结构并编译图，服务关闭时释放连接；
2. 将图实例挂载到 ``app.state``，供各业务接口共享（无需每个请求重复建连）；
3. 提供健康检查接口，用于探活与部署编排。

启动方式：
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse

from app.agent.graph import (
    DEFAULT_CHECKPOINT_DB_PATH,
    agent_graph_lifespan,
)
from app.api.chat import router as chat_router
from app.api.demo import router as demo_router
from app.api.deps import get_agent_graph
from app.api.documents import router as documents_router
from app.config import settings

logger = logging.getLogger(__name__)

# 单文件交互控制台（原生 HTML + Tailwind CDN），与 API 同源，避免跨域
CONSOLE_FILE = Path(__file__).resolve().parent.parent / "index.html"

# get_agent_graph 在这里一并导出，保持 `from app.main import get_agent_graph`
# 的旧引用方式可用；实际实现位于 app.api.deps
__all__ = ["app", "get_agent_graph", "lifespan"]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """应用生命周期管理。

    进入：创建 AsyncSqliteSaver 连接 -> 初始化检查点表 -> 编译 Agent 图；
    退出：``agent_graph_lifespan`` 的 ``async with`` 结束，SQLite 连接自动关闭。

    整个进程共享同一个图实例与连接，避免高并发下反复建立数据库连接。
    """
    # 统一日志格式；否则应用模块的 INFO 日志可能被默认丢弃。
    # 已存在处理器时（如测试环境）basicConfig 不会重复添加。
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    async with agent_graph_lifespan(DEFAULT_CHECKPOINT_DB_PATH) as graph:
        app.state.agent_graph = graph
        logger.info("应用启动完成：%s（环境=%s）", settings.app_name, settings.app_env)
        try:
            yield
        finally:
            # 连接已由上下文管理器关闭，这里只清理引用，防止误用已关闭的图
            app.state.agent_graph = None
            logger.info("应用已关闭，Agent 图与检查点连接释放完成")


app = FastAPI(
    title=settings.app_name,
    debug=settings.debug,
    lifespan=lifespan,
)

# 业务路由：POST /api/chat（SSE）
app.include_router(chat_router)

# 开发调试页：GET /demo（浏览器端 SSE 联调）
app.include_router(demo_router)

# 文档管理：POST /api/docs/upload（PDF 解析 + 入库）
app.include_router(documents_router)


@app.get("/", include_in_schema=False)
async def console() -> FileResponse:
    """返回单文件交互控制台。"""
    if not CONSOLE_FILE.is_file():
        raise HTTPException(status_code=404, detail="控制台页面不存在")
    return FileResponse(CONSOLE_FILE, media_type="text/html")


@app.get("/health", tags=["运维"])
async def health(request: Request) -> dict[str, object]:
    """健康检查：确认服务与 Agent 图均已就绪。"""
    return {
        "status": "ok",
        "app": settings.app_name,
        "env": settings.app_env,
        "agent_ready": getattr(request.app.state, "agent_graph", None) is not None,
    }
