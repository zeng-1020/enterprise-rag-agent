"""SSE 流式对话接口。

协议（每个 SSE 事件的 ``data`` 均为 JSON，UTF-8 编码）：

- ``status``: 内部流程状态
  ``{"node": "retrieve", "message": "正在检索知识库…"}``
- ``answer``: 大模型输出增量
  ``{"delta": "年假"}``；前端按顺序拼接即可得到完整答案
- ``done``: 本轮结束
  ``{"thread_id": "...", "dept_id": "...", "is_relevant": "YES", "retry_count": 0}``
- ``error``: 服务端异常
  ``{"message": "服务处理失败，请稍后重试"}``

说明：
- ``thread_id`` 通过 ``configurable.thread_id`` 传给 Checkpointer，
  同一 thread_id 的多轮请求会复用同一份持久化状态；
- 事件由 LangGraph ``astream_events`` 驱动：节点开始/结束时推送 status，
  generate 节点调用 ``llm.astream``，其 token 增量推送为 answer。
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field, field_validator
from sse_starlette.sse import EventSourceResponse

from app.agent.graph import build_thread_config
from app.agent.state import AgentState, create_initial_state
from app.api.deps import get_agent_graph

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["对话"])

# 节点开始时的状态提示
NODE_START_STATUS = {
    "retrieve": "正在检索知识库…",
    "evaluate": "正在评估检索结果的相关性…",
    "generate": "正在生成答案…",
}

# rewrite 节点带重试序号，单独处理
REWRITE_NODE = "rewrite"

_NODE_NAMES = {"retrieve", "evaluate", "rewrite", "generate"}


class ChatRequest(BaseModel):
    """对话请求体。"""

    dept_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="部门 ID，用于强制多租户数据隔离",
    )
    thread_id: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="会话/线程 ID，作为 Checkpointer 的 thread_id",
    )
    question: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="用户问题",
    )

    @field_validator("dept_id", "thread_id", "question")
    @classmethod
    def _strip_and_validate(cls, value: str) -> str:
        """去掉首尾空白，并拒绝纯空白字符串。"""
        normalized = value.strip()
        if not normalized:
            raise ValueError("不能为空或仅包含空白字符")
        return normalized


@router.post(
    "/chat",
    summary="流式问答（SSE）",
    response_description="text/event-stream",
)
async def chat(
    payload: ChatRequest,
    graph: CompiledStateGraph = Depends(get_agent_graph),
) -> EventSourceResponse:
    """以 Server-Sent Events 流式返回问答过程与最终答案。

    使用 ``POST`` 而非浏览器原生 ``EventSource``（其只支持 GET），
    前端应通过 ``fetch`` + ``ReadableStream`` 消费。
    """
    # thread_id 是 Checkpointer 的状态主键，必须随每次调用传入
    config = build_thread_config(payload.thread_id)
    initial_state = create_initial_state(payload.question, payload.dept_id)

    return EventSourceResponse(
        _stream_chat(graph, initial_state, config),
        # 定期发送注释帧，避免代理/网关在长 LLM 调用期间断开连接
        ping=15,
    )


async def _stream_chat(
    graph: CompiledStateGraph,
    initial_state: AgentState,
    config: dict[str, Any],
) -> AsyncIterator[dict[str, str]]:
    """消费 ``astream_events``，转换为 status / answer / done / error 事件。"""
    answer_started = False
    rewrite_attempt = 0

    try:
        yield _sse(
            "status",
            {"node": "start", "message": "已连接，正在处理问题…"},
        )

        async for event in graph.astream_events(
            initial_state,
            config=config,
            version="v2",
        ):
            kind = event.get("event")
            metadata = event.get("metadata") or {}
            node = metadata.get("langgraph_node")
            event_name = event.get("name")

            # ---------- 节点开始：检索 / 评估 / 重写 / 生成 ----------
            if kind == "on_chain_start" and node in _NODE_NAMES:
                if event_name != node:
                    continue

                if node == REWRITE_NODE:
                    rewrite_attempt += 1
                    message = (
                        f"评估不相关，正在进行第 {rewrite_attempt} 次重试，"
                        "并优化检索查询…"
                    )
                else:
                    message = NODE_START_STATUS[node]

                yield _sse("status", {"node": node, "message": message})

            # ---------- 评估结束：告知评估结论 ----------
            elif kind == "on_chain_end" and node == "evaluate":
                if event_name != node:
                    continue

                output = (event.get("data") or {}).get("output")
                verdict = (
                    output.get("is_relevant") if isinstance(output, dict) else None
                )
                if verdict == "YES":
                    yield _sse(
                        "status",
                        {
                            "node": "evaluate",
                            "message": "检索结果与问题相关，准备生成答案…",
                        },
                    )
                elif verdict == "NO":
                    yield _sse(
                        "status",
                        {
                            "node": "evaluate",
                            "message": "未找到足够相关的资料",
                        },
                    )

            # ---------- 生成阶段 token 增量 ----------
            elif kind == "on_chat_model_stream" and node == "generate":
                text = _extract_text((event.get("data") or {}).get("chunk"))
                if text:
                    answer_started = True
                    yield _sse("answer", {"delta": text})

            # ---------- 兜底：模型未产生流式 token 时一次性推送 ----------
            elif (
                kind == "on_chat_model_end"
                and node == "generate"
                and not answer_started
            ):
                text = _extract_text((event.get("data") or {}).get("output"))
                if text:
                    answer_started = True
                    yield _sse("answer", {"delta": text})

        # 图执行完成后读取最终状态（连接由 lifespan 托管，仍在同一事件循环内）
        snapshot = await graph.aget_state(config)
        values = snapshot.values or {}
        verdict = values.get("is_relevant", "NO")
        retry_count = values.get("retry_count", rewrite_attempt)

        if verdict == "NO" and not answer_started:
            yield _sse(
                "status",
                {
                    "node": "end",
                    "message": "重试次数已耗尽，未能找到相关资料",
                },
            )

        yield _sse(
            "done",
            {
                "thread_id": config["configurable"]["thread_id"],
                "dept_id": values.get("dept_id", initial_state["dept_id"]),
                "is_relevant": verdict,
                "retry_count": retry_count,
            },
        )

    except asyncio.CancelledError:
        # 客户端主动断开连接，属于正常结束，不记录为异常
        logger.info("SSE 客户端已断开，终止本轮 Agent 执行")
        raise
    except Exception:
        logger.exception("SSE 对话处理失败")
        yield _sse("error", {"message": "服务处理失败，请稍后重试"})


def _sse(event: str, data: dict[str, Any]) -> dict[str, str]:
    """构造 sse-starlette 可识别的事件字典。"""
    return {
        "event": event,
        "data": json.dumps(data, ensure_ascii=False, separators=(",", ":")),
    }


def _extract_text(message: Any) -> str:
    """从 BaseMessage / AIMessageChunk 中提取纯文本。

    兼容字符串内容与多模态内容块（``[{"type": "text", "text": ...}]``）。
    """
    if message is None:
        return ""

    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "".join(parts)

    return str(content) if content is not None else ""
