"""LangGraph 节点实现：检索、相关性评估、查询重写、答案生成。

Prompt 与 LLM 调用的组织方式：
- Prompt 统一集中在 ``build_evaluate_messages``、``build_generate_messages``
  和 ``build_rewrite_messages``，并已用 TODO 标注，后续替换 Prompt
  不需要修改节点逻辑；
- 所有节点均为 ``async def``：评估/重写使用 ``await llm.ainvoke(...)``，
  生成节点使用 ``llm.astream(...)`` 逐 token 消费，配合 ``astream_events``
  即可向前端转发流式答案；
- 检索器目前是同步实现，通过 ``asyncio.to_thread`` 卸载到线程池，
  避免 Embedding / Qdrant 的网络等待阻塞事件循环。
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Literal

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

from app.agent.state import AgentState
from app.config import settings
from app.core.llm import get_llm
from app.core.retriever import get_retriever

logger = logging.getLogger(__name__)


# =====================================================================
# Prompt 占位区
# ---------------------------------------------------------------------
# TODO(user): 请按业务需要替换下面两段 Prompt。
# 当前内容是可运行的最小模板，替换后无需改动任何节点逻辑。
# =====================================================================

# 相关性评估 Prompt：要求模型只输出 YES 或 NO
EVALUATE_SYSTEM_PROMPT = """你的任务是判断“候选资料”是否包含回答“用户问题”所需的信息。
判断标准：
1. 只要候选资料中存在与问题直接相关、可用于作答的信息，就判定为 YES；
2. 如果候选资料与问题无关，或信息不足以回答问题，判定为 NO；
3. 只能输出 YES 或 NO，不要输出任何解释或其他内容。"""

# 答案生成 Prompt：要求只依据父块上下文回答
GENERATE_SYSTEM_PROMPT = """你是一个严谨的企业知识库问答助手。
请严格依据“参考资料”回答用户问题：
1. 只能使用参考资料中的信息，不得编造或引入外部知识；
2. 引用信息时在句末标注对应编号，例如 [1][2]；
3. 如果参考资料不足以回答问题，请明确说明“根据现有资料无法回答”，并指出缺少的信息；
4. 使用与用户问题相同的语言，表达清晰、简洁。"""

# 查询重写 Prompt：评估为不相关时，用 LLM 改写检索查询
REWRITE_SYSTEM_PROMPT = """你是企业知识库的检索查询改写助手。
上一次检索未能找到相关资料，请把用户问题改写为更适合向量检索的查询：
1. 保留原始问题的核心意图和关键实体；
2. 补充同义词、相关术语或更具体的表述；
3. 不要回答问题，不要输出解释；
4. 只输出一条改写后的查询文本。"""


# =====================================================================
# 节点 1：检索
# =====================================================================
async def retrieve_node(state: AgentState) -> dict[str, Any]:
    """调用 QdrantRetriever 检索子块，并汇总去重后的父块上下文。

    返回：
    - ``sub_chunks``: 命中的子块列表（含 payload 和 score）；
    - ``parent_contexts``: 按命中顺序去重后的父块文本列表。
    """
    question = state["question"]
    dept_id = state["dept_id"]
    retry_count = state.get("retry_count", 0)

    # 首轮检索使用原始问题；触发重写后使用 retrieval_query。
    # 这样既能优化检索效果，又不会改变最终用于生成答案的 question。
    query = state.get("retrieval_query") if retry_count > 0 else question
    if not isinstance(query, str) or not query.strip():
        query = question

    retriever = get_retriever()
    # 检索器当前为同步实现（Embedding + Qdrant 均为阻塞调用），
    # 卸载到线程池执行，避免在长链路、高延迟、高并发下阻塞事件循环。
    sub_chunks = await asyncio.to_thread(
        retriever.search,
        query=query,
        dept_id=dept_id,
        top_k=settings.vector_top_k,
    )
    parent_contexts = _collect_parent_contexts(sub_chunks)

    logger.info(
        "检索完成：dept_id=%s, retry=%d, 子块=%d, 去重父块=%d",
        dept_id,
        retry_count,
        len(sub_chunks),
        len(parent_contexts),
    )

    return {
        "sub_chunks": sub_chunks,
        "parent_contexts": parent_contexts,
    }


# =====================================================================
# 节点 2：相关性评估
# =====================================================================
async def evaluate_node(state: AgentState) -> dict[str, Any]:
    """把 question 和 sub_chunks 交给 LLM，判断资料是否相关。

    返回：
    - ``is_relevant``: "YES" 或 "NO"；
    - 不向 messages 追加评估结果，避免内部判定污染对话上下文。
    """
    question = state["question"]
    sub_chunks = state.get("sub_chunks", [])

    # 没有候选资料时无需调用 LLM，直接判定为不相关
    if not sub_chunks:
        logger.info("未检索到子块，直接判定为 NO")
        return {"is_relevant": "NO"}

    llm = get_llm()
    messages = build_evaluate_messages(question, sub_chunks)

    # LLM 调用骨架：替换 Prompt 后此处无需修改
    response = await llm.ainvoke(messages)
    verdict = _parse_yes_no(_content_to_text(response.content))

    logger.info(
        "相关性评估完成：dept_id=%s, 判定=%s",
        state["dept_id"],
        verdict,
    )
    return {"is_relevant": verdict}


# =====================================================================
# 节点 3：查询重写
# =====================================================================
async def rewrite_node(state: AgentState) -> dict[str, Any]:
    """评估为 NO 时，用 LLM 重写检索查询，并累加重试次数。

    返回：
    - ``retrieval_query``: 改写后的查询，供下一轮 retrieve_node 使用；
    - ``retry_count``: 在原有基础上 +1。

    注意：不修改 ``question``，保证生成阶段仍围绕用户的原始问题作答。
    """
    question = state["question"]
    retry_count = state.get("retry_count", 0)
    current_query = state.get("retrieval_query") or question

    llm = get_llm()
    messages = build_rewrite_messages(question, current_query)

    # LLM 调用骨架：替换 Prompt 后此处无需修改
    response = await llm.ainvoke(messages)
    rewritten_query = _clean_query(_content_to_text(response.content))

    # 模型返回空内容时沿用上一次查询，且仍然累加重试次数，避免无限循环
    if not rewritten_query:
        logger.warning(
            "查询改写结果为空，沿用上一次查询：%r", current_query
        )
        rewritten_query = current_query

    logger.info(
        "查询重写完成：dept_id=%s, retry %d -> %d",
        state["dept_id"],
        retry_count,
        retry_count + 1,
    )

    return {
        "retrieval_query": rewritten_query,
        "retry_count": retry_count + 1,
    }


# =====================================================================
# 节点 4：答案生成
# =====================================================================
async def generate_node(state: AgentState) -> dict[str, Any]:
    """拼接 parent_contexts 和 question，流式生成最终答案。

    通过 ``llm.astream`` 逐 token 消费，LangGraph 的 ``astream_events``
    会为每个 token 发出 ``on_chat_model_stream`` 事件，供 SSE 层转发。

    返回：
    - ``messages``: 追加一条合并后的 AIMessageChunk，最终答案可通过
      ``state["messages"][-1].content`` 读取。
    """
    question = state["question"]
    parent_contexts = state.get("parent_contexts", [])

    llm = get_llm()
    messages = build_generate_messages(question, parent_contexts)

    # 累加流式 chunk：AIMessageChunk 支持合并（含 usage_metadata 等字段）
    accumulated = None
    async for chunk in llm.astream(messages):
        accumulated = chunk if accumulated is None else accumulated + chunk

    if accumulated is None:
        # 理论上流式调用至少返回一个 chunk；兜底避免写入空对象
        accumulated = AIMessage(content="")

    answer = _content_to_text(accumulated.content)

    logger.info(
        "答案生成完成：dept_id=%s, 上下文父块=%d, 答案长度=%d",
        state["dept_id"],
        len(parent_contexts),
        len(answer),
    )

    # add_messages reducer 会将消息追加到现有消息列表
    return {"messages": [accumulated]}


# =====================================================================
# Prompt 构建（占位，供后续替换）
# =====================================================================
def build_evaluate_messages(
    question: str,
    sub_chunks: list[dict[str, Any]],
) -> list[BaseMessage]:
    """构建相关性评估消息。

    TODO(user): 在这里调整评估 Prompt、判断标准或输出格式约束。
    """
    context = _format_sub_chunks(sub_chunks)
    return [
        SystemMessage(content=EVALUATE_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"用户问题：\n{question}\n\n"
                f"候选资料：\n{context}\n\n"
                "请只回答 YES 或 NO。"
            )
        ),
    ]


def build_generate_messages(
    question: str,
    parent_contexts: list[str],
) -> list[BaseMessage]:
    """构建答案生成消息。

    TODO(user): 在这里调整生成 Prompt、引用格式或拒答策略。
    """
    context = _format_parent_contexts(parent_contexts)
    return [
        SystemMessage(content=GENERATE_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"参考资料：\n{context}\n\n"
                f"用户问题：\n{question}\n\n"
                "请基于参考资料作答。"
            )
        ),
    ]


def build_rewrite_messages(
    question: str,
    current_query: str,
) -> list[BaseMessage]:
    """构建查询重写消息。

    TODO(user): 在这里调整重写 Prompt 或输出格式约束。
    """
    return [
        SystemMessage(content=REWRITE_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"原始问题：\n{question}\n\n"
                f"上一次检索查询：\n{current_query}\n\n"
                "请输出改写后的检索查询："
            )
        ),
    ]


# =====================================================================
# 内部工具函数
# =====================================================================
def _collect_parent_contexts(sub_chunks: list[dict[str, Any]]) -> list[str]:
    """从子块中提取父块文本，并按 parent_id 去重。

    多个子块可能来自同一个父块，生成阶段只需保留一份父块上下文，
    避免重复内容挤占 LLM 上下文窗口。
    """
    seen: set[str] = set()
    contexts: list[str] = []

    for chunk in sub_chunks:
        parent_text = chunk.get("parent_text")
        if not isinstance(parent_text, str) or not parent_text.strip():
            continue

        parent_id = chunk.get("parent_id")
        key = (
            parent_id
            if isinstance(parent_id, str) and parent_id
            else parent_text
        )
        if key in seen:
            continue

        seen.add(key)
        contexts.append(parent_text)

    return contexts


def _format_sub_chunks(sub_chunks: list[dict[str, Any]]) -> str:
    """将子块格式化为带编号的文本，供评估 Prompt 使用。"""
    if not sub_chunks:
        return "（未检索到候选资料）"

    parts: list[str] = []
    for index, chunk in enumerate(sub_chunks, start=1):
        text = chunk.get("text", "")
        source = chunk.get("doc_id") or chunk.get("parent_id") or f"chunk-{index}"
        parts.append(f"[{index}] 来源：{source}\n{text}")

    return "\n\n".join(parts)


def _format_parent_contexts(parent_contexts: list[str]) -> str:
    """将父块上下文格式化为带编号的文本，供生成 Prompt 使用。"""
    if not parent_contexts:
        return "（未检索到相关资料）"

    return "\n\n".join(
        f"[{index}] {context}"
        for index, context in enumerate(parent_contexts, start=1)
    )


def _content_to_text(content: Any) -> str:
    """兼容不同模型返回的消息内容格式（字符串或多模态列表）。"""
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


def _clean_query(text: str) -> str:
    """清理重写节点的模型输出，只保留一条干净的查询文本。

    处理模型常见的多余输出：多行解释、``查询：`` / ``Query:`` 前缀、引号。
    """
    lines = (text or "").strip().splitlines()
    if not lines:
        return ""

    first_line = lines[0].strip()
    first_line = re.sub(
        r"^(改写后(?:的)?查询|检索查询|查询|query)\s*[:：]\s*",
        "",
        first_line,
        flags=re.IGNORECASE,
    )
    return first_line.strip().strip("\"'`").strip()


def _parse_yes_no(text: str) -> Literal["YES", "NO"]:
    """解析模型输出为 YES / NO。

    策略：
    1. 优先匹配开头的 YES / NO；
    2. 其次匹配文本中独立出现的 YES / NO；
    3. 无法解析时保守判定为 NO，并记录告警。
    """
    normalized = (text or "").strip().upper()

    if normalized.startswith("YES"):
        return "YES"
    if normalized.startswith("NO"):
        return "NO"

    match = re.search(r"\b(YES|NO)\b", normalized)
    if match:
        return "YES" if match.group(1) == "YES" else "NO"

    logger.warning(
        "无法解析相关性判定结果，按 NO 处理：%r",
        (text or "")[:100],
    )
    return "NO"
