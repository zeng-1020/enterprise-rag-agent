"""LangGraph 状态定义。

状态在节点之间流转，每个节点只需返回自己更新的字段，
LangGraph 会自动合并到全局状态中。

字段说明：
- ``messages``: 对话消息列表，使用 ``add_messages`` reducer 实现追加合并；
- ``question``: 用户当前问题；
- ``retrieval_query``: 当前检索使用的查询；首轮等于 question，重写后更新；
- ``dept_id``: 部门 ID，贯穿检索链路，用于强制多租户过滤；
- ``sub_chunks``: 向量检索命中的子块（含 payload）；
- ``parent_contexts``: 去重后的父块文本，作为生成阶段的上下文；
- ``is_relevant``: 相关性判定结果，只允许 YES / NO；
- ``retry_count``: 检索/评估的重试次数，初始为 0，供后续条件边控制循环。
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """RAG Agent 的全局状态。"""

    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    retrieval_query: str
    dept_id: str
    sub_chunks: list[dict[str, Any]]
    parent_contexts: list[str]
    is_relevant: Literal["YES", "NO"]
    retry_count: int


def create_initial_state(question: str, dept_id: str) -> AgentState:
    """构造图执行所需的初始状态。

    TypedDict 本身不支持字段默认值，因此通过工厂函数统一初始化各字段：
    ``retry_count=0``、``is_relevant="NO"``、检索结果为空。

    Args:
        question: 用户问题。
        dept_id: 部门 ID，检索时用于强制过滤。

    Returns:
        初始化完成的 AgentState。

    Raises:
        ValueError: question 或 dept_id 为空。
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question 不能为空")
    if not isinstance(dept_id, str) or not dept_id.strip():
        raise ValueError("dept_id 不能为空")

    normalized_question = question.strip()
    normalized_dept_id = dept_id.strip()

    state: AgentState = {
        # 初始消息包含本轮用户问题，generate_node 会把模型回答追加进来
        "messages": [HumanMessage(content=normalized_question)],
        "question": normalized_question,
        # 首轮检索直接使用原始问题；rewrite_node 会在此基础上更新
        "retrieval_query": normalized_question,
        "dept_id": normalized_dept_id,
        "sub_chunks": [],
        "parent_contexts": [],
        # 初始视为不相关，只有评估节点输出 YES 后才进入生成阶段
        "is_relevant": "NO",
        "retry_count": 0,
    }
    return state
