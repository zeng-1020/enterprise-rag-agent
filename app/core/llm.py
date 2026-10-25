"""LLM 客户端封装（OpenAI 兼容协议）。

支持通过 ``LLM_BASE_URL`` 接入 DeepSeek、通义千问、智谱、OpenAI 等服务，
通过 ``LLM_MODEL`` / ``LLM_TEMPERATURE`` / ``LLM_MAX_TOKENS`` 控制生成行为。
"""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_openai import ChatOpenAI

from app.config import settings

logger = logging.getLogger(__name__)


class LLMConfigurationError(RuntimeError):
    """LLM 配置缺失时抛出的异常。"""


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    """获取全局 LLM 客户端单例。

    Raises:
        LLMConfigurationError: 未配置 ``LLM_API_KEY``。
    """
    if not settings.llm_configured:
        raise LLMConfigurationError(
            "LLM_API_KEY 未配置，无法调用大模型。"
            "请在 .env 中填写 LLM_API_KEY 后重启服务。"
        )

    llm = ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.llm_api_key.get_secret_value(),
        base_url=settings.llm_base_url,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout,
        max_retries=2,
    )

    logger.info(
        "LLM 客户端初始化完成：model=%s, base_url=%s",
        settings.llm_model,
        settings.llm_base_url,
    )
    return llm
