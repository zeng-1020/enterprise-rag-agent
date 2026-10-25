"""云端 Embedding 服务封装。

设计要点：
1. 基于 ``langchain_openai.OpenAIEmbeddings``，兼容所有 OpenAI 协议的
   Embedding 服务（OpenAI、通义千问、智谱、DeepSeek 等）；
2. 自动按 ``EMBEDDING_BATCH_SIZE`` 分批请求，避免单次请求体过大；
3. 调用失败时自动重试（默认重试 3 次，即最多请求 4 次），
   采用指数退避，并输出包含批次、尝试次数和异常信息的错误日志；
4. 延迟初始化客户端：未配置密钥时在首次调用处给出明确提示，
   避免应用在导入阶段直接崩溃。
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from functools import lru_cache
from typing import TypeVar

from langchain_openai import OpenAIEmbeddings

from app.config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T")


class EmbeddingServiceError(RuntimeError):
    """Embedding 调用在重试耗尽后仍然失败时抛出的业务异常。"""


class EmbeddingConfigurationError(EmbeddingServiceError):
    """Embedding 配置缺失或不合法时抛出的异常。"""


class CloudEmbeddingService:
    """云端 Embedding 封装类。

    Args:
        max_retries: 失败后的重试次数。默认 3，表示首次调用失败后最多再试 3 次，
            即单个批次最多请求 4 次。
        retry_backoff: 首次重试的等待基数（秒），后续按 2 的幂次递增。
        max_backoff: 单次重试等待时间上限（秒），避免退避时间无限增长。
    """

    def __init__(
        self,
        *,
        max_retries: int = 3,
        retry_backoff: float = 1.0,
        max_backoff: float = 8.0,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries 不能为负数")
        if retry_backoff < 0 or max_backoff < 0:
            raise ValueError("retry_backoff 与 max_backoff 不能为负数")

        self._max_retries = max_retries
        self._retry_backoff = retry_backoff
        self._max_backoff = max_backoff
        # 防御性处理：即使配置为 0 或负数，也至少保证每批 1 条
        self._batch_size = max(1, settings.embedding_batch_size)

        # 客户端延迟初始化，并使用锁保证多线程下只创建一次
        self._client: OpenAIEmbeddings | None = None
        self._client_lock = threading.Lock()

    # ------------------------------------------------------------------
    # 客户端初始化
    # ------------------------------------------------------------------
    def _get_client(self) -> OpenAIEmbeddings:
        """获取（并在必要时创建）LangChain Embedding 客户端。"""
        if self._client is not None:
            return self._client

        with self._client_lock:
            # 双重检查，避免并发场景下重复创建客户端
            if self._client is not None:
                return self._client

            if not settings.embedding_configured:
                raise EmbeddingConfigurationError(
                    "EMBEDDING_API_KEY 未配置，无法调用 Embedding 服务。"
                    "请在 .env 中填写 EMBEDDING_API_KEY 后重启服务。"
                )

            self._client = OpenAIEmbeddings(
                model=settings.embedding_model,
                api_key=settings.embedding_api_key.get_secret_value(),
                base_url=settings.embedding_base_url,
                request_timeout=settings.embedding_timeout,
                # 重试完全由本类接管，关闭 SDK 内置重试，
                # 避免两层重试叠加导致实际请求次数不可控
                max_retries=0,
                # 关键兼容项：非 OpenAI 官方服务通常不支持 tiktoken 的上下文
                # 长度预检查，关闭后可避免 tokenizer 相关报错
                check_embedding_ctx_length=False,
                # 与外部批处理保持一致的批大小
                chunk_size=self._batch_size,
            )

            logger.info(
                "Embedding 客户端初始化完成：model=%s, base_url=%s, batch_size=%d, dim=%d",
                settings.embedding_model,
                settings.embedding_base_url,
                self._batch_size,
                settings.embedding_dim,
            )
            return self._client

    # ------------------------------------------------------------------
    # 对外核心方法
    # ------------------------------------------------------------------
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """将一批文本转换为向量。

        Args:
            texts: 待向量化的文本列表，顺序与返回结果严格一一对应。

        Returns:
            与 ``texts`` 等长的向量列表，每个向量长度为 ``EMBEDDING_DIM``。

        Raises:
            ValueError: 输入包含非字符串或空白文本。
            EmbeddingConfigurationError: 未配置 Embedding 密钥。
            EmbeddingServiceError: 重试耗尽后仍调用失败，或返回结果不合法。
        """
        if not texts:
            return []

        self._validate_texts(texts)
        client = self._get_client()

        total_batches = (len(texts) + self._batch_size - 1) // self._batch_size
        all_vectors: list[list[float]] = []

        for batch_index, start in enumerate(
            range(0, len(texts), self._batch_size), start=1
        ):
            batch = texts[start : start + self._batch_size]
            vectors = self._call_with_retry(
                lambda b=batch: client.embed_documents(b),
                description=(
                    f"Embedding 请求（第 {batch_index}/{total_batches} 批，"
                    f"本批 {len(batch)} 条文本）"
                ),
            )
            self._validate_vectors(vectors, expected_count=len(batch))
            all_vectors.extend(vectors)

        return all_vectors

    def embed_query(self, text: str) -> list[float]:
        """将单条查询文本转换为向量（检索阶段使用）。

        与 ``embed_texts`` 共享同样的校验与重试策略。
        """
        self._validate_texts([text])
        client = self._get_client()
        vector = self._call_with_retry(
            lambda: client.embed_query(text),
            description="Embedding 查询向量请求（1 条文本）",
        )
        self._validate_vectors([vector], expected_count=1)
        return vector

    # ------------------------------------------------------------------
    # 内部工具方法
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_texts(texts: list[str]) -> None:
        """校验输入文本，提前拦截无法向量化的内容。"""
        for index, text in enumerate(texts):
            if not isinstance(text, str):
                raise ValueError(
                    f"第 {index} 条输入不是字符串，实际类型：{type(text).__name__}"
                )
            if not text.strip():
                raise ValueError(f"第 {index} 条输入为空字符串，无法向量化")

    def _call_with_retry(
        self,
        func: Callable[[], T],
        *,
        description: str,
    ) -> T:
        """执行一次带指数退避重试的调用。

        Args:
            func: 实际发起请求的无参可调用对象。
            description: 用于日志定位的调用描述（批次、条数等）。

        Returns:
            调用成功后的返回值。

        Raises:
            EmbeddingServiceError: 所有重试均失败后抛出，并保留原始异常链。
        """
        total_attempts = self._max_retries + 1

        for attempt in range(1, total_attempts + 1):
            try:
                result = func()
                if attempt > 1:
                    logger.warning(
                        "%s 在第 %d 次尝试后成功（此前失败 %d 次）",
                        description,
                        attempt,
                        attempt - 1,
                    )
                return result
            except Exception as exc:  # noqa: BLE001 - 需要兜底所有云端调用异常
                if attempt >= total_attempts:
                    logger.error(
                        "%s 调用失败，已重试 %d 次仍未成功：%s: %s",
                        description,
                        self._max_retries,
                        type(exc).__name__,
                        exc,
                        exc_info=True,
                    )
                    raise EmbeddingServiceError(
                        f"{description} 调用失败，已重试 {self._max_retries} 次："
                        f"{type(exc).__name__}: {exc}"
                    ) from exc

                # 指数退避：1x、2x、4x…… 并受 max_backoff 限制
                delay = min(
                    self._retry_backoff * (2 ** (attempt - 1)),
                    self._max_backoff,
                )
                logger.error(
                    "%s 第 %d/%d 次尝试失败：%s: %s；%.1f 秒后进行第 %d 次重试",
                    description,
                    attempt,
                    total_attempts,
                    type(exc).__name__,
                    exc,
                    delay,
                    attempt + 1,
                )
                time.sleep(delay)

        # 理论上不可达：循环内成功会 return，失败会 raise
        raise EmbeddingServiceError(f"{description} 调用失败：未知错误")

    @staticmethod
    def _validate_vectors(
        vectors: list[list[float]],
        *,
        expected_count: int,
    ) -> None:
        """校验模型返回结果的数量和维度，避免脏数据写入 Qdrant。"""
        if len(vectors) != expected_count:
            raise EmbeddingServiceError(
                f"Embedding 返回数量异常：期望 {expected_count} 条，实际 {len(vectors)} 条"
            )

        expected_dim = settings.embedding_dim
        for index, vector in enumerate(vectors):
            if not vector:
                raise EmbeddingServiceError(f"第 {index} 条文本返回了空向量")
            if len(vector) != expected_dim:
                raise EmbeddingServiceError(
                    f"向量维度不匹配：配置 EMBEDDING_DIM={expected_dim}，"
                    f"模型实际返回 {len(vector)} 维。"
                    "请修正 EMBEDDING_DIM，并重建 Qdrant 集合后重新入库。"
                )


@lru_cache(maxsize=1)
def get_embedding_service() -> CloudEmbeddingService:
    """获取 Embedding 服务单例，供各处依赖注入使用。"""
    return CloudEmbeddingService()
