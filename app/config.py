"""全局配置模块。

职责：
1. 使用 pydantic-settings 从环境变量 / `.env` 文件加载配置；
2. 对所有配置项做类型校验，避免程序运行到一半才发现参数错误；
3. 对外暴露单例 `settings`，供其他模块直接导入使用。

加载优先级（从高到低）：
    真实环境变量  >  `.env` 文件  >  代码中的默认值

安全约定：
- 所有密钥统一使用 `SecretStr` 类型，打印配置时不会泄露明文；
- `.env` 文件禁止提交到 Git，仅提交 `.env.example` 作为模板。
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用全局配置。

    字段名会自动映射到同名环境变量（大小写不敏感），
    例如 `qdrant_url` 对应环境变量 `QDRANT_URL`。
    """

    # ------------------------------------------------------------------
    # pydantic-settings 行为配置
    # ------------------------------------------------------------------
    model_config = SettingsConfigDict(
        env_file=".env",           # 从项目根目录的 .env 文件读取配置
        env_file_encoding="utf-8", # .env 文件编码，避免中文注释乱码
        case_sensitive=False,      # 环境变量名大小写不敏感（QDRANT_URL == qdrant_url）
        extra="ignore",            # .env 中存在未声明的变量时忽略，不报错
    )

    # ------------------------------------------------------------------
    # 一、应用基础配置
    # ------------------------------------------------------------------
    app_name: str = Field(
        default="enterprise-rag",
        description="服务名称，用于日志标识、OpenAPI 文档标题和监控指标前缀",
    )
    app_env: Literal["dev", "test", "prod"] = Field(
        default="dev",
        description="运行环境：dev=开发环境，test=测试环境，prod=生产环境",
    )
    debug: bool = Field(
        default=False,
        description="是否开启调试模式；生产环境必须为 false，避免泄露异常堆栈",
    )
    log_level: str = Field(
        default="INFO",
        description="日志级别：DEBUG / INFO / WARNING / ERROR",
    )
    api_prefix: str = Field(
        default="/api/v1",
        description="所有业务接口的统一路由前缀，便于后续版本升级和网关转发",
    )

    # ------------------------------------------------------------------
    # 二、多租户相关配置
    # ------------------------------------------------------------------
    tenant_header: str = Field(
        default="X-Tenant-ID",
        description="HTTP 请求头中用于标识租户的字段名，租户上下文由中间件解析",
    )
    qdrant_collection_prefix: str = Field(
        default="rag",
        description=(
            "Qdrant 集合命名前缀。多租户共用一个集合，"
            "集合名默认为 {前缀}_chunks，通过 payload 中的 dept_id 强制过滤实现隔离"
        ),
    )

    # ------------------------------------------------------------------
    # 三、Qdrant 向量数据库配置
    # ------------------------------------------------------------------
    qdrant_url: str = Field(
        default="http://127.0.0.1:6333",
        description=(
            "Qdrant 服务地址；本地默认暴露 6333 端口。"
            "Windows 下建议用 127.0.0.1 而非 localhost："
            "localhost 会优先解析到 IPv6 ::1，Qdrant 仅监听 IPv4，"
            "每次请求都要等待约 2 秒的连接回退"
        ),
    )
    qdrant_api_key: SecretStr | None = Field(
        default=None,
        description="Qdrant 访问密钥；本地开发可不填，Qdrant Cloud 或开启了鉴权的集群必须填写",
    )
    qdrant_timeout: float = Field(
        default=30.0,
        description="Qdrant 请求超时时间（秒），大批量写入时可适当调大",
    )
    qdrant_prefer_grpc: bool = Field(
        default=False,
        description="是否优先使用 gRPC 协议通信；高并发场景可开启以获得更低延迟",
    )

    # ------------------------------------------------------------------
    # 四、大语言模型（LLM）配置
    # 统一采用 OpenAI 兼容协议，可无缝切换 DeepSeek / 通义千问 / 智谱 / OpenAI 等
    # ------------------------------------------------------------------
    llm_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="LLM 服务的 API Key；留空时调用模型会抛出明确的配置异常",
    )
    llm_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="LLM 服务的 OpenAI 兼容接口地址（通常以 /v1 结尾）",
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="对话模型名称，需与所接入服务商提供的模型标识完全一致",
    )
    llm_temperature: float = Field(
        default=0.1,
        description="采样温度：RAG 问答建议使用 0~0.3，降低随机性、提高事实一致性",
    )
    llm_max_tokens: int = Field(
        default=2048,
        description="单次生成的最大 Token 数，用于控制响应长度和调用成本",
    )
    llm_timeout: float = Field(
        default=60.0,
        description="LLM 请求超时时间（秒），长文本生成场景可适当调大",
    )

    # ------------------------------------------------------------------
    # 五、Embedding 向量化模型配置
    # 可与 LLM 使用不同的服务商和密钥，因此单独配置
    # ------------------------------------------------------------------
    embedding_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="Embedding 服务的 API Key；留空时写入/检索文档会抛出明确的配置异常",
    )
    embedding_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="Embedding 服务的 OpenAI 兼容接口地址（通常以 /v1 结尾）",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="向量化模型名称，需与所接入服务商提供的模型标识一致",
    )
    embedding_dim: int = Field(
        default=1024,
        description=(
            "向量维度，必须与云端模型实际输出维度一致（当前模型为 1024 维）；"
            "创建 Qdrant 集合时使用该值，且同一集合内不可变更"
        ),
    )
    embedding_batch_size: int = Field(
        default=64,
        description="调用 Embedding 接口时每批处理的文本条数，用于平衡吞吐与请求大小限制",
    )
    embedding_timeout: float = Field(
        default=60.0,
        description="Embedding 请求超时时间（秒）",
    )

    # ------------------------------------------------------------------
    # 六、RAG 链路参数
    # ------------------------------------------------------------------
    chunk_size: int = Field(
        default=800,
        description="文档切分后的单块最大字符数；过大会稀释语义，过小会丢失上下文",
    )
    chunk_overlap: int = Field(
        default=120,
        description="相邻文本块的重叠字符数，用于避免关键信息被切断在块边界",
    )
    vector_top_k: int = Field(
        default=10,
        description="向量检索阶段召回的候选文档数量",
    )
    bm25_top_k: int = Field(
        default=10,
        description="BM25 关键词检索阶段召回的候选文档数量",
    )
    final_top_k: int = Field(
        default=5,
        description="融合、重排后最终送入 LLM 上下文窗口的文档数量",
    )
    rrf_k: int = Field(
        default=60,
        description="RRF（倒数排名融合）算法中的平滑常数 k，值越大各排名差距越平缓",
    )

    # ------------------------------------------------------------------
    # 七、校验逻辑
    # ------------------------------------------------------------------
    @field_validator("qdrant_url", "llm_base_url", "embedding_base_url")
    @classmethod
    def _normalize_url(cls, value: str) -> str:
        """去掉 URL 末尾多余的斜杠，避免拼接路径时出现双斜杠。"""
        return value.strip().rstrip("/")

    @field_validator("log_level")
    @classmethod
    def _normalize_log_level(cls, value: str) -> str:
        """将日志级别统一转为大写，兼容 info / INFO 等写法。"""
        level = value.strip().upper()
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if level not in allowed:
            raise ValueError(f"LOG_LEVEL 必须是 {allowed} 之一，当前值：{value}")
        return level

    @model_validator(mode="after")
    def _validate_rag_params(self) -> "Settings":
        """跨字段校验，确保 RAG 参数组合在业务上成立。"""
        # 重叠长度必须小于块大小，否则切分会陷入死循环或产生异常块
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP 必须小于 CHUNK_SIZE")
        # 最终召回数不能超过两路召回候选数之和，否则配置没有实际意义
        if self.final_top_k > self.vector_top_k + self.bm25_top_k:
            raise ValueError(
                "FINAL_TOP_K 不能大于 VECTOR_TOP_K 与 BM25_TOP_K 之和"
            )
        return self

    # ------------------------------------------------------------------
    # 八、便捷属性：供业务代码做启动前检查
    # ------------------------------------------------------------------
    @property
    def llm_configured(self) -> bool:
        """LLM 是否已完成必要配置（密钥非空）。"""
        return bool(self.llm_api_key.get_secret_value().strip())

    @property
    def embedding_configured(self) -> bool:
        """Embedding 模型是否已完成必要配置（密钥非空）。"""
        return bool(self.embedding_api_key.get_secret_value().strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """获取配置单例。

    使用 lru_cache 保证进程内只解析一次 .env，
    避免在高频请求路径上反复读取磁盘文件。
    """
    return Settings()


# 模块级单例，业务代码可直接 `from app.config import settings` 使用
settings = get_settings()
