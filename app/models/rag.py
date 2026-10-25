"""RAG 链路的数据模型。

使用 dataclass 而非 Pydantic 模型，原因：
- 这些对象只在内部服务层流转，不需要额外的序列化开销；
- 后续 API 层可以直接映射为 Pydantic 响应模型。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(slots=True)
class DocumentChunk:
    """待写入向量库的文档块。"""

    chunk_id: str                       # 块唯一 ID（由租户、文档、块序号确定性生成）
    tenant_id: str                      # 所属租户，用于审计和兜底校验
    doc_id: str                         # 原始文档 ID
    chunk_index: int                    # 块在文档中的序号，从 0 开始
    content: str                        # 块文本内容
    metadata: dict[str, Any] = field(default_factory=dict)  # 文档元数据（必须可 JSON 序列化）


@dataclass(slots=True)
class DocumentUpsertResult:
    """文档入库的版本控制结果。

    Attributes:
        doc_id: 文档的稳定业务标识（同一文档的多个版本共用）。
        doc_hash: 本次上传内容的哈希值。
        status:
            - ``created``: 首次入库；
            - ``updated``: 内容有变化，已软删除旧版本并写入新版本；
            - ``unchanged``: 哈希与当前有效版本一致，未写入。
        written_points: 本次写入的分块（点）数量。
        deactivated_points: 被软删除（``is_active=False``）的旧版本分块数量。
    """

    doc_id: str
    doc_hash: str
    status: Literal["created", "updated", "unchanged"]
    written_points: int
    deactivated_points: int

    @property
    def changed(self) -> bool:
        """内容是否发生了变化（首次入库视为发生变化）。"""
        return self.status != "unchanged"

    @property
    def message(self) -> str:
        """可直接返回给上传接口用户的中文提示。"""
        if self.status == "unchanged":
            return "文档内容未修改，已跳过入库"
        if self.status == "updated":
            return (
                f"文档已更新：停用旧版本 {self.deactivated_points} 个分块，"
                f"写入新版本 {self.written_points} 个分块"
            )
        return f"文档首次入库：写入 {self.written_points} 个分块"


@dataclass(slots=True)
class RetrievedChunk:
    """检索阶段返回的文档块。"""

    chunk_id: str
    doc_id: str
    chunk_index: int
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0                  # 向量相似度或 RRF 融合分数
    sources: tuple[str, ...] = ()       # 命中来源：vector / bm25


@dataclass(slots=True)
class IngestResult:
    """文档入库结果。"""

    tenant_id: str
    doc_id: str
    collection: str
    chunk_count: int                    # 本次写入的块数量
    deleted_stale: int                  # 清理掉的旧版本块数量
    elapsed_ms: float


@dataclass(slots=True)
class RAGAnswer:
    """一次 RAG 问答的完整结果。"""

    question: str
    answer: str
    sources: list[RetrievedChunk] = field(default_factory=list)
    elapsed_ms: float = 0.0
