"""基于 Qdrant 的多租户向量检索器。

多租户隔离策略：
- 所有部门共用同一个 Collection；
- 每个数据点的 payload 中写入 ``dept_id``；
- 检索时必须通过 Qdrant Filter 强制匹配 ``dept_id``，从查询层面阻止跨部门数据泄露；
- 为 ``dept_id`` 建立 keyword 载荷索引，保证过滤查询的性能。

文档版本控制策略（软删除）：
- 同一文档的多个版本共用稳定的 ``doc_id``；
- 每次入库计算内容哈希 ``doc_hash``，与当前有效版本一致时直接返回“未修改”；
- 内容变化时，先把该文档旧版本的所有点置为 ``is_active=False``，再写入新版本；
- 检索默认只返回 ``is_active=True`` 的点，旧版本数据仍保留在库中，便于审计或回滚。

注意：
- 本模块使用逻辑隔离，适合单部门数据量中部门数量多、等的场景；
- 若某个部门的数据量或合规要求极高，可扩展为“独立 Collection”物理隔离方案。
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from collections.abc import Sequence
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Literal

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    HasIdCondition,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from app.config import settings
from app.core.embedding import CloudEmbeddingService, get_embedding_service
from app.models.rag import DocumentUpsertResult

logger = logging.getLogger(__name__)

# 云端 Embedding 模型输出维度。必须与 EMBEDDING_DIM 保持一致。
DEFAULT_VECTOR_SIZE = 1024

# 单次 upsert 的最大点数，避免请求体过大
UPSERT_BATCH_SIZE = 128

# 单次检索允许返回的最大条数，防止恶意大 top_k 拖垮服务
MAX_TOP_K = 100

# 部门聚合（facet）单次最多返回的取值数量
MAX_DEPARTMENT_FACETS = 1000

# payload 中用于多租户过滤的字段名
DEPT_ID_FIELD = "dept_id"

# payload 中用于版本控制的字段名
DOC_ID_FIELD = "doc_id"        # 文档稳定标识，同一文档的多个版本共用
DOC_HASH_FIELD = "doc_hash"    # 文档内容哈希，用于判断内容是否变化
IS_ACTIVE_FIELD = "is_active"  # 是否为当前有效版本；旧版本软删除时置为 False


class QdrantRetriever:
    """Qdrant 多租户向量检索器。

    Args:
        url: Qdrant 服务地址，默认读取 ``QDRANT_URL``（本地为 http://localhost:6333）。
        collection_name: 集合名，默认 ``{QDRANT_COLLECTION_PREFIX}_chunks``。
        vector_size: 向量维度，默认读取 ``EMBEDDING_DIM``（当前为 1024）。
        client: 可注入的 Qdrant 客户端，便于测试；为空时自动创建。
        embedding_service: 可注入的 Embedding 服务，便于测试；为空时使用全局单例。

    Raises:
        ValueError: 向量维度配置不一致，或集合已存在但维度与配置不符。
    """

    def __init__(
        self,
        *,
        url: str | None = None,
        collection_name: str | None = None,
        vector_size: int | None = None,
        client: QdrantClient | None = None,
        embedding_service: CloudEmbeddingService | None = None,
    ) -> None:
        # 维度优先使用显式参数，否则跟随全局配置，确保与 Embedding 模型一致
        self._vector_size = vector_size or settings.embedding_dim
        if self._vector_size <= 0:
            raise ValueError("vector_size 必须大于 0")
        if self._vector_size != settings.embedding_dim:
            raise ValueError(
                f"vector_size={self._vector_size} 与 EMBEDDING_DIM="
                f"{settings.embedding_dim} 不一致；两者必须相同，"
                "否则写入和检索都会因向量维度不匹配而失败"
            )

        self._collection_name = (
            collection_name or f"{settings.qdrant_collection_prefix}_chunks"
        )
        self._embedding = embedding_service or get_embedding_service()

        # 允许注入客户端，方便单元测试；生产环境自动连接本地/远程 Qdrant
        if client is not None:
            self._client = client
        else:
            api_key = (
                settings.qdrant_api_key.get_secret_value()
                if settings.qdrant_api_key
                else None
            )
            self._client = QdrantClient(
                url=url or settings.qdrant_url,
                api_key=api_key or None,
                timeout=settings.qdrant_timeout,
                prefer_grpc=settings.qdrant_prefer_grpc,
            )

        # 初始化阶段完成集合创建、维度校验和过滤字段索引
        try:
            self._ensure_collection()
        except Exception:
            logger.exception(
                "Qdrant 初始化失败：collection=%s, url=%s",
                self._collection_name,
                url or settings.qdrant_url,
            )
            raise

    # ------------------------------------------------------------------
    # 初始化相关
    # ------------------------------------------------------------------
    def _ensure_collection(self) -> None:
        """确保集合存在，并建立 dept_id 载荷索引。"""
        if not self._client.collection_exists(self._collection_name):
            try:
                self._client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config=VectorParams(
                        size=self._vector_size,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(
                    "已创建 Qdrant 集合：%s（维度=%d, 距离=COSINE）",
                    self._collection_name,
                    self._vector_size,
                )
            except Exception:
                # 并发初始化时可能已被其他实例创建，确认存在即可继续
                if not self._client.collection_exists(self._collection_name):
                    raise
        else:
            self._check_existing_vector_size()

        self._ensure_payload_indexes()

    def _check_existing_vector_size(self) -> None:
        """校验已存在集合的向量维度，避免配置漂移导致运行期才报错。"""
        info = self._client.get_collection(self._collection_name)
        vectors_config = info.config.params.vectors
        existing_size = getattr(vectors_config, "size", None)
        if existing_size is not None and existing_size != self._vector_size:
            raise ValueError(
                f"集合 {self._collection_name} 已存在且向量维度为 {existing_size}，"
                f"与当前配置 {self._vector_size} 不一致。"
                "请改用新的集合名，或删除旧集合后重新入库。"
            )

    def _ensure_payload_indexes(self) -> None:
        """为多租户过滤和版本控制字段建立载荷索引。

        索引已存在时不同 Qdrant 版本的行为不完全一致，
        因此这里降级为告警：即使索引创建失败，过滤语义仍然正确，只是性能可能下降。
        """
        indexes = {
            DEPT_ID_FIELD: PayloadSchemaType.KEYWORD,
            DOC_ID_FIELD: PayloadSchemaType.KEYWORD,
            DOC_HASH_FIELD: PayloadSchemaType.KEYWORD,
            IS_ACTIVE_FIELD: PayloadSchemaType.BOOL,
        }

        for field_name, field_schema in indexes.items():
            try:
                self._client.create_payload_index(
                    collection_name=self._collection_name,
                    field_name=field_name,
                    field_schema=field_schema,
                )
            except Exception as exc:  # noqa: BLE001 - 索引已存在等情况不应阻断启动
                logger.warning(
                    "%s 载荷索引创建跳过（若已存在可忽略）：%s", field_name, exc
                )

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------
    def add_documents(
        self,
        docs: list[dict],
        *,
        doc_id: str | None = None,
        doc_hash: str | None = None,
    ) -> DocumentUpsertResult:
        """将子块写入 Qdrant，并完成文档版本控制（软删除旧版本）。

        Args:
            docs: 子块字典列表，通常来自 ``chunking.process_document``。
                每条必须包含 ``dept_id``、``text``、``parent_id``、``parent_text``；
                也可在每条中携带相同的 ``doc_id``。
            doc_id: 文档稳定标识（如文件名、业务主键或文档 UUID）。
                同一文档的多个版本必须使用同一个 ``doc_id``；
                未显式传入时读取 ``docs`` 中统一的 ``doc_id``。
            doc_hash: 文档内容哈希。建议在解析/切块前对原始文件字节计算，
                这样调整切块参数不会误判为“内容已修改”；
                未传入时基于子块文本自动计算。

        Returns:
            版本控制结果，状态为 ``created`` / ``updated`` / ``unchanged``。

        Raises:
            TypeError: docs 或其中元素类型错误。
            ValueError: 必填字段缺失、doc_id 缺失或冲突、跨部门混传。
            EmbeddingServiceError: 向量化失败。

        说明：
            - 哈希与当前有效版本一致时不会重复写入，返回 ``unchanged``，
              调用方可直接使用 ``result.message`` 提示用户“未修改”；
            - 哈希变化时先把旧版本的所有点置为 ``is_active=False``，再写入新版本；
            - Qdrant 不提供跨点的比较并交换事务，并发上传同一 ``doc_id`` 时，
              建议由 API 层按文档维度串行化。
        """
        if not isinstance(docs, list):
            raise TypeError(f"docs 必须是列表，实际类型：{type(docs).__name__}")
        if not docs:
            raise ValueError("docs 不能为空")

        normalized_doc_id = self._resolve_doc_id(docs, doc_id)
        normalized_hash = self._resolve_doc_hash(docs, doc_hash)

        # 1. 先完成全部字段校验和 payload 组装，避免向量化后才发现参数错误
        texts, payloads, point_ids = self._prepare_points(
            docs, normalized_doc_id, normalized_hash
        )

        dept_ids = {payload[DEPT_ID_FIELD] for payload in payloads}
        if len(dept_ids) != 1:
            raise ValueError(
                "同一文档的 dept_id 必须一致；跨部门文档请分多次上传"
            )
        normalized_dept_id = next(iter(dept_ids))

        # 2. 检查当前有效版本：哈希一致则直接提示未修改
        active_payload = self._get_active_version_payload(
            normalized_dept_id, normalized_doc_id
        )

        if (
            active_payload is not None
            and active_payload.get(DOC_HASH_FIELD) == normalized_hash
        ):
            active_count = self._count_active_version(
                normalized_dept_id, normalized_doc_id
            )
            if active_count == len(docs):
                logger.info(
                    "文档内容未修改，跳过入库：dept_id=%s, doc_id=%s, doc_hash=%s",
                    normalized_dept_id,
                    normalized_doc_id,
                    normalized_hash,
                )
                return DocumentUpsertResult(
                    doc_id=normalized_doc_id,
                    doc_hash=normalized_hash,
                    status="unchanged",
                    written_points=0,
                    deactivated_points=0,
                )

            # 极端情况：上次写入中途失败，哈希一致但分块缺失，需要重新写入修复
            logger.warning(
                "哈希一致但有效分块数异常（期望 %d，实际 %d），将重新写入："
                "dept_id=%s, doc_id=%s",
                len(docs),
                active_count,
                normalized_dept_id,
                normalized_doc_id,
            )

        status: Literal["created", "updated"] = (
            "updated" if active_payload is not None else "created"
        )

        # 3. 先向量化，再软删除旧版本：
        #    如果 Embedding 调用失败，旧版本仍然有效，不会导致文档在检索中消失
        vectors = self._embedding.embed_texts(texts)
        if len(vectors) != len(texts):
            raise RuntimeError(
                f"Embedding 返回数量异常：期望 {len(texts)} 条，实际 {len(vectors)} 条"
            )

        points = [
            PointStruct(id=point_id, vector=vector, payload=payload)
            for point_id, vector, payload in zip(
                point_ids, vectors, payloads, strict=True
            )
        ]

        # 4. 软删除旧版本：只把该部门、该文档的有效点置为 is_active=False。
        #    本次要写入的点 ID 必须排除在外——版本之间未改动的分块 ID 相同，
        #    若先停用再 upsert，它们会被重新置为 True，导致统计和状态不一致。
        deactivated_points = 0
        if active_payload is not None:
            deactivated_points = self._deactivate_active_version(
                normalized_dept_id,
                normalized_doc_id,
                keep_point_ids=point_ids,
            )

        # 分批写入，wait=True 保证写入后立即可检索
        for start in range(0, len(points), UPSERT_BATCH_SIZE):
            self._client.upsert(
                collection_name=self._collection_name,
                points=points[start : start + UPSERT_BATCH_SIZE],
                wait=True,
            )

        logger.info(
            "文档写入完成：collection=%s, dept_id=%s, doc_id=%s, "
            "状态=%s, 新版本点数=%d, 停用旧版本点数=%d",
            self._collection_name,
            normalized_dept_id,
            normalized_doc_id,
            status,
            len(points),
            deactivated_points,
        )

        return DocumentUpsertResult(
            doc_id=normalized_doc_id,
            doc_hash=normalized_hash,
            status=status,
            written_points=len(points),
            deactivated_points=deactivated_points,
        )

    def _prepare_points(
        self,
        docs: list[dict],
        doc_id: str,
        doc_hash: str,
    ) -> tuple[list[str], list[dict], list[str]]:
        """校验子块字段，生成文本、payload 与确定性点 ID。"""
        texts: list[str] = []
        payloads: list[dict] = []
        point_ids: list[str] = []

        # 记录 (dept_id, doc_id, parent_id, text) 在本次批次中的出现次数，
        # 保证完全相同的子块也能获得不同 ID，同时保持相同输入下的幂等性
        occurrences: dict[tuple[str, str, str, str], int] = {}

        for index, doc in enumerate(docs):
            if not isinstance(doc, dict):
                raise TypeError(
                    f"第 {index} 条文档必须是字典，实际类型：{type(doc).__name__}"
                )

            dept_id = self._validate_dept_id(doc.get(DEPT_ID_FIELD), index)

            # 文档级 doc_id 必须与本次入库的 doc_id 一致，防止混入其他文档
            doc_level_id = doc.get(DOC_ID_FIELD)
            if doc_level_id is not None and doc_level_id != doc_id:
                raise ValueError(
                    f"第 {index} 条文档的 doc_id={doc_level_id!r} 与本次入库的 "
                    f"doc_id={doc_id!r} 不一致"
                )

            text = doc.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"第 {index} 条文档的 text 不能为空")

            parent_id = doc.get("parent_id")
            if not isinstance(parent_id, str) or not parent_id.strip():
                raise ValueError(f"第 {index} 条文档的 parent_id 不能为空")

            parent_text = doc.get("parent_text")
            if not isinstance(parent_text, str) or not parent_text.strip():
                raise ValueError(f"第 {index} 条文档的 parent_text 不能为空")

            key = (dept_id, doc_id, parent_id, text)
            occurrence = occurrences.get(key, 0)
            occurrences[key] = occurrence + 1

            # payload 保留子块的全部字段，并补齐版本控制字段
            payload = dict(doc)
            payload[DEPT_ID_FIELD] = dept_id
            payload[DOC_ID_FIELD] = doc_id
            payload[DOC_HASH_FIELD] = doc_hash
            payload[IS_ACTIVE_FIELD] = True
            payload["indexed_at"] = datetime.now(timezone.utc).isoformat()

            payloads.append(payload)
            texts.append(text)
            point_ids.append(
                self._make_point_id(dept_id, doc_id, parent_id, text, occurrence)
            )

        return texts, payloads, point_ids

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        dept_id: str,
        top_k: int = 10,
        is_active: bool = True,
    ) -> list[dict]:
        """按部门隔离检索相似子块，默认只返回当前有效版本。

        Args:
            query: 用户查询文本。
            dept_id: 部门 ID，作为强制过滤条件，绝不跨部门检索。
            top_k: 返回条数，默认 10，最大 100。
            is_active: 版本过滤。
                - ``True``（默认）: 只检索 ``is_active=True`` 的当前有效版本；
                - ``False``: 只检索已软删除的历史版本，供审计/回溯使用。

        Returns:
            匹配的子块列表。每个元素包含该子块的原始 payload 字段，
            并额外附带 ``score`` 字段（余弦相似度，越大越相关）。

        Raises:
            ValueError: query / dept_id 为空，或 top_k 非法。
            TypeError: is_active 不是布尔值。
            EmbeddingServiceError: 查询向量化失败。
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query 不能为空")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k 必须是正整数")
        if not isinstance(is_active, bool):
            raise TypeError(
                f"is_active 必须是布尔值，实际类型：{type(is_active).__name__}"
            )

        normalized_dept_id = self._validate_dept_id(dept_id)
        effective_top_k = min(top_k, MAX_TOP_K)

        # 1. 查询向量化
        query_vector = self._embedding.embed_query(query.strip())

        # 2. 强制过滤：dept_id 精确匹配 + 版本有效性
        tenant_filter = Filter(
            must=[
                FieldCondition(
                    key=DEPT_ID_FIELD,
                    match=MatchValue(value=normalized_dept_id),
                ),
                FieldCondition(
                    key=IS_ACTIVE_FIELD,
                    match=MatchValue(value=is_active),
                ),
            ]
        )

        # 3. 向量检索
        response = self._client.query_points(
            collection_name=self._collection_name,
            query=query_vector,
            query_filter=tenant_filter,
            limit=effective_top_k,
            with_payload=True,
            with_vectors=False,
        )

        results: list[dict] = []
        for point in response.points:
            payload = dict(point.payload or {})

            # 防御性二次校验：即使过滤条件被绕过，也绝不返回其他部门的数据
            if payload.get(DEPT_ID_FIELD) != normalized_dept_id:
                logger.error(
                    "检测到跨部门检索结果，已丢弃：point_id=%s, 期望=%s, 实际=%s",
                    point.id,
                    normalized_dept_id,
                    payload.get(DEPT_ID_FIELD),
                )
                continue

            # 防御性二次校验：版本状态必须与请求一致，避免返回已软删除的旧版本
            if payload.get(IS_ACTIVE_FIELD) != is_active:
                logger.error(
                    "检测到版本状态不符的检索结果，已丢弃：point_id=%s, "
                    "期望 is_active=%s, 实际=%s",
                    point.id,
                    is_active,
                    payload.get(IS_ACTIVE_FIELD),
                )
                continue

            payload["score"] = float(point.score)
            results.append(payload)

        logger.info(
            "向量检索完成：dept_id=%s, is_active=%s, top_k=%d, 命中=%d",
            normalized_dept_id,
            is_active,
            effective_top_k,
            len(results),
        )
        return results

    # ------------------------------------------------------------------
    # 部门聚合
    # ------------------------------------------------------------------
    def list_departments(self, *, limit: int = MAX_DEPARTMENT_FACETS) -> list[dict]:
        """聚合当前有效版本中出现的部门及其分块数量。

        Args:
            limit: 最多返回的部门数量，默认 1000。

        Returns:
            ``[{"dept_id": "HR_dept", "count": 7}, ...]``，按 dept_id 升序排列。
            ``count`` 是该部门下 ``is_active=True`` 的向量点（子块）数量，
            不是文档数量。

        Raises:
            ValueError: limit 不是正整数。
            Exception: Qdrant 请求失败时原样抛出，由 API 层转换为 HTTP 错误。
        """
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise ValueError("limit 必须是正整数")

        response = self._client.facet(
            collection_name=self._collection_name,
            key=DEPT_ID_FIELD,
            facet_filter=Filter(
                must=[
                    FieldCondition(
                        key=IS_ACTIVE_FIELD,
                        match=MatchValue(value=True),
                    )
                ]
            ),
            limit=min(limit, MAX_DEPARTMENT_FACETS),
            exact=True,
        )

        departments = [
            {"dept_id": str(hit.value), "count": int(hit.count)}
            for hit in response.hits
            if hit.value is not None
        ]
        departments.sort(key=lambda item: item["dept_id"].lower())

        logger.info(
            "部门聚合完成：collection=%s, 部门数=%d",
            self._collection_name,
            len(departments),
        )
        return departments

    # ------------------------------------------------------------------
    # 版本控制
    # ------------------------------------------------------------------
    @classmethod
    def _resolve_doc_id(cls, docs: list[dict], doc_id: str | None) -> str:
        """确定本次入库的文档 ID，并校验 docs 中的 doc_id 是否一致。"""
        if doc_id is not None:
            return cls._validate_doc_id(doc_id)

        candidates = {
            doc[DOC_ID_FIELD].strip()
            for doc in docs
            if isinstance(doc, dict)
            and isinstance(doc.get(DOC_ID_FIELD), str)
            and doc[DOC_ID_FIELD].strip()
        }
        if not candidates:
            raise ValueError(
                "doc_id 不能为空：请传入稳定的文档标识（如文件名、业务主键或文档 UUID）。"
                "同一文档的多个版本必须使用同一个 doc_id，否则无法软删除旧版本"
            )
        if len(candidates) > 1:
            raise ValueError(
                f"docs 中存在多个不同的 doc_id：{sorted(candidates)}；"
                "一次只能入库一个文档"
            )
        return cls._validate_doc_id(candidates.pop())

    @classmethod
    def _resolve_doc_hash(cls, docs: list[dict], doc_hash: str | None) -> str:
        """使用调用方提供的哈希，或基于子块文本自动计算。"""
        if doc_hash is None:
            return cls.compute_document_hash(docs)
        if not isinstance(doc_hash, str) or not doc_hash.strip():
            raise ValueError("doc_hash 必须是非空字符串")
        return doc_hash.strip()

    @staticmethod
    def compute_document_hash(docs: list[dict]) -> str:
        """基于子块文本计算文档内容哈希。

        使用 ``\\x1e`` 分隔子块，避免不同切分结果产生歧义拼接。
        注意：调整切块参数会改变哈希；如需跨切块版本保持稳定，
        请在调用方对原始文件字节计算哈希，并通过 ``doc_hash`` 传入。
        """
        if not isinstance(docs, list) or not docs:
            raise ValueError("docs 不能为空，无法计算文档哈希")

        texts: list[str] = []
        for index, doc in enumerate(docs):
            if not isinstance(doc, dict):
                raise TypeError(
                    f"第 {index} 条文档必须是字典，实际类型：{type(doc).__name__}"
                )
            text = doc.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"第 {index} 条文档的 text 不能为空")
            texts.append(text)

        canonical = "\x1e".join(texts)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _get_active_version_payload(
        self,
        dept_id: str,
        doc_id: str,
    ) -> dict | None:
        """读取该文档当前有效版本的分块 payload（只取 1 条）。"""
        points, _ = self._client.scroll(
            collection_name=self._collection_name,
            scroll_filter=self._active_document_filter(dept_id, doc_id),
            limit=1,
            with_payload=[DOC_HASH_FIELD],
            with_vectors=False,
        )
        if not points:
            return None
        return dict(points[0].payload or {})

    def _count_active_version(self, dept_id: str, doc_id: str) -> int:
        """统计该文档当前有效版本的分块数量。"""
        result = self._client.count(
            collection_name=self._collection_name,
            count_filter=self._active_document_filter(dept_id, doc_id),
            exact=True,
        )
        return int(result.count)

    def _deactivate_active_version(
        self,
        dept_id: str,
        doc_id: str,
        *,
        keep_point_ids: Sequence[str] = (),
    ) -> int:
        """软删除该文档的旧版本分块，返回实际被停用的点数量。

        Args:
            dept_id: 部门 ID。
            doc_id: 文档稳定标识。
            keep_point_ids: 本次新版本会写入的点 ID。这些点与旧版本内容完全
                相同，会被 upsert 覆盖并保持有效，因此不能停用；否则
                ``is_active`` 状态与统计结果会与预期不一致。
        """
        target_filter = self._active_document_filter(
            dept_id,
            doc_id,
            exclude_point_ids=keep_point_ids,
        )
        deactivated = int(
            self._client.count(
                collection_name=self._collection_name,
                count_filter=target_filter,
                exact=True,
            ).count
        )

        if deactivated == 0:
            logger.info(
                "旧版本无需软删除：dept_id=%s, doc_id=%s（本次全部为重复分块）",
                dept_id,
                doc_id,
            )
            return 0

        # 用 Filter 批量更新 payload，避免把大量点 ID 拉回客户端
        self._client.set_payload(
            collection_name=self._collection_name,
            payload={IS_ACTIVE_FIELD: False},
            points=target_filter,
            wait=True,
        )

        logger.info(
            "旧版本已软删除：dept_id=%s, doc_id=%s, 停用点数=%d, 复用点数=%d",
            dept_id,
            doc_id,
            deactivated,
            len(keep_point_ids),
        )
        return deactivated

    @staticmethod
    def _active_document_filter(
        dept_id: str,
        doc_id: str,
        *,
        exclude_point_ids: Sequence[str] | None = None,
    ) -> Filter:
        """构造“部门 + 文档 + 当前有效”的过滤条件。

        ``exclude_point_ids`` 用于排除本次即将重写的点，避免它们被误停用。
        """
        must_not: list[HasIdCondition] = []
        if exclude_point_ids:
            must_not.append(HasIdCondition(has_id=list(exclude_point_ids)))

        return Filter(
            must=[
                FieldCondition(
                    key=DEPT_ID_FIELD,
                    match=MatchValue(value=dept_id),
                ),
                FieldCondition(
                    key=DOC_ID_FIELD,
                    match=MatchValue(value=doc_id),
                ),
                FieldCondition(
                    key=IS_ACTIVE_FIELD,
                    match=MatchValue(value=True),
                ),
            ],
            must_not=must_not or None,
        )

    @staticmethod
    def _validate_doc_id(doc_id: Any) -> str:
        """校验文档 ID。文档 ID 只作为 payload 值参与过滤，不做字符集限制。"""
        if not isinstance(doc_id, str):
            raise ValueError(
                f"doc_id 必须是字符串，实际类型：{type(doc_id).__name__}"
            )

        normalized = doc_id.strip()
        if not normalized:
            raise ValueError("doc_id 不能为空")
        if len(normalized) > 256:
            raise ValueError("doc_id 长度不能超过 256 个字符")
        return normalized

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_dept_id(dept_id: Any, index: int | None = None) -> str:
        """校验部门 ID。部门 ID 只作为 payload 值参与过滤，不做字符集限制。"""
        location = f"第 {index} 条文档的 " if index is not None else ""
        if not isinstance(dept_id, str):
            raise ValueError(
                f"{location}dept_id 必须是字符串，实际类型：{type(dept_id).__name__}"
            )

        normalized = dept_id.strip()
        if not normalized:
            raise ValueError(f"{location}dept_id 不能为空")
        if len(normalized) > 128:
            raise ValueError(f"{location}dept_id 长度不能超过 128 个字符")
        return normalized

    @staticmethod
    def _make_point_id(
        dept_id: str,
        doc_id: str,
        parent_id: str,
        text: str,
        occurrence: int,
    ) -> str:
        """生成确定性的 UUID 点 ID。

        相同输入重复入库会得到相同 ID，Qdrant 会覆盖旧点，实现幂等写入；
        ``doc_id`` 参与哈希，保证不同文档中完全相同的块不会互相覆盖。
        使用 \\x1f 分隔字段，避免字段拼接产生歧义。
        """
        raw = f"{dept_id}\x1f{doc_id}\x1f{parent_id}\x1f{occurrence}\x1f{text}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))


@lru_cache(maxsize=1)
def get_retriever() -> QdrantRetriever:
    """获取检索器单例，供 API / Agent 层依赖注入。"""
    return QdrantRetriever()
