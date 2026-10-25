"""租户标识校验与集合命名。

安全要点：
- tenant_id 会参与 Qdrant 集合名拼接，必须严格限制字符集，
  防止通过恶意 tenant_id 拼接出非法集合名或越权访问其他租户数据；
- 集合命名规则：``{QDRANT_COLLECTION_PREFIX}_{tenant_id}``，
  实现租户级物理隔离。
"""

from __future__ import annotations

import re

from app.config import settings

# 租户 ID 规则：字母或数字开头，允许字母、数字、下划线、中划线，长度 1~64
_TENANT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")

# Qdrant 集合名规则：仅允许字母、数字、下划线、中划线
_COLLECTION_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,255}$")


def validate_tenant_id(tenant_id: str) -> str:
    """校验并规范化租户 ID。

    Args:
        tenant_id: 由认证中间件解析出的租户标识。

    Returns:
        去除首尾空白后的租户 ID。

    Raises:
        ValueError: tenant_id 为空、类型错误或包含非法字符。
    """
    if not isinstance(tenant_id, str):
        raise ValueError(
            f"tenant_id 必须是字符串，实际类型：{type(tenant_id).__name__}"
        )

    normalized = tenant_id.strip()
    if not normalized:
        raise ValueError("tenant_id 不能为空")
    if not _TENANT_ID_PATTERN.fullmatch(normalized):
        raise ValueError(
            "tenant_id 不合法：仅允许字母、数字、下划线和中划线，"
            "长度 1~64，且必须以字母或数字开头"
        )
    return normalized


def tenant_collection_name(tenant_id: str) -> str:
    """根据租户 ID 生成 Qdrant 集合名。

    一个租户对应一个 Collection，检索和写入都只能落在本租户集合内，
    从物理层面避免跨租户数据泄露。
    """
    normalized = validate_tenant_id(tenant_id)
    name = f"{settings.qdrant_collection_prefix}_{normalized}"
    if not _COLLECTION_NAME_PATTERN.fullmatch(name):
        raise ValueError(
            f"生成的集合名不合法：{name!r}，请检查 QDRANT_COLLECTION_PREFIX 配置"
        )
    return name
