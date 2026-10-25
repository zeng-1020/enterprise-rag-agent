"""Parent-Child 文档切块逻辑。

设计目的：
- 父块（Parent）：按 Markdown 一/二级标题切分，保留完整的章节语义，
  用于给 LLM 提供充足上下文；
- 子块（Child）：在父块内部按固定长度滑动切分，粒度更小，
  用于向量检索时提高命中精度。

检索命中任意子块后，可通过 payload 中的 ``parent_text`` 回填完整父块，
兼顾“检索精度”和“上下文完整性”。

本模块只负责切块，不包含 PDF 解析逻辑；入参为已转换好的 Markdown 文本。
"""

from __future__ import annotations

import hashlib
import re

# 子块目标长度（字符数）。约 200 字既能保证语义完整，又便于精确召回。
CHILD_CHUNK_SIZE = 200

# 相邻子块的重叠字符数，避免关键信息恰好被切断在块边界。
CHILD_CHUNK_OVERLAP = 50

# 滑动步长 = 块大小 - 重叠大小
_CHILD_CHUNK_STEP = CHILD_CHUNK_SIZE - CHILD_CHUNK_OVERLAP

# Markdown ATX 标题：仅识别一级（#）和二级（##）标题；
# 允许标题前最多 3 个空格，兼容常见的 Markdown 排版。
_HEADING_PATTERN = re.compile(r"^ {0,3}(#{1,2})[ \t]+(.+?)[ \t]*$")


def process_document(markdown_text: str, dept_id: str) -> list[dict]:
    """将 Markdown 文档切分为 Parent-Child 结构。

    Args:
        markdown_text: 已由 PDF 转换得到的 Markdown 文本。
        dept_id: 部门 ID，原样写入每个子块的 payload，用于后续按部门隔离。

    Returns:
        子块字典列表。每个字典包含以下 payload：
        - ``dept_id``: 部门 ID；
        - ``text``: 子块文本；
        - ``parent_id``: 父块的 SHA-256 哈希 ID；
        - ``parent_text``: 父块完整内容（用于命中后回填上下文）。

    Raises:
        TypeError: 入参类型不正确。
        ValueError: ``dept_id`` 为空，或切块参数配置非法。

    说明：
        父块 ID 由 ``dept_id + 父块序号 + 父块内容`` 共同哈希得到：
        - 相同输入重复处理会得到相同 ID，便于幂等入库；
        - 不同部门即使文档内容相同也不会产生 ID 冲突；
        - 父块内容变化时 ID 随之变化，便于识别旧版本数据。
    """
    if not isinstance(markdown_text, str):
        raise TypeError(
            f"markdown_text 必须是字符串，实际类型：{type(markdown_text).__name__}"
        )
    if not isinstance(dept_id, str):
        raise TypeError(f"dept_id 必须是字符串，实际类型：{type(dept_id).__name__}")

    normalized_dept_id = dept_id.strip()
    if not normalized_dept_id:
        raise ValueError("dept_id 不能为空")

    if CHILD_CHUNK_SIZE <= 0:
        raise ValueError("CHILD_CHUNK_SIZE 必须大于 0")
    if not 0 <= CHILD_CHUNK_OVERLAP < CHILD_CHUNK_SIZE:
        raise ValueError("CHILD_CHUNK_OVERLAP 必须大于等于 0 且小于 CHILD_CHUNK_SIZE")

    # 第一步：按 Markdown 标题切分父块
    parent_blocks = _split_parent_blocks(markdown_text)
    if not parent_blocks:
        return []

    # 第二步：在每个父块内部按固定长度滑动切分子块
    results: list[dict] = []
    for parent_index, parent_text in enumerate(parent_blocks):
        parent_id = _make_parent_id(
            dept_id=normalized_dept_id,
            parent_index=parent_index,
            parent_text=parent_text,
        )

        for start in range(0, len(parent_text), _CHILD_CHUNK_STEP):
            child_text = parent_text[start : start + CHILD_CHUNK_SIZE]

            # 跳过纯空白子块，避免产生无意义的向量
            if not child_text.strip():
                continue

            results.append(
                {
                    "dept_id": normalized_dept_id,
                    "text": child_text,
                    "parent_id": parent_id,
                    "parent_text": parent_text,
                }
            )

            # 当前子块已覆盖到父块末尾，无需再生成完全被包含的冗余尾部块
            if start + CHILD_CHUNK_SIZE >= len(parent_text):
                break

    return results


def _split_parent_blocks(markdown_text: str) -> list[str]:
    """按 Markdown 一/二级标题切分父块。

    规则：
    - 每个 ``#`` / ``##`` 标题开启一个新的父块；
    - 第一个标题之前的内容作为独立父块保留；
    - 代码围栏（``` 或 ~~~）内部的 ``#`` 不会被误判为标题。
    """
    parents: list[str] = []
    buffer: list[str] = []
    in_fence = False

    def flush() -> None:
        """将当前缓冲区内容作为一个父块写入结果。"""
        block = "".join(buffer).strip()
        if block:
            parents.append(block)
        buffer.clear()

    for line in markdown_text.splitlines(keepends=True):
        stripped_line = line.lstrip()

        # 遇到代码围栏时切换状态；围栏行本身照常收入当前父块
        if stripped_line.startswith("```") or stripped_line.startswith("~~~"):
            in_fence = not in_fence
            buffer.append(line)
            continue

        # 仅在代码围栏之外识别标题
        if not in_fence and _HEADING_PATTERN.match(line.rstrip("\r\n")):
            # 标题出现前的内容（如文档前言）先落盘，再以该标题开启新父块
            flush()

        buffer.append(line)

    flush()
    return parents


def _make_parent_id(dept_id: str, parent_index: int, parent_text: str) -> str:
    """生成父块唯一哈希 ID。

    使用 \\x1f（单元分隔符）拼接字段，避免不同字段组合产生歧义；
    SHA-256 输出 64 位十六进制字符串，碰撞概率可忽略。
    """
    raw = f"{dept_id}\x1f{parent_index}\x1f{parent_text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
