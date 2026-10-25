"""上传接口冒烟测试。

用法（服务需已启动）：
    .venv\\Scripts\\python.exe scripts\\upload_smoke.py
    .venv\\Scripts\\python.exe scripts\\upload_smoke.py --base-url http://127.0.0.1:8000

覆盖场景：
1. 首次上传 -> ``created``；
2. 内容相同重复上传 -> ``unchanged``，不重复写向量；
3. 内容变化后上传 -> ``updated``，旧版本分块被停用；
4. 损坏 PDF / 空文件 / 非 PDF 后缀 -> 400；
5. 上传结束后 ``tmp/`` 不残留 UUID 命名的临时文件。

脚本使用随机 ``dept_id``，不会读写其他租户的数据；每次运行都会向
Qdrant 写入少量测试向量，建议只在开发/冒烟集合上执行。
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

import httpx
import pymupdf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TMP_DIR = PROJECT_ROOT / "tmp"

# 两个版本的正文：只有年假天数不同，用于验证版本更新
DOC_V1 = (
    "Employee Handbook\n"
    "Annual leave policy: employees receive 10 days of paid leave per year.\n"
    "Expense policy: submit receipts within 10 working days."
)
DOC_V2 = DOC_V1.replace("10 days of paid leave", "15 days of paid leave")

EXPECTED_ERRORS = [
    ("broken.pdf", b"%PDF-1.4 broken", "PDF 解析失败"),
    ("empty.pdf", b"", "上传文件为空"),
    ("not_pdf.txt", b"hello", "仅支持 PDF"),
]


def build_pdf(text: str) -> bytes:
    """用 PyMuPDF 生成一份最小 PDF，返回字节内容。"""
    document = pymupdf.open()
    try:
        page = document.new_page()
        y = 72.0
        for line in text.splitlines():
            page.insert_text((72, y), line, fontsize=12)
            y += 20
        return document.tobytes()
    finally:
        document.close()


def upload(
    client: httpx.Client,
    base_url: str,
    filename: str,
    content: bytes,
    dept_id: str,
    doc_id: str,
    content_type: str = "application/pdf",
) -> httpx.Response:
    return client.post(
        f"{base_url}/api/docs/upload",
        files={"file": (filename, content, content_type)},
        data={"dept_id": dept_id, "doc_id": doc_id},
        timeout=120.0,
    )


def uuid_temp_files() -> set[str]:
    """列出 tmp/ 下 UUID 命名的临时 PDF（接口只使用这种命名）。"""
    if not TMP_DIR.is_dir():
        return set()
    result = set()
    for path in TMP_DIR.glob("*.pdf"):
        stem = path.stem
        if len(stem) == 32:
            try:
                uuid.UUID(stem)
            except ValueError:
                continue
            result.add(path.name)
    return result


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    parser = argparse.ArgumentParser(description="POST /api/docs/upload 冒烟测试")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="服务地址，默认 http://127.0.0.1:8000",
    )
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")

    dept_id = f"upload_smoke_{uuid.uuid4().hex[:8]}"
    doc_id = "upload_smoke.pdf"
    v1 = build_pdf(DOC_V1)
    v2 = build_pdf(DOC_V2)
    before = uuid_temp_files()

    with httpx.Client() as client:
        # 1. 首次上传
        resp = upload(client, base_url, "upload_smoke.pdf", v1, dept_id, doc_id)
        check(resp.status_code == 200, f"首次上传失败：{resp.status_code} {resp.text}")
        body = resp.json()
        check(body["status"] == "created", f"首次上传状态应为 created：{body}")
        check(body["chunk_count"] >= 1, f"切块数量异常：{body}")
        check(body["written_points"] == body["chunk_count"], f"写入点数异常：{body}")
        check(body["deactivated_points"] == 0, f"首次上传不应停用分块：{body}")
        print(f"[1/5] created     切块={body['chunk_count']} 耗时={body['elapsed_seconds']}s")

        # 2. 相同内容重复上传
        resp = upload(client, base_url, "upload_smoke.pdf", v1, dept_id, doc_id)
        check(resp.status_code == 200, f"重复上传失败：{resp.status_code} {resp.text}")
        body = resp.json()
        check(body["status"] == "unchanged", f"重复上传状态应为 unchanged：{body}")
        check(body["written_points"] == 0, f"unchanged 不应写入向量：{body}")
        print("[2/5] unchanged   未重复写入")

        # 3. 内容变化
        resp = upload(client, base_url, "upload_smoke.pdf", v2, dept_id, doc_id)
        check(resp.status_code == 200, f"更新上传失败：{resp.status_code} {resp.text}")
        body = resp.json()
        check(body["status"] == "updated", f"更新上传状态应为 updated：{body}")
        check(body["written_points"] >= 1, f"更新后应写入新向量：{body}")
        check(body["deactivated_points"] >= 1, f"更新后应停用旧向量：{body}")
        print(
            f"[3/5] updated     写入={body['written_points']} "
            f"停用={body['deactivated_points']}"
        )

        # 4. 异常输入
        for filename, content, expected in EXPECTED_ERRORS:
            resp = upload(client, base_url, filename, content, dept_id, f"{dept_id}_{filename}")
            check(
                resp.status_code == 400,
                f"{filename} 应返回 400，实际 {resp.status_code}：{resp.text}",
            )
            check(
                expected in resp.text,
                f"{filename} 错误信息应包含“{expected}”：{resp.text}",
            )
        print("[4/5] errors      损坏 PDF / 空文件 / 非 PDF 均返回 400")

        # 5. 临时文件清理
        leaked = uuid_temp_files() - before
        check(not leaked, f"tmp/ 残留临时文件：{sorted(leaked)}")
        print("[5/5] cleanup     无临时文件残留")

    print(f"\n全部通过（dept_id={dept_id}, doc_id={doc_id}）")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as exc:
        print(f"\n冒烟测试失败：{exc}", file=sys.stderr)
        sys.exit(1)
    except httpx.HTTPError as exc:
        print(f"\n无法连接服务：{exc}", file=sys.stderr)
        print("请先启动服务：.venv\\Scripts\\python.exe -m uvicorn app.main:app --port 8000")
        sys.exit(2)
