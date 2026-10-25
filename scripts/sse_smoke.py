"""SSE 接口冒烟客户端：验证 /api/chat 的 status / answer / done 事件。

前提：先启动服务
    .\\.venv\\Scripts\\python.exe -m uvicorn app.main:app --port 8000
并且 .env 已配置可用的 Embedding / LLM，且知识库中已有可命中的文档。

用法：
    .\\.venv\\Scripts\\python.exe scripts\\sse_smoke.py
    .\\.venv\\Scripts\\python.exe scripts\\sse_smoke.py --question "年假有几天？" --dept-id smoke_dept_a
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import httpx


def _configure_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass


def main() -> int:
    _configure_stdout()

    parser = argparse.ArgumentParser(description="POST /api/chat SSE 冒烟测试")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--dept-id", default="smoke_dept_a")
    parser.add_argument("--thread-id", default=f"sse-smoke-{int(time.time())}")
    parser.add_argument("--question", default="入职满一年的员工每年有多少天年假？")
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    base = args.base_url.rstrip("/")

    # 健康检查：确认服务与 Agent 图均已就绪
    try:
        health = httpx.get(f"{base}/health", timeout=10)
        health.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] 无法访问 {base}/health：{type(exc).__name__}: {exc}")
        print("       请先启动服务：python -m uvicorn app.main:app --port 8000")
        return 2

    print(f"[health] {health.json()}")

    url = f"{base}/api/chat"
    payload = {
        "dept_id": args.dept_id,
        "thread_id": args.thread_id,
        "question": args.question,
    }
    print(f"[request] POST {url}")
    print(f"[request] {json.dumps(payload, ensure_ascii=False)}")

    statuses: list[dict] = []
    answer_parts: list[str] = []
    answer_events = 0
    done_payload: dict | None = None
    started = time.perf_counter()
    first_token_seconds: float | None = None

    timeout = httpx.Timeout(args.timeout, connect=10.0)
    with httpx.stream("POST", url, json=payload, timeout=timeout) as response:
        if response.status_code != 200:
            print(f"[FAIL] HTTP {response.status_code}: {response.read().decode('utf-8', 'replace')}")
            return 1

        event_name: str | None = None
        for line in response.iter_lines():
            if not line:
                event_name = None
                continue
            if line.startswith(":"):
                # sse-starlette 的 ping 注释帧
                continue
            if line.startswith("event:"):
                event_name = line[len("event:") :].strip()
                continue
            if not line.startswith("data:"):
                continue

            raw = line[len("data:") :].strip()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                print(f"[warn] 无法解析 data：{raw[:120]}")
                continue

            if event_name == "status":
                statuses.append(data)
                print(f"[status] {data.get('message', data)}")
            elif event_name == "answer":
                delta = data.get("delta", "")
                if delta:
                    answer_events += 1
                    if first_token_seconds is None:
                        first_token_seconds = time.perf_counter() - started
                        print(f"[answer] 首 token 延迟 {first_token_seconds:.2f}s")
                    answer_parts.append(delta)
                    print(delta, end="", flush=True)
            elif event_name == "done":
                done_payload = data
                print()
                print(f"[done] {json.dumps(data, ensure_ascii=False)}")
            elif event_name == "error":
                print(f"[error] {data.get('message', data)}")
                return 1

    elapsed = time.perf_counter() - started
    answer = "".join(answer_parts)

    print("-" * 72)
    passed = True
    if done_payload is None:
        print("[FAIL] 未收到 done 事件")
        passed = False
    else:
        print(
            f"[PASS] 收到 done 事件：is_relevant={done_payload.get('is_relevant')}, "
            f"retry_count={done_payload.get('retry_count')}"
        )
        if done_payload.get("thread_id") != args.thread_id:
            print(
                f"[FAIL] done.thread_id={done_payload.get('thread_id')} "
                f"与请求 {args.thread_id} 不一致"
            )
            passed = False

    if answer:
        print(
            f"[PASS] 收到 answer 共 {len(answer)} 字符 / {answer_events} 个分片，"
            f"总耗时 {elapsed:.2f}s"
        )
        print(f"[answer] {answer}")
        if answer_events <= 1:
            print(
                "[WARN] 只收到 1 个分片：模型或链路未逐 token 返回，"
                "需检查服务商是否支持流式输出"
            )
    else:
        print("[WARN] 未收到 answer（若 is_relevant=NO 则属于预期兜底路径）")

    if statuses:
        print(f"[PASS] 收到 {len(statuses)} 个 status 事件")
    else:
        print("[FAIL] 未收到任何 status 事件")
        passed = False

    if first_token_seconds is not None:
        print(f"[PASS] 流式生效：首 token {first_token_seconds:.2f}s，总耗时 {elapsed:.2f}s")

    print("-" * 72)
    print("SSE 冒烟测试" + ("通过" if passed else "失败"))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
