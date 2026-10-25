"""浏览器端 SSE 调试页。

说明：
- ``GET /demo`` 返回一个纯静态页面，不依赖任何外部 CDN；
- 页面使用 ``fetch`` + ``ReadableStream`` 消费 ``POST /api/chat``，
  因为浏览器原生 ``EventSource`` 只支持 GET，无法携带 JSON 请求体；
- 左侧展示 status 时间线，右侧逐字显示 answer，底部可展开原始事件。

该路由仅用于开发联调，生产环境可按需移除或加鉴权。
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["调试"])

_DEMO_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RAG SSE 流式对话调试</title>
<style>
  :root { color-scheme: light; }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 24px; background: #f6f7f9; color: #1f2328;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif;
  }
  .wrap { max-width: 960px; margin: 0 auto; display: grid; gap: 16px; }
  .card { background: #fff; border: 1px solid #d8dee4; border-radius: 10px; padding: 18px; }
  h1 { font-size: 19px; margin: 0 0 4px; }
  .hint { color: #57606a; font-size: 13px; margin: 0 0 14px; }
  label { display: block; font-size: 13px; color: #57606a; margin: 10px 0 4px; }
  input, textarea {
    width: 100%; padding: 8px 10px; border: 1px solid #d0d7de; border-radius: 6px;
    font: inherit; background: #fff;
  }
  textarea { min-height: 62px; resize: vertical; }
  .row { display: flex; gap: 12px; flex-wrap: wrap; }
  .row > div { flex: 1 1 220px; }
  .actions { display: flex; gap: 10px; margin-top: 14px; align-items: center; }
  button {
    padding: 8px 16px; border-radius: 6px; border: 1px solid #d0d7de;
    background: #f6f8fa; cursor: pointer; font: inherit;
  }
  button.primary { background: #1f6feb; border-color: #1f6feb; color: #fff; }
  button:disabled { opacity: .5; cursor: not-allowed; }
  .panels { display: grid; grid-template-columns: 320px 1fr; gap: 16px; }
  @media (max-width: 760px) { .panels { grid-template-columns: 1fr; } }
  .panel-title { font-size: 13px; font-weight: 600; color: #57606a; margin: 0 0 8px; }
  #timeline { list-style: none; margin: 0; padding: 0; font-size: 13px; }
  #timeline li { padding: 5px 0; border-bottom: 1px dashed #eaeef2; line-height: 1.5; }
  #timeline time { color: #8c959f; margin-right: 8px; font-variant-numeric: tabular-nums; }
  #answer { white-space: pre-wrap; line-height: 1.75; font-size: 15px; min-height: 96px; }
  #answer.streaming::after { content: "▍"; color: #1f6feb; animation: blink 1s steps(2) infinite; }
  @keyframes blink { 50% { opacity: 0; } }
  .badge { display: inline-block; padding: 2px 9px; border-radius: 999px; font-size: 12px; background: #ddf4ff; color: #0969da; }
  .badge.no { background: #ffebe9; color: #cf222e; }
  .badge.err { background: #ffebe9; color: #cf222e; }
  .meta { font-size: 12px; color: #57606a; margin-top: 10px; }
  details { margin-top: 12px; }
  summary { cursor: pointer; font-size: 13px; color: #57606a; }
  pre { background: #0d1117; color: #c9d1d9; padding: 12px; border-radius: 8px; overflow: auto; font-size: 12px; max-height: 240px; }
</style>
</head>
<body>
<div class="wrap">
  <div class="card">
    <h1>RAG SSE 流式对话调试</h1>
    <p class="hint">POST /api/chat · 事件类型：status / answer / done / error</p>
    <div class="row">
      <div>
        <label for="dept">dept_id（部门隔离）</label>
        <input id="dept" value="smoke_dept_a">
      </div>
      <div>
        <label for="thread">thread_id（会话主键）</label>
        <input id="thread">
      </div>
    </div>
    <label for="question">question</label>
    <textarea id="question">入职满一年的员工每年有多少天年假？</textarea>
    <div class="actions">
      <button id="send" class="primary">发送</button>
      <button id="abort" disabled>中止</button>
      <span id="state" class="meta"></span>
    </div>
  </div>

  <div class="panels">
    <div class="card">
      <p class="panel-title">流程状态</p>
      <ul id="timeline"></ul>
    </div>
    <div class="card">
      <p class="panel-title">模型输出</p>
      <div id="answer"></div>
      <div id="result" class="meta"></div>
      <details>
        <summary>原始 SSE 事件</summary>
        <pre id="raw"></pre>
      </details>
    </div>
  </div>
</div>

<script>
(function () {
  const deptEl = document.getElementById("dept");
  const threadEl = document.getElementById("thread");
  const questionEl = document.getElementById("question");
  const sendBtn = document.getElementById("send");
  const abortBtn = document.getElementById("abort");
  const stateEl = document.getElementById("state");
  const timeline = document.getElementById("timeline");
  const answerEl = document.getElementById("answer");
  const resultEl = document.getElementById("result");
  const rawEl = document.getElementById("raw");

  threadEl.value = "web-" + Date.now();

  let controller = null;

  function now() {
    return (performance.now() / 1000).toFixed(2);
  }

  function addStatus(message, node) {
    const li = document.createElement("li");
    const time = document.createElement("time");
    time.textContent = "+" + now() + "s";
    li.appendChild(time);
    li.appendChild(document.createTextNode(message));
    if (node) {
      const badge = document.createElement("span");
      badge.className = "badge";
      badge.textContent = node;
      badge.style.marginLeft = "8px";
      li.appendChild(badge);
    }
    timeline.appendChild(li);
    timeline.scrollTop = timeline.scrollHeight;
  }

  function reset() {
    timeline.innerHTML = "";
    answerEl.textContent = "";
    answerEl.classList.remove("streaming");
    resultEl.textContent = "";
    rawEl.textContent = "";
    stateEl.textContent = "";
  }

  function handleFrame(frame, stats) {
    let event = "message";
    const dataLines = [];
    for (const line of frame.split("\\n")) {
      if (!line || line.startsWith(":")) continue;
      if (line.startsWith("event:")) {
        event = line.slice(6).trim();
      } else if (line.startsWith("data:")) {
        dataLines.push(line.slice(5).trimStart());
      }
    }
    if (!dataLines.length) return;

    let data;
    try {
      data = JSON.parse(dataLines.join("\\n"));
    } catch (err) {
      return;
    }

    rawEl.textContent = "[" + now() + "s] " + event + " " +
      JSON.stringify(data, null, 0) + "\\n" + rawEl.textContent;

    if (event === "status") {
      addStatus(data.message || JSON.stringify(data), data.node);
      stateEl.textContent = data.message || "";
    } else if (event === "answer") {
      stats.chunks += 1;
      stats.chars += (data.delta || "").length;
      answerEl.textContent += data.delta || "";
      answerEl.classList.add("streaming");
    } else if (event === "done") {
      answerEl.classList.remove("streaming");
      const ok = data.is_relevant === "YES";
      resultEl.innerHTML = "";
      const badge = document.createElement("span");
      badge.className = "badge" + (ok ? "" : " no");
      badge.textContent = "is_relevant=" + data.is_relevant;
      resultEl.appendChild(badge);
      resultEl.appendChild(document.createTextNode(
        "  retry_count=" + data.retry_count +
        "  thread_id=" + data.thread_id +
        "  分片=" + stats.chunks + "  字符=" + stats.chars
      ));
      stateEl.textContent = "完成，耗时 " + now() + "s";
    } else if (event === "error") {
      answerEl.classList.remove("streaming");
      const badge = document.createElement("span");
      badge.className = "badge err";
      badge.textContent = "error";
      resultEl.innerHTML = "";
      resultEl.appendChild(badge);
      resultEl.appendChild(document.createTextNode(" " + (data.message || "")));
      stateEl.textContent = "服务端返回错误";
    }
  }

  async function send() {
    reset();
    const payload = {
      dept_id: deptEl.value.trim(),
      thread_id: threadEl.value.trim(),
      question: questionEl.value.trim(),
    };
    if (!payload.dept_id || !payload.thread_id || !payload.question) {
      stateEl.textContent = "三个字段都不能为空";
      return;
    }

    sendBtn.disabled = true;
    abortBtn.disabled = false;
    controller = new AbortController();
    const stats = { chunks: 0, chars: 0 };
    stateEl.textContent = "连接中…";

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });

      if (!response.ok) {
        const text = await response.text();
        stateEl.textContent = "HTTP " + response.status;
        resultEl.textContent = text;
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        buffer = buffer.replace(/\\r\\n/g, "\\n");

        let index;
        while ((index = buffer.indexOf("\\n\\n")) >= 0) {
          const frame = buffer.slice(0, index);
          buffer = buffer.slice(index + 2);
          handleFrame(frame, stats);
        }
      }
    } catch (err) {
      if (err.name === "AbortError") {
        stateEl.textContent = "已中止（客户端断开）";
      } else {
        stateEl.textContent = "请求失败：" + err.message;
      }
    } finally {
      answerEl.classList.remove("streaming");
      sendBtn.disabled = false;
      abortBtn.disabled = true;
      controller = null;
    }
  }

  sendBtn.addEventListener("click", send);
  abortBtn.addEventListener("click", function () {
    if (controller) controller.abort();
  });
  questionEl.addEventListener("keydown", function (event) {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) send();
  });
})();
</script>
</body>
</html>
"""


@router.get("/demo", response_class=HTMLResponse, summary="SSE 对话调试页")
async def demo() -> HTMLResponse:
    """返回浏览器调试页（仅用于开发联调）。"""
    return HTMLResponse(_DEMO_HTML)
