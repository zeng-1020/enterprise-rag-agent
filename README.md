# 企业级多租户 RAG 系统 · Vibe Coding 实录

一个面向企业内部的、多租户隔离的 RAG（检索增强生成）问答系统。
本文档既是项目说明书，也是开发过程中的**踩坑与修复日志**，后续每次优化都会在
[问题与解决记录](#5-问题与解决记录) 中追加条目。

## 1. 技术栈与能力

| 模块 | 选型 |
|---|---|
| 运行时 | Python 3.12，Windows 11 + PowerShell 5.1 |
| Web 框架 | FastAPI + Uvicorn |
| Agent 编排 | LangGraph（异步节点 + 条件路由 + 重试） |
| 状态持久化 | AsyncSqliteSaver（按 `thread_id` 保存会话状态） |
| 向量库 | Qdrant 1.19.2（本地 Windows 二进制） |
| LLM / Embedding | OpenAI 兼容云端 API（DeepSeek + BAAI/bge-m3 1024 维） |
| 文档解析 | PyMuPDF + pymupdf4llm（PDF → Markdown） |
| 流式输出 | SSE（`sse-starlette`） |

核心链路：

```
PDF 上传
  └─ 临时文件 → PDF 解析为 Markdown → Parent-Child 切块
       └─ Embedding → Qdrant（单集合 + dept_id 过滤 + 版本软删除）

用户提问
  └─ retrieve（按 dept_id 检索）
       └─ evaluate（YES / NO）
            ├─ YES → generate（父块上下文 → LLM 流式回答）
            └─ NO 且 retry < 2 → rewrite → retrieve
                 └─ 重试耗尽 → 结束（兜底提示）
```

## 2. 目录结构

```
.
├─ app/
│  ├─ api/
│  │  ├─ chat.py          # POST /api/chat，SSE 流式问答
│  │  ├─ documents.py     # POST /api/docs/upload，PDF 解析与入库
│  │  ├─ demo.py          # GET /demo，浏览器调试页
│  │  └─ deps.py          # 从 app.state 获取 Agent 图
│  ├─ agent/
│  │  ├─ state.py         # AgentState 定义
│  │  ├─ nodes.py         # retrieve / evaluate / rewrite / generate
│  │  └─ graph.py         # 图组装、条件路由、AsyncSqliteSaver
│  ├─ core/
│  │  ├─ embedding.py     # 云端 Embedding 封装（重试 + 分批）
│  │  ├─ chunking.py      # Parent-Child 切块
│  │  ├─ retriever.py     # Qdrant 多租户检索与版本控制
│  │  └─ pdf_parser.py    # PDF → Markdown
│  ├─ models/rag.py
│  ├─ config.py           # pydantic-settings 配置
│  └─ main.py             # FastAPI 入口 + lifespan
├─ scripts/
│  ├─ qdrant.ps1          # 本地 Qdrant 启停
│  ├─ smoke_test.py       # 核心链路冒烟（入库/隔离/版本/问答）
│  ├─ sse_smoke.py        # SSE 接口冒烟
│  ├─ upload_smoke.py     # 上传接口冒烟
│  └─ list_documents.py   # 查询已入库文档
├─ tools/qdrant/          # 本地 Qdrant 二进制与存储（不提交）
├─ data/                  # Agent 会话检查点（不提交）
├─ tmp/                   # 上传临时目录（不提交）
├─ index.html             # 单文件交互控制台（原生 HTML + Tailwind CDN）
├─ requirements.txt
├─ .env.example           # 配置模板（不得写入真实密钥）
├─ .gitignore             # 排除密钥、虚拟环境、Qdrant 数据与日志
└─ README.md
```

## 3. 快速开始

### 3.1 启动 Qdrant

```powershell
.\scripts\qdrant.ps1 start
.\scripts\qdrant.ps1 status
# 停止：.\scripts\qdrant.ps1 stop
```

REST 端口 `6333`，gRPC 端口 `6334`，数据目录 `tools/qdrant/storage`。

### 3.2 安装依赖

本项目虚拟环境由 `uv` 创建，**`.venv` 内没有 pip**，请使用：

```powershell
uv pip install --python .\.venv\Scripts\python.exe -r requirements.txt
```

若机器上没有 `uv`，先安装：`pip install uv`（或 `python -m ensurepip --upgrade`）。

### 3.3 配置环境变量

```powershell
# 仅当 .env 不存在时才复制模板；已有 .env 时绝对不要覆盖
if (-not (Test-Path .env)) {
    copy .env.example .env
}
# 编辑 .env：填入 LLM_API_KEY、EMBEDDING_API_KEY 等
```

> **警告**：已有 `.env` 时执行 `copy .env.example .env` 会用模板覆盖真实密钥。
> 需要新增配置项时，请手动对比 `.env.example` 后只追加缺少的行。

关键配置：

- `QDRANT_URL=http://127.0.0.1:6333`（**不要用 localhost**，原因见踩坑记录）
- `EMBEDDING_MODEL=BAAI/bge-m3`，`EMBEDDING_DIM=1024`
- `QDRANT_COLLECTION_PREFIX`：集合名为 `{前缀}_chunks`，默认 `rag`

### 3.4 启动服务

```powershell
# 开发环境建议使用独立前缀，避免污染正式集合
$env:QDRANT_COLLECTION_PREFIX = "rag_smoke"   # 实际集合名为 rag_smoke_chunks
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

启动后：

- 交互控制台：<http://127.0.0.1:8000/>（`index.html`，与 API 同源）
- 健康检查：<http://127.0.0.1:8000/health>
- Swagger：<http://127.0.0.1:8000/docs>
- SSE 调试页：<http://127.0.0.1:8000/demo>

> 控制台必须通过 FastAPI 访问，不能直接双击打开 `index.html`（`file://`
> 协议无法调用同源 API）。切换部门时控制台会自动重置 `thread_id` 并清空
> 对话，避免把上一个部门的会话上下文带入新部门。
>
> 控制台启动时会调用 `GET /api/docs/departments` 动态渲染部门下拉框；也可以
> 点击「+ 新建部门」临时输入新的 `dept_id`（如 `marketing_dept`）。接口不可用时
> 回退到 HTML 里的三项兜底选项。`dept_id` 区分大小写，Qdrant 中 `HR_dept` 与
> `hr_dept` 属于两个不同租户，不能互相检索。

> ⚠️ **前缀必须一致**：服务启动时用的 `QDRANT_COLLECTION_PREFIX` 决定它读写哪个
> 集合。重启时漏掉这个环境变量，会连到默认的 `rag_chunks`，表现为“上传的文档
> 全都不见了”。

## 4. 接口说明

### 4.1 `POST /api/docs/upload`

`multipart/form-data`：

| 字段 | 必填 | 说明 |
|---|---|---|
| `file` | 是 | PDF 文件，最大 50 MB |
| `dept_id` | 是 | 部门 ID，作为租户隔离字段 |
| `doc_id` | 否 | 文档稳定标识；同一文档的多个版本必须保持一致，默认取文件名 |

返回：

```json
{
  "doc_id": "manual_handbook.pdf",
  "dept_id": "dept_A",
  "status": "created",
  "message": "文档首次入库：写入 3 个分块",
  "chunk_count": 3,
  "parent_count": 3,
  "written_points": 3,
  "deactivated_points": 0,
  "doc_hash": "cf0fb608...",
  "size_bytes": 2054,
  "elapsed_seconds": 0.89
}
```

- `status=created`：首次入库
- `status=updated`：内容变化，旧版本已软删除
- `status=unchanged`：`doc_hash` 相同，没有重复写入
- 错误码：`400` 文件为空/非 PDF/解析失败；`413` 超大；`422` 缺少表单字段；
  `502` Embedding 服务失败；`503` Embedding 未配置或 Qdrant 不可用；`500` 其他异常

> 原始 PDF **不会保存**：解析完成后临时文件立即删除，Qdrant 中只有切块文本、
> 向量和元数据。若业务需要保留原件，应另外接入对象存储。

### 4.2 `GET /api/docs/departments`

按 `dept_id` 聚合当前有效版本（`is_active=True`）的分块数量，返回按
`dept_id` 升序排列的列表：

```json
[
  {"dept_id": "HR_dept", "count": 7},
  {"dept_id": "finance_dept", "count": 7}
]
```

`count` 是**向量点（子块）数量**，不是文档数量。Qdrant 不可用时返回 `503`，
其他异常返回 `500`。

> 该接口会枚举所有已入库的 `dept_id`。生产环境必须加鉴权，避免租户信息泄露。

### 4.3 `POST /api/chat`（SSE）

请求体：

```json
{"dept_id": "HR_dept", "thread_id": "web-123", "question": "年假有几天？"}
```

事件（每个事件的 `data` 均为 JSON）：

| 事件 | 示例 | 说明 |
|---|---|---|
| `status` | `{"node":"retrieve","message":"正在检索知识库…"}` | 流程状态 |
| `answer` | `{"delta":"入职满"}` | 模型输出增量，前端按序拼接 |
| `done` | `{"thread_id":"web-123","dept_id":"HR_dept","is_relevant":"YES","retry_count":0}` | 本轮结束 |
| `error` | `{"message":"服务处理失败，请稍后重试"}` | 服务端异常 |

浏览器原生 `EventSource` 只支持 GET，无法携带 JSON 请求体，因此 `/demo` 使用
`fetch` + `ReadableStream` 消费。SSE 每 15 秒发送一次注释帧，防止代理断开长连接。

### 4.4 测试与运维脚本

| 脚本 | 用途 | 注意事项 |
|---|---|---|
| `scripts/smoke_test.py` | 核心链路：入库、幂等、租户隔离、版本控制、真实 LLM 问答 | **会删除并重建当前前缀的集合**，勿在有数据的环境执行；加 `--full` 可测 NO/重试路径 |
| `scripts/sse_smoke.py` | 验证 SSE 事件与首 token 延迟 | 需服务已启动且有可命中文档 |
| `scripts/upload_smoke.py` | 自动生成 PDF，验证 created/unchanged/updated 与 400 错误 | 会写入测试数据 |
| `scripts/list_documents.py` | 查询已入库文档与版本 | 只读；默认前缀来自配置 |

```powershell
.\.venv\Scripts\python.exe scripts\smoke_test.py --full
.\.venv\Scripts\python.exe scripts\sse_smoke.py --dept-id HR_dept
.\.venv\Scripts\python.exe scripts\upload_smoke.py
.\.venv\Scripts\python.exe scripts\list_documents.py --prefix rag_smoke
.\.venv\Scripts\python.exe scripts\list_documents.py --prefix rag_smoke --dept-id HR_dept --show-chunks
```

## 5. 问题与解决记录

> 记录格式：现象 → 根因 → 解决 → 涉及文件 → 验证。
> 新增条目请追加在本节末尾，并同步更新[已知限制](#6-已知限制与后续优化)。

### [2026-10-21] Windows 下 SSE 首 token 要等 16 秒

- **现象**：`/api/chat` 首 token 延迟约 16 秒，Qdrant 每次 REST 调用稳定耗时约 2.01 秒。
- **根因**：Windows 将 `localhost` 优先解析为 IPv6 `::1`，而 Qdrant 只监听 IPv4
  `127.0.0.1`，每次连接都要等待 IPv6 失败后再回退到 IPv4。
- **解决**：全部改用 `127.0.0.1`（`.env`、`.env.example`、`app/config.py` 默认值），
  并在配置说明中写明原因。
- **验证**：Qdrant 单次调用从约 2 秒降到 1–12 ms，端到端从 16.2 秒降到约 1.6 秒。

### [2026-10-21] 版本更新时 `deactivated_points` 统计错误

- **现象**：文档只改了 1 个分块，接口却显示停用了 3 个旧分块。
- **根因**：未改动的分块在版本间复用同一个确定性点 ID。先软删除旧版本、再
  `upsert` 新版本时，这些复用点被重新置为 `is_active=True`，导致统计与实际状态不一致。
- **解决**：软删除时用 `HasIdCondition` 把“本次即将写入的点 ID”放进 `must_not`，
  只停用真正消失的分块。
- **涉及文件**：`app/core/retriever.py`（`_deactivate_active_version`、
  `_active_document_filter`）。
- **验证**：`smoke_test.py` 断言 `deactivated_points` 等于更新前有效点数减复用点数；
  历史版本仍可 `is_active=False` 检索到。

### [2026-10-21] Qdrant 进程停止导致上传 500

- **现象**：上传过程中 Qdrant 进程退出，接口返回 500，错误信息不直观。
- **解决**：
  1. 用 `scripts/qdrant.ps1 start` 重新启动；
  2. 在 `app/api/documents.py` 中捕获 `qdrant_client.http.exceptions.ResponseHandlingException`，
     返回 `503 向量数据库不可用，请稍后重试`。
- **验证**：Qdrant 在线时上传正常；异常时返回 503 而非 500。

### [2026-10-21] PowerShell 5.1 读取脚本中文乱码 / 解析报错

- **现象**：`scripts/qdrant.ps1` 中的中文提示乱码，甚至出现语法解析错误。
- **根因**：PowerShell 5.1 会把**无 BOM 的 UTF-8** 当作系统 ANSI 编码读取。
- **解决**：用 UTF-8 **带 BOM** 重写脚本：

  ```powershell
  [System.IO.File]::WriteAllText(
    $path, $text,
    (New-Object System.Text.UTF8Encoding($true))
  )
  ```

- **经验**：本项目里所有 `.ps1` 文件都应保存为“UTF-8 with BOM”。

### [2026-10-21] `.venv` 中没有 pip

- **现象**：`.venv\Scripts\python.exe -m pip` 报 `No module named pip`。
- **解决**：依赖统一用 `uv` 安装：

  ```powershell
  uv pip install --python .\.venv\Scripts\python.exe pymupdf pymupdf4llm
  ```

- **注意**：不要直接激活环境后执行 `pip install`，会再次踩坑。

### [2026-10-21] 仓库里没有 `parse_pdf_to_markdown`

- **现象**：上传接口需要调用 PDF 解析函数，但项目里并不存在该实现。
- **解决**：新增 `app/core/pdf_parser.py`，使用 PyMuPDF + `pymupdf4llm`：
  - 保留 `#` / `##` 标题层级，保证 Parent-Child 切块能按章节切分；
  - 加密、无页面、无文本层（扫描件）等情况统一抛 `PDFParseError`；
  - 用户可见错误信息不包含临时文件路径，详细异常只写日志。
- **依赖**：`requirements.txt` 增加 `pymupdf>=2.0.0`、`pymupdf4llm>=2.0.0`。
- **验证**：中文 PDF 解析出正确标题与正文；损坏 PDF 返回 400。

### [2026-10-21] 上传后 `tmp/` 残留 UUID 临时文件

- **现象**：损坏 PDF 返回 400 是正确的，但 `tmp/` 中留下 `{uuid}.pdf`；
  服务日志出现 `PermissionError: [WinError 32]`。
- **根因**：Windows 上 `pymupdf.open(path)` 会持有文件句柄，句柄释放时机受
  线程/GC 影响；`unlink` 重试 3 次（约 0.6 秒）仍可能失败。
- **解决**：
  1. `pdf_parser.py` 改为先 `path.read_bytes()`，再用
     `pymupdf.open(stream=data, filetype="pdf")` 打开，并把 `Document` 直接传给
     `pymupdf4llm.to_markdown()`，不再按路径二次打开；
  2. 临时文件清理函数改为 `async`，重试等待使用 `asyncio.sleep`，避免阻塞事件循环。
- **涉及文件**：`app/core/pdf_parser.py`、`app/api/documents.py`。
- **验证**：损坏 PDF / 空文件 / 非 PDF 均返回 400，上传结束后 `tmp/` 无新增文件。

### [2026-10-21] 空的 `dept_id` 返回 422 而不是自定义 400

- **现象**：上传时传空字符串 `dept_id=`，FastAPI 返回 422。
- **根因**：Starlette 将“空表单值”视为字段缺失，请求在进入处理函数前就被校验拦截。
- **处理**：作为可接受行为记录在案；非空但非法的值仍由业务代码返回 400。
  若前端需要统一错误结构，可在网关层或自定义异常处理器中转换。

### [2026-10-21] 浏览器无法用 `EventSource` 调 `POST /api/chat`

- **现象**：原生 `EventSource` 只支持 GET，无法发送 JSON 请求体。
- **解决**：`app/api/demo.py` 使用 `fetch` + `ReadableStream` 手动解析 SSE 帧
  （按 `\n\n` 分帧，处理 `event:` / `data:`），并支持中止请求。
- **验证**：调试页可实时展示 status 时间线、流式答案与原始事件。

### [2026-10-21] 日志文件中文乱码

- **现象**：用 `Get-Content` 查看 `tools/uvicorn.err.log` 时中文乱码。
- **解决**：日志是 UTF-8，读取时显式指定编码：

  ```powershell
  Get-Content tools\uvicorn.err.log -Encoding UTF8 -Tail 50
  ```

### [2026-10-21] 扫描件 PDF 解析为空

- **现象**：纯图片 PDF 返回 `400 未能从 PDF 中提取到文本`。
- **原因**：没有文本层，需要 OCR。`pymupdf4llm` 会提示
  `RapidOCR is missing`，这只是能力提示，不是异常。
- **处理**：目前明确提示用户先做 OCR；后续可接入 OCR 服务或启用 RapidOCR。

### [2026-10-21] `.env.example` 中残留真实形式的 API Key

- **现象**：配置模板的 `LLM_API_KEY` 不是占位符，而是真实形式的密钥。
- **根因**：早期从 `.env` 复制配置时没有替换密钥。
- **解决**：模板改为 `sk-xxxxxxxxxxxxxxxxxxxxxxxx`；真实密钥只保留在
  未提交的 `.env` 中。
- **涉及文件**：`.env.example`。
- **验证**：模板中不再出现真实密钥；README 本身也不包含任何真实密钥。

### [2026-10-22] 执行 `copy .env.example .env` 后真实 API Key 丢失

- **现象**：`.env` 中的 `LLM_API_KEY` / `EMBEDDING_API_KEY` 变成 `sk-xxxx...`
  占位符，文件内容、大小、哈希和修改时间都与 `.env.example` 完全一致。
- **根因**：`.env` 已经存在时仍执行了 `copy .env.example .env`，模板覆盖了
  正在使用的配置。`Copy-Item` 默认保留源文件的修改时间，因此两个文件的
  时间戳也会完全相同，很容易误以为是“密钥自己被改了”。
- **解决**：从备份或服务商控制台恢复密钥；启动文档已改为“仅当 `.env`
  不存在时才复制模板”。
- **预防**：永远不要覆盖已有 `.env`；需要新增配置项时手动对比模板，只追加
  缺少的行。排查时先比对两个文件的哈希与修改时间：

  ```powershell
  Get-FileHash .env, .env.example
  Get-Item .env, .env.example | Select-Object Name, LastWriteTime
  ```

## 6. 已知限制与后续优化

按优先级从高到低：

1. **租户身份未认证**：`dept_id` 由客户端直接传入，没有与登录态 / 网关认证绑定。
   当前隔离只在“请求内逻辑”成立；生产环境必须从认证上下文派生 `dept_id`，
   并由中间件统一注入。
2. **并发上传同一 `doc_id` 未串行化**：Qdrant 没有跨点事务，两个并发上传可能
   导致版本状态互相覆盖。建议按 `dept_id + doc_id` 加进程内锁或分布式锁。
3. **原始文件未留存**：只保存切块与向量；若需要下载原件、审计或重新解析，
   需接入对象存储（如 MinIO / S3）。
4. **检查点使用 SQLite**：适合单实例；多实例部署应换成 PostgresSaver。
5. **Hybrid Search 未接通**：配置里已有 `BM25_TOP_K` / `RRF_K`，但检索仍以向量
   为主，尚未实现 BM25 + RRF 融合。
6. **`API_PREFIX` 配置未生效**：路由实际硬编码为 `/api`，与配置默认值
   `/api/v1` 不一致，后续应统一。
7. **`/`、`/demo`、`/docs` 与 `/api/docs/departments` 未做生产保护**：控制台、
   调试页和部门枚举接口均无鉴权，上线前应移除或加认证。
8. **测试仍以脚本为主**：建议把 `scripts/smoke_test.py` 的断言迁移到 pytest，
   接入 CI。
9. **`list_documents.py` 是 CLI 工具**：如果要做管理后台，需要新增带鉴权的
   `GET /api/docs` 接口（不要直接暴露脚本能力）。

## 7. 安全注意事项

- `.env` 存放真实密钥，**绝对不能提交到 Git**；`.env.example` 只能放占位符。
- `.env.example` 中的真实密钥已替换为占位符。如果该密钥曾经被提交、分享或
  推送过，请立即到服务商控制台**轮换**，仅改模板并不能让已泄露的密钥失效。
- 上传接口已限制后缀与 50 MB 大小，解析失败统一返回 400，不向用户泄露临时路径。
- `GET /api/docs/departments` 会枚举所有租户 ID，生产环境必须加鉴权，且只应
  返回当前登录用户有权访问的部门。
- 生产环境关闭 `DEBUG`，隐藏 Swagger / 调试页，并为所有接口加认证与限流。

## 8. 本文档的维护约定

- 每次修复 bug 或调整关键设计，在[问题与解决记录](#5-问题与解决记录) **末尾追加**
  一条，格式保持一致：

  ```markdown
  ### [YYYY-MM-DD] 一句话标题

  - **现象**：
  - **根因**：
  - **解决**：
  - **涉及文件**：
  - **验证**：
  ```

- 如果是“解决了已知限制”，同时从[已知限制](#6-已知限制与后续优化)中删除或更新对应条目。
- 命令、字段名、错误码必须与代码保持一致；不要粘贴真实密钥、完整日志中的敏感数据。
- 新增脚本或接口时，同步更新[目录结构](#2-目录结构)和[接口说明](#4-接口说明)。

## 9. 发布到 GitHub

仓库已通过 `.gitignore` 排除真实密钥、虚拟环境、Qdrant 二进制/向量数据、
会话检查点、临时文件和日志。提交前建议再确认一次：

```powershell
git status --short
git check-ignore -v .env tools\qdrant\qdrant.exe data\agent_checkpoints.sqlite
git grep --cached -n -E "sk-[A-Za-z0-9_-]{16,}"
```

`git grep` 只应命中 `.env.example` 和 README 中的 `sk-xxxxxxxx...` 占位符；
如果命中 `.env` 或其他真实密钥，先停止提交并轮换密钥。

首次推送：

```powershell
git init -b main
git add .
git commit -m "feat: 初始化企业级多租户 RAG Agent 系统"
git remote add origin https://github.com/<你的用户名>/<仓库名>.git
git push -u origin main
```

注意事项：

- `tools/qdrant/`（可执行文件约 86 MB，本地向量数据约 1.4 GB）、`.venv/`、
  `data/`、`tmp/` 和 `*.log` 都不应提交；Qdrant 二进制请按 3.1 节自行下载。
- GitHub 单个文件上限为 100 MB，当前忽略规则已避开该限制。
- 如果 `.env` 曾经被提交过，仅删除文件不会清除 Git 历史；需要使用
  `git filter-repo` 或 BFG 清理历史，并立即到服务商控制台轮换密钥。
- 远端仓库已有 README 时，先执行 `git pull --rebase origin main` 再推送。
