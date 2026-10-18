# M5 工作流、MCP 与调用链：运行和验收

## 实现边界

搜索仍使用现有 `SearchPlanner`、`PaperSearcher`、`SearchEvaluator`、`SearchReflector`；`app/workflows/search.py` 中的 LangGraph 接管执行顺序和循环。节点为 `plan → search → evaluate → reflect`，质量达标、无下一计划、达到轮次上限或每轮结束时发现超时后进入 `finish`。一次耗时很久的外部搜索请求不会被强行中断。每次运行有独立 `graph_thread_id`，中间状态保存在 `data/m2/search_graph.sqlite3`。这支持运行后检查状态；M5 不自动续跑被中断的外部搜索，重启后仍由 MySQL 将原任务标记为可重试。

问答的 `rag_reader/evidence_graph.py` 先取全局 Top-K，检查所选论文的片段覆盖，再为不足两条的论文定向补检；每篇只补到两条，以限制提示词长度。默认最多补检一次，可在 `RAG_EVIDENCE_MAX_EXPANSIONS` 设为 0–2。缺少某篇论文任何原文时，跨论文比较直接返回证据不足；仍有少量证据时将回答置信度上限设为 0.5。覆盖度只是一道确定性检查，不能代替 M6 的引用支持率人工评估。

`app/mcp/server.py` 提供独立 MCP 服务，工具直接调用原 FastAPI 接口，复用 MySQL 任务、入库与问答服务：

| 工具 | 用途 |
| --- | --- |
| `search_papers` | 创建候选论文搜索任务 |
| `get_job_status` | 获取搜索或入库任务进度与候选论文 |
| `start_ingestion` | 下载并入库选中的论文 |
| `ask_papers` | 对已入库的 1–5 篇论文提问 |
| `get_evidence` | 查看答案引用的论文片段与页码 |

通过 `python -m app.mcp.server --transport stdio` 可供本机进程启动；Compose 中的独立服务使用 Streamable HTTP，默认地址为 `http://127.0.0.1:8001/mcp`。MCP 端口仅绑定本机回环地址。远程部署前需再设计身份认证和访问控制。

Langfuse 追踪记录研究任务、搜索各节点、证据检索各节点和答案生成的耗时与结构化结果，答案生成还记录模型名和 token 用量。追踪不上传完整论文片段或密钥。将 `LANGFUSE_PUBLIC_KEY`、`LANGFUSE_SECRET_KEY`、`LANGFUSE_BASE_URL` 配在本机 `.env` 后启用；未配置时业务正常运行。Langfuse 官方自建方案需要额外的 PostgreSQL、ClickHouse、对象存储、Redis、Web 与 Worker，可使用[官方 Docker Compose](https://github.com/langfuse/langfuse/blob/main/docker-compose.yml) 单独部署。本项目的基础 Compose 不隐式启动这些服务。

## 启动与验证

```powershell
Copy-Item .env.example .env
# 设置 MySQL 密码；已有 .env 可以直接沿用。
docker compose up --build -d
docker compose ps
```

如端口已被占用，可在 `.env` 设置 `SCHOLARTRACE_MCP_PORT`。MCP 客户端地址为 `http://127.0.0.1:<端口>/mcp`。M5 离线测试：

```powershell
python -m pytest -q test_m5_workflow.py test_m4_async.py test_m3_retrieval.py test_m3_api.py test_m2_api.py test_m1_rag.py test_m1_service.py
```

真实验收依次检查：搜索图的持久化状态与退出理由、MCP 客户端 `list_tools` 和只读任务/证据调用、Milvus 多论文证据分支、Langfuse 实例中的同一请求子节点。实时搜索与生成会调用外部服务，需使用本机配置的密钥；离线回归不需要这些密钥。

## 本机验收记录

2026-10-17，本机使用已验收的 M4 镜像叠加 M5 依赖，并将当前源码挂载进 Compose 做联调。MySQL、Redis、Milvus、API、MCP、worker、beat、Streamlit 及 Milvus 的两个依赖服务共 10 个容器正常运行；API、UI 与 MCP 分别映射到本机 18000、18501、18001 端口。

| 检查 | 结果 |
| --- | --- |
| LangGraph 搜索退出与持久化 | 假检索器验证质量达标、最大轮次、轮后超时三条退出路径；SQLite 中有多步 checkpoint |
| 证据补检 | 假检索器验证仅扩展覆盖不足论文及补检次数上限；真实 Milvus 集合 `paper_chunks_m4_accept_20261017` 对两篇已入库论文补检后得到 7 条有页码片段，分别为 2、5 条 |
| MCP 协议 | 独立 MCP 客户端通过真实 HTTP 端点列出 5 个工具，读取现有任务状态，读取已有答案的 5 条引用证据 |
| Langfuse SDK 导出 | SDK 向本机临时 OTLP 接收器发出 1 次请求，包含研究任务、真实搜索图的 5 个节点和答案生成共 8 个 span；抽查的 3 条父子关系正确 |

2026-10-18 在隔离环境中运行 M1–M5 受影响路径回归，共 **32 项通过**；生产 API 容器自带 `DATABASE_URL`，会覆盖这些测试的临时 SQLite 配置，因此回归测试须在不注入生产数据库环境变量的容器中执行。

同日从 `deploy/Dockerfile` 构建 M5 正式镜像，`pip check` 无依赖冲突；通过 Compose 启动全部 10 个服务，MySQL、Redis、Milvus、API、UI 的健康检查通过，MCP、worker、beat 保持运行。正式镜像中的 MCP 客户端列出 5 个工具，并从现有答案读取 5 条证据；Milvus 上两篇已入库论文经一次补检得到 7 条片段，分别为 2、5 条，覆盖检查结果为 `covered`；本机 OTLP 接收器也收到了 8 个有父子关系的追踪节点。本机验收使用 `tmp/m4_accept_override.yaml` 将 API/UI 映射到 18000/18501，并复用已有测试集合；MCP 仍映射到 8001。

这轮未调用新的 DeepSeek 生成接口，也未连接实际 Langfuse Web 实例；因此不能把上述结果写成真实回答质量提升或 Langfuse 页面验收。真实质量收益将在 M6 固定问题集上衡量。
