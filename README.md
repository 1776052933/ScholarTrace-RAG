# AutoPaper：学术论文检索与证据问答

输入研究问题后，项目依次完成查询理解与拆解、多来源论文检索、引文扩展、论文排序、开放全文获取，以及基于论文内容的摘要、知识抽取和问答。

M2 已接入 FastAPI、MySQL/Alembic 和 Streamlit HTTP 客户端；M3 已完成 Milvus 检索消融和复杂 PDF 首轮对照；M4 已接入 Redis/Celery 选文入库、失败重试和多论文证据问答；M5 接入 LangGraph 搜索与证据补检工作流、MCP 服务及可选的 Langfuse 追踪。运行及验收见 [M5 运行与验收](docs/M5_RUN_AND_ACCEPTANCE.md)、[M4 运行与验收](docs/M4_RUN_AND_ACCEPTANCE.md)、[M3 运行与验收](docs/M3_RUN_AND_ACCEPTANCE.md)和[升级计划](docs/UPGRADE_PLAN.md)。数据集位置见[数据集说明](docs/DATASETS.md)；[M0 评测说明](evaluation/M0.md)、[M0 十题基线](evaluation/M0_BASELINE.md)和[M1 对照](evaluation/M1_COMPARISON.md)可用于同输入复测。[版本归档与 M1 工程验收](docs/VERSION_ARCHIVE_AND_M1_ACCEPTANCE.md)记录了早期源码和验收边界。

项目采用模块化工作流。LangGraph 编排 `SearchAgent` 的规划、检索、评价和多轮调整，并控制 RAG 证据补检；其他以 Agent 命名的模块承担排序等职责。当前实现没有多个自主 Agent 之间的任务委派与协作机制。

## 主要模块

| 目录 | 职责 |
| --- | --- |
| `query_understanding/`、`query_decomposition/` | 理解和拆解研究问题 |
| `search_agent/` | 搜索 Google Scholar、arXiv、OpenAlex，并扩展引文关系 |
| `ranking_agent/` | 用 BGE-M3、元数据特征和大模型评分排序论文 |
| `paper_downloader/`、`literature_database/` | 获取开放全文，管理下载状态及论文记录 |
| `rag_reader/` | 解析全文、切分、向量检索、摘要、知识抽取与证据问答 |
| `evaluation/` | M0 固定语料检索、QASPER 全文问答、M3 检索消融与在线搜索评测 |
| `Auto_Paper_UI/` | Streamlit HTTP 界面、候选选文、入库进度、多论文问答与导出 |
| `app/api/`、`app/services/`、`app/tasks.py` | FastAPI 契约、MySQL 持久化状态、Celery 入库、论文问答与领域工作流 |
| `app/storage/`、`migrations/` | MySQL 业务表和 Alembic 迁移 |
| `app/workflows/`、`app/mcp/`、`observability/` | LangGraph 搜索与证据图、MCP 工具、Langfuse 追踪 |

## 运行

Compose 配置包含 MySQL、Milvus、Redis、API、MCP、Celery worker/beat 和 UI。启动前参阅[M5 运行与验收](docs/M5_RUN_AND_ACCEPTANCE.md)：

```powershell
Copy-Item .env.example .env
# 在 .env 中填写本地数据库密码；实时检索/生成需要对应 API Key
docker compose up --build -d
```

界面位于 `http://127.0.0.1:8501`，API 文档位于 `http://127.0.0.1:8000/docs`，MCP 位于 `http://127.0.0.1:8001/mcp`；端口被占用时可在 `.env` 中设置 `SCHOLARTRACE_UI_PORT`、`SCHOLARTRACE_API_PORT` 和 `SCHOLARTRACE_MCP_PORT`。模型和检索密钥由 API 服务端读取；不要提交 `.env` 或本地密钥。首次摄取论文需要下载 BGE-M3 模型，亦可按 M4 验收说明挂载已下载的模型。无 API Key 仍可运行离线测试和已有论文入库；实时搜索及生成回答需要相应密钥。

运行离线测试：

```powershell
python -m pytest -q -m "not slow"
```

数据集、PDF、缓存、运行结果和本地密钥均被 `.gitignore` 排除。
