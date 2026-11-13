# ScholarTrace RAG：学术论文检索与证据问答

输入研究问题，系统会拆解检索式、搜索并排序候选论文。选中文献或上传本地 PDF 后，系统在后台建立索引，支持带原文页码的流式问答和跨论文比较。

## 快速启动

准备 Docker Compose 和 Python 3.11。在仓库根目录执行：

```powershell
python -m scripts.init_deployment_env
# 在 .env 填写模型配置，或按下方说明接入本地 Qwen
docker compose --env-file .env build api
docker compose --env-file .env up -d --no-build --wait
```

首次运行需准备 BGE-M3、BGE 重排和 Docling 模型缓存，或者允许首次下载。完整配置、端口和停止方式见[部署说明](docs/REPRODUCIBLE_DEPLOYMENT.md)，本地模型见[Qwen 接入](docs/LOCAL_QWEN.md)。打开 Streamlit（默认 `http://127.0.0.1:8501`），API 文档在 `http://127.0.0.1:8000/docs`，MCP 入口在 `http://127.0.0.1:8001/mcp`。

**演示顺序**：上传两篇 PDF → 查看入库进度 → 查询论文表格中的具体数值 → 连续追问 → 跨论文比较并点击引用查看 PDF 原页 → 刷新界面恢复会话。若使用论文搜索入口，可在入库前先展示候选列表和选文。

## 架构与结果

- Streamlit 通过 HTTP 调用 FastAPI，MySQL 保存论文、任务与会话。Redis 和 Celery 承接后台 PDF 入库。
- 入库用 Docling 保留章节和页码，BGE-M3 生成向量。Milvus 混合召回后用 BGE 重排，再由 LangGraph 编排证据检索、生成与引用解析。更多细节见 [RAG Reader](rag_reader/README.md)和[工作流结构](docs/LANGCHAIN_ARCHITECTURE.md)。
- 固定 QASPER 200 道开发题和 50 道留出题的原版答案 F1 分别为 0.3333、0.4156。候选提示词修正了回译语言错例，但留出集 F1 降到 0.3848，因此保留原版。这里的 F1 按官方词汇重叠算法计算，逐题事实与拒答核查见[扩展回归报告](docs/QASPER_PROMPT_REGRESSION_20261113.md)。
- 本地部署默认绑定回环地址。LangSmith 追踪和严格支持性核验均为可选，严格核验实验因过度拒答默认关闭。[追踪配置](docs/LANGSMITH_TRACING.md) · [核验对照](docs/ANSWER_GATE_EVALUATION_20261104.md)

## 主要模块

| 目录 | 职责 |
| --- | --- |
| `ai_components/` | 共享模型配置与 OpenAI 兼容调用层、LangChain ChatModel 适配及 LCEL 生成链 |
| `app/research/understanding/`、`app/research/decomposition/` | 理解和拆解研究问题 |
| `app/research/search/`、`app/research/ranking/` | 多来源论文搜索、引文扩展与候选排序 |
| `paper_downloader/`、`app/papers/` | 获取开放全文，管理下载状态及论文记录 |
| `rag_reader/` | 解析全文、切分、向量检索、摘要、知识抽取与证据问答 |
| `evaluation/` | 固定语料检索、QASPER 消融、跨论文证据对照、离线 Ragas 与人工复核工具 |
| `Auto_Paper_UI/` | Streamlit HTTP 界面、候选选文、入库进度、多论文问答与导出 |
| `main.py`、`app/cli.py` | 命令行启动入口与参数处理；业务执行由 `app/services/workflow.py` 负责 |
| `app/api/`、`app/services/`、`app/tasks.py` | FastAPI 契约、MySQL 持久化状态、Celery 入库、论文问答与领域工作流 |
| `app/storage/`、`migrations/` | MySQL 业务表和 Alembic 迁移 |
| `app/workflows/`、`app/mcp/`、`observability/` | LangGraph 论文发现与搜索图、MCP 工具、LangSmith 追踪 |

论文发现相关实现统一放在 [`app/research/`](app/research/README.md)，论文资源与持久化放在 [`app/papers/`](app/papers/README.md)。旧顶层 `query_understanding`、`query_decomposition`、`search_agent`、`ranking_agent` 和 `literature_database` 导入路径已移除。命令行继续使用 `python main.py`，程序化调用工作流请从 `app.services.workflow` 导入 `AutoPaperWorkflow`。测试代码统一放在 `tests/`，按 `ui/`、`evaluation/`、`downloader/`、`papers/`、`research/` 和根目录功能测试组织。

项目对外统一使用 **ScholarTrace RAG**。`AutoPaperWorkflow` 类名、`Auto_Paper_UI/` 目录及 `AUTO_PAPER_*` 环境变量是现有代码和部署的兼容接口，暂不改名；早期文档中的 AutoPaper 表示同一项目的历史阶段。

数据库由 `DATABASE_URL` 明确指定；未设置时，API、worker 和独立论文服务统一使用 `AUTO_PAPER_DATA_ROOT/app.sqlite3`（默认 `data/m2/app.sqlite3`）。Compose 配置 MySQL 并校验数据库类型，启动后可在界面侧栏查看实际后端和库名。旧 `literature.db` 不会被自动读取或删除，迁移方法见 [M2 数据迁移说明](docs/M2_RUN_AND_ACCEPTANCE.md)。

## 测试与文档

```powershell
python -m pip install -r requirements-test.txt
python -m pytest -q tests -m "not slow and not network"
```

- [部署与固定 PDF 回归](docs/REPRODUCIBLE_DEPLOYMENT.md)
- [当前评测入口](evaluation/CURRENT.md) · [数据集](docs/DATASETS.md)
- [会话与流式输出](docs/CONVERSATIONS_AND_STREAMING.md) · [本地论文库](docs/LOCAL_PDF_LIBRARY.md)
- [M0～M6 与历史验收索引](docs/HISTORY.md)

配置密钥保存在本地 `.env`。论文、模型缓存及原始运行结果保存在 Git 忽略的数据目录中。
