# AutoPaper：学术论文检索与证据问答

输入研究问题后，项目依次完成查询理解与拆解、多来源论文检索、引文扩展、论文排序、开放全文获取，以及基于论文内容的摘要、知识抽取和问答。

M2 已接入 FastAPI、MySQL/Alembic 和 Streamlit HTTP 客户端；运行方式见[M2 运行与验收](docs/M2_RUN_AND_ACCEPTANCE.md)。M3 已开始接入 Milvus 混合检索、重排和复杂 PDF 解析，真实 Milvus 联调仍待完成；当前状态见[M3 运行与验收](docs/M3_RUN_AND_ACCEPTANCE.md)。后续升级目标与阶段见[升级计划](docs/UPGRADE_PLAN.md)。数据集位置见[数据集说明](docs/DATASETS.md)；[M0 评测说明](evaluation/M0.md)、[M0 十题基线](evaluation/M0_BASELINE.md)和[M1 对照](evaluation/M1_COMPARISON.md)可用于同输入复测。[版本归档与 M1 工程验收](docs/VERSION_ARCHIVE_AND_M1_ACCEPTANCE.md)记录了源码、运行包校验和及验收边界。

项目采用模块化工作流。`SearchAgent` 在单个搜索流程内进行规划、检索、评价和多轮调整；其他以 Agent 命名的模块承担排序等职责。当前实现没有多个自主 Agent 之间的任务委派与协作机制。

## 主要模块

| 目录 | 职责 |
| --- | --- |
| `query_understanding/`、`query_decomposition/` | 理解和拆解研究问题 |
| `search_agent/` | 搜索 Google Scholar、arXiv、OpenAlex，并扩展引文关系 |
| `ranking_agent/` | 用 BGE-M3、元数据特征和大模型评分排序论文 |
| `paper_downloader/`、`literature_database/` | 获取开放全文，管理下载状态及论文记录 |
| `rag_reader/` | 解析全文、切分、向量检索、摘要、知识抽取与证据问答 |
| `evaluation/` | M0 固定语料检索、QASPER 全文问答、M3 检索消融与在线搜索评测 |
| `Auto_Paper_UI/` | Streamlit HTTP 界面、历史任务恢复和结果导出 |
| `app/api/`、`app/services/` | FastAPI 契约、持久化任务执行、论文问答与领域工作流 |
| `app/storage/`、`migrations/` | MySQL 业务表和 Alembic 迁移 |

## 运行

Compose 配置包含 MySQL、Milvus、API 和 UI；Milvus 容器运行路径尚未在当前 Windows 机器验收。启动前参阅[M3 运行与验收说明](docs/M3_RUN_AND_ACCEPTANCE.md)：

```powershell
Copy-Item .env.example .env
# 在 .env 中填写本地数据库密码和所需 API Key
docker compose up --build -d
```

界面位于 `http://127.0.0.1:8501`，API 文档位于 `http://127.0.0.1:8000/docs`。模型和检索密钥由 API 服务端读取；不要提交 `.env` 或本地密钥。没有 Docker 时，可按验收说明使用独立 SQLite 文件启动相同的 API/UI 链路；MySQL 实例验收仍需单独执行。

运行离线测试：

```powershell
python -m pytest -q -m "not slow"
```

数据集、PDF、缓存、运行结果和本地密钥均被 `.gitignore` 排除。
