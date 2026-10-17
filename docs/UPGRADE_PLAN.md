# ScholarTrace-RAG 升级计划

> 状态：规划稿；目标技术栈尚未实现。项目定位：面向论文发现、选文、阅读与可追溯问答的 RAG 系统。独立的多 Agent 项目另行建设。

## 1. 项目目标与成功标准

用户可从研究问题出发，先找到候选论文、选择要处理的论文，再完成 PDF 下载、解析、索引和单篇/多篇问答。回答须给出论文、章节、PDF 页码和原文证据；证据不足时说明。已有论文库可以直接问答，新论文可以增量入库。系统在本地可复现运行，并用固定数据、基线和消融实验说明主要改造的效果与代价。

最终交付：交互界面、FastAPI 服务、MCP 工具服务、MySQL 元数据、Milvus 检索索引、可靠入库任务、评测报告、Docker Compose、CI、演示说明。技术栈必须服务于真实功能和可解释的技术选型。

## 2. 当前实现与差距

| 方面 | 当前实现 | 目标变化 |
| --- | --- | --- |
| 流程 | main.py 串行组织查询理解、检索、排序、下载、RAG；SearchAgent 有规划/搜索/评估/反思循环 | 提取领域服务；给需要状态和循环的搜索/证据流程使用 LangGraph |
| 前后端 | Streamlit 经 Auto_Paper_UI/backend.py 启动 main.py 子进程；任务状态保存在内存 | Streamlit 经 HTTP 调 FastAPI；任务状态持久化和恢复 |
| 存储 | SQLite/SQLAlchemy、Chroma、本地 PDF | MySQL 业务数据、Milvus 检索索引、PDF 持久化卷 |
| 文档 | PyMuPDF、文本清洗、切块、BGE-M3 dense embedding | 保留快速解析，复杂版式按需引入 Docling；页码与章节贯穿全链路 |
| 检索 | dense Top-K，无片段级重排；问答来源主要为 chunk ID | dense + BM25 混合召回、RRF、片段重排、可定位的结构化引用 |
| 评测 | 现有 LitSearch 相关评测主要测实时搜索/排序，PDF/RAG 关闭 | 固定语料检索、全文 QA、自建 PDF 页码集分别评测 |
| UI | 已有检索页、关系图、单篇追问、导出 | 候选选文、进度、论文库、多篇对比、证据阅读和失败恢复 |

先修复已发现的正确性问题：rag_reader/parser.py 能提取逐页内容，但 rag_reader/pipeline.py 清洗后把合并文本交给切块器，未传入页对象，chunk 页码容易丢失；rag_reader/embedding.py 模型加载失败会静默回退到哈希向量，存在不同向量混入索引的风险。这两项优先于框架迁移。

## 3. 目标使用流程

1. 用户输入研究问题，系统先匹配已有论文库；可直接问答，也可发现新论文。
2. 新搜索先获取元数据、去重与排序。首屏展示约 10–20 篇候选的题目、摘要、年份、来源、相关理由、PDF 状态。默认预选约 3–5 篇，允许调整与查看更多。
3. 用户确认后才下载与索引；另提供一键自动分析。用户指定的论文失败时显示原因和替代候选，不静默替换；自动模式可尝试下一篇。
4. 每篇论文展示下载、解析、索引、完成/失败阶段，允许重试。已入库论文避免重复处理。
5. 用户可对单篇或多篇问答。引用可展开原文、打开 PDF 对应页；跨论文结论标明各自证据和分歧；证据不足时拒答或建议补充文献。
6. 保留关系图和导出，增加研究会话、论文库和预建索引的演示论文。现场演示不依赖实时学术 API。

## 4. 目标架构与责任

架构主链：Streamlit → FastAPI → 领域服务 → MySQL / Milvus / 模型适配层。MCP 服务作为领域服务的另一入口。耗时任务经 Redis + Celery → 下载/解析/索引 Worker → PDF 持久化卷、MySQL、Milvus。LangGraph 承接搜索扩展与证据检查流程。Langfuse 观察 API、工作流和 Worker 的关键调用。

### 推荐的代码边界

| 目录（建议） | 职责 |
| --- | --- |
| app/api/ | FastAPI 路由、请求/响应 schema；不直接实现检索算法 |
| app/services/ | 论文发现、入库、问答、导出等可被 API/MCP 共同调用的业务服务 |
| app/workflows/ | LangGraph 状态定义、节点、条件边与检查点 |
| app/ingestion/ | 下载、解析、清洗、切块及 Celery 任务 |
| app/retrieval/ | embedding、Milvus/BM25、融合、重排和证据组装 |
| app/storage/ | MySQL 模型/仓库、PDF 资源、Milvus 适配器及 Alembic 迁移 |
| app/llm/ 与 app/observability/ | 模型适配、提示词版本和 Langfuse 追踪 |
| app/mcp/、Auto_Paper_UI/ | MCP 协议入口、Streamlit HTTP 客户端和界面 |
| evaluation/、tests/、deploy/ | 固定基准、关键测试、Compose 与演示配置 |

从现有目录逐步移动代码，先抽象接口与数据契约，再替换调用者。旧 CLI 可以保留为调用领域服务的薄入口，避免迁移期间同时维护两套业务逻辑。
- 领域服务统一实现论文发现、选择/入库、任务查询、证据检索、问答和导出。API 与 MCP 不复制业务规则。
- LangChain 用于模型、提示词、文档和工具抽象，统一重复的 DeepSeek/OpenAI 兼容调用；不强行改写所有业务代码。Milvus 混合检索若封装不足，直接使用 PyMilvus。
- LangGraph 只管理现有 SearchAgent 循环，以及证据充分性判断、扩展搜索、重试与退出条件。给工作流设置最大迭代次数、超时和检查点。普通固定步骤的 PDF 处理交给 Celery。
- MySQL 是论文、会话和任务状态的业务真相；Milvus 是可重建的检索索引。PDF 首期放本地持久化卷。Worker 在 Docker/WSL 内运行。
- Langfuse 记录模型/提示词版本、检索结果、工具调用、耗时、token、费用和错误；不记录密钥。它用于运行诊断，不能代替离线效果评测。

## 5. 技术栈替换与新增

| 层次 | 当前 → 目标 | 选型目的与约束 |
| --- | --- | --- |
| 接口 | Python/Pydantic → 增加 FastAPI | 前端、MCP 和测试共用稳定的请求/响应契约 |
| 界面 | Streamlit/Plotly → 保留并升级 | 复用已有关系图和导出，重点完善选文与证据体验 |
| 工作流 | 自写 SearchAgent 循环 → LangGraph 局部迁移 | 显式状态、分支与循环退出；不承担全部业务逻辑 |
| 模型 | 多处自写客户端 → LangChain + 集中模型适配层 | 统一超时、重试、限流、结构化输出和版本配置 |
| PDF | PyMuPDF → PyMuPDF + Docling 按需回退 | 普通 PDF 保持速度；复杂版式按标注集验证解析收益 |
| 切块 | 普通文本切块 → 页码/章节感知切块 | 保留可追溯原文和页面位置 |
| Embedding | BGE-M3 dense → 保留并锁定模型/维度/索引版本 | 模型失败显式报错，禁止静默回退 |
| 检索 | Chroma dense → Milvus dense + 内建 BM25 sparse + RRF | 语义与关键词互补；用固定集调融合参数 |
| 重排 | 无片段级重排 → BGE reranker 等 cross-encoder | 对召回段落重排，和现有论文级排序区分 |
| 关系库 | SQLite → MySQL + SQLAlchemy 2 + Alembic | 业务状态持久化、可重复迁移和旧数据迁移 |
| 异步 | 内存任务/子进程 → Redis + Celery | 耗时索引、失败重试、服务重启后状态可查询 |
| 协议 | 无 → MCP Python SDK | 供外部客户端调用真实论文能力 |
| 观测/评测 | 基础日志/局部评测 → Langfuse + Ragas + 金标准指标 | 同时衡量质量、速度、成本和失败率 |
| 交付 | 手工环境 → Docker Compose + CI + .env.example | 本地复现与自动检查，观测组件可分 profile |

上表是目标清单，不代表目前已接入。框架迁移本身不预设质量提升，每项优化应有输入、输出和验收依据。

## 6. 数据与接口契约

建议实体：Paper（稳定 DOI/arXiv ID）、PaperAsset（PDF 路径、哈希、来源）、IngestionJob（阶段、错误、重试）、Chunk（paper_id、页码、章节、原文、解析版本）、IndexVersion（embedding 模型、维度、检索配置）、ResearchSession、Question、Answer、Evidence。以 paper_id + PDF hash + pipeline version 保证重复入库幂等。写入 Milvus 并验证可检索后才将论文标为 READY；重建索引时清理旧版本 chunk。

任务状态：PENDING → DOWNLOADING → PARSING → INDEXING → READY；失败进入 FAILED，保留失败阶段、原因和可重试标记。界面通过状态接口查询，不从命令行日志判断结果。

首批 HTTP 接口建议：POST /search、POST /papers/ingestions、GET /jobs/{id}、GET /papers、POST /questions、GET /answers/{id}/evidence。回答引用至少包含 paper_id、标题、chunk_id、PDF 页码、章节、原文摘录与 evidence_id。MCP 首批工具：search_papers、start_ingestion、get_job_status、ask_papers、get_evidence；共用领域服务，提供外部客户端演示。

## 7. 分阶段实施与验收

主依赖按 M0 → M1 → M2 → M3 → M4 → M5 → M6 推进。数据准备与 UI 草图可提前开展；每阶段保留可运行的核心链路。编号不是工期。

| 阶段 | 工作内容 | 完成标准 |
| --- | --- | --- |
| M0 基线冻结 | 记录当前架构、依赖、运行方式和已知问题；固定论文/问题样例，保存当前检索、问答输出和耗时 | 同一输入可重跑；实时搜索成绩不冒充 LitSearch 固定语料官方分数 |
| M1 正确性与边界 | 修复页码丢失和 embedding 静默回退；统一论文 ID、引用结构、异常、模型配置；提取领域服务 | 回答能追到 PDF 页；模型加载失败显式报错；原 CLI/界面核心流程仍可用 |
| M2 API 与 MySQL | FastAPI 契约、Streamlit 改 HTTP、MySQL/Alembic、Docker Compose 基础服务、SQLite 数据迁移/备份、任务状态落库 | API 文档可访问；服务重启后元数据与状态仍在；前端不再启动 main.py 子进程 |
| M3 解析与检索 | Milvus 替换 Chroma；dense 基线、BM25/RRF、cross-encoder；页/节感知切块与 Docling 复杂 PDF 路径 | 同一测试集有逐项消融；引用可定位页与原文；新旧索引不混用 |
| M4 异步与交互 | Redis/Celery 入库、重试与幂等；候选选文、预建库、进度、失败处理、多篇问答和导出 | 正常/下载失败/解析失败/服务重启可演示；用户可选择论文 |
| M5 工作流与 MCP | 搜索循环迁入 LangGraph、证据不足分支和次数上限；MCP 共享服务；Langfuse 追踪 | 图的节点与退出条件可解释；外部 MCP 客户端可搜索/问答；调用链可追踪 |
| M6 评测与交付 | 固定检索/QA 基准、人工审查和错误分析；完善 Compose、CI、种子数据、README、架构图、演示脚本 | 新环境按文档启动；无密钥核心测试自动通过；报告含基线、收益、代价与失败案例 |

### 各阶段首次接入的技术

| 阶段 | 首次接入或明确升级 | 沿用的已有能力 |
| --- | --- | --- |
| M0 | PyArrow 读取 LitSearch Parquet；固定数据适配与自定义评测脚本 | Python、现有检索流程和 pytest |
| M1 | LangChain 的模型/提示词抽象（在统一模型适配层时引入） | Pydantic、PyMuPDF、BGE-M3、现有 RAG 主链 |
| M2 | FastAPI、Uvicorn、MySQL 驱动、Alembic、Docker Compose 基础服务 | SQLAlchemy、Streamlit |
| M3 | Milvus、PyMilvus、Milvus 内建 BM25、RRF 融合、cross-encoder 重排、Docling 按需解析 | BGE-M3 embedding、PyMuPDF |
| M4 | Redis、Celery；Streamlit 界面和 Plotly 图表升级 | FastAPI、MySQL、Milvus |
| M5 | LangGraph、MCP Python SDK、Langfuse | LangChain、领域服务和异步入库 |
| M6 | Ragas、GitHub Actions；完善 Compose 配置、评测报告与演示交付 | 复用并联调 M0–M5 的完整系统 |

BM25、RRF 和 RAG 是检索方法或系统范式，并非必须单独安装的框架。Docker Compose 从 M2 引入，之后随 Milvus、Redis/Celery、Langfuse 等服务逐步扩充；M6 才做最终集成与复现检查。M0 已建立评测基线，M6 再用冻结版本做全面评测。

## 8. 数据集、训练与评测

| 数据 | 用途 | 使用边界 |
| --- | --- | --- |
| [LitSearch](https://huggingface.co/datasets/princeton-nlp/LitSearch) | 固定语料论文发现/排序 | 用其固定 corpus 与 gold paper ID；实时学术 API 检索另报 |
| [QASPER](https://huggingface.co/datasets/allenai/qasper) | 单论文全文 QA 与段落级证据 | 主要是结构化全文，不能直接证明 PDF 页码定位 |
| 自建 PDF 留出集 | PDF 页码引用、多论文综合、拒答、解析异常 | 建议约 10–20 篇合法取得的 PDF、30–50 个人工核对问题；记录 gold 论文/页/片段/答案 |
| [ScholarQABench](https://github.com/AkariAsai/ScholarQABench) | 多论文综合的后续扩展 | 先完成基础证据评测，再决定接入 |

无需训练大模型。解析、切块、embedding 入库属于索引构建，不是训练。开发集用于调参；测试集仅用于冻结版本的最终评价。不能把 gold 答案或证据传给问答模型。记录数据集修订、模型版本、提示词、索引版本、参数与随机种子；公开基准可能被模型见过，需用新建留出集补充。

对于中文提问、英文论文的跨语言场景，检索前可做可追踪的英文查询改写与术语扩展，并用中英文问题分组验证收益，避免 BM25 只匹配中文词而漏掉英文证据。

指标分层：论文发现看候选召回、Recall@K、MRR、去重质量、API 失败率；证据检索看 Recall@K、Precision@K、nDCG、页码定位正确率；回答看人工正确率、引用支持率、拒答准确率，Ragas 的 faithfulness 等仅作辅助；工程看入库成功率、P50/P95 延迟、token/费用、重试与恢复。

固定同一语料、问题、模型和候选规模，逐步比较：当前 Chroma dense Top-K → Milvus dense → dense + BM25/RRF → 加重排 → 结构切块/解析优化 → 证据检查工作流。每步同时报告质量、延迟和成本，保存逐题结果及失败案例，不只展示最佳样例。

## 9. UI 升级清单

- 首页：研究问题、已有库/发现新论文入口、示例问题和预建语料。
- 候选列表：摘要、相关理由、来源、PDF 状态、排序筛选、勾选、查看更多；确认后下载。
- 处理进度：逐篇显示阶段和失败原因，提供重试；处理时仍可使用已入库论文。
- 阅读问答：单篇/多篇选择、追问历史、编号引用、PDF 页与摘录、证据不足提示。
- 研究结果：保留关系图、概览和导出；导出必须包含完整来源，不仅是内部 chunk ID。
- 视觉与体验：统一字体、间距、色彩和论文卡片层级；答案旁提供固定的证据面板；明确空状态、加载状态与错误恢复；窄屏可用，不把队列 ID、Milvus 集合名直接放在普通用户界面。

界面按“搜索 → 选文 → 等待/恢复 → 引用问答 → 导出”的完整任务走查验收。

## 10. 风险、取舍与面试材料

1. 学术 API 可能限流或缺 PDF。演示和正式基准用固定语料；在线搜索单独测试。
2. Milvus、MySQL、Redis、Langfuse、Docling/重排模型可能占用较多本机资源。Compose 区分核心与观测 profile，解析/重排按需启用并记录硬件；目标版本仍包含 Milvus、MySQL、MCP。
3. PDF、MySQL、Milvus 跨系统写入可能部分成功。任务状态、幂等键、重试、索引版本和重建工具必须配套。
4. 面试时要能解释为什么搜索循环用图、为什么文档入库用任务队列、两者的替代方案和代价。
5. 只提交可分发的示例 PDF 或元数据；真实密钥通过环境变量注入。

后置事项：模型微调、全量知识图谱、复杂权限/多租户、Kubernetes、对象存储、高并发压测和额外多 Agent 角色。只在基线暴露相应瓶颈或目标岗位明确要求时加入。

最终面试材料至少包括架构图、核心链路 demo、固定数据与标注方法、逐项对照结果（质量/延迟/成本）、典型失败案例、选型理由、个人负责代码和复现说明。

## 11. 参考资料

- [LangGraph 官方文档](https://docs.langchain.com/oss/python/langgraph/overview)、[Milvus BM25 全文检索](https://milvus.io/docs/full-text-search.md)
- [FastAPI 后台任务](https://fastapi.tiangolo.com/tutorial/background-tasks/)、[Celery](https://docs.celeryq.dev/en/stable/getting-started/introduction.html)
- [Docling](https://github.com/docling-project/docling)、[Langfuse](https://langfuse.com/docs/observability/data-model)、[Ragas 指标](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/)
- [MCP 工具规范](https://modelcontextprotocol.io/specification/2025-11-25/server/tools)、[Docker Compose](https://docs.docker.com/compose/)、[Alembic](https://alembic.sqlalchemy.org/en/latest/)



## 当前实施记录（2026-10-15）

M0 已实现固定 QASPER 全文问答与 LitSearch 固定语料 dense 评测入口，保存 manifest、逐题输出和耗时；QASPER dev 固定十题已完成基线，具体结果见 `evaluation/M0_BASELINE.md`。LitSearch 完整 64,183 篇索引尚未在当前 CPU 环境运行，32 篇仅用于流程验证。

M1 已完成逐页 PDF 入库与结构化引用、BGE-M3 失败显式报错、索引模型指纹与维度保护、统一 RAG 配置和错误代码、LangChain Core 提示词模板、稳定论文 ID 的搜索到注册传递，以及业务工作流抽取至 `app/services/workflow.py`。旧 CLI 继续调用同一服务。旧 Chroma 集合需重新摄取到 `paper_chunks_m1`。M2 及之后的目标技术栈仍按上文计划实施。


## M2 当前进度（2026-10-15）

已实现 FastAPI 契约与任务/问答接口、MySQL/Alembic 数据模型与初始迁移、旧 SQLite 论文库备份导入、Docker Compose 基础服务和 Streamlit HTTP 客户端。API/UI 的真实界面与 API 重启验收先在独立 SQLite 数据库完成，随后在 Windows 原生 MySQL 8.4.11 上完成实际 Alembic 迁移、M1 旧库导入、有效论文重建索引、真实浏览器检索/下载/问答以及服务重启恢复。下载/入库部分失败的任务状态与界面告警已补齐。Docker Compose 运行路径按当前本机开发安排后移，仍需有 Docker/WSL 的环境复核，详见 [M2 运行与验收](M2_RUN_AND_ACCEPTANCE.md)。


## M3 当前进度（2026-10-17）

已加入 Milvus 2.6 dense/BM25/RRF 索引适配、可选 BGE cross-encoder 片段重排、Docling 复杂 PDF 解析路径和论文索引版本字段。本机 MySQL 迁移、旧索引拦截、Milvus 服务端联调、六服务 Compose 启动与重启持久性已验证。QASPER dev 固定 30 题的六路检索消融全部完成；默认 BGE 重排在开发样本上改善证据召回代理，但 CPU 延迟显著增加，因此默认关闭。十篇 PDF 的首轮标注、解析及 22 题问答对照也已完成；金标准尚待独立复核，扫描件未覆盖。详见 [M3 运行与验收](M3_RUN_AND_ACCEPTANCE.md)及[Docling 质量对照](M3_DOCLING_QUALITY_LABELING.md)。


## M4 当前进度（2026-10-17）

已实现 Redis/Celery 选文入库、MySQL 逐篇持久化状态、幂等提交、失败重试、心跳与中断恢复、Streamlit 候选选择与进度、多论文证据问答和 Markdown 导出。九服务 Compose、两篇真实 PDF 入库、解析失败后重试、API/worker 重启持久化、真实 Milvus 多论文检索及界面连接 API 均已验收。生成质量本轮只做无外部模型的契约测试，后续在固定多论文问题集评估；浏览器视觉布局仍待人工走查。详见 [M4 运行与验收](M4_RUN_AND_ACCEPTANCE.md)。
