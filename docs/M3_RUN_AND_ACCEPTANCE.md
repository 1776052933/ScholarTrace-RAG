# M3 解析与检索：运行、评测与验收

## 本阶段做了什么

- 增加 PyMilvus 2.6 的版本化 collection：BGE-M3 dense、Milvus 内建 BM25 sparse、RRF 混合召回。集合描述写入模型指纹、维度、解析与切块版本；不兼容时直接报错。
- 可选 BGE reranker v2 m3 cross-encoder 对候选片段重排，保留第一阶段分数、重排分数和来源页码。
- 可选 Docling 复杂 PDF 路径，按元素 provenance 保留页码、章节和表格 Markdown；常规 PDF 仍使用 PyMuPDF。
- API、搜索工作流、旧 CLI 问答入口统一根据 `RAG_VECTOR_BACKEND` 选择索引。
- MySQL 新增 `papers.index_signature`。切到 Milvus 时，旧 Chroma 的 `COMPLETED` 论文展示 `REINDEX_REQUIRED`，问答返回 409，避免把不存在于新索引的论文当成可用。
- 固定 QASPER 开发集的逐题消融脚本：Chroma dense → Chroma dense + rerank → Milvus dense → Milvus BM25 → Milvus hybrid/RRF → Milvus hybrid + rerank。所有路径使用同一清洗文本、同一 chunk 和同一 BGE-M3 向量。

## 2026-10-16 本机验收

2026-10-17 又完成了六服务的完整 Compose 启动、离线 PDF 入库检索和全套重启持久性验收，结果及边界见 [完整 Compose 本机验收](COMPOSE_FULL_ACCEPTANCE.md)。

| 检查 | 结果 |
| --- | --- |
| Docker 与 Milvus | Docker Desktop 4.95.0、WSL 2.7.13、Milvus 2.6.24、etcd、MinIO 已在本机启动并通过健康检查；Docker 数据位于 D 盘。首次真实启动修复了 Compose 中带下划线的存储服务主机名 |
| 新检索、重排、解析单元测试及回归 | 当前 M3 API/检索针对性测试 7/7 通过；完整非慢速回归 453 通过、22 跳过、3 个 slow 未运行 |
| Alembic M3 迁移和历史论文 | 原生 MySQL revision `9f1a43cd621e`。27 篇记录中，16 篇有完整资源的论文已重建到 `paper_chunks_m3`，2,435 个片段均可查询，16 个 MySQL 索引签名一致；其余 11 篇保持 `PENDING` |
| API、界面和容器重启 | `/api/v1/health` 正常，`/api/v1/papers` 显示 16 篇 `COMPLETED`、11 篇 `PENDING`。Streamlit 浏览器实测能打开，显示研究 API 已连接、MySQL 可用。刷新并重启 Milvus 后，2,435 个片段及 dense/BM25 查询仍可用 |
| QASPER dev 固定 10 题 | 10 篇、10 题，Chroma dense、Milvus dense/BM25/hybrid 四组均 10/10 完成。真实服务首次运行发现 PyMilvus 2.6 返回 `chunk_id` 主键而非 `id`，已修复并重跑 |
| BGE cross-encoder 本机实测 | 2026-10-16 尚未取得默认权重；2026-10-17 已完整取得 BGE reranker v2 m3 的约 2.27 GB 权重并在固定 QASPER dev 30 题上完成真实重排，结果见下文。上面的四组原验收结果本身仍不包含重排 |
| Docling 真实复杂 PDF | 截至 2026-10-16，接口和模拟 provenance 测试已通过；当时尚未安装 Docling 或用真实复杂版式 PDF 检验质量。2026-10-17 的补充结果见下文 |
| 外部生成问答 | 经用户明确授权，对《Attention Is All You Need》执行 1 次真实 DeepSeek 问答；`POST /api/v1/questions` 返回 200，回答和 2 条引用已写入本机 MySQL，证据读取接口返回 2 条 |

固定 10 题的结果保存在 `evaluation/runs/m3_ablation_20261016_165804`，运行参数为 `--split dev --limit 10 --seed 42 --top-k 5 --candidate-k 20 --skip-reranker`：

| 方法 | 成功题数 | 证据召回代理 | 查询 P50 |
| --- | ---: | ---: | ---: |
| Chroma dense | 10/10 | 0.667 | 0.146 秒 |
| Milvus dense | 10/10 | 0.667 | 0.150 秒 |
| Milvus BM25 | 10/10 | 0.313 | 0.007 秒 |
| Milvus hybrid/RRF | 10/10 | 0.490 | 0.146 秒 |

这些时间只计查询，不含模型加载、嵌入建库或生成。这个小样本上 hybrid 没有优于 dense，因此示例配置暂以 `dense` 为默认值；BM25 和 hybrid 保留为可切换策略，后续用更大的开发集及留出集复核。

### 2026-10-17 扩大开发集验收

在独立 Milvus 2.6 服务上，以 QASPER dev、`seed=42`、`limit=30`、`top_k=5`、`candidate_k=20` 和同一 BGE-M3 embedding 运行四组检索。29 篇论文产生 913 个可查询片段，入库耗时 520.6 秒；四组均完成 30/30 题。结果在本机 `evaluation/runs/m3_accept_20261017_30b`，包含 manifest、逐题记录和汇总。

| 方法 | 证据召回代理 | 查询 P50 | 查询 P95 |
| --- | ---: | ---: | ---: |
| Chroma dense | 0.711 | 0.167 秒 | 0.200 秒 |
| Milvus dense | 0.711 | 0.155 秒 | 0.189 秒 |
| Milvus BM25 | 0.431 | 0.005 秒 | 0.006 秒 |
| Milvus hybrid/RRF | 0.656 | 0.159 秒 | 0.202 秒 |

30 题中有 6 题的证据召回代理为 `null`，其余 24 题可比较；hybrid 对 dense 赢 1 题、输 5 题、平 18 题。这只支持保留 dense 作为当前默认检索方式，不支持宣称 Milvus 比 Chroma 提高了检索质量。查询耗时包含查询向量化，不含模型首次加载、入库或答案生成。BGE-M3 使用本机缓存组合的配置与权重，权重文件 SHA-256 为 `993b2248881724788dcab8c644a91dfd63584b6e5604ff2037cb5541e1e38e7e`。manifest 中模型修订为 `null`、Git HEAD 为 `null`，但保存了数据集哈希、代码哈希、依赖版本、样本 ID 和全部逐题输出；对外发布基准前需固定可复现的完整模型快照。

同一真实 collection 的错误 embedding 指纹被拒绝，且回读计数仍为 913，验证了新旧索引隔离。此前真实问答的两个引用也从本机 MySQL 回读到 PDF 第 3 页、`3.1 Encoder and Decoder Stacks`，保留原文片段与检索分数。

重排适配器另用已下载的 `cross-encoder/ms-marco-MiniLM-L6-v2` 在本机离线运行了真实推理：对“法国首都”样例，相关片段从首阶段第二名升至第一名，并保留原始分数、页码和章节；含模型加载总耗时 6.27 秒。此检查只证明 cross-encoder 代码链路可运行，不能代表默认 `BAAI/bge-reranker-v2-m3` 的质量或延迟，也不是 QASPER 重排消融。

Docling `2.140.0` 的 CPU 镜像完成真实论文 PDF 解析：15 页均有可定位记录，共 153 条，其中章节标题 28 条、表格类型 5 条。模型已缓存后，离线复跑耗时 41.8 秒。清洗和切块后得到 158 个片段，其中 9 个保留 `item_type=table`、页码和章节；第 8 页的 Table 2 仍可定位到 `5.4 Regularization`。人工检查发现第 1 页的作者排版被误判为表格，其余 4 条对应论文 Table 1–4，因此只能确认 Docling 路径可运行和来源信息可保留，不能据此宣称表格识别准确率或相对 PyMuPDF 的质量提升。逐条结果与复跑日志位于本机被 Git 忽略的 `data/m3/docling_records.json` 和 `data/m3/docling_smoke_offline.log`。

针对这篇 PDF 已补[逐页人工辅助标注及 PyMuPDF/Docling 对照](M3_DOCLING_QUALITY_LABELING.md)：Docling 对 4 张真实表格全检出，另将第 1 页作者区误判为表格；两条路径对所标 4 行关键表格值和 4 个正确页答案片段均为 4/4。这是扩展到十篇 PDF 之前的试标结果，金标准仍待独立复核。

### 2026-10-17 解析质量、下游问答与默认重排补充验收

十篇 PDF、115 页的首轮逐页标注得到 48 张数据表、21 条行值抽样和 22 个固定问题。[完整方法及失败例](M3_DOCLING_QUALITY_LABELING.md)与本机 `evaluation/runs/m3_pdf_layout_10pdf_20261017/report.json` 记录了：Docling 表格检测 TP 47、FP 3、FN 1；关键行值 21/21，PyMuPDF 页面纯文本为 20/21。两条解析路径使用相同的 BGE-M3、切块和 Chroma dense 检索后，Top-5 答案片段均为 20/22；Top-10 为 PyMuPDF 21/22、Docling 22/22。22 题各用两组 Top-5 片段调用 `deepseek-chat`，44/44 次均成功，答案包含金标准字面片段分别为 21/22、20/22；人工检查直接回答与指定表格出处后，两组均为 20/22。以上 PDF 标注是首轮，缺独立复核；这些数字不构成 Docling 整体优于 PyMuPDF 的结论。本地报告分别在 `evaluation/runs/m3_parser_retrieval_10pdf_20261017`、`evaluation/runs/m3_parser_answers_10pdf_20261017`。

[固定 QASPER 30 题默认 BGE 重排脚本](../evaluation/run_m3_cached_reranker_eval.py)在既有 Chroma dense Top-20 候选上运行真实 `BAAI/bge-reranker-v2-m3`，并核对 30/30 个 Top-5 基线与旧报告一致。24 道可评分题的 Top-5 证据召回代理从 0.711 升至 0.753；逐题 3 胜、3 负、18 平。CPU 上模型加载约 5.1 秒，重排 P50 9.43 秒、P95 10.85 秒。权重 SHA-256 为 `d9e3e081faff1eefb84019509b2f5558fd74c1a05a2c7db22f74174fcedb5286`；本机报告位于 `evaluation/runs/m3_bge_reranker_30dev_20261017/report.json`。此指标沿用 M0 词覆盖代理，不是 QASPER 官方 evidence F1；这次没有测试 Milvus hybrid 加重排，也没有测试重排后的生成答案。样本小且 CPU 延迟较高，**暂不把重排默认开启**。

### 2026-10-17 M3 收口：同题六路消融

旧评测集合 `m3_qasper_accept_20261017` 在收口时仍存在，但可查询片段数为 0，不能复用其候选片段。为保证对照有效，新建 `m3_qasper_close_20261017`，用与此前相同 SHA-256 的 QASPER dev 文件、相同 `seed=42` 的 30 个问题、相同 BGE-M3 权重、1000/200 切块和 Top-20 候选重新建立 29 篇论文索引。六组均完成 30/30 题、无执行失败。完整 manifest、逐题记录和汇总在本机 `evaluation/runs/m3_close_20261017_30`。

| 检索方案 | Top-5 证据召回代理 | 查询 P50 |
| --- | ---: | ---: |
| Chroma dense | 0.711 | 0.162 秒 |
| Chroma dense + 默认 BGE 重排 | 0.753 | 9.609 秒 |
| Milvus dense | 0.711 | 0.152 秒 |
| Milvus BM25 | 0.431 | 0.005 秒 |
| Milvus hybrid/RRF | 0.656 | 0.153 秒 |
| Milvus hybrid/RRF + 默认 BGE 重排 | 0.753 | 9.646 秒 |

24 道题有可计算的证据召回代理；Milvus hybrid 加重排相对不重排逐题 5 胜、2 负、17 平。重排在该开发样本上提高代理分数，但本机 CPU 延迟增加约 9.5 秒，仍保持默认关闭。六路消融只测检索，不测生成答案；指标不是 QASPER 官方 evidence F1，也不是未调参留出集结论。旧集合为何变空尚未查明，故本次使用隔离的新集合，没有把空集合当作有效基线。

**阶段判定：M3 核心检索、存储迁移、引用、完整 Compose、十篇 PDF 首轮对照、真实问答对照及同题六路消融均已完成。** 独立复核 PDF 金标准、扫描件和更大留出集仍属于后续质量验证，不能把首轮 Docling 数字写成普遍准确率。

Milvus 的 `get_collection_stats().row_count` 在刚完成写入时曾滞后；验收用 `query(..., output_fields=["count(*)"])` 核对了实际可查询的 2,435 个片段，重启后结果相同。浏览器截图保存在本机被 Git 忽略的 `data/m3/m3_ui_milvus_smoke.png`。

真实问答仅执行一次：对 `arxiv:1706.03762` 提问“原版 Transformer 的编码器和解码器各有多少层”，模型回答两者各有 6 层。保存的回答 ID 为 `73c4f897ee6a4ff2bc54bc08867118b0`；两条引用分别指向论文 PDF 第 3 页、`3.1 Encoder and Decoder Stacks` 中明确写有编码器和解码器 `N = 6` 的片段。随后通过 `/api/v1/answers/{answer_id}/evidence` 从本机 MySQL 回读到 2 条相同证据。此案例验证端到端链路，不能代替问答质量的批量评测。

`evidence_recall_proxy` 沿用 M0 的词覆盖口径，并非 QASPER 官方 evidence F1。固定 10 题仅用于开发检查，不可写成质量提升或泛化结论。QASPER 是结构化全文，不提供原始 PDF 页码金标准；PDF 页码准确率需另建留出集。

## 本机运行

先安装核心依赖并执行迁移：

```powershell
python -m pip install -r requirements.txt
python -m alembic -c alembic.ini upgrade head
```

Milvus 可使用可连接的 2.6 服务；本机已通过 Docker/WSL 验证 `compose.yaml` 的 Milvus 2.6.24、etcd、MinIO 组合。只启动 Milvus 时执行 `docker compose --env-file .env.example up -d milvus`；其中 MySQL 占位密码不会用于这三个容器。启动完整 Compose 前应先创建本地 `.env` 并设置真实 MySQL 密码。Milvus 官方目前不支持在原生 Windows 上运行 Lite；Windows 本地 Standalone 路径依赖 Docker/WSL，或连接远端 Milvus。参见[官方 Windows 部署](https://milvus.io/docs/install_standalone-windows.md)与[Milvus Lite 平台](https://milvus.io/docs/milvus_lite.md)。

将以下设置交给 API、重建脚本和界面进程使用，不能只设置在一个终端：

```powershell
$env:RAG_VECTOR_BACKEND = "milvus"
$env:RAG_COLLECTION_NAME = "paper_chunks_m3"
$env:MILVUS_URI = "http://127.0.0.1:19530"
$env:RAG_RETRIEVAL_MODE = "dense"
$env:RAG_CANDIDATE_K = "20"
$env:RAG_RERANKER_ENABLED = "false"
```

旧论文必须重新向量化入库，不能直接复制 Chroma 索引：

```powershell
python -m scripts.reindex_migrated_papers --backend milvus --data-root data/m2
```

需提供与 API 相同的 `DATABASE_URL`。重建完成后，`/api/v1/papers` 的对应论文从 `REINDEX_REQUIRED` 变为 `COMPLETED`，并应能问答且拿到原文证据。新集合的签名不符时换新集合名重新入库，不覆盖旧集合。原生 Windows 继续运行 M2 演示时可显式设 `RAG_VECTOR_BACKEND=chroma`；它只承担历史基线，不产生 Milvus/BM25 结果。

复杂版式 PDF 需要时安装可选依赖，并在重建前设置 `RAG_PDF_PARSER_BACKEND=docling`。不同解析器使用不同集合名，不混用旧 chunk：

```powershell
python -m pip install -r requirements-m3-docling.txt
$env:RAG_PDF_PARSER_BACKEND = "docling"
$env:RAG_COLLECTION_NAME = "paper_chunks_m3_docling"
```

## 固定开发集消融

在 Milvus 服务已就绪时运行：

```powershell
python -m evaluation.run_m3_ablation --split dev --limit 10 --seed 42 --top-k 5 --candidate-k 20
```

结果写入被 Git 忽略的 `evaluation/runs/m3_ablation_...`：`manifest.json` 保存数据与代码指纹、模型修订、样本 ID 和参数；`records/` 保存逐题证据及错误；`summary.json` 保存每种方法的成功率、召回代理和 P50/P95 延迟。要先扩大开发样本，最后冻结参数，再在未调参的留出集上报告结果。消融脚本不会调用生成模型，因此没有回答质量或 token 费用结论。

无 Milvus 时仅验证历史 dense 与重排链路：

```powershell
python -m evaluation.run_m3_ablation --split dev --limit 10 --seed 42 --skip-milvus
```

禁止把 `--skip-milvus` 的结果称作 Milvus 性能或 M3 完整验收。

## 选型与限制

- Milvus 内建 BM25 和 dense 共存于同一 collection，服务端执行 RRF；比客户端各维护一套倒排索引更容易保证数据同步。查询英文论文时，中文问题的 BM25 匹配可能弱于 dense；需要按中英文问题分组分析。
- Cross-encoder 只对 Top-N 候选运行，成本高于向量召回；是否默认启用由开发集收益和延迟决定。
- Docling 适用于表格、复杂布局等失败案例；简单 PDF 保留 PyMuPDF 快速路径。Docling 若无法提供元素页码，当前实现不把它伪装成可定位证据。
- MySQL 是论文和任务的业务记录，Milvus 是可重建索引；M4 将补充异步队列和更完整的失败恢复。当前 M3 对单篇重建的写入验证已经有回读检查，但跨 MySQL/Milvus 的完全原子提交尚未实现。

技术依据：[Milvus BM25](https://milvus.io/docs/bm25-function.md)、[Milvus 混合检索](https://milvus.io/docs/search_patterns.md)、[BAAI 重排模型](https://huggingface.co/BAAI/bge-reranker-v2-m3)、[Docling 页面来源](https://docling-project.github.io/docling/reference/docling_document/)。
