# M3 解析与检索：运行、评测与当前验收

## 本阶段做了什么

- 增加 PyMilvus 2.6 的版本化 collection：BGE-M3 dense、Milvus 内建 BM25 sparse、RRF 混合召回。集合描述写入模型指纹、维度、解析与切块版本；不兼容时直接报错。
- 可选 BGE reranker v2 m3 cross-encoder 对候选片段重排，保留第一阶段分数、重排分数和来源页码。
- 可选 Docling 复杂 PDF 路径，按元素 provenance 保留页码、章节和表格 Markdown；常规 PDF 仍使用 PyMuPDF。
- API、搜索工作流、旧 CLI 问答入口统一根据 `RAG_VECTOR_BACKEND` 选择索引。
- MySQL 新增 `papers.index_signature`。切到 Milvus 时，旧 Chroma 的 `COMPLETED` 论文展示 `REINDEX_REQUIRED`，问答返回 409，避免把不存在于新索引的论文当成可用。
- 固定 QASPER 开发集的逐题消融脚本：Chroma dense → Chroma dense + rerank → Milvus dense → Milvus BM25 → Milvus hybrid/RRF → Milvus hybrid + rerank。所有路径使用同一清洗文本、同一 chunk 和同一 BGE-M3 向量。

## 当前状态

| 检查 | 结果 |
| --- | --- |
| 新检索、重排、解析单元测试 | 已通过 |
| M0–M2 回归 | 最终常规集 453 通过、22 跳过、3 个 slow 未运行；旧重建脚本接口回归已修复 |
| Alembic M3 迁移 | 本机原生 MySQL 执行成功，revision `9f1a43cd621e`；27 篇旧论文中 16 篇曾标为已完成，索引签名均为空。真实 API 检查显示 16 篇 `REINDEX_REQUIRED`、旧论文问答 409 |
| QASPER dev 固定 10 题 Chroma dense 对照 | 10/10 完成，证据召回代理 0.667；与 M1 记录相同 |
| Milvus 服务端真实联调及完整消融 | 尚未运行。本机 Windows 无 Milvus 服务，Docker 按用户决定暂缓安装 |
| BGE cross-encoder 本机实测 | 首次模型下载持续停滞，已中止；单元测试通过，但无可报告的质量/延迟数字 |
| Docling 真实复杂 PDF | 接口和模拟 provenance 测试已通过；尚未安装 Docling 或用真实复杂版式 PDF 检验质量 |

`evidence_recall_proxy` 沿用 M0 的词覆盖口径，并非 QASPER 官方 evidence F1。固定 10 题仅用于开发检查，不可写成质量提升或泛化结论。QASPER 是结构化全文，不提供原始 PDF 页码金标准；PDF 页码准确率需另建留出集。

## 本机运行

先安装核心依赖并执行迁移：

```powershell
python -m pip install -r requirements.txt
python -m alembic -c alembic.ini upgrade head
```

Milvus 可使用可连接的 2.6 服务；`compose.yaml` 已加入官方 2.6.24 对应的 etcd、MinIO、Standalone 服务配置。当前不需要为阅读代码安装 Docker。Milvus 官方目前不支持在原生 Windows 上运行 Lite；Windows 本地 Standalone 路径依赖 Docker/WSL，或连接远端 Milvus。参见[官方 Windows 部署](https://milvus.io/docs/install_standalone-windows.md)与[Milvus Lite 平台](https://milvus.io/docs/milvus_lite.md)。

将以下设置交给 API、重建脚本和界面进程使用，不能只设置在一个终端：

```powershell
$env:RAG_VECTOR_BACKEND = "milvus"
$env:RAG_COLLECTION_NAME = "paper_chunks_m3"
$env:MILVUS_URI = "http://127.0.0.1:19530"
$env:RAG_RETRIEVAL_MODE = "hybrid"
$env:RAG_CANDIDATE_K = "20"
$env:RAG_RERANKER_ENABLED = "true"
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
