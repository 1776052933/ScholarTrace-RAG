# M6 固定集评测与交付报告

## 评测边界

- QASPER test 固定 30 题、30 篇结构化全文；六路检索使用相同语料、切块、BGE-M3 向量与 Top-5。分数是项目自定义 token 覆盖代理，非官方 QASPER F1。
- PDF 集共 10 篇、22 道单论文题；跨论文题 4 道，dev/test 各 2 道。标注来自 M3 首轮，尚待独立人工复核。
- 已把跨论文 test 四处金标准 PDF 页渲染检查表格数值与页码；这是开发过程中的抽查，不代替独立人工复核。
- 跨论文对照用同一个旧 Chroma 索引比较普通 Top-5 与 M5 证据补检图；它不代表 Milvus 混合检索的效果。
- 以下是已保存的离线结果。M6 本轮没有重新向 DeepSeek 提交问答。

## QASPER 留出集：六路检索消融

| 检索方式 | 成功/30 | 证据覆盖代理 | P50 延迟 | P95 延迟 |
| --- | ---: | ---: | ---: | ---: |
| `chroma_dense` | 30/30 | 48.6% | 0.20 s | 0.27 s |
| `chroma_dense_rerank` | 30/30 | 61.6% | 10.59 s | 12.38 s |
| `milvus_dense` | 30/30 | 48.6% | 0.18 s | 0.21 s |
| `milvus_bm25` | 30/30 | 41.3% | 0.01 s | 0.01 s |
| `milvus_hybrid` | 30/30 | 45.7% | 0.18 s | 0.21 s |
| `milvus_hybrid_rerank` | 30/30 | 61.6% | 10.62 s | 11.38 s |

与 Chroma dense 对照，覆盖代理最高的 `chroma_dense_rerank` 高 **+13.1 个百分点**。这只是同一批 30 题的检索代理指标；CPU 重排成本见延迟列。
Milvus dense 与 Chroma dense 的代理分数相同；Milvus hybrid 比 Milvus dense **-2.9 个百分点**，没有展示收益。Milvus hybrid + rerank 的 P50 是 Milvus dense 的约 **58 倍**，因此当前不宜默认开启 CPU 重排。

## PDF 解析与已缓存问答

| 解析路径 | 完成/22 | 答案含金标准片段 | 引用金标准页 | 总 token |
| --- | ---: | ---: | ---: | ---: |
| `pymupdf` | 22/22 | 21/22 | 22/22 | 36,663 |
| `docling` | 22/22 | 20/22 | 21/22 | 27,305 |

片段命中和页码命中只做自动检查。答案正确率、引用支持率需要检查原始 PDF；token 总量来自 M3 已缓存的 44 次调用，不是本轮费用。

## 跨论文证据检索

| 解析 | 分组 | 检索 | 页面召回 | 答案片段召回 | 两篇都命中 | 平均片段数 | P50 延迟 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| pymupdf | dev (2 题) | plain_dense | 75.0% | 50.0% | 0/2 | 5.0 | 1.45 s |
| pymupdf | dev (2 题) | bounded_evidence_graph | 75.0% | 50.0% | 0/2 | 5.0 | 1.51 s |
| pymupdf | test (2 题) | plain_dense | 50.0% | 25.0% | 0/2 | 5.0 | 1.43 s |
| pymupdf | test (2 题) | bounded_evidence_graph | 75.0% | 25.0% | 0/2 | 6.0 | 2.40 s |
| docling | dev (2 题) | plain_dense | 50.0% | 50.0% | 0/2 | 5.0 | 0.94 s |
| docling | dev (2 题) | bounded_evidence_graph | 75.0% | 50.0% | 0/2 | 7.0 | 1.67 s |
| docling | test (2 题) | plain_dense | 50.0% | 25.0% | 0/2 | 5.0 | 0.80 s |
| docling | test (2 题) | bounded_evidence_graph | 75.0% | 50.0% | 1/2 | 5.5 | 1.13 s |

页面/答案片段召回的分母是每题两篇论文的金标准引用；两篇都命中的分母是题数。两题延迟样本过小，不能用来宣称性能优势。

## 失败案例与原因

- `pymupdf / test_scene_ap`：金标准页的答案片段未进入检索结果，缺失论文：multi_label, global_view。需要检查解析文本和排序；当前图按论文片段数补检，不能判断片段是否真的含答案。
- `pymupdf / test_rag_metrics`：金标准页的答案片段未进入检索结果，缺失论文：rag_benchmark。需要检查解析文本和排序；当前图按论文片段数补检，不能判断片段是否真的含答案。
- `docling / test_scene_ap`：金标准页的答案片段未进入检索结果，缺失论文：multi_label, global_view。需要检查解析文本和排序；当前图按论文片段数补检，不能判断片段是否真的含答案。

## Ragas 与回答复核

Ragas `StringPresence` 对 M3 缓存答案进行离线片段检查：

| 分组 | 答案含全部片段 | 平均片段存在率 |
| --- | ---: | ---: |
| dev:pymupdf | 12/12 | 100.0% |
| dev:docling | 11/12 | 92.3% |
| test:pymupdf | 9/10 | 90.0% |
| test:docling | 9/10 | 90.0% |

该指标只检查字串出现；不会判断句子真伪或证据是否支持。

AI 辅助初审（逐条对照原 PDF，**不是独立人工复核**）：回答正确 40/44；严格按引用摘录判定支持 38/44。这是 M3 单论文缓存回答的初步判断，不能代表 M5/M6 多论文在线回答质量。[逐项结果、六条重点复核记录与口径](M6_AI_ASSISTED_REVIEW.md)。

人工复核表：**0/44** 条答案已填写正确性判断；全量审查未完成，不能报告全部 44 条的人工正确率或引用支持率。

## 成本、限制与下一步

- QASPER 检索本轮完成 30 题 × 6 路；运行记录包含逐题耗时和失败状态。M6 额外生成调用 0 次、在线判分调用 0 次。
- 这次 CPU 本机 QASPER 运行约 20.3 分钟，其中论文清洗、切块和两种索引写入约 9.3 分钟；查询延迟单独见表格，不包括入库。
- 现有首轮 PDF 标注未独立复核；四道跨论文题不足以代表复杂研究任务；结构化 QASPER 不能验证 PDF 页码。
- LitSearch 全量 64,183 篇语料尚未跑官方固定集；早期 32 篇烟测没有金标准覆盖，不能报告发现质量。
- 若要报告完整 44 条的人工正确率，须逐条独立复核；也可先完整复核 test 的 10 题 × 2 种解析路径，仅报告该 20 条的分组结果。后续扩大多论文 test 集，补无答案/拒答题，并对比真实回答质量与引用支持率。

## 复现入口

- `python -m pytest -q test_m6_evaluation.py`：无密钥评测契约。
- `python -m evaluation.run_m3_ablation --split test --limit 30 --seed 42 ...`：六路检索；参数见 M6 运行说明。
- `python -m evaluation.run_m6_multi_evidence ...`：复用固定 PDF 索引的跨论文对照。
- `python -m evaluation.m6_review`：生成本地审查表；填写后加 `--summary-only` 汇总。
- `python -m evaluation.run_m6_ragas`：安装可选 Ragas 后，对缓存答案做离线字串检查。
- [检索优化后续记录](M6_RETRIEVAL_FOLLOWUP.md)：新增留出题的负结果、失败案例和默认策略决定。
