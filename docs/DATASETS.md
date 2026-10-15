# 基准数据集来源与本地位置

本项目将原始基准数据放在 data/benchmarks/（已被 .gitignore 排除），不提交到 GitHub。评测代码应读取本地文件，并在运行结果中记录以下固定版本。

| 数据集 | 官方来源与固定版本 | 本地路径 | 当前用途 |
| --- | --- | --- | --- |
| LitSearch | [Princeton NLP 数据集](https://huggingface.co/datasets/princeton-nlp/LitSearch)，提交 9573fb284a1026c998df47024b888a163f0f0e25 | data/benchmarks/litsearch/query/ 和 corpus_clean/ | 597 个带 gold paper ID 的查询、固定论文标题/摘要语料；论文发现与排序 |
| QASPER | [AllenAI 数据集卡](https://huggingface.co/datasets/allenai/qasper)中的官方 v0.3 原始包 | data/benchmarks/qasper/ | 单论文全文 QA、段落级证据与不可回答问题 |
| ScholarQABench | [官方仓库](https://github.com/AkariAsai/ScholarQABench)，提交 95e6fc52b0a8a0ce0a74956029991e3bb00c38b9 | data/benchmarks/scholarqabench/ScholarQABench-95e6fc52b0a8a0ce0a74956029991e3bb00c38b9/data/ | 多论文综合与单论文任务的后续评测 |

2026-10-14 校验结果：LitSearch 的 7 个 Parquet 文件均与官方 LFS SHA-256 一致；QASPER 的 train/dev/test JSON 共 1,585 篇论文、5,049 个问题，均可解析；ScholarQABench 压缩包 CRC 校验通过，10 个数据文件均可解析。详细文件大小和 SHA-256 见本地 manifest.json。

LitSearch 下载 query 的一个 Parquet 文件和 corpus_clean 的六个 Parquet 分片；corpus_s2orc 是同一批论文的扩展元数据版本，当前检索对照实验不需要，故暂不下载。若以后需要原始 S2ORC 元数据，可使用同一固定提交补齐。

QASPER 下载官方 qasper-train-dev-v0.3.tgz 与 qasper-test-and-evaluator-v0.3.tgz，解压后使用原始 JSON。QASPER 是结构化全文及标注，不是原论文 PDF，不能据此声称验证了 PDF 页码定位。

ScholarQABench 取得固定提交的完整仓库压缩包，保留 data/ 和评测脚本。其各子任务评测方式不同；使用时需按官方 README 明确选定子任务，不能将所有样本混成同一指标。数据集本身不提供训练数据，使用前仍需核对各组成数据的许可。

本地另生成 data/benchmarks/manifest.json，记录实际文件大小、SHA-256 和下载完成状态。评测脚本只在校验通过后读取数据。自建 PDF 页码证据集将另外建立，不与上述公开基准混淆。

