# M0 基线记录（2026-10-14）

## 冻结输入与运行

- 命令：`python -m evaluation.run_m0_qasper --split dev --limit 10 --seed 42 --top-k 5 --run-dir evaluation/runs/m0_qasper_dev10`
- 数据：QASPER v0.3 dev，文件 SHA256 `2ae7ee62a65b1c4225791c70de80c2aad4e8998cf1fd4f09a53103db4f21af93`；具体 10 个 question_id 见本地 `manifest.json`。
- 现有链路：结构化论文正文 → 文本清洗 → 1000 字符切块、200 字符重叠 → BGE-M3 → Chroma 单篇 Top-5 → DeepSeek `deepseek-chat`（temperature 0）。
- Embedding：`BAAI/bge-m3`，本机模型修订 `5617a9f61b028005a4858fdac845db406aefb181`；未触发哈希向量回退。
- 运行环境：Windows 10、Python 3.10.20、Chroma 1.5.8、sentence-transformers 5.6.0、PyTorch 2.5.1。

## 结果

| 指标 | M0 结果 | 解释 |
| --- | ---: | --- |
| 成功 / 总题数 | 10 / 10 | 固定种子抽出的开发集小样本；不能代表完整测试集 |
| 证据召回代理值 | 0.667 | 8 题有可评估的人类证据，逐题先对多个标注员取最大值，再做宏平均 |
| 回答词级 F1 代理值 | 0.176 | 只检查词面重叠；中文提示词与英文参考答案、同义改写均会压低分数 |
| 有引用的回答占比 | 1.000 | 仅检查非空 `sources`，不说明引用足以支持结论 |
| 总耗时 | 140.1 秒 | 其中摄取 124.3 秒；不含模型启动与加载的时间 |
| DeepSeek token | 14,002 | prompt 12,633，completion 1,369 |

逐题原始输入、证据、回答、错误和指标保存在被 Git 忽略的 `evaluation/runs/m0_qasper_dev10/records/`；`manifest.json` 保存精确题目 ID、模型修订、依赖版本和代码 SHA256。后续 M1 比较须使用相同输入、Top-K 和指标代码，同时另报新增的页码与引用正确性指标。

固定 LitSearch 入口完成过 32 篇、2 查询的流程验证：32 篇成功索引，2/2 查询完成，两个问题的标准答案均不在抽样索引内（gold coverage=0），因此这些 Recall@K=0 **不能用于评估检索质量**。完整 64,183 篇的 BGE-M3 索引在当前 CPU 环境耗时较高，尚未运行正式固定语料成绩。实时学术 API 检索也未在本次冻结时重跑；已有独立评测入口及口径见 `evaluation/README.md`。

## 已知限制与 M1 对照重点

1. 当前入库切块没有传递逐页信息，PDF 引用无法稳定定位页码；QASPER 是结构化全文，不能检验 PDF 页码。
2. 当前 embedding 失败会静默切到哈希向量；M0 评测器会阻止这种回退，产品链路仍需在 M1 修复。
3. 无可靠的引用支持率自动判分；应保留逐题记录并人工核查。
4. 基准仅 10 题，可能受样本构成和模型服务变化影响。它是工程改造的同输入对照，不是最终对外宣称的整体效果。
