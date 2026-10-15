# AutoPaper LitSearch 测评

M0 固定基线评测（QASPER 全文问答与 LitSearch 固定语料检索）见 [M0.md](M0.md)。M1 同题对照见 [M1_COMPARISON.md](M1_COMPARISON.md)。

该目录独立调用项目现有的 `AutoPaperWorkflow`，不修改查询理解、查询拆解、
搜索 Agent 或精排 Agent 的实现。测评时关闭 PDF 下载和 RAG，只评估赛题核心的
“查询 -> 检索 -> 去重 -> 精排 -> 论文列表”流程。

## 测评口径

- 查询和标准答案来自官方 `princeton-nlp/LitSearch` 查询配置，共 597 条。
- 标准答案使用 Semantic Scholar `CorpusId`。
- 系统预测通过 DOI、arXiv ID 批量解析；缺少标识时使用严格标题匹配回退。
- 输出 Precision、Recall、F1、Hit Rate 和 MRR@5/10/20。
- 未解析出 `CorpusId` 的预测按错误结果计入 Precision 分母，不会虚高成绩。
- 当前脚本使用 AutoPaper 的实时学术搜索 API，不使用 LitSearch 约 1.2 GB 的固定
  `corpus_clean`。因此结果适合项目调优，但不能写成 LitSearch 官方榜单成绩。

官方数据集说明：<https://huggingface.co/datasets/princeton-nlp/LitSearch>

## 安装

在 PyCharm Terminal 中执行：

```powershell
conda activate pywork
cd D:\files\Auto_Paper-model
python -m pip install -r evaluation\requirements.txt
```

## 第一次测试

先运行一条查询，避免一次消耗过多 DeepSeek Token：

```powershell
python -m evaluation.run_litsearch --limit 1 --top-k 20 --max-search-rounds 1
```

查询文件约 50 KB，第一次运行会自动下载到 `evaluation/data`。结果写入带时间戳的
`evaluation/runs/litsearch_*` 目录。

确认流程正常后测试 5 条随机查询：

```powershell
python -m evaluation.run_litsearch --limit 5 --shuffle --seed 42
```

测试 20 条，并启用基于 DeepSeek 的搜索规划、评价和反思：

```powershell
python -m evaluation.run_litsearch `
  --limit 20 `
  --shuffle `
  --seed 42 `
  --max-search-rounds 2 `
  --search-agent-llm
```

该模式比项目当前默认设置消耗更多 API 调用和 Token，应先完成 1–5 条测试。

## 断点续跑

为运行指定固定目录：

```powershell
python -m evaluation.run_litsearch `
  --limit 20 `
  --run-dir evaluation\runs\litsearch_experiment_01
```

中断后继续：

```powershell
python -m evaluation.run_litsearch `
  --limit 20 `
  --run-dir evaluation\runs\litsearch_experiment_01 `
  --resume
```

## 输出文件

- `manifest.json`：本次参数和测评口径。
- `summary.json`：Macro 指标、平均耗时和论文身份解析率。
- `per_query.csv`：每条查询的指标，便于 Excel 分析。
- `records/*.json`：标准答案、预测论文、身份解析和单条指标。
- `raw_results/*.json`：AutoPaper 原始工作流结果。
- `benchmark.log`：运行日志。
- `semantic_scholar_cache.json`：论文标识解析缓存。

其中 Semantic Scholar 身份解析属于测评后处理，其 API 调用次数和耗时会单独记录，
不计入 AutoPaper 工作流耗时。

单条记录还包含 `candidate_recall`、`best_gold_rank` 和 `gold_ranks`：它们检查标准
答案是否已进入完整去重精排列表，以及最终排在第几名，可用于区分“搜索漏检”和
“已经搜到但未进入 Top K”。这些诊断字段不改变最终 Precision、Recall 和 F1 的口径。

## PyCharm 运行配置

无需创建额外启动文件，在 **Run/Debug Configurations** 中新建 Python 配置：

- Module name：`evaluation.run_litsearch`
- Parameters：`--limit 1 --top-k 20 --max-search-rounds 1`
- Working directory：`D:\files\Auto_Paper-model`
- Python interpreter：`pywork`

## PaSa RealScholarQuery

`run_pasa`直接调用现有`AutoPaperWorkflow`，关闭PDF下载和RAG，使用PaSa官方
RealScholarQuery的50条真实科研查询与相关论文标题进行测评。输出PaSa风格的
Crawler Recall、Selector Precision/Recall、Recall@20/50/100，并额外输出
Selector F1、Top-K Precision/F1和Hit Rate。Selector集合对应当前系统自适应阈值
标记的高度相关论文。

PaSa数据集需要先在下列页面登录Hugging Face并同意访问条款：

<https://huggingface.co/datasets/CarlanLark/pasa-dataset>

随后在PyCharm Terminal中设置具有读取权限的Token：

```powershell
$env:HF_TOKEN="你的Hugging Face Read Token"
```

首次建议只测1条：

```powershell
python -m evaluation.run_pasa --limit 1 --top-k 100 --cutoffs 20,50,100
```

固定随机测试5条：

```powershell
python -m evaluation.run_pasa --limit 5 --shuffle --seed 42
```

测试全部50条并支持中断续跑：

```powershell
python -m evaluation.run_pasa `
  --limit 50 `
  --run-dir evaluation\runs\pasa_real_full `
  --resume
```

也可以手动下载`RealScholarQuery/test.jsonl`，然后使用：

```powershell
python -m evaluation.run_pasa --dataset-path "D:\你的路径\test.jsonl" --limit 1
```

结果保存在`evaluation/runs/pasa_real_*`，包含`summary.json`、`per_query.csv`、
逐查询记录、AutoPaper原始结果和`benchmark.log`。

注意：PaSa官方实现把检索截止时间设置为源论文发表日期前7天。为了不改变AutoPaper
现有运行逻辑，本适配器使用原始查询和实时搜索后端，没有强制复刻该后端时间截断；
因此该结果适合项目内部对比，但不能直接宣称与PaSa论文成绩完全同口径。

## ScholarGym PaSa-RealScholar（无需 Hugging Face）

无法登录 Hugging Face 时，可以使用 ScholarGym 公开整合的 PaSa-RealScholar 子集。
评测器会自动从 GitHub 下载约 1.3 MB 的 `scholargym_bench.jsonl`，不需要下载
ScholarGym 的 860 MB 静态论文库，也不需要 `HF_TOKEN`。标准答案优先按 arXiv ID
匹配；文献源缺少该字段时，使用严格规范化后的同名标题回退，避免同一论文的会议版漏算。

先做 1 条冒烟测试：

```powershell
python -m evaluation.run_scholargym --limit 1 --top-k 20 --cutoffs 20 --max-search-rounds 1
```

测 5 条并随机抽样：

```powershell
python -m evaluation.run_scholargym --limit 5 --shuffle --seed 42 --top-k 50 --cutoffs 20,50 --max-search-rounds 1
```

继续一次被中断的运行：

```powershell
python -m evaluation.run_scholargym --limit 5 --top-k 50 --cutoffs 20,50 --run-dir evaluation\runs\scholargym_test --resume
```

结果保存在 `evaluation/runs/scholargym_pasa_real_*`。其中 `summary.json` 是汇总指标，
`per_query.csv` 是逐查询结果，`records/` 保存预测和命中明细，`benchmark.log` 保存日志。

注意：该入口使用 ScholarGym 提供的查询和 arXiv 标准答案，但仍调用 AutoPaper 的实时
文献源，并未使用 ScholarGym 的静态 57 万篇论文检索环境。因此它适合项目内部对比，
结果应标注为“ScholarGym PaSa-RealScholar subset with live AutoPaper backends”，不能直接
与 ScholarGym 论文表格或 PaSa 原论文中的数字横向比较。
