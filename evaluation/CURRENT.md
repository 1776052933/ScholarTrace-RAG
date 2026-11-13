# 当前版本评测入口

入口：`python -m evaluation.run_current`。历史 `run_m0_*` 保持原样。

## 范围

复用当前 `RAGSettings → RAGReader → PaperRetriever → EvidenceGraph / AnswerGraph`。
生成继续使用 LangChain 模型适配器和 LangGraph 问答图；不是另写一套提示词。
这是 **RAG 核心离线数据集入口**，不经过 HTTP、MySQL、Celery、UI，
不代表上传、会话、多轮历史、SSE 或完整部署的端到端验收。

| 数据集 | 输入与执行方式 | 限制 |
|---|---|---|
| LitSearch | 固定标题/摘要语料，一篇一个向量文档，跨库检索 | 不实时搜索；裁剪语料只适合调试 |
| QASPER | 官方结构化全文，当前清洗/切分/向量化；限定论文问答 | 不测试 PDF 解析，没有真实 PDF 页码 |
| ScholarQA-CS | 官方问题 + 显式提供的独立来源语料；先检索候选再选最多五篇问答 | 缺少语料时仅能 plan；这是项目适配协议，不是官方原协议 |

LitSearch 使用官方 `Title: ...\nAbstract: ...` 模板，保留标题摘要都为空的记录，
不补造内容、不删除标准相关论文；本地全量数据缺失情况与建库记录见
[LITSEARCH_LOCAL_BUILD.md](LITSEARCH_LOCAL_BUILD.md)。

ScholarQA-CS 语料 JSONL 每行必须且只能包含 `paper_id`、`title`、`text` 三个非空字符串。
需要 `--corpus-jsonl PATH --corpus-provenance "来源及准备方式"`。
字段校验不能判断文本语义，准备者仍必须保证正文不是参考答案、评分标准或模型样例输出。
所有参考答案写入单独 `references.json`，不会传给模型或作为检索文本。

## 分阶段操作（本次只实现入口，不执行以下正式评测）

在项目根目录、已安装项目及 `evaluation/requirements.txt` 依赖的 Python 环境运行。
脚本读取**进程环境变量**，不自动加载 `.env` 或读取运行中容器配置。
复现实测前，需要显式使用与部署一致的 embedding、切分、检索、重排、模型等环境配置。
Milvus 地址必须从当前执行环境可访问。

```powershell
# 默认 plan：读取数据与配置，不连接数据库、不加载模型、不写运行目录
python -m evaluation.run_current qasper --run-dir evaluation/runs/current_qasper

# 建库：独立索引，只有本地 embedding；首次加载权重可能需要下载
python -m evaluation.run_current qasper --phase index --run-dir evaluation/runs/current_qasper

# 检索：当前 EvidenceGraph，不调用生成模型
python -m evaluation.run_current qasper --phase retrieve --run-dir evaluation/runs/current_qasper

# 生成：必须明确打开模型调用，并限制本次最多新尝试的题数
python -m evaluation.run_current qasper --phase answer --run-dir evaluation/runs/current_qasper --allow-model-calls --max-answers 10

# LitSearch 全量语料计划，检索前需要以相同参数执行 index
python -m evaluation.run_current litsearch --run-dir evaluation/runs/current_litsearch --top-k 20

# ScholarQA-CS 缺少独立语料时显示 ready=false
python -m evaluation.run_current scholarqa-cs --run-dir evaluation/runs/current_scholarqa
```

默认 `--limit 0` 表示全部题。QASPER 可用 `--limit N --seed 42` 固定抽样；
另外两个数据集使用固定前 N 题。LitSearch 的 `--corpus-limit` 只用于调试，
不会自动补齐被裁掉的相关论文，不能把这种结果报告成全库成绩。
同一 run-dir 的所有阶段必须保持数据选择和配置一致。

后端默认 Milvus，`RAG_VECTOR_BACKEND` / `--backend` 可显式覆盖。
Chroma 仅允许 dense，避免把 hybrid 配置悄悄降级成 dense。
其他参数沿用项目 `RAG_*`、`MILVUS_*`、`DEEPSEEK_*` 环境变量。
`--retrieval-mode` 和 `--top-k` 可覆盖环境值。

## 后续正式测试的重排约定

用户要求后续正式检索/问答测试默认启用本地 `BAAI/bge-reranker-v2-m3`
CrossEncoder；同时保留关闭重排的同题对照，以便量化收益与额外耗时。
LitSearch 使用完整固定语料、相同问题和 embedding，对照时只改变检索阶段的重排设置。
重排只发生在查询时，不参与当前正在运行的 BGE-M3 向量建库。

当前统一入口把建库与检索配置锁在同一 manifest；在支持“同一索引、分开记录
检索配置”的入口完成之前，不能直接在现有 run-dir 把
`RAG_RERANKER_ENABLED=false` 改为 `true` 后继续运行，否则会触发配置校验。
正式评测启动前需要先接通该复用方式，并确认重排候选数大于最终 Top-K。

## 隔离、恢复与结果

- collection 名由运行目录派生，始终为 `eval_current_*`，不会沿用业务 collection。
  新运行拒绝接管已存在且不属于它的索引；不删除业务数据。
- `manifest.json` 固定数据文件 SHA256、题目 ID、配置、核心代码哈希和证据策略。
  修改代码、配置或数据后必须新建 run-dir。
- `progress.sqlite3` 记录每篇入库完成情况；每批写入后读回验证。
  只有全部入库才产生 `index_complete.json` 并允许检索/生成。
- 重启会检查已记录 chunk 是否仍存在；缺失时停止，不当作有效结果。
- `retrieve/`、`answer/` 每题一个 JSON，包含检索片段、论文 ID、图节点、
  回答与引用（生成阶段）、耗时和可用的 token 用量。结果文件使用题目 ID 的哈希命名。
- 每次生成前先写 started；已有 completed、failed、started 记录都不会自动重发。
  pending 题可使用同参数继续执行。failed/started 需先调查，再用新运行目录重测。
- `--max-answers` 是每次命令的**新题尝试上限**，不是 HTTP 请求硬上限或金额上限；
  SDK 内部可能重试。预算耗尽时剩余题保留 pending。
- summary 按全部选中题记录 completed/failed/started/pending/no_evidence，
  不隐藏失败题；`quality_scores` 当前为 null。
- API key、Milvus token 不写 manifest；含凭证或查询参数的服务 URL 会被拒绝。
  此入口关闭 LangSmith 外发，结果保留本地；answer 阶段才允许向配置的模型发送题目和片段。
- 索引阶段不会生成答案，但真实 embedding / reranker 权重加载仍需足够磁盘与内存。
  禁止把哈希向量回退当作真实模型结果。

## 本地评分（不调用模型）

```powershell
python -m evaluation.score_current --run-dir evaluation/runs/current_qasper --phase answer
python -m evaluation.score_current --run-dir evaluation/runs/current_litsearch --phase retrieve --cutoffs 1 5 10 20
```

评分读取已保存结果，不连接 Milvus，不加载模型，不调用 API。输出 `scores_<phase>.json`
和 `scored_cases_<phase>.json`；保留原始结果和原始运行 summary。
全部选中题为分母，未完成/失败题计零分，同时报告完成率和状态分布。
没有运行过生成的记录不能得到有意义的答案质量成绩。

- LitSearch：按去重后的论文排名计算宏平均 Recall@K、HitRate@K、MRR@K、
  二值相关性的 nDCG@K。K 不得超过运行时的 top-k；报告是否为完整语料。
- QASPER：调用下载数据包内的 `qasper_evaluator.py`，记录该脚本和数据 SHA256。
  答案原文直接进入官方 token-F1；不根据标准答案改写拒答、布尔值或答案内容。
  输出 Answer F1、分答案类型 F1、Evidence F1，另报 text-evidence-only F1。
  这不是人工正确率，较冗长的正确回答也可能只有较低 token-F1。
- QASPER 证据映射：只查看原始 full_text 段落，不查看 gold evidence。
  合并空白后，引用完整包含某原段落，或长度至少 40 字符的引用完整位于某段落内，
  即选择该原始段落；原段落原文交给官方脚本。去重，未匹配引用单独计数。
  此保守规则会漏掉跨段落的截断引用；不会以模糊匹配标准证据提高分数。
  answer 阶段使用实际 citations，retrieve 阶段使用检索 chunks，指标名明确区分。
  当前未入库图表/图注；完整证据 F1 的这一限制不能隐去。
- ScholarQA-CS：输出覆盖率、状态及“待人工/模型 rubric 评审”，质量分数为 null。
  不伪造官方成绩，不自动调用收费裁判。

`scored_cases_answer.json` 中 QASPER 每项含 question_id、predicted_answer、
predicted_evidence，可导出为 JSONL 交给官方脚本交叉验证；应使用同一选中问题子集的 gold。

## 后续工作

确定生成模型与预算、准备 ScholarQA-CS 来源语料、正式执行问答和检索评测。
本入口没有新增 Docling/PyMuPDF 对照，这部分仍使用原有 PDF 专项评测。

离线测试：`python -m pytest -q tests/test_current_scoring.py tests/test_current_evaluation.py tests/test_langchain_architecture.py`。
QASPER 官方脚本对照测试在本地下载数据包存在时执行，CI 缺少脚本时明确 skip。
使用小型构造数据、假模型，以及临时 Chroma，验证接入行为；不证明真实模型质量或 Milvus 性能。

## 固定错题的生成瓶颈诊断

`python -m evaluation.answer_bottleneck prepare` 验证上一轮 30 题的冻结输入，并将官方证据定位回同一篇论文。
`run` 使用本机 Qwen 重放系统片段与标注片段，`score` 仅重算本地结果。标注答案不进入模型输入；
不可回答题的空证据控制与有答案题分开解释。图表说明缺少单元格时单列限制。
2026-11-08 的 60 次本地调用、重点 7 题核查和标注疑点见
[错题复核与证据对照报告](../docs/QASPER_ANSWER_BOTTLENECK_20261108.md)。

使用 `--case-ids-file` 可为 QASPER 指定固定题目清单。200 道开发题与 50 道留出题的
提示词对照见 [单项提示词优化与扩展回归](../docs/QASPER_PROMPT_REGRESSION_20261113.md)。
