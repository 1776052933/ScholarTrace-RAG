# M3 Docling 解析质量：标注与对照办法

## 要回答什么

比较同一份 PDF 在当前 PyMuPDF 路径和可选 Docling 路径下，能否保留回答所需的文字、表格行与正确页码；另外测量 Docling 的表格识别误报和漏报。解析质量与最终 RAG 问答质量是两个问题，后者还受切块、检索和生成影响。

## 标注口径

1. **先冻结 PDF 文件和待评文档。** 保存 PDF SHA-256、物理页数、来源、评审页范围；页码从 PDF 第 1 页算起。若只评部分页，表格检测的分母只能使用这些逐页看过的页面。不要事后按解析结果挑“好看”的页面。
2. **看原始 PDF 页面后填写金标准。** 每张数据表记录页码、表题、至少一行关键行及对应值；作者名单、公式排列、图片等疑似表格的区域记为负例。章节标题和可用于问答的原文证据另记。Docling/PyMuPDF 输出不能反过来充当正确答案。
3. **同一 PDF 跑两个解析路径。** Docling 的 `table` 元素按“页码 + 表题”匹配人工表格，计算 TP/FP/FN、精确率和召回率；逐行检查关键值是否与行名保持关联。PyMuPDF 现有生产路径只有按页纯文本，因此比较其关键行、答案片段和页码是否仍可获得，不给它虚构“表格检测准确率”。章节标题也要区分 Docling 的结构标签与 PyMuPDF 文本中是否出现标题。
4. **复核和归档。** 第二人独立核对全部分歧、误报、漏报及至少部分一致样本；修订金标准时保留记录。原 PDF 和解析输出可留在本机 `data/`，可提交的标注与评测程序放在 `evaluation/`，运行报告放在被 Git 忽略的 `evaluation/runs/`。

对一个项目级小基准，建议先选 **10–20 篇不同版式的 PDF**，覆盖原生双栏、复杂表格、跨页表格、公式密集、扫描或混合扫描件。逐篇检查所有页面上的表格，另外每篇准备 2–5 个需要指定页证据的问题；先用开发部分确定解析策略，最后只在未调参的留出文档上报告结论。扫描 PDF 应单列，因为 PyMuPDF 纯文本路径可能根本没有可抽取文字。不要把多个同一版式的论文当成广泛泛化证据。

## 已完成的人工辅助样例

样例是 15 页的《Attention Is All You Need》。我逐页查看渲染后的 PDF，并在[金标准标注](../evaluation/gold/m3_pdf_layout/attention_is_all_you_need.json)中记录了全部 4 张数据表、第 1 页作者区这个非表格、3 个标题以及 4 个页码证据问题。标注绑定 PDF SHA-256 `bdfaa68d8984f0dc02beaca527b76f207d99b666d31d1da728ee0728182df697`，其状态仍是 **试标，待独立二次复核**。

本机 [评测程序](../evaluation/run_m3_pdf_layout_eval.py) 使用已有 Docling 解析记录和同一 PDF 的现跑 PyMuPDF 输出。当前[运行报告](../evaluation/runs/m3_pdf_layout_pilot_20261017/report.json)得到：

| 项目 | Docling | 当前 PyMuPDF 路径 |
| --- | ---: | ---: |
| 真实表格检测 | 4/4；另有 1 个作者区误报，精确率 0.80、召回率 1.00 | 不输出表格实体，不能计算 |
| 所标关键行和值仍同处一行 | 4/4 | 4/4 |
| 所标标题 | 3/3 被标为章节标题 | 3/3 在页面文字中出现 |
| 所标答案片段在正确页出现 | 4/4 | 4/4 |

表格在物理页 6、8、9、10；第 1 页作者名单被 Docling 误判为表格。4 条行检查和 4 个问题只验证了少数值是否可取得，没有测所有单元格的精确结构、跨页表、检索召回或模型回答。PyMuPDF 这次表现也好，所以**不能据此声称 Docling 优于现有解析器**。Docling 的结构化表格和章节标签可能有价值，是否带来问答收益须继续做下游固定问题对照。两种解析耗时也未在同一独立计时实验中比较。

复跑命令（需有相同 PDF、Docling 记录及 PyMuPDF 依赖）：

```powershell
python -m evaluation.run_m3_pdf_layout_eval `
  --gold evaluation/gold/m3_pdf_layout/attention_is_all_you_need.json `
  --pdf data/compose_acceptance/assets/papers/attention_is_all_you_need.pdf `
  --docling-records data/m3/docling_records.json `
  --output evaluation/runs/m3_pdf_layout_pilot/report.json
```

## 2026-10-17 十篇 PDF 对照

在 [PDF 清单](../evaluation/gold/m3_pdf_layout/manifest.json) 中固定了十篇不同版式、共 115 页的论文；[首轮标注](../evaluation/gold/m3_pdf_layout/batch_labels.json)与上面的试标合计 48 张数据表、21 条抽样行检查和 22 个页码证据问题。逐页看原 PDF 渲染图并在查看 Docling 输出之前冻结标注。十篇均为可提取文本的 PDF，**没有扫描件**；标注仍需另一人独立复核。

使用 [批量 Docling 解析脚本](../evaluation/run_m3_docling_batch.py)与 [版式评测脚本](../evaluation/run_m3_pdf_layout_batch_eval.py)处理全部十篇。结果在本机 `evaluation/runs/m3_pdf_layout_10pdf_20261017/report.json`：

| 指标 | Docling | 当前 PyMuPDF 路径 |
| --- | ---: | ---: |
| 表格检测 | TP 47、FP 3、FN 1；精确率 47/50、召回率 47/48 | 只输出按页文本，不计算表格检测率 |
| 抽样行名与数值同处一行 | 21/21 | 20/21 |
| 答案片段出现在标注的物理页 | 22/22 | 21/22 |

Docling 把 Transformer 的作者区及两篇论文的算法伪代码误判成表格，漏检一篇综述第 3 页的图像丰富表格。表格匹配依据页码、表题及少量唯一页回退，未按单元格边框计算 IoU；行检查只是抽样，没有逐格转写所有表格。解析耗时取自不同运行环境，暂不能作为同机性能对照。

## 下游固定问题对照

[检索对照脚本](../evaluation/run_m3_parser_retrieval_ablation.py)将两条解析路径的结果使用相同清洗、1000/200 字符切块、相同 BGE-M3 embedding 分别建入 Chroma dense；只让 PDF 解析器变化。十篇论文共 22 题，固定 12 题开发、10 题留出。详细记录在本机 `evaluation/runs/m3_parser_retrieval_10pdf_20261017/report.json`。

| Top-5/Top-10 指标 | PyMuPDF | Docling |
| --- | ---: | ---: |
| Top-5 含答案片段 | 20/22 | 20/22 |
| Top-10 含答案片段 | 21/22 | 22/22 |
| Top-5 开发集 | 12/12 | 11/12 |
| Top-5 留出集 | 8/10 | 9/10 |

对同一批 Top-5，已在用户明确授权的最多 44 次调用范围内使用 `deepseek-chat` 生成 22×2 个答案。[问答对照脚本](../evaluation/run_m3_parser_answer_ablation.py)按题保存结果并支持中断续跑；本机记录为 `evaluation/runs/m3_parser_answers_10pdf_20261017/summary.json`。两组均 22/22 调用成功。按答案包含金标准字面片段自动统计，PyMuPDF 为 21/22，Docling 为 20/22；正确页引用分别为 22/22、21/22。人工通读 44 个答案后，PyMuPDF 有一题虽然提到正确数字，却误称题目指定的表格无法确定；若要求直接回答且不否认出处，两组均为 20/22。两种解析器各有独有失败题：Docling 缺少 MMGL 的 Dice 值与 SCENE 的一项 AP；PyMuPDF 缺少 SCENE 的另一项 AP，另有上述表格出处判断错误。这是小规模问题集上的描述性结论，**不能据此断言某解析器整体更优**。生成答案未检查所有引用片段是否充分支持每个附加说法，正式发布前仍需独立评审。

## 尚需完成的质量复核

请第二位评审者独立核对 115 页的表格清单、全部分歧及部分一致样本，并逐题核对金标准数值、表格编号、回答与引用。出现标注修订时保留旧版和修订理由，重新运行上述三份报告。再补扫描或混合扫描 PDF，分别报告 OCR 失败率。十篇小样本不能代表所有学术版式；若 Docling 只在特定布局有收益，可按文档类型选解析器。
