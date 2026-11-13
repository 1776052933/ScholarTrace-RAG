# 论文发现领域模块

这里保存论文发现阶段正在使用的实现。应用入口在 `app/api/main.py` 和 `app/cli.py`；`app/services/workflow.py` 负责调用领域模块并处理论文注册、下载与阅读；`app/workflows/discovery.py` 和 `app/workflows/search.py` 负责 LangGraph 编排。

| 包 | 职责 | 主要入口 |
| --- | --- | --- |
| `understanding/` | 从研究问题提取意图、实体和约束 | `QueryUnderstandingPipeline.run` |
| `decomposition/` | 判断查询复杂度并生成检索子问题 | `QueryDecompositionPipeline.run` |
| `search/` | 多来源搜索、迭代反思和引文扩展 | `SearchAgent.run`、`CitationTracker.expand` |
| `ranking/` | 元数据特征、嵌入及模型评分排序 | `RankingAgent.run` |

数据流是：用户查询 → 理解 → 拆解 → 搜索与可选引文扩展 → 排序。图中的阶段名继续沿用现有 API 契约，以保持任务进度、界面和评测结果可读。领域模块不直接负责 HTTP、任务持久化或 PDF 入库。

候选论文的发现、去重和排序只走这里的 `search/` 与 `ranking/`。选中的论文元数据随后交给 `app/papers/`；下载器不会重新决定候选集合或精排顺序。

离线单元测试放在 `tests/research/`。仓库根目录的 `main.py` 只启动 CLI；通过 `python main.py --help` 查看参数。
