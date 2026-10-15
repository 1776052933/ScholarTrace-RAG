# M1 补充验收：真实界面、模型配置与旧下载器测试（2026-10-15）

本记录是 `m1-accepted-2026-10-15` 标签之后的补充验收，不改动已归档的 M0/M1 运行记录、质量指标或标签。测试环境为 Windows、`pywork` Python 3.10.20；真实界面在本机 Streamlit 服务上通过 Chrome 操作。

## 真实界面流程

在浏览器中输入 `Attention Is All You Need Vaswani 2017 arxiv 1706.03762`，设置论文数量 1、最大搜索轮次 1，启用 PDF 下载与 RAG 阅读，关闭非本次必需的摘要、知识抽取、查询 LLM 与引文扩展。点击“开始检索”，等待任务完成，再进入“智能阅读 → 问答”，提问 `How many encoder layers does the Transformer use?`。

| 检查点 | 实际结果 |
| --- | --- |
| 检索与精排 | 检索 20 篇，去重后 20 篇，选出 1 篇 `Attention Is All You Need` |
| PDF 与向量库 | `arxiv:1706.03762` 下载成功，RAG 摄取成功，生成 67 个片段 |
| 前端状态 | 出现完成提示、结果指标、目标论文和智能阅读页，任务结果 JSON 为 `completed` |
| 界面追问 | 收到“两条聊天消息”，回答编码器由 `N = 6` 个相同层组成；显示“证据片段（1）”入口 |
| 实际引用内容 | 用相同已入库论文、相同问题调用问答后端，返回 1 条结构化引用：PDF 第 3 页、`3.1 Encoder and Decoder Stacks`、非空摘录及 chunk ID |

本机留有被 Git 忽略的验收截图 `Auto_Paper_UI/runtime/acceptance_result.png`、`Auto_Paper_UI/runtime/acceptance_qa.png` 和运行结果 `Auto_Paper_UI/outputs/result_20261015_180826_7d4ea2.json`。其中问答截图显示回答及证据入口；页码和摘录由真实后端返回并由界面组件测试验证展示。界面现在优先显示结构化引用的论文、页码、章节和摘录；对于旧结果仍兼容只含 `sources` 的 chunk ID。

验收范围是单篇论文的完整交互路径。尚未把摘要、知识抽取、取消任务和失败恢复都纳入本次真实浏览器路径；本记录也不代替完整数据集质量评测。

## 模型配置现状

| 位置 | 当前模型及入口 | 已确认的行为或问题 |
| --- | --- | --- |
| 查询理解、查询拆解、搜索规划/反思、精排 LLM | 各模块独立的 `DeepSeekClient`，默认 `deepseek-chat`，多数读取 `DEEPSEEK_MODEL`、`DEEPSEEK_BASE_URL`、`DEEPSEEK_API_KEY` | 同一工作流可运行，但客户端、超时和密钥回退路径分散，改模型时必须核对多个入口 |
| `query_understanding/config.py` | `QueryUnderstandingConfig.from_env()` 定义 `QUERY_UNDERSTANDING_MODEL` 等字段 | 当前主流水线未实例化该配置；这些专用环境变量不能视为已生效的模型切换入口 |
| 精排向量模型 | `ranking_agent.EmbeddingModel` 默认 `BAAI/bge-m3` | 可经依赖注入更换；主流程没有独立的精排模型环境变量 |
| RAG 向量模型 | `RAGSettings.from_env()` 默认 `BAAI/bge-m3`，读取 `RAG_EMBEDDING_MODEL`、设备、批量大小、切分与检索参数 | 设置会进入 Reader、向量化和检索；模型指纹与维度校验已在 M1 验收 |
| RAG 生成模型 | `RAGSettings` 读取 `DEEPSEEK_MODEL`、`DEEPSEEK_BASE_URL`、`DEEPSEEK_TIMEOUT` | Reader 的默认惰性客户端使用这些值；本次把界面追问的显式客户端也改为使用同一组值 |
| 前端密钥 | UI 可以将提供的 DeepSeek/SerpAPI 密钥传给后端任务；OpenAlex 也可使用环境或本地配置 | 追问会传入当前 UI 的 DeepSeek 密钥；不要把本地密钥或截图中的敏感信息提交到仓库 |

短期约定：设置 `DEEPSEEK_MODEL`、`DEEPSEEK_BASE_URL` 和 `DEEPSEEK_API_KEY` 可以覆盖当前工作流中主要生成模型的默认值；`DEEPSEEK_TIMEOUT` 目前是 RAG 专用配置，不能声称对其他 Agent 的客户端生效。下一阶段若改成统一模型配置，应明确每个模块的模型、超时、密钥来源与优先级，然后为不同任务保留可解释的覆盖项，避免仅为了“技术栈统一”重写已验收的推理逻辑。

## 两份旧测试的处理

`paper_downloader/tests/test_config_cache.py` 混合了配置、校验器、搜索工厂和已缺失的 SQLite/内存缓存模块测试；`paper_downloader/tests/test_integration.py` 混合了下载/API 集成用例及一条缓存集成用例。此前两个文件在收集时因导入不存在的 `paper_downloader.src.cache` 而整体报错，导致其中仍可运行的用例无法执行。

现改为仅对依赖该模块的用例标注条件跳过，其他测试正常收集。两文件的非慢速结果：**58 passed、22 skipped、3 deselected**。跳过原因明确显示为 `paper_downloader.src.cache is absent from this source tree`。这些跳过项没有通过验证，不能写成“缓存功能已通过验收”。源码中 `ConfigManager` 仍有 `cache.enabled` 默认配置，`SearchFactory` 仍有可选缓存接入点；但缺少实际 `CacheManager` 与装饰器实现。

后续决策应在缓存架构确定时作出：若 M2/M3 使用新的共享缓存/Redis，先确定接口再迁移或替换这 22 条测试；若保留下载器本地缓存，则补回实现并取消跳过。当前 M1 主路径并不依赖这个旧缓存包。

全仓库非慢速复查命令与结果：

```powershell
python -m pytest -q -m "not slow"
# 437 passed, 22 skipped, 3 deselected, 2 warnings
```

两条警告分别来自已弃用的 PyPDF2 和 Windows 异步 SSL 资源清理。被跳过的 22 项都与不存在的缓存包有关；3 项真实 arXiv 慢速测试未在本次运行。
