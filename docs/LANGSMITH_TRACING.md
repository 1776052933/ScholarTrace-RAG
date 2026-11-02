# LangSmith 追踪

本项目使用 LangSmith 观察研究搜索、入库和论文问答。`observability/tracing.py` 为 FastAPI 服务、Celery 任务和领域工作流提供应用层节点；开启追踪后，LangChain LCEL 与 LangGraph 自动记录图节点、Retriever、提示词和模型调用。模型 token 用量由 `ProviderChatModel` 的 `AIMessage.usage_metadata` 提供。业务状态仍保存在 MySQL，搜索检查点仍保存在 SQLite。

## 启用

在本机 `.env` 中设置：

```dotenv
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=scholartrace-rag
LANGSMITH_API_KEY=你的密钥
```

如果账户不在默认区域，再配置 `LANGSMITH_ENDPOINT`；一个密钥关联多个 workspace 时配置 `LANGSMITH_WORKSPACE_ID`。Compose 将这些变量传给 API 和 Celery Worker。改动 `.env` 后需重新创建相应容器。未配置密钥或 `LANGSMITH_TRACING=false` 时，应用层追踪不会发送，业务正常运行。常规离线评测和本地对话验收脚本会主动关闭追踪；本页的 LangSmith 连通性脚本除外。

可先运行 `python -m scripts.acceptance_langsmith`，它只发送一条不含论文内容的测试轨迹，不调用模型。然后在 LangSmith 项目中检查 `scholartrace.smoke_test`。这只能验证凭据和网络；要验收完整链路，还需运行一次实际搜索或问答并查看节点层级。

## 数据范围

应用层节点只显式传递任务 ID、论文 ID、chunk ID、计数、模型名及 token 用量等摘要。**LangChain/LangGraph 的自动追踪会记录其运行输入和输出**，其中可能包含用户问题、会话历史、模型提示词、检索到的论文片段以及生成答案。仅在确认这些内容可以发送到所配置的 LangSmith 服务后开启追踪；不要把 API 密钥放入问题、论文文本或提示词。关闭追踪不影响模型供应商调用本身。

## 验收边界

本仓库有离线测试验证应用层节点名称、模型用量元数据与无凭据时的行为。SDK 与真实 LangSmith 页面之间的端到端链路需要有效账号和网络，不能由离线测试代替。搜索、问答与 Celery 是不同进程或请求时，轨迹按各自的应用根节点记录；当前没有实现跨进程 trace ID 传播。

参考：[LangSmith 追踪 LangChain](https://docs.langchain.com/langsmith/trace-with-langchain)。
