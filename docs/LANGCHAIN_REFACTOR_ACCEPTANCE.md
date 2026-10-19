# LangChain / LangGraph 架构重构验收

日期：2026-10-20（Asia/Shanghai）。API 版本：4.2.0。
架构与运行边界见 [架构说明](LANGCHAIN_ARCHITECTURE.md)。

## 完成范围

- LangChain ChatModel、Retriever、Embeddings 标准接口；共享 LCEL 提示词→模型→文本解析链。
- 查询理解、拆解、规划/评价/反思、论文评分、摘要、知识抽取、问答和追问改写使用共享生成入口。
- DiscoveryGraph 编排理解、拆解、搜索、可选引文扩展与排序；原搜索图保留循环与检查点。
- AnswerGraph 编排改写、证据子图、上下文预算、生成、引用校验和无证据分支；单轮与多轮共用。
- 会话并发、MySQL 事务、Celery 入库和 HTTP/SSE 协议留在应用层。
- 拆出工作流数据契约和回答校验模块，避免图模块反向导入应用服务。
- README、架构说明和 CI 测试清单已更新。

## 自动化回归

最终运行：**89 passed，1 skipped**，耗时 45.29 秒。
跳过项为 Linux 容器下无法执行的既有 Windows Chroma 路径专用测试。
另有原有 PyPDF2 弃用提示，不影响测试结果。

覆盖：

- 原生 LangChain 模型注入、标准组件回调传递、参数和 usage 保留。
- 流式增量与关闭、Document 元数据无损转换、检索范围保护。
- 单轮/多轮图路径、引用过滤、无证据分支、无效输出拒绝。
- 发现图可选引文分支、日期过滤、取消后不写入、实际搜索子图嵌套与 SQLite checkpoint。
- 真实 HTTP SSE 在生成完成前到达客户端；断连释放会话并且不保存为成功回答。
- 会话隔离、重启恢复、输入预算、API、MySQL 迁移契约、检索、UI、评测、并行任务等既有回归。

本地 JUnit 报告：`tmp/langchain_architecture_pytest.xml`（不入 Git）。
测试使用隔离容器和 `--network none`，没有访问外部模型。

## 本机真实存储与界面联调

通过 `python -m scripts.chat_acceptance` 执行，结果：

| 项目 | 结果 |
| --- | --- |
| 论文 | `arxiv:1706.03762`、`swin-unetr` |
| MySQL、Milvus、BGE-M3 | 使用本机真实服务和模型 |
| 连续问答 | 2 轮完成并持久化 |
| 第二轮历史 | 1 对历史消息 |
| 第一轮 / 第二轮证据片段 | 5 / 6 |
| 引用页码 | 8、10 |
| Streamlit 提交、追问、恢复、导出 | 通过 AppTest |
| 模型调用 | loopback 夹具：2 次流式生成、1 次问题改写 |
| 外部模型调用 | 0 |
| 用时 | 11.95 秒；不是性能基准 |

本地独立记录：`data/m2/langchain_architecture_acceptance.json`；日志：`tmp/langchain_acceptance_latest.log`。
脚本清理了本轮创建的临时会话及对应答案记录，没有删除已有论文或向量。
占位回答只验证工程链路，不能作为真实模型质量或人工正确率结果。
本轮没有运行真实浏览器视觉验收，也没有重新执行完整数据集质量评测。

## 本机运行状态

API、worker、beat、MCP 已同步当前源码并重启；API 返回 4.2.0，数据库健康检查成功，现有 UI 健康检查为 `ok`。
UI 地址仍为 `http://127.0.0.1:18501`；API 为 `http://127.0.0.1:18000`。
Compose 配置静态校验通过，Git 差异空白检查通过（保留仓库约定的 CRLF）。

本次部署为现有容器内同步代码，**没有重新构建镜像或验收全新 Compose 启动**。
当前工作区缺少 `.env`，沿用现有容器配置；后续重建需要恢复原有数据库和模型配置并执行 `--build`。
没有推送 GitHub，因此本次 CI 结果为本机执行，未声称云端 Actions 已通过。
