# 当前版本真实回归（2026-11-06）

本次在 Windows 本机执行；使用当前工作区代码构建镜像 `scholartrace:accept-20261102`（镜像 ID `sha256:f80b437f0dc89ef0904f1041a3831827d7da321cc700438fc7e9c6450a4c09a0`，Git HEAD `119419a`，工作区有未提交修改）。生成模型为 F 盘本地 Qwen3-4B，追踪关闭；本轮没有调用 DeepSeek API。PDF 全栈验收使用隔离 Compose 项目 `scholartrace_repro_20261102`，QASPER 回归只读复用原 `scholartrace_repro` Milvus 索引。

## 部署、数据库与界面

- 按 [部署说明](REPRODUCIBLE_DEPLOYMENT.md)构建并启动 MySQL、Redis、Milvus、etcd、MinIO、FastAPI、MCP、Celery worker/beat、Streamlit，共 10 个容器。API 和界面健康接口均返回 200。
- API 和 worker 实际连接 `mysql/scholartrace`；两者均配置 `hybrid` 与 BGE 重排。API 重启后的检索返回 5 个片段，5 个均含重排分数与 `hybrid` 标记。
- 使用 Streamlit `AppTest` 在真实 API 配置下运行页面：无异常，状态显示已连接，能看到 ScholarTrace、MySQL 与 PDF 上传入口。此检查覆盖页面组件运行，不包含真人浏览器操作。
- 首次真实问答因离线容器仍按线上名称查找重排模型而失败。已将本机 `.env` 与隔离验收环境的 `RAG_RERANKER_MODEL` 指向 `/app/models/reranker_windows_model`，并更新部署说明；重新启动后问答成功。模型文件、旧索引和原有数据库未改动。

## 两篇公开 PDF 的真实模型问答

原有隔离库中的《Attention Is All You Need》和机器人灵巧操作综述用于本次问答。原始调用、回答、引用与会话记录位于 `data/repro_acceptance_20261102/runtime/current_qwen_20261106_v3.json`（本机忽略目录）。

| 检查 | 结果 |
| --- | --- |
| 表格事实问答 | 回答包含 EN-DE 28.4 和 EN-FR 41.8，引用可从证据端点读取。 |
| 两轮流式追问 | 两轮都回答编码器/解码器为 6 层；第二轮读取 1 轮历史。短答案各收到 1 个正文增量和完成事件。另一个解释性问题收到 72 个正文增量、1 个引用。 |
| 跨论文问答 | 返回两篇论文的引用，**内容未通过事实检查**：Qwen 将综述论文描述为具体软体机械手实验，并用英文回答中文问题。 |
| 证据不足 | 对“作者每天喝什么咖啡”明确拒答，置信度 0、引用 0。 |
| 模型不可用 | 短暂关闭本地 Qwen 后，SSE 返回 `error`，数据库将该轮标记为 `failed`，未保存答案；随后恢复 Qwen。 |

短答案只有一个增量事件是正常流式输出；验收脚本原先强制要求两个，已改为至少一个，并继续核对增量先于完成事件且拼接结果与最终答案一致。模型不可用时 SSE 传输本身返回 HTTP 200，错误由 `event: error` 表达。

完整 Compose 停止后重新启动，10 个服务恢复健康；重启前后的两轮会话 JSON 完全一致，MySQL 仍为 `scholartrace`，Milvus 可检索相同论文。单独上传一篇 11 页的公开 ACL 论文后，Celery 入库任务 `4bb2593560d0ce63d0dc619c59765840` 完成，Docling/BGE-M3/Milvus 形成 148 个片段；论文库状态为 `COMPLETED`，PDF 原文接口返回 200 和完整 855,501 字节。

## QASPER 固定 30 题

从此前冻结的 QASPER dev 80 题中按固定顺序选出抽取 10、概括 8、是非 6、标注不可回答 6。复用同一 151 篇论文、4,962 个片段的 Milvus 索引，重新执行 `hybrid` 候选 20、BGE 重排 Top-5 和本地 Qwen 生成。正式提示词与旧实验字节一致，输出上限统一为 2048；30/30 新旧 Top-5 片段 ID 与顺序一致。运行目录为 `evaluation/runs/current_qasper_regression_20261106_2048/`，旧答案来自 `evaluation/runs/prompt_hybrid_80_dev_20261104/`。结果均在本机忽略目录。

| 同一组 30 题 | 旧保存答案 | 当前重新生成 |
| --- | ---: | ---: |
| 官方词面 Answer F1 | 0.3460 | 0.3227 |
| 官方 Evidence F1（引用段落投影） | 0.4393 | 0.4393 |
| 成功运行 | 30/30 | 30/30 |

26/30 题答案文本相同。4 题有文字变化，其中一题将数据集语言从正确的 English 扩展为 English、German、French、Greek，是实质性错误；其余包括附加解释或近义改写。6 道标注不可回答题中，只有 1 道给出明确拒答，其余给出具体答案；该类题的官方词面 F1 均为 0。部分 QASPER 标注与可见论文文本存在口径差异，因此需逐题核对，不能直接把所有具体回答都判为幻觉。本次 30 题的平均完整检索加生成耗时约 15.38 秒/题。

QASPER 使用结构化论文文本，本轮只验证现有索引上的检索和问答；PDF 解析、HTTP、Streamlit 与会话由上面的全栈验收单独覆盖。官方词面 F1 不是人工事实正确率，30 题来自已见论文的 dev 集，不是新论文盲测。

## 代码检查与结论

数据库配置、模型调用、检索、流式会话和界面关键测试共 **29 项通过**；`git diff --check` 通过。当前部署链路、数据恢复、文档上传和流式异常处理可运行；本地 Qwen 的跨论文事实混淆以及 QASPER 个别回答波动需要作为下一轮质量优化的输入。进一步修改提示词或拒答策略时，应复用这批固定题并增加新的未见论文样本。
