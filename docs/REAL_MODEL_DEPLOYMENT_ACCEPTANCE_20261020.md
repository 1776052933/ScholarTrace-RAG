# 当前版本可复现部署与真实模型回归验收

日期：2026-10-20（Asia/Shanghai）。本机 Windows + Docker Desktop Linux 容器。验收使用独立项目 `scholartrace_repro`，没有替换旧项目的数据卷或 44 条人工复核。镜像 `scholartrace:local` 的本次内容摘要为 `sha256:d3e25adbfc89c65791b548987c7893ab71c9bbd8ff819ddfc58c443e75744780`。本次工作区含未提交修改，因此镜像标签中的源码 revision 为 `unknown`；复现本次镜像应保留当前源码与约束文件，并以内容摘要核对构建产物。

## 部署结果

按 [部署步骤](REPRODUCIBLE_DEPLOYMENT.md)以全新 Compose 项目启动 10 个服务；MySQL、etcd、MinIO、Milvus、Redis、FastAPI、MCP、Celery worker/beat 和 Streamlit 全部启动。API、MySQL、Milvus 和界面的健康检查通过；MCP 返回 5 个预期工具。统一镜像从 `deploy/Dockerfile` 重建，`pip check` 通过，Docling 已包含在镜像中。实装版本包括 Docling 2.140.0、PyTorch 2.5.1+cpu、LangChain Core 1.6.12、LangGraph 1.2.16、FastAPI 0.143.5、PyMilvus 2.6.17、Streamlit 1.66.0。

新镜像上的无密钥回归结果为 **71 passed，1 warning**。警告来自现有 PyPDF2 弃用提示。运行清单覆盖 LangChain/LangGraph 架构、SSE 和会话、M2–M6 关键契约、Streamlit 界面以及中文复核页面；命令见 `tmp/repro-pytest.log` 和 CI 清单。这不是云端 GitHub Actions 的结果。

## 论文入库

本次使用两篇固定公开论文。`Attention Is All You Need` 的 PDF SHA-256 为 `bdfaa68d8984f0dc02beaca527b76f207d99b666d31d1da728ee0728182df697`；`Dexterous Manipulation for Multi-Fingered Robotic Hands With Reinforcement Learning: A Review` 的 SHA-256 为 `e28afaf268f2f372fd8389526ece09a40bf306586ee508a5b7a68976c8ac8d65`。两篇经 Docling 解析和 BGE-M3 向量化，分别在隔离 Milvus 集合中生成 **158**、**287** 个片段。重复提交同一选文请求返回同一任务 ID。

首次执行时，验收脚本把第二篇论文的标题误写成另一个名称，项目的 PDF 标题校验拒绝了文件，任务状态为 `partial_success`。修正测试元数据后，同一任务重试成功；第一篇已完成论文没有重新处理。这个失败暴露的是测试数据准备错误，且确认了标题校验与失败重试路径实际起作用。旧失败和重试结果仍保存在本机报告中。

首次执行测试时尚未使用验收覆盖层，系统从本机环境继承了 OpenAlex 等搜索密钥；错误标题触发过 arXiv、OpenAlex、Crossref 的在线找论文。此事发生在真正问答之前。后续加入 `deploy/compose.acceptance.yaml`，确认 API 和 worker 的搜索/追踪密钥均为空，DeepSeek 密钥仍可用；覆盖层启动后再次通过了会话与 Milvus 的只读恢复检查。验收仍应如实记录这次意外在线搜索。

## 真实模型问答

使用已配置的 `deepseek-chat`。固定问题通过项目的 FastAPI/业务服务，检索两篇公开论文片段后请求真实模型。脚本计划 **5 道问答、6 次逻辑模型操作**（多轮追问包含一次问题改写）；底层 SDK 可能重试，本次没有独立计量供应商的实际 HTTP 请求次数。下表为本次本机结果，属于小规模功能回归，不是准确率测评。

| 场景 | 检查结果 | 引用 |
| --- | --- | --- |
| 表 2 数值问答 | 回答 EN-DE 28.4、EN-FR 41.8，事实检查通过 | 原论文第 8 页，1 条 |
| 第一轮流式问答 | 回答编码器 N=6；收到 19 个正文增量，首个增量早于完成事件 | 第 3 页，1 条 |
| 第二轮追问 | 结合 1 轮历史改写检索问题，回答解码器 N=6；收到 65 个正文增量 | 第 3 页，1 条 |
| 跨论文问答 | 回答涉及机器翻译与多指机器人灵巧操作，两篇均有引用 | 各 1 条 |
| 证据不足 | 对作者日常咖啡品牌明确拒答，置信度 0.0 | 0 条 |

每条答案还核对了引用论文范围、页码、片段、API 证据端点和 Markdown 导出。跨论文回答较保守：检索片段中包含章节标题、论文标题等浅层证据，模型只给出任务级对比，不能将本次通过解读成“复杂跨论文分析质量已经达标”。也没有重新跑 44 条人工复核或由本轮独立人工核对全部生成答案。

界面链路另用本地模型夹具验收：Streamlit 提交、两轮追问、历史恢复和导出通过；该脚具产生的文字是占位回答，外部模型调用为 0，不属于上表的真实答案。

## 全栈重启

完成问答后执行 `docker compose stop`，随后 `up -d --no-build --wait`。10 个服务重新启动；会话快照与重启前完全一致，已保存答案、引用证据和导出可重新读取，Milvus 对编码器/解码器层数问题仍检索到同一论文的 5 个片段。重启验证未再调用真实模型。

本机原始运行报告：`data/repro_acceptance_20261020/runtime/deployment_regression.json`（被 Git 忽略，含完整测试问答和公开论文证据）。构建、启动、无密钥回归、MCP、界面与重启日志位于本机 `tmp/repro-*.log`（被 Git 忽略）。当前隔离环境 API 为 `http://127.0.0.1:18010`，界面为 `http://127.0.0.1:18510`；它只装有两篇验收论文。
