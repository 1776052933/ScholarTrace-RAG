# M4 异步入库与交互：运行和验收

## 目标与实现

M4 把“检索候选论文”和“下载、解析、向量入库”拆成两个用户动作。检索完成后，用户在 Streamlit 勾选论文；FastAPI 创建可恢复的入库任务，MySQL 保存任务和逐篇状态，Redis 传递任务 ID，Celery worker 处理 PDF，Celery beat 每 60 秒检查遗留任务。入库成功的论文可在论文库中选择 2–5 篇提问，并导出带 PDF 页码和片段摘录的 Markdown 答案。

同一检索任务、论文集合和下载选项得到同一个入库任务 ID。工作进程通过数据库条件更新抢占 `queued` 任务；重复投递不会重复执行正在运行或已完成的任务。失败后只重试未完成论文，每篇最多三次；运行中的任务每 10 秒更新心跳，超过 120 秒无心跳的任务由定时检查重新排队。取消在当前论文处理结束后生效。MySQL 是任务状态来源，Redis 仅负责传递任务 ID。

跨论文问答先取所选论文的全局 Top-K 片段；如果某篇未出现在候选证据中，再为该论文补取一条片段。回答中的来源 ID 只接受本次检索到的片段，引用保留论文 ID、页码、章节和原文摘录。此补取保证模型能看到各篇论文的证据，不保证每篇都会被答案引用。

主要代码：`app/services/ingestions.py`、`app/tasks.py`、`app/services/questions.py`、`rag_reader/qa.py`、`app/api/main.py`、`Auto_Paper_UI/ui_app.py`、`migrations/versions/bd7492a4c18e_m4_async_jobs.py`。

## 本机启动

```powershell
Copy-Item .env.example .env
# 在 .env 中填写 MYSQL_PASSWORD、MYSQL_ROOT_PASSWORD；需要实时搜索/生成时再填写 API Key。
docker compose up --build -d
docker compose ps
```

默认界面为 `http://127.0.0.1:8501`，API 文档为 `http://127.0.0.1:8000/docs`。端口被占用时在 `.env` 中设置 `SCHOLARTRACE_UI_PORT`、`SCHOLARTRACE_API_PORT`。容器首次进行 embedding 时会按 `RAG_EMBEDDING_MODEL` 下载模型；离线机器可将已下载的 BGE-M3 目录挂载到 API 和 worker 的相同容器路径，并把两者的 `RAG_EMBEDDING_MODEL` 都设成该路径。API 与 worker 还必须使用相同的 Milvus 集合和检索配置。

使用步骤：输入研究问题并保持“先看候选，再选择入库”开启 → 检索完成后勾选论文 → 查看逐篇下载和入库进度 → 失败时重试未完成论文 → 在“论文库 · 多篇证据问答”选择 2–5 篇 → 查看来源并导出。刷新后可从“最近任务”恢复查看。

离线验收可以使用 `scripts/m4_acceptance_seed.py` 把已经合法取得且位于 API 可见路径的 PDF 登记为候选，不触发在线搜索或生成模型调用。例如在 API 容器中运行 `python -m scripts.m4_acceptance_seed --paper "paper-id|Paper title|/app/data/m2/papers/example.pdf"`，再用返回的 `discovery_job_id` 调用 `POST /api/v1/ingestions`。`scripts/m4_acceptance_retrieve.py` 可用本地 embedding 检查多论文 Milvus 证据检索，不调用生成模型。

## 2026-10-17 本机验收结果

环境：Windows Docker Compose、MySQL 8.4、Redis 7、Milvus 2.6、API、worker、beat、Streamlit。为避开本机已有服务，验收时 API/UI 分别映射到 18000/18501；BGE-M3 使用此前下载的本地模型目录。九个服务均启动，MySQL、Redis、Milvus、API 和 UI 的健康检查通过。

| 检查 | 结果 |
| --- | --- |
| M4/M3/M2/API/旧 UI 回归 | 22 项通过 |
| M4 离线 Streamlit 选文、问答、导出组件测试 | 2 项通过 |
| Streamlit 组件连接真实 API，提交选文并恢复任务 | 1 项通过 |
| 真实选文入库 | 两篇本地 PDF 通过 Redis/Celery 入库，任务 `completed`；分别生成 67、91 个片段 |
| 逐篇解析失败与重试 | 损坏 PDF 首次显示 `PDFOpenError`、任务 `failed`；替换有效 PDF 后原任务第二次尝试 `completed`，生成 67 个片段 |
| 重启与持久化 | 重启 API 和 worker 后两个任务仍为 `completed`，三篇论文在 API 中均为 `FULL_AVAILABLE / COMPLETED` |
| Milvus 与多论文检索 | 当前集合确认两篇论文均存在；本地 BGE-M3 检索跨论文问题返回两篇论文的页码证据 |

初次真实入库暴露了 Celery 命令启动时找不到项目内 `paper_downloader` 包的问题；改为 `python -m celery` 后，从 MySQL 中保留的原任务直接重试成功。上述结果使用修复后的 Compose 配置。

本轮未调用 DeepSeek，跨论文回答的 API 契约、引用保存与导出由模拟生成响应的测试覆盖；真实跨论文生成质量应在后续固定问题集上单独评估。Streamlit 通过组件测试和真实 API 交互验收；浏览器视觉布局尚未进行人工走查。
