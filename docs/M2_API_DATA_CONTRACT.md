# M2 API 与数据契约

M2 保留现有 `AutoPaperWorkflow`、BGE-M3、PyMuPDF 和 Chroma。Streamlit 只调用 HTTP；FastAPI 复用领域服务，SQLAlchemy/MySQL 保存业务状态。任务执行在单个 API 进程的后台线程中；服务重启后未完成任务标记为 `interrupted`，保留错误和历史，自动恢复留待 M4 队列化。

## HTTP v1

| 方法和路径 | 作用 | 响应 |
| --- | --- | --- |
| `GET /api/v1/health` | 数据库及外部密钥可用性 | 布尔状态，不返回密钥 |
| `POST /api/v1/jobs` | 启动当前完整研究流程 | `202` 与持久化任务快照；已有任务执行时 `409` |
| `GET /api/v1/jobs` | 最近任务，用于界面恢复 | 任务摘要列表 |
| `GET /api/v1/jobs/{job_id}` | 查询阶段、日志、结果和错误 | 任务快照 |
| `POST /api/v1/jobs/{job_id}/cancel` | 请求阶段间协作式取消 | 任务快照 |
| `GET /api/v1/papers` | 查询已登记论文 | 论文摘要 |
| `POST /api/v1/questions` | 对已入库论文追问 | 回答与 `answer_id` |
| `GET /api/v1/answers/{answer_id}/evidence` | 取持久化引用 | 结构化证据 |

任务状态为 `queued → running → completed/failed`，取消时为 `running → cancelling → cancelled`。启动时将旧的 `queued/running/cancelling` 标记为 `interrupted`。阶段、进度与有限条日志在数据库中更新，原始密钥不进入请求、日志和任务表。

## 表结构

沿用 `papers`、`paper_files`、`download_tasks`、`upload_queue`；增加 `research_jobs`（输入、阶段、状态、结果、错误、时间）和 `question_answers`（论文 ID、问题、回答及引用）。PDF 与 Chroma 索引保存在持久化目录，数据库记录可用于重启后找回任务和论文。MySQL 表结构由 Alembic 管理；旧 SQLite 文件先备份再迁移。
