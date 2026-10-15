# M2 运行与验收

## 使用 Docker Compose（MySQL）

复制 `.env.example` 为本地 `.env`，填写 MySQL 密码和所需模型/检索 API Key。`.env` 和运行数据已排除在 Git 与镜像构建上下文外。执行：

```powershell
docker compose up --build -d
```

浏览器打开 `http://127.0.0.1:8501`；API 文档为 `http://127.0.0.1:8000/docs`。Compose 在启动 API 前执行 `alembic upgrade head`，MySQL 数据保存在命名卷，PDF/Chroma 数据保存在本地 `data/m2`。API 按单 worker 运行；任务记录在 MySQL 中，运行中的任务在 API 重启后标为 `interrupted`，需要人工重新提交。M4 引入 Redis/Celery 后再实现自动恢复。

### 导入原 M1 SQLite 论文库

先确认旧库位于 `Auto_Paper_UI/runtime/literature.db`，旧 PDF 在其 `papers` 目录。下面的命令在容器中备份 SQLite，拷贝 PDF/文本到新持久化目录，并导入四张论文资源表；已导入的相同记录会跳过，主键内容冲突会显式报错：

```powershell
docker compose run --rm -v "${PWD}/Auto_Paper_UI/runtime:/app/legacy_runtime:ro" api python -m scripts.migrate_sqlite_to_mysql --source /app/legacy_runtime/literature.db --assets-root /app/data/m2/papers --backup-dir /app/data/m2/backups
docker compose run --rm api python -m scripts.reindex_migrated_papers --data-root /app/data/m2
```

导入后必须重建 Chroma 索引，才能对旧论文问答。旧 SQLite、备份、PDF 和向量库都不提交到 Git。导入前建议停止正在写入旧 SQLite 的 M1 进程。

## 无 Docker 的本地 API 验收

可用独立 SQLite 验证 HTTP 调用、任务持久化和重启行为；这不能代替 MySQL 实例联调。先安装 `requirements.txt` 与 `Auto_Paper_UI/requirements-ui.txt`，再在项目根目录执行：

```powershell
$env:DATABASE_URL='sqlite:///data/m2/app.sqlite3'
$env:AUTO_PAPER_DATA_ROOT='data/m2'
python -m alembic -c alembic.ini upgrade head
python -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000 --workers 1
```

另开终端：

```powershell
$env:AUTO_PAPER_API_URL='http://127.0.0.1:8000'
python -m streamlit run Auto_Paper_UI/ui_app.py
```

M2 继续使用 Chroma/BGE-M3；MySQL 中保存论文、任务和问答结果。固定语料质量改进属于 M3/M6 的检索与评测工作。

## 本次 M2 验收记录（2026-10-15）

- **真实 MySQL 迁移**：在 Windows 本机 MySQL 8.4.11 上执行 Alembic `upgrade head`，得到 6 张业务表和 `alembic_version`，版本为 `2c63dc4ac6b4`。Docker/WSL 在这台机器上不可用，所以 Compose 运行路径尚未实测。
- **旧数据迁移**：从 M1 SQLite 导入 27 篇论文、44 条文件记录、32 条下载任务和 2 条上传队列记录；PDF/文本文件均在新持久化目录中。重复执行导入时，已有记录被跳过。随后真实界面流程新增 1 条文件记录。
- **旧论文索引**：只对 PDF 和解析文本均存在且状态有效的 16 篇论文重建 Chroma 索引，共 2435 个片段；`MISMATCH` 文件没有入库。其余 11 篇仍为 `PENDING`，需要补齐或修复原始资源后才能问答。
- **真实界面流程**：浏览器经 Streamlit → FastAPI → MySQL 完成论文检索、下载、PDF 入库和追问。`arxiv:1706.03762` 下载及索引成功，产生 67 个片段；问答给出 Transformer 编码器 6 层，并能展开定位到 PDF 第 3 页。
- **重启恢复**：停止并重启 API 后，同一个已完成任务仍有完整结果；27 篇论文和已保存的问答证据可由 API 读取。新浏览器会话从侧边栏恢复了任务和 11 个结果指标。另以模拟运行中任务验证了启动时会标记为 `interrupted` 且可重试。
- **实测修复**：MySQL 对包含大 JSON 结果的任务列表排序出现内存错误，现列表只查询轻量摘要，详情按任务 ID 获取；Windows 中文绝对路径导致 Chroma HNSW 重载失败，向量存储改用指向同目录的 ASCII 相对路径。两项均通过真实服务复测。
- **任务状态修复**：单篇下载或入库失败不再被统一标记为 `completed`。有可用论文但存在失败时为 `partial_success`，无可用论文时为 `failed`；两者保留完整结果、失败提示和可重试标记。真实 MySQL 验收中原先误标完成的失败任务已纠正，浏览器验证了告警和结果明细可恢复。
- **回归范围**：API 生命周期、任务恢复、旧库迁移、索引状态筛选、Windows 路径、界面逻辑等纳入自动化测试；完整非慢速测试结果为 **446 passed、22 skipped、3 deselected**。

### 在原生 MySQL 上复核

准备 MySQL 8.4 实例和独立数据库账号，设置 `DATABASE_URL` 与 `AUTO_PAPER_DATA_ROOT`，然后在项目根目录执行：

```powershell
python -m alembic upgrade head
python -m scripts.migrate_sqlite_to_mysql --source Auto_Paper_UI/runtime/literature.db --assets-root data/m2/papers --backup-dir data/m2/backups
python -m scripts.reindex_migrated_papers --data-root data/m2
python -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000 --workers 1
```

另开终端运行界面：

```powershell
$env:AUTO_PAPER_API_URL='http://127.0.0.1:8000'
python -m streamlit run Auto_Paper_UI/ui_app.py
```

使用 API 的 `/api/v1/health`、`/api/v1/jobs` 和 `/api/v1/papers` 检查持久化结果，再从 Streamlit 执行检索和问答。真实服务与旧库路径必须指向同一套持久化数据。Windows 中文目录下建议从项目根目录启动服务，使 Chroma 能使用 ASCII 相对路径。
