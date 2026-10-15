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

## 本次 M2 验收记录

- 契约与表结构：Alembic 在全新 SQLite 数据库升级到 head；实际 Alembic 迁移离线编译了 MySQL SQL（7 张表，含 Alembic 版本表及论文标题的前缀索引）。
- API 与界面：用真实浏览器经 Streamlit → FastAPI 完成论文检索、下载、PDF 入库与追问；论文 `arxiv:1706.03762` 产生 67 个检索片段，回答可展开定位至 PDF 第 3 页。
- 重启持久化：停止并重启 API 后，同一数据库中的任务、论文和问答证据仍可读取；新浏览器会话从侧边栏恢复已完成任务。
- 回归测试：包含 API 任务生命周期、取消、中断恢复、SQLite 导入和统一模型配置的自动化测试；完整非慢速测试集结果为 441 passed、22 skipped、3 deselected。
- 环境边界：此工作环境没有可运行的 Docker/MySQL 服务，因此 **MySQL 实例上的 Compose 启动、实际迁移和重启验收尚未执行**。上述真实浏览器和重启验证使用独立 SQLite 作为 M2 API 的本地存储；部署到有 Docker 的环境后，应再运行本页的 Compose 路径并复核 MySQL 数据留存。
