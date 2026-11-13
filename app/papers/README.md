# 论文资源与持久化

此包负责论文元数据、PDF 文件、下载任务及数据库读写。主要入口：

| 模块 | 职责 |
| --- | --- |
| `schema.py` | 论文与资源的数据契约 |
| `models.py`、`database.py`、`repository.py` | SQLAlchemy 表模型、会话和仓储 |
| `downloader.py`、`download_tracker.py`、`upload_queue.py` | 调用独立的 `paper_downloader` 包，并跟踪下载或上传任务 |
| `service.py` | 注册候选论文、下载开放全文并交给 `rag_reader` 入库 |

HTTP 路由和任务状态由 `app/api/`、`app/services/`、`app/tasks.py` 管理。此包不负责 LangGraph 编排，也不直接处理界面请求。离线测试放在 `tests/papers/`。

下载时优先使用发现阶段传来的 PDF URL、DOI 或 arXiv ID 直接获取全文；这些定位信息无效或下载失败时，`paper_downloader` 才按标题搜索开放全文来源，并用标题匹配及 PDF 身份检查拦截错篇。这里的补源搜索不产生新的候选论文列表，`paper_downloader.search_papers()` 保留给独立下载器调用方。对应回归在 `tests/downloader/test_metadata_first_download.py`。
