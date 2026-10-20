# 本地 PDF、论文集合与文档管理

在 Streamlit 首页底部展开“论文库 · 本地上传、集合与证据问答”。上传 PDF 后，FastAPI 校验文件大小（最多 50 MB）、PDF 文件头、可读取性、页数和加密状态；原件保存到持久化数据目录的 `uploads/`。同一内容以 SHA-256 去重，论文 ID 为 `upload:<sha256>`。页面上的标题只作为展示元数据；用户上传的 PDF 已由本地文件校验，因此不会套用网络下载的标题匹配规则。

如果论文已经由检索流程找到、但缺少本地 PDF，上传时可以选择“关联已有检索论文”。系统保留原论文 ID、元数据和集合关系，并用这份 PDF 触发入库；已有可用 PDF 的论文不能重复关联。作为新论文上传时才使用文件内容哈希作为论文 ID。

上传随后创建持久化入库任务，由原有 Celery worker 调用 Docling/PyMuPDF 解析、BGE-M3 向量化和 Milvus/Chroma 入库。页面显示最近上传的任务状态，并可重试失败任务；完整任务仍可从已有任务接口查询。重建索引会为本地有 PDF 的论文创建新任务；正在入库的论文不能重复重建索引。

论文集合保存在 MySQL 表 `paper_collections` 和 `paper_collection_items` 中，支持创建、改名、删除、加入和移出论文。删除集合不会删除论文。集合是业务分组，向量仍保存在现有共享 Milvus collection 内，检索时按论文 ID 限定；选择集合后可以从其中最多 5 篇已完成索引的论文发起现有连续问答。

文档管理可以查看来源、PDF 状态、索引状态，下载原始 PDF，重建索引。只有本地上传的论文可永久删除，且入库任务、问答和会话不能仍引用它；删除会清除其向量、数据库记录和受管理的原始/解析文件。检索得到的论文可以从集合移出，但不会通过此接口从全局论文库删除。

主要 API：

- `POST /api/v1/papers/upload`：multipart 字段 `file`、可选 `title`、`collection_id` 与用于关联已有检索论文的 `paper_id`；返回论文 ID 和入库任务 ID。
- `GET /api/v1/papers/{paper_id}/pdf`、`POST /api/v1/papers/{paper_id}/reindex`、`DELETE /api/v1/papers/{paper_id}`。
- `GET/POST /api/v1/collections`、`GET/PATCH/DELETE /api/v1/collections/{collection_id}`、`POST /api/v1/collections/{collection_id}/papers`、`DELETE /api/v1/collections/{collection_id}/papers/{paper_id}`。

升级已有数据库前运行 `python -m alembic upgrade head`。Compose 的 API、worker 和界面共用现有持久化数据目录，无需新的卷。当前版本没有账号或集合权限隔离，适合现阶段的单用户本机使用。
