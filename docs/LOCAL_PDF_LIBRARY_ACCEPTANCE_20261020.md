# 本地 PDF 与论文集合验收（2026-10-20）

本轮在既有隔离 Compose 项目 `scholartrace_repro` 中升级镜像和 Alembic 表结构。API、Streamlit、MySQL、Redis、Celery worker/beat、Milvus、etcd、MinIO 和 MCP 10 个服务均通过健康检查。隔离 API：`http://127.0.0.1:18010`；界面：`http://127.0.0.1:18510`。

无密钥回归按 `.github/workflows/ci.yml` 的测试清单执行，加上 `test_paper_library_api.py`，结果为 **74 passed、1 warning**。唯一警告是原有 PyPDF2 弃用提示。新增测试覆盖无效 PDF 拒绝、同内容去重、关联已有检索论文、集合增删改与重启持久化、原件下载、任务冲突和安全删除。

真实链路使用程序生成的一页英文测试 PDF，通过 `/api/v1/papers/upload` 上传；其展示标题与 PDF 文本中的标题不同。Celery 后台任务一次完成，Docling/BGE-M3/Milvus 产生 **3 个片段**。验收检查原始 PDF 下载字节完全一致、集合成员正确、Milvus 中有该论文向量，删除后向量与论文记录清除。测试 PDF 与集合均已删除；验收结束后隔离论文库仍为之前的 2 篇，集合数为 0。该测试未调用 DeepSeek，也没有重新执行 44 条人工复核。

界面通过 Streamlit 自动化回归和 Compose 健康检查。此处的 1 页测试 PDF 用于证明上传到索引的工作流可用，不代表对复杂学术 PDF 解析准确率或真实问答质量进行了新一轮评估。
