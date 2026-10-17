# 完整 Compose 本机验收（2026-10-17）

## 验收方式

使用独立项目名 `scholartrace_compose_accept` 启动 `compose.yaml` 中的全部六个服务：MySQL、etcd、MinIO、Milvus、FastAPI 和 Streamlit。仅在被 Git 忽略的 `data/compose_acceptance/` 中覆盖端口和测试数据路径：API 为 `127.0.0.1:18000`，界面为 `127.0.0.1:18501`，Milvus 为 `127.0.0.1:19531`。这避免占用原本正在运行的 `8000`、`8501`、`19530`；其余服务与正式 Compose 定义相同。

验收命令使用 `up -d --no-build --wait`，等待 MySQL、etcd、MinIO、Milvus、API 和界面的健康检查。API 启动时自动执行 Alembic，从空库依次迁移到 `2c63dc4ac6b4`、`9f1a43cd621e`。测试配置显式清空 DeepSeek、SerpAPI、OpenAlex 密钥，避免本机环境变量覆盖验收文件的空值。BGE-M3 使用本机已有的模型配置和权重，只在容器内离线读取。

## 结果

| 检查 | 结果 |
| --- | --- |
| 构建与启动 | 两个 Python 镜像成功构建；六个服务由 Compose 启动并全部通过健康检查 |
| API 与界面 | `GET /api/v1/health` 返回 200，MySQL 连通；Streamlit 首页和 `/_stcore/health` 返回 200；界面容器通过 `http://api:8000` 访问 API 返回 200 |
| API 与向量库 | API 容器通过 `http://milvus:19530` 连接 Milvus 并读取集合列表 |
| 迁移与持久化 | 空 MySQL 数据卷自动完成 M2、M3 迁移；测试论文记录及其索引签名保存在 MySQL |
| 离线论文摄取 | 从《Attention Is All You Need》本地 PDF 用 PyMuPDF 提取并切成 67 个片段，BGE-M3 生成 1024 维向量，Milvus 可实际查询到 67 行 |
| 检索 | 对编码器和解码器层数提问，返回 5 个同论文片段，包含第 3 页明确写有 `N = 6` 的证据 |
| 全套停止再启动 | `stop` 后用 `up -d --no-build --wait` 重启六个服务，全部再次健康；MySQL 论文仍为 `COMPLETED`，Milvus 仍为 67 行，检索再次返回同一证据 |
| 原有服务 | 原来占用 `8000`、`8501`、`19530` 的本机服务仍可连接 |

最终镜像中的关键版本为 PyTorch `2.5.1+cpu`、Sentence Transformers `5.4.1`、Transformers `4.57.6`。

## 验收中修复的问题

1. 原依赖范围允许安装 Sentence Transformers 6 / Transformers 5；实际构建选中该组合后，Transformers 5 因 PyTorch 2.5.1 低于其要求而禁用模型。现在在 `requirements.txt` 固定了已实测兼容的版本。
2. 默认从 PyPI 安装 PyTorch 会引入不需要的 CUDA 运行库；`deploy/Dockerfile` 改为先安装官方 CPU wheel，减少本机镜像体积和下载量，并让后续改动依赖文件时可复用这一层。
3. 原 Compose 没有 API、界面的健康检查，`--wait` 只能保证进程已运行。现在检查真实 HTTP 端点，并让界面等 API 健康后再启动。

首次下载 Docker Hub 基础镜像时出现过一次认证端点连接失败；单独拉取 `python:3.11-slim` 后构建成功。该问题没有再出现。验收用的 BGE-M3 缓存原本分散在两个本机修订目录；测试通过只读挂载权重和复制配置文件组合成离线目录，没有修改上游模型文件。

## 范围与复现

本次是**完整 Compose 启动与离线 RAG 主链路验收**。没有向 DeepSeek 发起新请求，也没有测试在线论文搜索、论文下载、Docling 或 reranker；这些结果不能当作生成答案质量评估。测试使用 PyMuPDF、Milvus dense 检索和一篇固定 PDF。

隔离验收的覆盖文件、随机数据库密码、模型缓存路径和原始日志均保存在被 Git 忽略的 `data/compose_acceptance/`，路径只适用于这台电脑。普通运行请准备 `.env` 中的 MySQL 密码，再执行 `docker compose --env-file .env up -d --build --wait`；默认端口若已被本机进程占用，需要先释放端口或使用自己的端口覆盖文件。首次摄取论文前须确保容器能取得完整的 BGE-M3 模型。Compose 会优先读取当前终端中已有的同名环境变量，离线运行时应明确清空第三方密钥。

验收结束后停止了隔离项目的六个容器以释放资源，未删除其 MySQL 和 Milvus 数据卷；原有的 Milvus 服务保持运行。

兼容性依据：[Transformers 4.57 安装要求](https://huggingface.co/docs/transformers/v4.57.0/installation)、[Transformers 5 安装要求](https://huggingface.co/docs/transformers/installation)、[PyTorch CPU 安装指引](https://pytorch.org/get-started/previous-versions/)。
