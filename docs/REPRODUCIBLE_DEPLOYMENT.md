# 可复现部署与真实模型回归

这份流程针对本机 Docker Compose。运行在独立项目名和数据卷时，已有论文库及 44 条人工复核不会被覆盖。默认服务包括 MySQL、etcd、MinIO、Milvus、Redis、FastAPI、MCP、Celery worker/beat 和 Streamlit，共 10 个容器。向量模型与 Docling 模型缓存是部署依赖，需要随环境配置或首次联网获取。

## 准备

1. 启动 Docker Desktop。仓库根目录执行 `python -m scripts.init_deployment_env`，创建本地 `.env`。脚本只生成本地数据库密码，不覆盖现有文件。
2. 在 `.env` 中填写 `DEEPSEEK_API_KEY`。如已下载 BGE-M3，把完整模型目录放入 `SCHOLARTRACE_MODELS_DIR`，并把 `RAG_EMBEDDING_MODEL` 设为容器路径，例如 `/app/models/bge_m3_model`。Docling 缓存目录由 `SCHOLARTRACE_MODEL_CACHE` 指定；没有缓存时设 `HF_HUB_OFFLINE=0`，首次入库会下载模型。有缓存且希望完全离线解析时设 `HF_HUB_OFFLINE=1`。
3. 如果默认端口已被占用，在 `.env` 中调整 `SCHOLARTRACE_API_PORT`、`SCHOLARTRACE_UI_PORT`、`SCHOLARTRACE_MCP_PORT`、`SCHOLARTRACE_MILVUS_PORT`。不要把 `.env`、模型、PDF 或运行报告提交 Git。

## 启动与停止

```powershell
docker compose --env-file .env config --quiet
docker compose --env-file .env build api
docker compose --env-file .env up -d --no-build --wait
docker compose --env-file .env ps
```

`api`、`mcp`、`worker`、`beat`、`ui` 共用同一个镜像；`api` 启动时执行 Alembic。镜像使用固定 Python 3.11 基础镜像摘要、CPU Torch 和 `deploy/constraints-py311.txt` 中已实测的包版本。镜像包含 Docling；默认解析器为 Docling，可通过 `RAG_PDF_PARSER_BACKEND=pymupdf` 改用 PyMuPDF。目前切换需要重新摄取论文以更新索引，运行时没有自动解析回退。

检查：API `/api/v1/health`、界面 `/_stcore/health`、MCP `/mcp`；`docker compose --env-file .env ps` 应显示相关容器健康。停止可用 `docker compose --env-file .env stop`；重新启动用 `up -d --no-build --wait`。有状态数据位于 Compose 命名卷及 `SCHOLARTRACE_DATA_DIR`，重启不应丢失。不要使用 `down -v` 作为常规停止命令。

验收时若要禁用搜索和追踪密钥，在以下 Compose 命令的 `--env-file .env` 后增加 `-f compose.yaml -f deploy/compose.acceptance.yaml`。Compose 可能从当前终端继承同名环境变量；此覆盖层明确清空 API 和 worker 的搜索、追踪密钥，仍保留本地配置的 DeepSeek 密钥。

## 固定回归

测试材料是固定的两篇公开 PDF：`Attention Is All You Need` 和 `Dexterous Manipulation for Multi-Fingered Robotic Hands With Reinforcement Learning: A Review`。将其分别放到 `SCHOLARTRACE_DATA_DIR/fixtures/attention.pdf` 与 `dexterous.pdf`。 本次实测文件的 SHA-256 分别为 `bdfaa68d8984f0dc02beaca527b76f207d99b666d31d1da728ee0728182df697` 和 `e28afaf268f2f372fd8389526ece09a40bf306586ee508a5b7a68976c8ac8d65`；回归报告也保存实际哈希，便于核对是否为同一输入。为避免与已有业务数据混淆，建议使用全新 Compose 项目名、数据库卷、向量库卷和集合名，或复制本次本机验收目录的隔离配置。此回归脚本要求数据库中没有 `repro-attention`、`repro-dexterous` 两条记录。

```powershell
docker compose --env-file .env exec -T api python -m scripts.deployment_regression --phase ingest
docker compose --env-file .env exec -T api python -m scripts.deployment_regression --phase model --real-model
docker compose --env-file .env stop
docker compose --env-file .env up -d --no-build --wait
docker compose --env-file .env exec -T api python -m scripts.deployment_regression --phase verify
```

第一阶段经 Celery worker 使用 Docling 解析、BGE-M3 向量化、Milvus 入库，并验证重复入库请求合并到同一任务。第二阶段向配置的真实模型发起固定问题，覆盖表格事实、两轮 SSE 追问、跨论文引用及证据不足拒答。计划六次逻辑上的模型操作；SDK 网络重试可能增加实际 HTTP 次数。脚本将尝试状态和每个结果写到 `SCHOLARTRACE_DATA_DIR/deployment_regression.json`。为防重复计费，模型阶段一旦开始，使用相同报告文件就不会自动再次运行；如需重测，先检查旧报告和实际调用记录，再使用新的独立环境和报告文件。第三阶段重新打开数据库和向量库，核对会话、答案导出及检索是否保持一致。测试的答案正确性只覆盖几项确定事实，不能替代 44 条人工复核。

无密钥代码回归命令见 `.github/workflows/ci.yml`。本次本机结果及未通过项记录在后续验收报告中。
