# RAG Reader：当前实现说明

本页描述当前代码中的 `rag_reader/`，与项目根目录的 [README](../README.md) 配合阅读。M0/M1 等历史验收文档保留当时的架构和结果，不作为当前运行配置。

## 职责与边界

`RAGReader`（`pipeline.py`）组合 PDF 解析、清洗、分块、Embedding、向量库、检索、问答、摘要和知识抽取。它不是 HTTP 服务：`app/api/` 暴露接口，`app/services/` 管理业务状态，`app/tasks.py` 运行后台入库；界面通过 FastAPI 访问这些能力。论文资源对象来自 `app/papers/`。

五个领域模型客户端共享 `ai_components/provider.py` 的配置与调用层；查询理解、拆解、搜索、排序和 RAG 各自保留输出 JSON 解析及面向该环节的异常契约。生成链由 `ai_components/chains.py` 提供。

## 当前主运行路径

1. 用户搜索论文或上传本地 PDF，API 在 MySQL 记录论文、任务和会话。Celery 从 Redis 接收任务 ID，负责后台下载和入库。
2. `RAGReader.prepare_paper()` 读取 PDF，使用 Docling 或 PyMuPDF 提取页面文本，清洗后按页和章节用 LangChain `RecursiveCharacterTextSplitter` 分块，保留论文 ID、页码等元数据。
3. `ingest_prepared()` 用 BGE-M3 生成向量并写入向量库。Compose 主运行采用 Milvus；重新摄取同一论文会替换其旧片段，并校验索引签名。Chroma 是显式可选的轻量后端。
4. 问答使用 `PaperQA` 和 `AnswerGraph`：追问先结合近期历史改写，再由 `EvidenceGraph` 检索证据；生成、解析引用和校验走同一张 LangGraph。多篇论文可并行取证。针对询问论文本身任务或贡献的问题，检索器会先排除标题和书目条目；若旧索引的正文候选不足，会在同一论文内补检一次。引用类问题仍可检索参考文献。模型调用由共享 LangChain 生成链适配 OpenAI 兼容接口，支持 DeepSeek 或配置的本地兼容模型。
5. 应用层将有效对话轮次写入 MySQL，FastAPI 通过 SSE 传输增量回答，Streamlit 显示引用并可打开原始 PDF 页。生成失败或断流的轮次不写入有效历史。

`AnswerGraph` 本身不持久化检查点；持久状态由应用层管理。`RAGReader.get_document()` 和 `get_chunks()` 保存的是当前进程内副本，进程重启后的论文目录、任务与会话状态来自数据库，检索索引来自向量库。

## 检索默认值与覆盖

Compose 的运行默认值是 **Milvus 混合检索（dense + BM25）+ RRF 合并 + BAAI/bge-reranker-v2-m3 重排**，先取候选再返回 Top-K。主要环境变量：

```text
RAG_VECTOR_BACKEND=milvus
RAG_COLLECTION_NAME=paper_chunks_m3
RAG_RETRIEVAL_MODE=hybrid
RAG_CANDIDATE_K=20
RAG_RERANKER_ENABLED=true
RAG_RERANKER_MODEL=BAAI/bge-reranker-v2-m3
RAG_PDF_PARSER_BACKEND=docling
```

实际部署值可在 `.env` 覆盖；`.env.example` 和 `compose.yaml` 给出默认配置。`RAGSettings` 的检索模式和重排开关也分别默认为 `hybrid`、`true`。直接在代码中无环境变量构造 `RAGReader()` 时，向量库仍默认 Chroma；Chroma 只支持 dense，`pipeline.py` 会自动使用 dense 检索，重排仍可启用。需要混合检索时必须选择 Milvus，并在入库和问答两端使用同一索引配置。

配置 `RAG_RERANKER_ENABLED=false` 可做无重排消融；`RAG_RETRIEVAL_MODE=dense` 或 `bm25` 可做检索消融。默认的严格支持性二次模型核验仍关闭，因现有对照实验发现过度拒答；问答继续走引用解析和已有答案约束。

## 关键文件

| 文件 | 当前作用 |
| --- | --- |
| `config.py`、`store_factory.py` | 参数读取、Chroma/Milvus 选择与索引范围 |
| `parser.py`、`docling_parser.py`、`cleaner.py`、`splitter.py` | PDF 解析、清洗、保留页码与分块 |
| `embedding.py`、`vector_store.py`、`milvus_store.py` | BGE-M3、Chroma、Milvus 向量索引 |
| `retriever.py`、`reranker.py`、`evidence_graph.py` | 召回、BGE 重排、多论文证据补检 |
| `provenance.py`、`answer_contracts.py` | 区分正文、标题、表格和参考文献；将论文标题与证据类型传给回答模型 |
| `qa.py`、`answer_graph.py`、`prompts.py` | 问题改写、证据问答、流式生成与引用校验 |
| `extractor.py`、`summarizer.py` | 可调用的知识抽取和结构化摘要 |
| `examples.py` | 显式使用 Chroma 的单文件演示，不是 Compose 主运行入口 |

## 运行与测试

在项目根目录按 [可复现部署说明](../docs/REPRODUCIBLE_DEPLOYMENT.md) 配置服务。离线测试入口统一为：

```powershell
python -m pytest -q tests -m "not slow and not network"
```

检索与回答的真实效果以 [评测目录](../evaluation/README.md) 中固定数据集和运行报告为准；接口与数据流可用 `tests/` 下的无密钥测试检查。

QASPER 的新评测索引直接使用数据集提供的章节名；重新解析 PDF 时会为书目条目写入 `References` 和来源类型。已有 Milvus 索引不会自动重写，查询时会尽量按旧片段的章节与文本补判来源；要让持久化元数据也更新，需重新入库对应论文。固定 30 题和真实 PDF 对照记录见 [来源与证据修复报告](../docs/SOURCE_PROVENANCE_REGRESSION_20261107.md)。
