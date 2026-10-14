# RAG Reader

`rag_reader` 是 Auto_paper 的独立论文深度理解模块。它接收
`literature_database` 提供的 `PaperResource`，将论文全文转换为可检索的向量索引，
并提供结构化知识抽取、基于证据的论文问答和论文摘要能力。

模块不提供 Web 或 FastAPI 接口，其他 Agent 可以直接调用 `RAGReader`。

## 1. 模块作用

主要功能包括：

- 加载 PDF、TXT 和 Markdown 论文资源；
- 使用 PyMuPDF 提取 PDF 页面文本；
- 清理页眉页脚、乱码、重复内容和异常断行；
- 将论文切分为适合检索的 `DocumentChunk`；
- 使用 BGE-M3 生成中英文文本向量；
- 将文本、Embedding 和元数据写入 Chroma；
- 根据问题检索相关论文片段；
- 使用 DeepSeek 抽取 `PaperKnowledge`；
- 基于检索证据回答论文问题；
- 生成背景、方法、实验、结果、局限和结论摘要。

## 2. 数据流

论文摄取流程：

```text
Literature Database / PaperResource
                 ↓
          DocumentLoader
                 ↓
       PDFParser（PDF 文件）
                 ↓
            TextCleaner
                 ↓
         DocumentSplitter
                 ↓
       BGE-M3 EmbeddingModel
                 ↓
          Chroma VectorStore
```

论文问答流程：

```text
paper_id + question
         ↓
   PaperRetriever
         ↓
   Top-K 相关 Chunk
         ↓
      DeepSeek
         ↓
RAGAnswer（回答、来源 Chunk、置信度）
```

论文总结流程：

```text
已摄取的 PaperDocument
           ↓
       DeepSeek
           ↓
背景 / 方法 / 实验 / 结果 / 局限 / 结论
```

重新摄取相同 `paper_id` 时，`RAGReader` 会替换该论文原有的 Chroma Chunk，避免旧索引残留。

## 3. 依赖

核心依赖：

```text
pydantic
openai
pymupdf
sentence-transformers
chromadb
langchain-text-splitters
```

可以使用以下命令安装：

```bash
pip install pydantic openai pymupdf sentence-transformers chromadb langchain-text-splitters
```

各依赖用途：

- `pydantic`：文档、Chunk、知识和问答结果校验；
- `openai`：通过 OpenAI 兼容接口调用 DeepSeek；
- `pymupdf`：PDF 文本提取；
- `sentence-transformers`：加载 `BAAI/bge-m3` 并生成向量；
- `chromadb`：向量存储和相似度检索；
- `langchain-text-splitters`：递归文本切片。

首次使用 BGE-M3 时，`sentence-transformers` 会下载模型文件，需要可用的模型下载环境和足够的磁盘空间。

## 4. DeepSeek 配置

模块使用 DeepSeek 的 OpenAI 兼容接口，不会在代码中写死 API Key。

PowerShell：

```powershell
$env:DEEPSEEK_API_KEY="your-api-key"
$env:DEEPSEEK_BASE_URL="https://api.deepseek.com"
$env:DEEPSEEK_MODEL="deepseek-chat"
```

Linux 或 macOS：

```bash
export DEEPSEEK_API_KEY="your-api-key"
export DEEPSEEK_BASE_URL="https://api.deepseek.com"
export DEEPSEEK_MODEL="deepseek-chat"
```

其中：

- `DEEPSEEK_API_KEY`：必需；
- `DEEPSEEK_BASE_URL`：可选，默认 `https://api.deepseek.com`；
- `DEEPSEEK_MODEL`：可选，默认 `deepseek-chat`。

如果项目的 `API_key/key_manager.py` 已配置 `deepseek`，客户端也会尝试通过
`key_get("deepseek")` 读取密钥。推荐优先使用环境变量，便于部署和密钥轮换。

## 5. 向量数据库配置

`VectorStore` 使用 Chroma。默认构造方式创建进程内 Chroma 客户端：

```python
from rag_reader.pipeline import RAGReader

reader = RAGReader()
```

如需在程序重启后保留向量数据，应配置持久化目录：

```python
from rag_reader.pipeline import RAGReader
from rag_reader.vector_store import VectorStore

vector_store = VectorStore(
    collection_name="paper_chunks",
    persist_directory="./data/chroma",
)

reader = RAGReader(vector_store=vector_store)
```

每条向量记录包含：

- Chunk Embedding；
- Chunk 原文；
- `paper_id`；
- 页码（可获得时）；
- 章节；
- Chunk 扩展元数据。

## 6. 调用示例

以下示例导入一篇本地 PDF，然后执行问答和摘要：

```python
import json

from literature_database.schema import (
    PaperAvailability,
    PaperMetadata,
    PaperResource,
)
from rag_reader.pipeline import RAGReader
from rag_reader.vector_store import VectorStore


resource = PaperResource(
    paper=PaperMetadata(
        paper_id="paper-001",
        title="Swin UNETR: Swin Transformers for Semantic Segmentation",
    ),
    availability=PaperAvailability.FULL_AVAILABLE,
    file_path="./papers/swin_unetr.pdf",
)

reader = RAGReader(
    vector_store=VectorStore(
        collection_name="paper_chunks",
        persist_directory="./data/chroma",
    )
)

# 1. 导入论文、切片、生成向量并建立索引
chunks = reader.ingest_paper(resource)
print(f"已写入 {len(chunks)} 个 Chunk")

# 2. 基于论文片段问答
answer = reader.ask(
    "paper-001",
    "该论文提出的方法是什么？",
)
print(answer.model_dump_json(indent=2))

# 3. 生成结构化摘要
summary = reader.summarize("paper-001")
print(json.dumps(summary, ensure_ascii=False, indent=2))
```

`reader.ask()` 返回 `RAGAnswer`：

```json
{
  "question": "该论文提出的方法是什么？",
  "answer": "论文提出……",
  "sources": ["paper-001-chunk-00003"],
  "confidence": 0.92
}
```

`reader.summarize()` 返回：

```json
{
  "background": "",
  "method": "",
  "experiment": "",
  "result": "",
  "limitation": "",
  "conclusion": ""
}
```

如需运行完整命令行示例：

```bash
python -m rag_reader.examples ./papers/paper.pdf --title "论文标题"
```

该示例会依次完成 PDF 导入、Chroma 建库、知识抽取、摘要生成和论文问答，并输出回答所引用的来源 Chunk。
