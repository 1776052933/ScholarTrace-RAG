# Literature Database

`literature_database` 是 Auto_paper 的论文资源数据库管理模块，位于论文精排与 RAG 文档处理之间。它负责保存论文元数据、管理本地全文和解析文本、追踪下载任务、识别重复论文，并为无法自动下载的核心论文生成用户上传提醒。

## 1. 模块作用

模块提供以下能力：

- 保存 Ranking Agent 输出的论文元数据和精排分数；
- 按 `paper_id`、DOI 和标题检查论文是否已经存在；
- 记录 PDF、解析文本等本地论文文件；
- 判断 PDF、文本和 Embedding 是否全部可用；
- 创建并追踪论文下载任务；
- 调用 `paper_downloader` 自动搜索并下载 PDF；
- 保存下载失败原因和重试次数；
- 根据精排分数与引用数划分论文重要等级；
- 为下载失败的 CORE 论文创建人工上传任务；
- 向后续 RAG 模块返回已经准备完成的 `PaperResource`。

实际 PDF 下载由 `paper_downloader` 执行；Literature Database 负责调用其稳定接口并同步数据库状态。文档解析和 Embedding 由 `rag_reader` 负责。

## 2. 数据库结构

默认数据库为 SQLite：

```text
sqlite:///literature.db
```

可以通过环境变量切换数据库：

```powershell
$env:DATABASE_URL="postgresql+psycopg://user:password@host/database"
```

ORM 使用 SQLAlchemy 2.0 通用字段类型，可切换到 PostgreSQL。

### papers

保存论文元数据和资源处理状态。

| 字段 | 说明 |
| --- | --- |
| `paper_id` | 论文主键 |
| `title` | 论文标题 |
| `authors` | JSON 作者列表 |
| `abstract` | 摘要 |
| `doi` | DOI，唯一索引 |
| `arxiv_id` | arXiv ID，唯一索引 |
| `year` | 发表年份 |
| `venue` | 期刊、会议或预印本平台 |
| `citation_count` | 引用次数 |
| `rank_score` | Ranking Agent 综合分数 |
| `availability` | 论文资源状态 |
| `embedding_status` | Embedding 生成状态 |

### paper_files

保存论文相关文件记录。

| 字段 | 说明 |
| --- | --- |
| `id` | 文件记录主键 |
| `paper_id` | 关联论文 |
| `file_path` | 本地文件路径 |
| `status` | 文件处理状态 |
| `file_type` | `pdf`、`text` 等文件类型 |

### download_tasks

保存自动下载任务。

| 字段 | 说明 |
| --- | --- |
| `id` | 下载任务主键 |
| `paper_id` | 关联论文 |
| `status` | 下载状态 |
| `retry_count` | 重试次数 |
| `error` | 最近一次失败原因 |

下载状态包括：

- `PENDING`
- `DOWNLOADING`
- `SUCCESS`
- `FAILED`

### upload_queue

保存需要用户人工上传全文的论文。

| 字段 | 说明 |
| --- | --- |
| `id` | 上传任务主键 |
| `paper_id` | 关联论文 |
| `importance` | `CORE`、`IMPORTANT` 或 `NORMAL` |
| `status` | `PENDING` 或 `COMPLETED` |
| `reason` | 需要人工上传的原因 |

## 3. 论文状态流转

论文资源状态包括：

- `METADATA_ONLY`：尚未完成全部资源准备；下载成功后也可能表示 PDF 已有、但解析或 Embedding 未完成；
- `DOWNLOADING`：全文下载任务等待执行或正在执行；
- `DOWNLOAD_FAILED`：自动下载失败；
- `USER_UPLOAD_REQUIRED`：核心论文需要用户人工上传；
- `FULL_AVAILABLE`：PDF、解析文本和 Embedding 均已准备完成。

正常流程：

```text
METADATA_ONLY
      ↓ 创建下载任务
DOWNLOADING
      ↓ paper_downloader 下载 PDF
METADATA_ONLY（PDF 已就绪）
      ↓ RAG Reader 解析、清洗、切片并生成向量
RAG 索引建立完成
      ↓
FULL_AVAILABLE
```

失败流程：

```text
DOWNLOADING
      ↓ 自动下载失败
DOWNLOAD_FAILED
      ├── NORMAL / IMPORTANT：保留重试需求
      └── CORE：创建人工上传任务
                    ↓
          USER_UPLOAD_REQUIRED
                    ↓ 用户完成上传
             Document Parser
                    ↓
                Embedding
                    ↓
             FULL_AVAILABLE
```

下载状态为 `SUCCESS` 只表示 PDF 下载任务完成。只有 PDF 文件、解析文本和 Embedding 同时存在时，`AvailabilityChecker` 才会返回 `FULL_AVAILABLE`。

## 4. 与 Ranking Agent 的关系

Ranking Agent 负责相关性、质量、方法和任务匹配评分，并输出排序后的候选论文。Literature Database 消费这些候选论文，保存元数据和 `rank_score`，然后管理全文资源。

```text
Ranking Agent
      ↓ RankingResult + last_scores
LiteratureService.register_ranked_papers()
      ↓
论文去重、元数据保存、全文检查、下载任务创建
```

调用示例：

```python
from literature_database.service import LiteratureService

service = LiteratureService()

states = service.register_ranked_papers(
    ranking_result,
    scores=ranking_agent.last_scores,
)
```

如果论文已存在，模块会按 `paper_id`、DOI、标题识别，不重复写入论文记录或活动中的下载任务。

## 5. 与 RAG Reader 的关系

完整处理链路为：

```text
Ranking Agent
      ↓
Literature Database
      ↓
Document Parser
      ↓
Embedding
      ↓
RAG
```

各模块职责：

- Ranking Agent：对搜索结果进行精排，确定论文相关性和重要性依据；
- Literature Database：保存论文和文件状态，创建下载或上传任务；
- Document Parser：将 PDF 解析为可检索文本，并写入文本文件记录；
- Embedding：为解析文本生成向量，更新 `embedding_status`；
- RAG Reader：摄取已下载的本地 PDF，完成解析、切片、向量化和索引。

启用自动下载并交给 RAG Reader：

```python
from literature_database.service import LiteratureService
from rag_reader.pipeline import RAGReader

literature = LiteratureService()
reader = RAGReader()

resources = literature.prepare_for_reader(
    ranking_result,
    scores=ranking_agent.last_scores,
    output_dir="./papers",
    engine="academic_oa",
)

for resource in resources:
    reader.ingest_paper(resource)
```

`prepare_for_reader()` 会注册精排论文、执行待处理下载任务，并返回具有真实本地
PDF 的 `PaperResource`。`engine="academic_oa"` 默认使用 arXiv、OpenAlex、
Semantic Scholar 和 CrossRef；如需尝试包括 Google Scholar 在内的全部来源，
可改为 `engine="auto"`。

原有 `prepare_documents()` 保持兼容，只返回数据库中 PDF、解析文本和 Embedding
都已完成的 `FULL_AVAILABLE` 资源。

如果需要同时把 RAG Reader 生成的清洗文本和 Embedding 状态回写数据库，可直接
调用完整编排接口：

```python
results = literature.prepare_and_ingest(
    ranking_result,
    reader,
    scores=ranking_agent.last_scores,
    output_dir="./papers",
    text_output_dir="./papers/text",
)
```

成功记录的状态为 `FULL_AVAILABLE`；单篇下载或摄取失败会记录在对应结果的
`error` 中，不会中断其他论文。

RAG 调用入口：

```python
documents = service.prepare_documents(
    ranking_result,
    scores=ranking_agent.last_scores,
)

for document in documents:
    print(document.paper.title)
    print(document.file_path)
    print(document.text_path)
```

`prepare_documents()` 返回 `List[PaperResource]`，只包含当前可以进入 RAG 的论文。缺少全文的论文会进入下载流程，但不会出现在返回列表中。

查询需要人工上传的核心论文：

```python
missing_core_papers = service.get_missing_core_papers()

for request in missing_core_papers:
    print(request.paper_id, request.title, request.reason)
```

运行完整示例：

```powershell
python -m literature_database.examples
```
