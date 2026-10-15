"""RAG Reader 的论文摄取、问答和总结统一入口。"""

import logging
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional, Union

from literature_database.schema import PaperResource

from .cleaner import TextCleaner
from .config import RAGSettings
from .document_loader import DocumentLoader
from .embedding import EmbeddingModel, embedding_fingerprint
from .errors import RAGNotFoundError
from .extractor import PaperKnowledgeExtractor
from .llm_client import DeepSeekClient, LazyDeepSeekClient
from .parser import PDFParser
from .qa import PaperQA
from .retriever import PaperRetriever
from .schema import DocumentChunk, PaperDocument, PaperKnowledge, RAGAnswer
from .splitter import DocumentSplitter
from .summarizer import PaperSummarizer
from .vector_store import VectorStore


logger = logging.getLogger(__name__)


class PaperNotIngestedError(RAGNotFoundError, LookupError):
    """请求的论文尚未通过 ``ingest_paper`` 加入当前 RAGReader。"""


class RAGReader:
    """组合论文加载、解析、向量索引、检索和 LLM 生成组件。

    所有组件均支持依赖注入，便于测试以及后续替换向量库、Embedding 模型或
    知识抽取器。默认情况下，问答和摘要共享同一个 DeepSeek 客户端（如果调用方
    提供）以及同一个 Retriever。
    """

    def __init__(
        self,
        document_loader: Optional[DocumentLoader] = None,
        pdf_parser: Optional[PDFParser] = None,
        cleaner: Optional[TextCleaner] = None,
        splitter: Optional[DocumentSplitter] = None,
        embedding_model: Optional[EmbeddingModel] = None,
        vector_store: Optional[VectorStore] = None,
        extractor: Optional[Any] = None,
        qa: Optional[PaperQA] = None,
        summarizer: Optional[PaperSummarizer] = None,
        *,
        retriever: Optional[PaperRetriever] = None,
        llm_client: Optional[DeepSeekClient] = None,
        settings: Optional[RAGSettings] = None,
    ) -> None:
        self.settings = settings or RAGSettings.from_env()
        self.document_loader = document_loader or DocumentLoader()
        self.pdf_parser = pdf_parser or PDFParser()
        self.cleaner = cleaner or TextCleaner()
        self.splitter = splitter or DocumentSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        self.embedding_model = embedding_model or EmbeddingModel(
            model_name=self.settings.embedding_model,
            device=self.settings.embedding_device,
            batch_size=self.settings.embedding_batch_size,
        )
        self.vector_store = vector_store or VectorStore(
            collection_name=self.settings.collection_name,
        )
        self.retriever = retriever or PaperRetriever(
            embedding_model=self.embedding_model,
            vector_store=self.vector_store,
            default_top_k=self.settings.retrieval_top_k,
        )
        shared_client = llm_client or LazyDeepSeekClient(self.settings)
        self.extractor = extractor or PaperKnowledgeExtractor(llm_client=shared_client)
        self.qa = qa or PaperQA(
            self.retriever, llm_client=shared_client, top_k=self.settings.retrieval_top_k,
        )
        self.summarizer = summarizer or PaperSummarizer(llm_client=shared_client)

        self._documents: Dict[str, PaperDocument] = {}
        self._chunks: Dict[str, List[DocumentChunk]] = {}
        self._lock = RLock()

    def ingest_paper(self, resource: PaperResource) -> List[DocumentChunk]:
        """加载论文、清洗切片、生成向量并写入向量数据库。

        PDF 使用 ``PDFParser`` 提取正文；TXT/Markdown（或已有 ``text_path``）
        使用 ``DocumentLoader`` 读取的文本。重新摄取同一论文时会替换该论文原有
        的向量记录，避免旧 Chunk 残留。
        """
        if not isinstance(resource, PaperResource):
            raise TypeError(
                "resource must be a literature_database.schema.PaperResource"
            )

        document = self.document_loader.load(resource)
        raw_pages = self._load_pages(document)
        cleaned_pages = self.cleaner.clean_page_records(raw_pages)
        cleaned_content = "\n\n".join(page["text"] for page in cleaned_pages)
        if not cleaned_content:
            raise ValueError(f"paper contains no usable text: {document.paper_id}")

        document = document.model_copy(update={"content": cleaned_content}, deep=True)
        chunks = self.splitter.split(document, pages=cleaned_pages)
        if not chunks:
            raise ValueError(f"paper produced no document chunks: {document.paper_id}")

        # 先完成最耗时且可能失败的向量化，再替换旧索引，降低重摄取时的数据风险。
        embeddings = self.embedding_model.embed_chunks(chunks)
        if hasattr(self.vector_store, "ensure_embedding_compatible"):
            self.vector_store.ensure_embedding_compatible(
                embedding_fingerprint(self.embedding_model), len(embeddings[0])
            )
        self.vector_store.delete(paper_id=document.paper_id)
        stored = self.vector_store.add_documents(chunks, embeddings)
        if stored != len(chunks):
            raise RuntimeError(
                f"vector store wrote {stored} of {len(chunks)} chunks: "
                f"{document.paper_id}"
            )

        with self._lock:
            self._documents[document.paper_id] = document.model_copy(deep=True)
            self._chunks[document.paper_id] = [
                chunk.model_copy(deep=True) for chunk in chunks
            ]
        logger.info(
            "论文摄取完成：paper_id=%s, chunks=%d",
            document.paper_id,
            len(chunks),
        )
        return [chunk.model_copy(deep=True) for chunk in chunks]

    def ask(self, paper_id: str, question: str) -> RAGAnswer:
        """检索指定论文的相关片段并通过 DeepSeek 回答问题。"""
        paper_id = _required_text(paper_id, "paper_id")
        question = _required_text(question, "question")
        return self.qa.ask(paper_id, question)

    def remove_paper(self, paper_id: str) -> int:
        """删除身份失效论文的向量和本次进程内文档缓存。"""
        paper_id = _required_text(paper_id, "paper_id")
        removed = self.vector_store.delete(paper_id=paper_id)
        with self._lock:
            self._documents.pop(paper_id, None)
            self._chunks.pop(paper_id, None)
        logger.info("清理论文 RAG 缓存：paper_id=%s, chunks=%d", paper_id, removed)
        return removed

    def summarize(
        self,
        paper: Union[str, PaperDocument, PaperResource],
    ) -> Dict[str, str]:
        """生成结构化论文总结。

        ``paper`` 可传已摄取的论文 ID、完整 ``PaperDocument``，或可直接加载的
        ``PaperResource``。传入论文 ID 时使用摄取阶段缓存的清洗后全文。
        """
        document = self._resolve_document(paper)
        return self.summarizer.summarize(document)

    def extract_knowledge(
        self,
        paper: Union[str, PaperDocument, PaperResource],
    ) -> PaperKnowledge:
        """调用注入的 Extractor 抽取结构化论文知识。"""
        if self.extractor is None:
            raise RuntimeError(
                "knowledge extractor is unavailable; pass extractor=... to RAGReader"
            )
        document = self._resolve_document(paper)
        method = getattr(self.extractor, "extract", None)
        if not callable(method):
            raise TypeError("extractor must provide extract(document)")
        result = method(document)
        if isinstance(result, PaperKnowledge):
            return result
        return PaperKnowledge.model_validate(result)

    def get_document(self, paper_id: str) -> PaperDocument:
        """返回已摄取文档的副本。"""
        paper_id = _required_text(paper_id, "paper_id")
        with self._lock:
            document = self._documents.get(paper_id)
            if document is None:
                raise PaperNotIngestedError(f"paper not ingested: {paper_id}")
            return document.model_copy(deep=True)

    def get_chunks(self, paper_id: str) -> List[DocumentChunk]:
        """返回指定论文在本次运行中生成的全部 Chunk 副本。"""
        paper_id = _required_text(paper_id, "paper_id")
        with self._lock:
            chunks = self._chunks.get(paper_id)
            if chunks is None:
                raise PaperNotIngestedError(f"paper not ingested: {paper_id}")
            return [chunk.model_copy(deep=True) for chunk in chunks]

    def _load_pages(self, document: PaperDocument) -> List[Dict[str, Any]]:
        if document.file_path:
            path = Path(document.file_path)
            if path.suffix.casefold() == ".pdf":
                return [dict(page) for page in self.pdf_parser.parse(path)]
        if document.content.strip():
            return [{"page": None, "text": document.content}]
        raise ValueError(f"paper has no readable content: {document.paper_id}")

    def _resolve_document(
        self,
        paper: Union[str, PaperDocument, PaperResource],
    ) -> PaperDocument:
        if isinstance(paper, str):
            return self.get_document(paper)
        if isinstance(paper, PaperDocument):
            return paper.model_copy(deep=True)
        if isinstance(paper, PaperResource):
            document = self.document_loader.load(paper)
            pages = self._load_pages(document)
            content = self.cleaner.clean(pages)
            if not content:
                raise ValueError(f"paper contains no usable text: {document.paper_id}")
            return document.model_copy(update={"content": content}, deep=True)
        raise TypeError("paper must be a paper_id, PaperDocument, or PaperResource")


def _required_text(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    result = value.strip()
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result
