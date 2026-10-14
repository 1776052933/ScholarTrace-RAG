"""使用 sentence-transformers 生成查询和论文的多语言向量表示。"""

from hashlib import blake2b
import logging
import math
import re
from threading import Lock
from typing import Any, ClassVar, Dict, Iterable, List, Optional, Tuple


logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"


class EmbeddingModel:
    """支持中英文科研文本的 SentenceTransformer 向量模型。

    模型采用进程级缓存。创建多个相同 ``model_name`` 和 ``device`` 的实例时，
    底层模型只会加载一次，避免重复占用内存和初始化时间。
    """

    _model_cache: ClassVar[Dict[Tuple[str, Optional[str]], Any]] = {}
    _cache_lock: ClassVar[Lock] = Lock()

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        *,
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
    ) -> None:
        model_name = model_name.strip()
        if not model_name:
            raise ValueError("model_name must not be empty")
        self.model_name = model_name
        self.device = device.strip() if isinstance(device, str) and device.strip() else None
        self.normalize_embeddings = bool(normalize_embeddings)

    def encode_text(self, text: str) -> List[float]:
        """将任意中文或英文文本编码为一维浮点向量。"""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        model = self._get_model()
        vector = model.encode(
            text.strip(),
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        if not isinstance(vector, (list, tuple)):
            raise TypeError("sentence-transformers returned an invalid embedding")
        return [float(value) for value in vector]

    def encode_texts(
        self,
        texts: Iterable[str],
        *,
        batch_size: int = 32,
    ) -> List[List[float]]:
        """批量编码文本，减少逐篇调用模型产生的固定开销。"""
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        values = list(texts)
        if any(not isinstance(text, str) for text in values):
            raise TypeError("texts must contain only strings")
        if not values:
            return []
        model = self._get_model()
        vectors = model.encode(
            [text.strip() for text in values],
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        if hasattr(vectors, "tolist"):
            vectors = vectors.tolist()
        if not isinstance(vectors, (list, tuple)) or len(vectors) != len(values):
            raise TypeError("sentence-transformers returned an invalid embedding")
        result = []
        for vector in vectors:
            if hasattr(vector, "tolist"):
                vector = vector.tolist()
            if not isinstance(vector, (list, tuple)):
                raise TypeError("sentence-transformers returned an invalid embedding")
            result.append([float(value) for value in vector])
        return result

    def encode_query(self, query: str) -> List[float]:
        """编码科研检索查询。BGE-M3 原生支持中文和英文查询。"""
        return self.encode_text(query)

    def encode_paper(self, title: str, abstract: str) -> List[float]:
        """组合标题和摘要后生成论文向量。"""
        return self.encode_text(_paper_text(title, abstract))

    def encode_papers(
        self,
        papers: Iterable[Tuple[str, str]],
        *,
        batch_size: int = 32,
    ) -> List[List[float]]:
        """批量组合论文标题和摘要并生成向量。"""
        texts = []
        for title, abstract in papers:
            texts.append(_paper_text(title, abstract))
        return self.encode_texts(texts, batch_size=batch_size)
    def _get_model(self) -> Any:
        cache_key = (self.model_name, self.device)
        model = self._model_cache.get(cache_key)
        if model is not None:
            return model

        # 双重检查防止多个搜索/排序线程同时重复加载大模型。
        with self._cache_lock:
            model = self._model_cache.get(cache_key)
            if model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as exc:
                    raise ImportError(
                        "EmbeddingModel requires sentence-transformers; "
                        "install it with: pip install sentence-transformers"
                    ) from exc

                logger.info("加载向量模型：%s，device=%s", self.model_name, self.device or "auto")
                kwargs = {"device": self.device} if self.device else {}
                try:
                    # Transformers 5.x 会拒绝在 Torch < 2.6 上读取 pickle/bin
                    # 权重。BGE-M3 提供 safetensors，因此显式禁止回退到 bin。
                    model = SentenceTransformer(
                        self.model_name,
                        model_kwargs={"use_safetensors": True},
                        **kwargs,
                    )
                except Exception as exc:
                    # 模型缓存损坏、网络不可用或 Torch 版本不兼容时，
                    # 精排仍应继续。本地哈希向量保留中英文词项重叠信号。
                    logger.warning(
                        "向量模型 %s 加载失败，使用本地哈希向量回退：%s",
                        self.model_name,
                        exc,
                    )
                    model = _HashingEncoder()
                self._model_cache[cache_key] = model
        return model

    @classmethod
    def clear_cache(cls) -> None:
        """清除进程级模型缓存，主要用于测试或显存资源切换。"""
        with cls._cache_lock:
            cls._model_cache.clear()


def _paper_text(title: str, abstract: str) -> str:
    """构造与原单篇编码完全一致的论文文本。"""
    if not isinstance(title, str) or not isinstance(abstract, str):
        raise TypeError("title and abstract must be strings")
    title = title.strip()
    abstract = abstract.strip()
    if title and abstract:
        return f"Title: {title}\nAbstract: {abstract}"
    return title or abstract


class _HashingEncoder:
    """无外部模型时的确定性多语言向量回退。"""

    dimension = 1024  # 与 BGE-M3 的 dense embedding 维度保持一致。

    def encode(self, texts: Any, **_: Any) -> Any:
        if isinstance(texts, str):
            return _hash_vector(texts, self.dimension)
        return [_hash_vector(str(text), self.dimension) for text in texts]


def _hash_vector(text: str, dimension: int) -> List[float]:
    normalized = text.casefold().strip()
    words = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", normalized)
    features = [*words, *(f"{a}:{b}" for a, b in zip(words, words[1:]))]
    vector = [0.0] * dimension
    for feature in features:
        digest = blake2b(feature.encode("utf-8"), digest_size=8).digest()
        number = int.from_bytes(digest, "big")
        vector[number % dimension] += -1.0 if number & 1 else 1.0
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector
