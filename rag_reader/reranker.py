"""Optional multilingual cross-encoder reranking for retrieved passages."""
from __future__ import annotations

import math
from threading import Lock
from typing import Any, Sequence


class CrossEncoderReranker:
    _cache: dict[str, Any] = {}
    _lock = Lock()

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", *, model: Any = None) -> None:
        if not model_name.strip():
            raise ValueError("model_name is required")
        self.model_name = model_name.strip()
        self._model = model

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        with self._lock:
            if self.model_name not in self._cache:
                try:
                    from sentence_transformers import CrossEncoder
                except ImportError as exc:
                    raise ImportError("cross-encoder reranking requires sentence-transformers") from exc
                self._cache[self.model_name] = CrossEncoder(self.model_name, max_length=512)
            self._model = self._cache[self.model_name]
        return self._model

    def rerank(
        self, question: str, results: Sequence[dict[str, Any]], *, top_k: int
    ) -> list[dict[str, Any]]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if not results:
            return []
        pairs = [(question, str(item["text"])) for item in results]
        scores = self._get_model().predict(pairs)
        if len(scores) != len(results):
            raise RuntimeError("cross-encoder returned the wrong number of scores")
        ranked = []
        for index, (item, raw) in enumerate(zip(results, scores)):
            raw_score = float(raw)
            if not math.isfinite(raw_score):
                raise ValueError("cross-encoder produced a non-finite score")
            # CrossEncoder models may return either logits or activated values.
            # The order uses raw scores; the bounded value is only for citations.
            bounded = raw_score if 0.0 <= raw_score <= 1.0 else (
                1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, raw_score))))
            )
            updated = dict(item)
            metadata = dict(updated.get("metadata") or {})
            metadata.update({
                "first_stage_score": item.get("score"),
                "rerank_score": bounded,
                "raw_rerank_score": raw_score,
            })
            updated["metadata"] = metadata
            updated["score"] = bounded
            updated["distance"] = 1.0 - bounded
            ranked.append((raw_score, index, updated))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [item for _, _, item in ranked[:top_k]]
