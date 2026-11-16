from __future__ import annotations

from functools import lru_cache

from app.rag.hybrid import normalize_scores

DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str, device: str):
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "The reranker is enabled but sentence-transformers is not installed. "
            "Run: pip install sentence-transformers"
        ) from exc
    return CrossEncoder(model_name, device=device)


class CrossEncoderReranker:
    """Cross-encoder scoring for the final retrieval candidate set."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER,
        device: str = "cpu",
        batch_size: int = 16,
        model=None,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = model

    def _get_model(self):
        if self._model is None:
            self._model = _load_cross_encoder(self.model_name, self.device)
        return self._model

    def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        model = self._get_model()
        raw_scores = model.predict(
            [(query, document) for document in documents],
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        scores: list[float] = []
        for value in raw_scores:
            if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
                score = float(list(value)[0])
            else:
                score = float(value)
            scores.append(score)
        return scores


def combine_rerank_scores(
    base_scores: dict[str, float],
    cross_scores: dict[str, float],
    weight: float = 0.7,
) -> dict[str, float]:
    """Blend cross-encoder relevance with the hybrid base ranking."""
    if not cross_scores:
        return dict(base_scores)
    cross_norm = normalize_scores(cross_scores)
    return {
        doc_id: weight * cross_norm.get(doc_id, 0.0)
        + (1.0 - weight) * base_scores.get(doc_id, 0.0)
        for doc_id in cross_scores
    }
