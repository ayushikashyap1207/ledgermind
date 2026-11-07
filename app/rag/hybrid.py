"""
Small hybrid-retrieval helpers.

The retriever combines a dense candidate list (Chroma) with a sparse BM25
candidate list. Reciprocal Rank Fusion (RRF) merges the two ranked lists
without needing score normalization, then a feature-based reranker and MMR
select the final chunks.

These functions are pure and unit-testable; Chroma access stays in
app/rag/pipeline.py.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[str]], k: int = 60
) -> dict[str, float]:
    """Fuse ranked ID lists: score = sum(1 / (k + rank + 1))."""
    scores: dict[str, float] = {}
    for ids in ranked_lists:
        for rank, doc_id in enumerate(ids):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return scores


def normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    """Min-max normalize to [0, 1]; all-equal scores become 1.0."""
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        # All-equal: a nonzero signal is equally strong for every candidate;
        # an all-zero signal carries no information at all.
        return {key: (1.0 if hi > 0.0 else 0.0) for key in scores}
    return {key: (value - lo) / (hi - lo) for key, value in scores.items()}


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Dot product of already-normalized embeddings (all our embeddings are)."""
    return float(sum(x * y for x, y in zip(a, b)))


def mmr_select(
    candidate_ids: Iterable[str],
    relevance: dict[str, float],
    embeddings: dict[str, Sequence[float]],
    top_k: int,
    lambda_: float = 0.85,
) -> list[str]:
    """Select top_k with MMR, using reranked relevance as the utility."""
    pool = set(candidate_ids)
    selected: list[str] = []

    while pool and len(selected) < top_k:
        best_id: str | None = None
        best_score = -math.inf

        for doc_id in pool:
            utility = relevance.get(doc_id, 0.0)
            diversity = 0.0
            if selected:
                diversity = max(
                    cosine_similarity(embeddings[doc_id], embeddings[other])
                    for other in selected
                )
            score = lambda_ * utility - (1.0 - lambda_) * diversity
            if score > best_score or (score == best_score and best_id is not None and doc_id < best_id):
                best_score = score
                best_id = doc_id

        if best_id is None:
            break
        selected.append(best_id)
        pool.remove(best_id)

    return selected
