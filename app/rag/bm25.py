"""
Dependency-free BM25 index for the document chunks.

This gives the retriever a lexical signal that the hashed embeddings do not
provide: exact terms like "reimbursement", "Net 45", or "single-vendor"
survive instead of being averaged away. It is intentionally small and
in-process; for a production corpus, replace it with OpenSearch, Postgres
full-text search, or a dedicated sparse-vector store.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Function words add noise on a small corpus and can outweigh the content
# terms in BM25's IDF calculation. Keep domain words such as "within",
# "before", and "over" because they change policy semantics.
_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "been", "by", "for",
        "from", "has", "have", "how", "in", "is", "it", "its", "of", "on",
        "or", "that", "the", "their", "there", "this", "to", "was", "were",
        "what", "when", "where", "which", "who", "why", "will", "with",
    }
)


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in _TOKEN_RE.findall(text.lower())
        if token not in _STOPWORDS
    ]


@dataclass(frozen=True)
class BM25Hit:
    doc_id: str
    score: float
    rank: int


class BM25Index:
    def __init__(
        self,
        documents: dict[str, str],
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self._ids = list(documents.keys())
        self._tokens = [tokenize(documents[doc_id]) for doc_id in self._ids]
        self._term_freqs = [Counter(tokens) for tokens in self._tokens]
        self._doc_lengths = [len(tokens) for tokens in self._tokens]
        self._k1 = k1
        self._b = b
        self._avgdl = (
            sum(self._doc_lengths) / len(self._doc_lengths)
            if self._doc_lengths
            else 1.0
        )

        doc_freq: Counter[str] = Counter()
        for tokens in self._tokens:
            doc_freq.update(set(tokens))
        n_docs = len(self._ids)
        self._idf = {
            term: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_freq.items()
        }

    def __len__(self) -> int:
        return len(self._ids)

    def score_all(self, query: str) -> dict[str, float]:
        """Return positive BM25 scores for every matching document."""
        query_terms = tokenize(query)
        if not query_terms:
            return {}

        scores: dict[str, float] = {}
        for doc_id, term_freqs, doc_len in zip(
            self._ids, self._term_freqs, self._doc_lengths
        ):
            score = 0.0
            for term in query_terms:
                freq = term_freqs.get(term, 0)
                if freq == 0:
                    continue
                idf = self._idf.get(term, 0.0)
                denom = freq + self._k1 * (
                    1.0 - self._b + self._b * doc_len / self._avgdl
                )
                score += idf * (freq * (self._k1 + 1.0) / denom)
            if score > 0.0:
                scores[doc_id] = score
        return scores

    def search(self, query: str, top_n: int | None = None) -> list[BM25Hit]:
        scores = self.score_all(query)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        if top_n is not None:
            ranked = ranked[:top_n]
        return [
            BM25Hit(doc_id=doc_id, score=score, rank=rank)
            for rank, (doc_id, score) in enumerate(ranked)
        ]
