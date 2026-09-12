from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class HashEmbeddingFunction(EmbeddingFunction[Documents]):
    """Deterministic local embedding function to avoid external API coupling."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def __call__(self, input: Documents) -> Embeddings:
        return [self.embed_text(text) for text in input]

    def embed_text(self, text: str) -> list[float]:
        tokens = _TOKEN_RE.findall(text.lower())
        counts = Counter(tokens)
        vec = [0.0] * self.dim

        for token, count in counts.items():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dim
            sign = -1.0 if digest[4] % 2 else 1.0
            vec[idx] += sign * float(count)

        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            return vec
        return [v / norm for v in vec]
