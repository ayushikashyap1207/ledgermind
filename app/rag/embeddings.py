from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class EmbeddingProfile:
    query_prefix: str = ""
    document_prefix: str = ""


def embedding_profile_for_model(model_name: str) -> EmbeddingProfile:
    """Return the query/document prefixes required by a model family."""
    name = model_name.lower()
    if "bge" in name:
        # Optional for BGE v1.5, but recommended for short-query -> passage.
        return EmbeddingProfile(
            query_prefix="Represent this sentence for searching relevant passages: "
        )
    if "e5" in name:
        return EmbeddingProfile(query_prefix="query: ", document_prefix="passage: ")
    if "nomic" in name:
        return EmbeddingProfile(
            query_prefix="search_query: ", document_prefix="search_document: "
        )
    if "qwen3-embedding" in name:
        return EmbeddingProfile(
            query_prefix=(
                "Instruct: Given a search query, retrieve relevant passages that "
                "answer the query\nQuery: "
            )
        )
    return EmbeddingProfile()


def _model_slug(model_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", model_name).strip("-")


class HashEmbeddingFunction(EmbeddingFunction[Documents]):
    """Deterministic local fallback used by tests and offline development."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    @staticmethod
    def name() -> str:
        return "hash"

    def collection_name_suffix(self) -> str:
        return f"hash-{self.dim}"

    def get_config(self) -> dict:
        return {"dim": self.dim}

    @classmethod
    def build_from_config(cls, config: dict):
        return cls(dim=int(config.get("dim", 384)))

    def __call__(self, input: Documents) -> Embeddings:
        return self.embed_documents(list(input))

    def embed_documents(self, texts: list[str]) -> Embeddings:
        return [self.embed_text(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_text(text)

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


@lru_cache(maxsize=2)
def _load_sentence_transformer(model_name: str, device: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "The sentence-transformers backend is selected but the package is "
            "not installed. Run: pip install sentence-transformers"
        ) from exc
    return SentenceTransformer(model_name, device=device)


class SentenceTransformerEmbeddingFunction(EmbeddingFunction[Documents]):
    """Local sentence-transformers embeddings with model-specific prefixes."""

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        batch_size: int = 32,
        model=None,
    ):
        self.model_name = model_name
        self.profile = embedding_profile_for_model(model_name)
        self.device = device
        self.batch_size = batch_size
        self._model = model

    @staticmethod
    def name() -> str:
        return "sentence_transformer"

    def collection_name_suffix(self) -> str:
        return f"st-{_model_slug(self.model_name)}"

    def get_config(self) -> dict:
        return {
            "model_name": self.model_name,
            "device": self.device,
            "batch_size": self.batch_size,
        }

    @classmethod
    def build_from_config(cls, config: dict):
        return cls(
            model_name=config["model_name"],
            device=config.get("device", "cpu"),
            batch_size=int(config.get("batch_size", 32)),
        )

    def _get_model(self):
        if self._model is None:
            self._model = _load_sentence_transformer(self.model_name, self.device)
        return self._model

    def _encode(self, texts: list[str]) -> Embeddings:
        model = self._get_model()
        vectors = model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [list(map(float, vector)) for vector in vectors]

    def embed_documents(self, texts: list[str]) -> Embeddings:
        return self._encode(
            [self.profile.document_prefix + text for text in texts]
        )

    def embed_query(self, text: str) -> list[float]:
        return self._encode([self.profile.query_prefix + text])[0]

    def __call__(self, input: Documents) -> Embeddings:
        return self.embed_documents(list(input))
