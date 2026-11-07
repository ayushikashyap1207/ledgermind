from __future__ import annotations

from functools import lru_cache

import chromadb
from chromadb.api.models.Collection import Collection

from app.core.config import settings
from app.rag.bm25 import BM25Index
from app.rag.embeddings import HashEmbeddingFunction

_bm25_index: BM25Index | None = None


@lru_cache
def get_embedding_function() -> HashEmbeddingFunction:
    return HashEmbeddingFunction()


@lru_cache
def get_chroma_client() -> chromadb.ClientAPI:
    if settings.chroma_use_http:
        return chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)

    # Use local persistent client by default; this works even when Docker
    # is unavailable but keeps the same collection/query interface.
    return chromadb.PersistentClient(path=settings.chroma_persist_dir)


@lru_cache
def get_collection() -> Collection:
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=settings.chroma_collection_name,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )


def clear_bm25_cache() -> None:
    global _bm25_index
    _bm25_index = None


def get_bm25_index() -> BM25Index | None:
    """Build the lexical index lazily from the chunks currently in Chroma."""
    global _bm25_index
    if _bm25_index is None:
        result = get_collection().get(include=["documents"])
        ids = result.get("ids") or []
        documents = result.get("documents") or []
        if not ids:
            return None
        _bm25_index = BM25Index(dict(zip(ids, documents)))
    return _bm25_index


def reset_collection() -> None:
    client = get_chroma_client()
    try:
        client.delete_collection(settings.chroma_collection_name)
    except Exception:
        pass
    get_collection.cache_clear()
    clear_bm25_cache()


def upsert_chunks(ids: list[str], documents: list[str], metadatas: list[dict]) -> None:
    collection = get_collection()
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    clear_bm25_cache()
