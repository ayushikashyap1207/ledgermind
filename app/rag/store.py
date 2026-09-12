from __future__ import annotations

from functools import lru_cache

import chromadb
from chromadb.api.models.Collection import Collection

from app.core.config import settings
from app.rag.embeddings import HashEmbeddingFunction


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


def reset_collection() -> None:
    client = get_chroma_client()
    try:
        client.delete_collection(settings.chroma_collection_name)
    except Exception:
        pass
    get_collection.cache_clear()


def upsert_chunks(ids: list[str], documents: list[str], metadatas: list[dict]) -> None:
    collection = get_collection()
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
