from __future__ import annotations

import re
from functools import lru_cache

import chromadb
from chromadb.api.models.Collection import Collection

from app.core.config import settings
from app.rag.bm25 import BM25Index
from app.rag.embeddings import HashEmbeddingFunction

_bm25_indexes: dict[str, BM25Index] = {}
_SAFE_TENANT_RE = re.compile(r"[^a-zA-Z0-9_-]")


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


def collection_name_for_tenant(tenant_id: str) -> str:
    safe_tenant = _SAFE_TENANT_RE.sub("_", tenant_id)
    return f"{settings.chroma_collection_name}_{safe_tenant}"


@lru_cache
def get_collection(tenant_id: str = "default") -> Collection:
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=collection_name_for_tenant(tenant_id),
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine", "tenant_id": tenant_id},
    )


def clear_bm25_cache(tenant_id: str = "default") -> None:
    _bm25_indexes.pop(tenant_id, None)


def get_bm25_index(tenant_id: str = "default") -> BM25Index | None:
    """Build the lexical index lazily from this tenant's chunks."""
    if tenant_id not in _bm25_indexes:
        result = get_collection(tenant_id).get(include=["documents"])
        ids = result.get("ids") or []
        documents = result.get("documents") or []
        if not ids:
            return None
        _bm25_indexes[tenant_id] = BM25Index(dict(zip(ids, documents)))
    return _bm25_indexes.get(tenant_id)


def reset_collection(tenant_id: str = "default") -> None:
    client = get_chroma_client()
    try:
        client.delete_collection(collection_name_for_tenant(tenant_id))
    except Exception:
        pass
    get_collection.cache_clear()
    clear_bm25_cache(tenant_id)


def upsert_chunks(
    ids: list[str],
    documents: list[str],
    metadatas: list[dict],
    tenant_id: str = "default",
) -> None:
    collection = get_collection(tenant_id)
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    clear_bm25_cache(tenant_id)
