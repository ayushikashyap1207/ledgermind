from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader

from app.core.config import settings
from app.core.tracing import trace_chain
from app.rag.chunker import split_into_chunks
from app.rag.hybrid import mmr_select, normalize_scores, reciprocal_rank_fusion
from app.rag.reranker import combine_rerank_scores
from app.rag.store import (
    get_bm25_index,
    get_collection,
    get_embedding_function,
    get_reranker,
    reset_collection,
    upsert_chunks,
)

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    chunk_id: str
    source: str
    chunk_index: int
    text: str
    score: float


@dataclass
class IngestionReport:
    files: int
    chunks: int
    chunk_size: int
    chunk_overlap: int
    seconds: float


@dataclass
class DocumentInput:
    source: str
    text: str
    metadata: dict = field(default_factory=dict)


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_pdf_file(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = [(p.extract_text() or "") for p in reader.pages]
    return "\n\n".join(pages)


def _read_document(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return _read_pdf_file(path)
    return _read_text_file(path)


def _iter_supported_files(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for path in paths:
        if path.is_file() and path.suffix.lower() in {".txt", ".md", ".pdf"}:
            out.append(path)
        elif path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file() and child.suffix.lower() in {".txt", ".md", ".pdf"}:
                    out.append(child)
    return out


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


@trace_chain("rag_ingest")
def ingest_text_records(
    records: list[DocumentInput],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    reset: bool = False,
    tenant_id: str = "default",
) -> IngestionReport:
    """Chunk and upsert already-extracted documents.

    Shared by local-file ingestion and the optional S3 loader.
    """
    start = time.perf_counter()
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap

    if reset:
        reset_collection(tenant_id)

    ids: list[str] = []
    docs: list[str] = []
    metas: list[dict] = []

    for record in records:
        source = record.source
        chunks = split_into_chunks(
            record.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        for idx, chunk in enumerate(chunks):
            chunk_id = f"{source}::chunk-{idx}"
            ids.append(chunk_id)
            docs.append(chunk.text)
            metas.append(
                {
                    "source": source,
                    "chunk_index": idx,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "tenant_id": tenant_id,
                    **record.metadata,
                }
            )

    if ids:
        upsert_chunks(
            ids=ids, documents=docs, metadatas=metas, tenant_id=tenant_id
        )

    return IngestionReport(
        files=len(records),
        chunks=len(ids),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        seconds=round(time.perf_counter() - start, 3),
    )


def ingest_documents(
    paths: list[str],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    reset: bool = False,
    tenant_id: str = "default",
) -> IngestionReport:
    files = _iter_supported_files([Path(p) for p in paths])
    records = [
        DocumentInput(source=str(file), text=_read_document(file)) for file in files
    ]
    return ingest_text_records(
        records,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        reset=reset,
        tenant_id=tenant_id,
    )


@trace_chain("rag_retrieve")
def retrieve(
    question: str, top_k: int = 3, tenant_id: str = "default"
) -> list[RetrievedChunk]:
    """Hybrid retrieval: dense candidates + BM25 candidates, fused with RRF,
    reranked with lexical+dense features, then diversified with MMR."""
    collection = get_collection(tenant_id)
    candidate_n = max(settings.retrieval_k, top_k * 3)
    embedding_function = get_embedding_function()
    query_embedding = embedding_function.embed_query(question)

    dense = collection.query(
        query_embeddings=[query_embedding],
        n_results=candidate_n,
        include=["documents", "metadatas", "distances"],
    )
    dense_ids = list((dense.get("ids") or [[]])[0])

    bm25 = get_bm25_index(tenant_id)
    bm25_scores = bm25.score_all(question) if bm25 is not None else {}
    bm25_ranked_ids = [
        doc_id
        for doc_id, _ in sorted(
            bm25_scores.items(), key=lambda item: (-item[1], item[0])
        )
    ][:candidate_n]

    if not dense_ids and not bm25_ranked_ids:
        return []

    candidate_ids = list(dict.fromkeys([*dense_ids, *bm25_ranked_ids]))
    fetched = collection.get(
        ids=candidate_ids,
        include=["documents", "metadatas", "embeddings"],
    )
    ids = fetched.get("ids") or []
    docs = fetched.get("documents") or []
    metas = fetched.get("metadatas") or []
    raw_embeddings = fetched.get("embeddings")
    if raw_embeddings is None:
        raw_embeddings = []

    doc_by_id = dict(zip(ids, docs))
    meta_by_id = dict(zip(ids, metas))
    emb_by_id = {
        doc_id: [float(value) for value in embedding]
        for doc_id, embedding in zip(ids, raw_embeddings)
    }
    ids = [doc_id for doc_id in ids if doc_id in emb_by_id]
    if not ids:
        return []

    dense_scores = {
        doc_id: _cosine(query_embedding, embedding)
        for doc_id, embedding in emb_by_id.items()
    }

    rrf_scores = reciprocal_rank_fusion([dense_ids, bm25_ranked_ids])
    rrf_norm = normalize_scores({doc_id: rrf_scores.get(doc_id, 0.0) for doc_id in ids})
    bm25_norm = normalize_scores({doc_id: bm25_scores.get(doc_id, 0.0) for doc_id in ids})
    dense_norm = normalize_scores(dense_scores)

    relevance = {
        doc_id: (
            0.4 * rrf_norm.get(doc_id, 0.0)
            + 0.4 * bm25_norm.get(doc_id, 0.0)
            + 0.2 * dense_norm.get(doc_id, 0.0)
        )
        for doc_id in ids
    }

    reranker = get_reranker()
    if reranker is not None:
        try:
            cross_scores = reranker.score(
                question, [doc_by_id[doc_id] for doc_id in ids]
            )
            if len(cross_scores) != len(ids):
                raise RuntimeError(
                    f"reranker returned {len(cross_scores)} scores for {len(ids)} candidates"
                )
            relevance = combine_rerank_scores(
                base_scores=relevance,
                cross_scores=dict(zip(ids, cross_scores)),
            )
            selected = sorted(
                ids, key=lambda doc_id: (-relevance[doc_id], doc_id)
            )[: min(top_k, len(ids))]
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Cross-encoder rerank failed; falling back to hybrid MMR: %s", exc
            )
            selected = mmr_select(
                candidate_ids=ids,
                relevance=relevance,
                embeddings=emb_by_id,
                top_k=min(top_k, len(ids)),
                lambda_=0.85,
            )
    else:
        selected = mmr_select(
            candidate_ids=ids,
            relevance=relevance,
            embeddings=emb_by_id,
            top_k=min(top_k, len(ids)),
            lambda_=0.85,
        )

    ranked: list[RetrievedChunk] = []
    for doc_id in selected:
        md = meta_by_id.get(doc_id) or {}
        ranked.append(
            RetrievedChunk(
                chunk_id=doc_id,
                source=str(md.get("source", "unknown")),
                chunk_index=int(md.get("chunk_index", -1)),
                text=doc_by_id.get(doc_id, ""),
                score=float(relevance.get(doc_id, 0.0)),
            )
        )

    return ranked
