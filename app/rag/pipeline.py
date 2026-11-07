from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from app.core.config import settings
from app.core.tracing import trace_chain
from app.rag.chunker import split_into_chunks
from app.rag.hybrid import mmr_select, normalize_scores, reciprocal_rank_fusion
from app.rag.store import (
    get_bm25_index,
    get_collection,
    get_embedding_function,
    reset_collection,
    upsert_chunks,
)


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
def ingest_documents(
    paths: list[str],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    reset: bool = False,
    tenant_id: str = "default",
) -> IngestionReport:
    start = time.perf_counter()
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap

    files = _iter_supported_files([Path(p) for p in paths])
    if reset:
        reset_collection(tenant_id)

    ids: list[str] = []
    docs: list[str] = []
    metas: list[dict] = []

    for file in files:
        raw_text = _read_document(file)
        source = str(file)
        chunks = split_into_chunks(raw_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

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
                }
            )

    if ids:
        upsert_chunks(
            ids=ids, documents=docs, metadatas=metas, tenant_id=tenant_id
        )

    return IngestionReport(
        files=len(files),
        chunks=len(ids),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        seconds=round(time.perf_counter() - start, 3),
    )


@trace_chain("rag_retrieve")
def retrieve(
    question: str, top_k: int = 3, tenant_id: str = "default"
) -> list[RetrievedChunk]:
    """Hybrid retrieval: dense candidates + BM25 candidates, fused with RRF,
    reranked with lexical+dense features, then diversified with MMR."""
    collection = get_collection(tenant_id)
    candidate_n = max(settings.retrieval_k, top_k * 3)

    dense = collection.query(
        query_texts=[question],
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

    query_embedding = get_embedding_function().embed_text(question)
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
