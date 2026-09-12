from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from app.core.config import settings
from app.core.tracing import trace_chain
from app.rag.chunker import split_into_chunks
from app.rag.store import get_collection, get_embedding_function, reset_collection, upsert_chunks


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


def _mmr_rank(query_embedding: list[float], candidate_embeddings: list[list[float]], k: int, lam: float = 0.7) -> list[int]:
    if candidate_embeddings is None or len(candidate_embeddings) == 0:
        return []

    selected: list[int] = []
    pool = set(range(len(candidate_embeddings)))

    while pool and len(selected) < k:
        best_idx = -1
        best_score = float("-inf")

        for idx in pool:
            relevance = _cosine(query_embedding, candidate_embeddings[idx])
            diversity_penalty = 0.0
            if selected:
                diversity_penalty = max(
                    _cosine(candidate_embeddings[idx], candidate_embeddings[sel])
                    for sel in selected
                )
            score = lam * relevance - (1 - lam) * diversity_penalty
            if score > best_score:
                best_score = score
                best_idx = idx

        selected.append(best_idx)
        pool.remove(best_idx)

    return selected


@trace_chain("rag_ingest")
def ingest_documents(paths: list[str], chunk_size: int | None = None, chunk_overlap: int | None = None, reset: bool = False) -> IngestionReport:
    start = time.perf_counter()
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap

    files = _iter_supported_files([Path(p) for p in paths])
    if reset:
        reset_collection()

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
                }
            )

    if ids:
        upsert_chunks(ids=ids, documents=docs, metadatas=metas)

    return IngestionReport(
        files=len(files),
        chunks=len(ids),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        seconds=round(time.perf_counter() - start, 3),
    )


@trace_chain("rag_retrieve")
def retrieve(question: str, top_k: int = 3) -> list[RetrievedChunk]:
    collection = get_collection()
    query_res = collection.query(
        query_texts=[question],
        n_results=max(settings.retrieval_k, top_k),
        include=["documents", "metadatas", "embeddings", "distances"],
    )

    docs = query_res.get("documents", [[]])[0]
    metas = query_res.get("metadatas", [[]])[0]
    raw_embeddings = query_res.get("embeddings", [[]])[0]
    embeddings = [list(map(float, e)) for e in raw_embeddings]
    ids = query_res.get("ids", [[]])[0]

    if not docs:
        return []

    query_embedding = get_embedding_function().embed_text(question)
    ranked_idx = _mmr_rank(
        query_embedding=query_embedding,
        candidate_embeddings=embeddings,
        k=min(top_k, len(docs)),
    )

    ranked: list[RetrievedChunk] = []
    for idx in ranked_idx:
        md = metas[idx] or {}
        ranked.append(
            RetrievedChunk(
                chunk_id=ids[idx],
                source=str(md.get("source", "unknown")),
                chunk_index=int(md.get("chunk_index", -1)),
                text=docs[idx],
                score=float(_cosine(query_embedding, embeddings[idx])),
            )
        )

    return ranked
