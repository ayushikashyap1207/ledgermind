from __future__ import annotations

import logging

from fastapi import FastAPI
from pydantic import BaseModel, Field
from sqlalchemy import text

from app.core.config import settings
from app.router.pipeline import answer_query
from app.rag.pipeline import ingest_documents
from app.sql.db import get_engine
from eval.full_eval import run_full_stack_eval
from eval.rag_eval import run_rag_eval
from eval.router_eval import run_router_eval
from eval.sql_eval import run_full_eval
from ingestion.seed_synthetic_data import seed

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger("ledgermind")

app = FastAPI(title=settings.app_name)


class QueryRequest(BaseModel):
    question: str = Field(min_length=3)


class IngestRequest(BaseModel):
    paths: list[str] = Field(default_factory=lambda: ["docs/rag_corpus"])
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    reset: bool = True


class EvalRequest(BaseModel):
    suite: str = "full"  # rag | router | sql | full


@app.get("/health")
def health() -> dict:
    """
    Reports health of the app process AND its dependencies (DB, vector
    store) so `docker-compose up` + a health check can actually tell you
    something is wrong, rather than always returning 200.
    """
    checks = {"app": "ok"}

    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as exc:  # noqa: BLE001
        checks["database"] = f"error: {exc}"

    try:
        import chromadb

        client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        client.heartbeat()
        checks["chroma"] = "ok"
    except Exception as exc:  # noqa: BLE001
        checks["chroma"] = f"error: {exc}"

    overall_ok = all(v == "ok" for v in checks.values())
    return {"status": "ok" if overall_ok else "degraded", "checks": checks}


@app.post("/query")
def query(req: QueryRequest) -> dict:
    ans = answer_query(req.question)
    return {
        "question": req.question,
        "route": ans.route,
        "route_reason": ans.route_reason,
        "answer": ans.answer,
        "citations": ans.citations,
        "sql_status": ans.sql_status,
        "rag_hits": ans.rag_hits,
    }


@app.post("/ingest")
def ingest(req: IngestRequest) -> dict:
    report = ingest_documents(
        paths=req.paths,
        chunk_size=req.chunk_size,
        chunk_overlap=req.chunk_overlap,
        reset=req.reset,
    )
    return report.__dict__


@app.post("/ingest/sql")
def ingest_sql() -> dict:
    report = seed()
    return {"status": "ok", "seeded": report}


@app.post("/eval/run")
def run_eval(req: EvalRequest) -> dict:
    suite = req.suite.lower().strip()
    if suite == "rag":
        return run_rag_eval()
    if suite == "router":
        return run_router_eval()
    if suite == "sql":
        return run_full_eval()
    if suite == "full":
        return run_full_stack_eval()
    return {"status": "error", "message": f"Unknown suite: {req.suite}"}
