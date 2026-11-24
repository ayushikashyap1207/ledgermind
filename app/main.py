from __future__ import annotations

import hashlib
import logging

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import text

from app.core.audit import AuditMiddleware
from app.core.auth import AuthContext, require_role
from app.core.config import settings
from app.core.identity import (
    create_api_key,
    create_tenant,
    init_identity,
    revoke_api_key,
)
from app.router.pipeline import answer_query
from app.rag.pipeline import ingest_documents
from app.rag.s3 import ingest_s3_documents
from app.sql.db import get_engine
from eval.full_eval import run_full_stack_eval
from eval.rag_eval import run_rag_eval
from eval.router_eval import run_router_eval
from eval.sql_eval import run_full_eval
from ingestion.seed_synthetic_data import seed

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger("ledgermind")

app = FastAPI(title=settings.app_name)
app.add_middleware(AuditMiddleware)
init_identity()


class QueryRequest(BaseModel):
    question: str = Field(min_length=3)


class IngestRequest(BaseModel):
    paths: list[str] = Field(default_factory=lambda: ["docs/rag_corpus"])
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    reset: bool = True


class S3IngestRequest(BaseModel):
    bucket: str | None = None
    prefix: str | None = None
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    reset: bool = True


class EvalRequest(BaseModel):
    suite: str = "full"  # rag | router | sql | full


class TenantCreateRequest(BaseModel):
    tenant_id: str = Field(min_length=2, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")
    name: str = Field(min_length=1, max_length=200)
    sql_enabled: bool = False


class ApiKeyCreateRequest(BaseModel):
    tenant_id: str = Field(min_length=2, max_length=64)
    email: str = Field(min_length=3, max_length=320)
    role: str = Field(pattern=r"^(viewer|analyst|controller|admin)$")


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
def query(
    req: QueryRequest,
    request: Request,
    auth: AuthContext = Depends(require_role("viewer")),
) -> dict:
    request.state.question_hash = hashlib.sha256(req.question.encode("utf-8")).hexdigest()
    ans = answer_query(
        req.question,
        tenant_id=auth.tenant_id,
        allow_sql=auth.sql_enabled,
    )
    return {
        "question": req.question,
        "tenant_id": auth.tenant_id,
        "route": ans.route,
        "route_reason": ans.route_reason,
        "route_confidence": ans.route_confidence,
        "route_method": ans.route_method,
        "sql_question": ans.sql_question,
        "rag_question": ans.rag_question,
        "decomposition_method": ans.decomposition_method,
        "answer": ans.answer,
        "answer_method": ans.answer_method,
        "claims": ans.claims,
        "insufficient_evidence": ans.insufficient_evidence,
        "conflicts": ans.conflicts,
        "citations": ans.citations,
        "sql_status": ans.sql_status,
        "rag_hits": ans.rag_hits,
    }


@app.post("/ingest")
def ingest(
    req: IngestRequest,
    auth: AuthContext = Depends(require_role("analyst")),
) -> dict:
    report = ingest_documents(
        paths=req.paths,
        chunk_size=req.chunk_size,
        chunk_overlap=req.chunk_overlap,
        reset=req.reset,
        tenant_id=auth.tenant_id,
    )
    return {**report.__dict__, "tenant_id": auth.tenant_id}


@app.post("/ingest/s3")
def ingest_s3(
    req: S3IngestRequest,
    auth: AuthContext = Depends(require_role("analyst")),
) -> dict:
    try:
        report = ingest_s3_documents(
            bucket=req.bucket,
            prefix=req.prefix,
            chunk_size=req.chunk_size,
            chunk_overlap=req.chunk_overlap,
            reset=req.reset,
            tenant_id=auth.tenant_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        **report.__dict__,
        "tenant_id": auth.tenant_id,
        "source": "s3",
    }


@app.post("/ingest/sql")
def ingest_sql(auth: AuthContext = Depends(require_role("admin"))) -> dict:
    if not auth.sql_enabled:
        raise HTTPException(
            status_code=403,
            detail="SQL execution is not enabled for this tenant.",
        )
    report = seed()
    return {"status": "ok", "tenant_id": auth.tenant_id, "seeded": report}


@app.post("/eval/run")
def run_eval(
    req: EvalRequest,
    auth: AuthContext = Depends(require_role("controller")),
) -> dict:
    if auth.tenant_id != settings.default_tenant_id:
        raise HTTPException(
            status_code=403,
            detail="Evaluation runs are restricted to the default tenant.",
        )
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


@app.post("/admin/tenants")
def admin_create_tenant(
    req: TenantCreateRequest,
    auth: AuthContext = Depends(require_role("admin")),
) -> dict:
    try:
        tenant = create_tenant(req.tenant_id, req.name, req.sql_enabled)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "tenant_id": tenant.id,
        "name": tenant.name,
        "sql_enabled": tenant.sql_enabled,
    }


@app.post("/admin/api-keys")
def admin_create_api_key(
    req: ApiKeyCreateRequest,
    auth: AuthContext = Depends(require_role("admin")),
) -> dict:
    try:
        token = create_api_key(req.tenant_id, req.email, req.role)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "api_key": token,
        "tenant_id": req.tenant_id,
        "email": req.email,
        "role": req.role,
        "note": "Store this key now; it will not be shown again.",
    }


@app.post("/admin/api-keys/{key_id}/revoke")
def admin_revoke_api_key(
    key_id: int,
    auth: AuthContext = Depends(require_role("admin")),
) -> dict:
    if not revoke_api_key(key_id):
        raise HTTPException(status_code=404, detail="API key not found")
    return {"status": "revoked", "key_id": key_id}
