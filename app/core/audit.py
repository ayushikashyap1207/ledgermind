from __future__ import annotations

import json
import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.core.config import settings
from app.core.identity import record_audit

audit_logger = logging.getLogger("ledgermind.audit")


class AuditMiddleware(BaseHTTPMiddleware):
    """Request-level audit trail.

    Stores metadata in the audit_events table and emits a structured log line.
    Never logs question text, document contents, SQL results, or API keys.
    """

    async def dispatch(self, request: Request, call_next):
        if not settings.audit_log_enabled:
            return await call_next(request)

        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        started = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - started) * 1000, 2)

        context = getattr(request.state, "auth", None)
        question_hash = getattr(request.state, "question_hash", None)
        tenant_id = context.tenant_id if context else settings.default_tenant_id
        user_id = context.user_id if context else None
        role = context.role if context else None
        authenticated = bool(context and context.method != "disabled")

        payload = {
            "request_id": request_id,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "role": role,
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "duration_ms": duration_ms,
            "client": request.client.host if request.client else None,
            "authenticated": authenticated,
            "question_hash": question_hash,
        }
        audit_logger.info(json.dumps(payload))

        try:
            record_audit(
                tenant_id=tenant_id,
                user_id=user_id,
                role=role,
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                duration_ms=duration_ms,
                question_hash=question_hash,
            )
        except Exception:  # noqa: BLE001
            # Auditing must never take down a request; log loudly instead.
            audit_logger.exception("Failed to persist audit event")

        response.headers["x-request-id"] = request_id
        return response
