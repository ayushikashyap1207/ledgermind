from __future__ import annotations

import json
import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.core.config import settings

audit_logger = logging.getLogger("ledgermind.audit")


class AuditMiddleware(BaseHTTPMiddleware):
    """Request-level audit trail.

    Logs only request metadata, never question text, document contents, SQL
    results, or API keys. Replace the stdout logger with a write-once sink
    (for example, a dedicated audit table or SIEM) for regulated deployments.
    """

    async def dispatch(self, request: Request, call_next):
        if not settings.audit_log_enabled:
            return await call_next(request)

        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        started = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - started) * 1000, 2)

        audit_logger.info(
            json.dumps(
                {
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "duration_ms": duration_ms,
                    "client": request.client.host if request.client else None,
                    "authenticated": bool(request.headers.get("authorization")),
                }
            )
        )
        response.headers["x-request-id"] = request_id
        return response
