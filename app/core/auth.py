from __future__ import annotations

import secrets

from fastapi import Header, HTTPException

from app.core.config import settings


def require_api_key(authorization: str | None = Header(default=None)) -> None:
    """Bearer-token guard for business endpoints.

    Auth is deliberately disabled when API_KEY is unset so local development
    and the test suite remain frictionless. Production deployments must set it.
    """
    if not settings.api_key:
        return

    expected = f"Bearer {settings.api_key}"
    if authorization is None or not secrets.compare_digest(authorization, expected):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Bearer"},
        )
