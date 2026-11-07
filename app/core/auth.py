from __future__ import annotations

from fastapi import Header

from app.core.identity import AuthContext, authenticate, require_role


def require_api_key(authorization: str | None = Header(default=None)) -> AuthContext:
    """Backward-compatible authentication dependency.

    New endpoints should use `require_role(...)` so authorization is explicit.
    """
    return authenticate(authorization)


__all__ = ["AuthContext", "authenticate", "require_api_key", "require_role"]
