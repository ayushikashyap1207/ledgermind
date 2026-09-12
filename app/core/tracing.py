from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

from app.core.config import settings

F = TypeVar("F", bound=Callable[..., Any])


def trace_chain(name: str) -> Callable[[F], F]:
    """Best-effort LangSmith tracing decorator with no-op fallback."""

    def decorator(func: F) -> F:
        if not settings.langsmith_tracing or not settings.langsmith_api_key:
            return func

        try:
            from langsmith.run_helpers import traceable
        except Exception:
            return func

        traced = traceable(name=name)(func)

        @wraps(func)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            return traced(*args, **kwargs)

        return wrapped  # type: ignore[return-value]

    return decorator
