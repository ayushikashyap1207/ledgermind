"""
Identity, tenants, API keys, and the durable audit trail.

The financial data models live in app/sql/models.py and are intentionally
separate: the Text-to-SQL allowlist is built by introspecting that Base, so
identity tables must never leak into the generated-schema prompt.

Production note: this module uses SQLAlchemy create_all() for local/test
bootstrap. A real deployment should replace it with an Alembic migration and
apply appropriate database grants for an append-only audit table.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import secrets
from dataclasses import dataclass
from functools import lru_cache

from fastapi import Header, HTTPException, Request
from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, UniqueConstraint, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from app.core.config import settings
from app.sql.db import get_engine

ROLE_RANK = {
    "viewer": 1,
    "analyst": 2,
    "controller": 3,
    "admin": 4,
}


class IdentityBase(DeclarativeBase):
    pass


class Tenant(IdentityBase):
    __tablename__ = "tenants"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    sql_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)


class User(IdentityBase):
    __tablename__ = "users"
    __table_args__ = (UniqueConstraint("tenant_id", "email", name="uq_user_tenant_email"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id"), index=True)
    email: Mapped[str] = mapped_column(String(320))
    role: Mapped[str] = mapped_column(String(20), default="viewer")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)


class ApiKey(IdentityBase):
    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id"), index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    key_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    key_prefix: Mapped[str] = mapped_column(String(16))
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
    last_used_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)


class AuditEvent(IdentityBase):
    __tablename__ = "audit_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    user_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    role: Mapped[str | None] = mapped_column(String(20), nullable=True)
    request_id: Mapped[str] = mapped_column(String(64), index=True)
    method: Mapped[str] = mapped_column(String(10))
    path: Mapped[str] = mapped_column(String(200), index=True)
    status: Mapped[int] = mapped_column(Integer)
    duration_ms: Mapped[float] = mapped_column()
    question_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow, index=True)


@dataclass(frozen=True)
class AuthContext:
    tenant_id: str
    tenant_name: str
    user_id: int | None
    email: str | None
    role: str
    key_id: int | None
    method: str  # "db_key" | "legacy" | "disabled"
    sql_enabled: bool


@lru_cache
def _session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


def get_identity_session() -> Session:
    return _session_factory()()


def hash_api_key(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def init_identity() -> None:
    """Create identity tables and seed the default tenant.

    Idempotent. Called at application startup.
    """
    IdentityBase.metadata.create_all(get_engine())

    with get_identity_session() as session:
        tenant = session.get(Tenant, settings.default_tenant_id)
        if tenant is None:
            tenant = Tenant(
                id=settings.default_tenant_id,
                name="Default tenant",
                sql_enabled=True,
            )
            session.add(tenant)
            session.commit()


def create_tenant(tenant_id: str, name: str, sql_enabled: bool = False) -> Tenant:
    if sql_enabled and tenant_id != settings.default_tenant_id:
        raise ValueError(
            "SQL execution is only available for the default tenant. "
            "A non-default tenant needs its own isolated database connection."
        )

    with get_identity_session() as session:
        existing = session.get(Tenant, tenant_id)
        if existing is not None:
            return existing
        tenant = Tenant(id=tenant_id, name=name, sql_enabled=sql_enabled)
        session.add(tenant)
        session.commit()
        session.refresh(tenant)
        return tenant


def create_api_key(tenant_id: str, email: str, role: str) -> str:
    """Create a user if needed and return the raw API key exactly once."""
    if role not in ROLE_RANK:
        raise ValueError(f"Unknown role: {role}")

    with get_identity_session() as session:
        tenant = session.get(Tenant, tenant_id)
        if tenant is None:
            raise ValueError(f"Unknown tenant: {tenant_id}")

        user = session.scalar(
            select(User).where(User.tenant_id == tenant_id, User.email == email)
        )
        if user is None:
            user = User(tenant_id=tenant_id, email=email, role=role)
            session.add(user)
            session.flush()
        else:
            user.role = role

        token = secrets.token_urlsafe(32)
        session.add(
            ApiKey(
                tenant_id=tenant_id,
                user_id=user.id,
                key_hash=hash_api_key(token),
                key_prefix=token[:8],
                active=True,
            )
        )
        session.commit()
        return token


def revoke_api_key(key_id: int) -> bool:
    with get_identity_session() as session:
        key = session.get(ApiKey, key_id)
        if key is None:
            return False
        key.active = False
        session.commit()
        return True


def _legacy_context() -> AuthContext:
    return AuthContext(
        tenant_id=settings.default_tenant_id,
        tenant_name="Default tenant",
        user_id=None,
        email=None,
        role="admin",
        key_id=None,
        method="legacy",
        sql_enabled=True,
    )


def _disabled_context() -> AuthContext:
    return AuthContext(
        tenant_id=settings.default_tenant_id,
        tenant_name="Default tenant",
        user_id=None,
        email=None,
        role="admin",
        key_id=None,
        method="disabled",
        sql_enabled=True,
    )


def authenticate(authorization: str | None) -> AuthContext:
    """Resolve a bearer token to user/tenant/role.

    Auth is enforced when AUTH_REQUIRED=true, ENVIRONMENT=production, or a
    legacy API_KEY is configured. Otherwise local development uses a default
    admin context.
    """
    auth_required = (
        settings.auth_required
        or settings.environment == "production"
        or bool(settings.api_key)
    )

    if authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise HTTPException(
                status_code=401,
                detail="Expected Authorization: Bearer <token>",
                headers={"WWW-Authenticate": "Bearer"},
            )

        if settings.api_key and secrets.compare_digest(token, settings.api_key):
            return _legacy_context()

        key_hash = hash_api_key(token)
        with get_identity_session() as session:
            row = session.execute(
                select(ApiKey, User, Tenant)
                .join(User, ApiKey.user_id == User.id)
                .join(Tenant, ApiKey.tenant_id == Tenant.id)
                .where(ApiKey.key_hash == key_hash, ApiKey.active.is_(True))
            ).first()
            if row is None:
                raise HTTPException(
                    status_code=401,
                    detail="Unauthorized",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            key, user, tenant = row
            key.last_used_at = dt.datetime.utcnow()
            session.commit()
            return AuthContext(
                tenant_id=tenant.id,
                tenant_name=tenant.name,
                user_id=user.id,
                email=user.email,
                role=user.role,
                key_id=key.id,
                method="db_key",
                sql_enabled=tenant.sql_enabled,
            )

    # Once a database key exists, do not allow anonymous local access even
    # if AUTH_REQUIRED was forgotten. This makes key creation opt-in to auth.
    with get_identity_session() as session:
        any_active_key = session.scalar(
            select(ApiKey.id).where(ApiKey.active.is_(True)).limit(1)
        )
    if auth_required or any_active_key is not None:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return _disabled_context()


def require_role(min_role: str):
    """FastAPI dependency factory: authenticate and enforce a minimum role."""

    def dependency(
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> AuthContext:
        context = authenticate(authorization)
        if ROLE_RANK.get(context.role, 0) < ROLE_RANK[min_role]:
            raise HTTPException(status_code=403, detail="Insufficient role")
        request.state.auth = context
        return context

    return dependency


def record_audit(
    *,
    tenant_id: str | None,
    user_id: int | None,
    role: str | None,
    request_id: str,
    method: str,
    path: str,
    status: int,
    duration_ms: float,
    question_hash: str | None,
) -> None:
    with get_identity_session() as session:
        session.add(
            AuditEvent(
                tenant_id=tenant_id,
                user_id=user_id,
                role=role,
                request_id=request_id,
                method=method,
                path=path,
                status=status,
                duration_ms=duration_ms,
                question_hash=question_hash,
            )
        )
        session.commit()
