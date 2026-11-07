"""
Auto-seeds the local sqlite DB before the test session if it doesn't
exist yet, so `pytest` works standalone on a fresh clone without
requiring a manual `python -m ingestion.seed_synthetic_data` step first.
"""

import os

import pytest

from app.core.config import settings


@pytest.fixture(scope="session", autouse=True)
def _ensure_seeded_db():
    if settings.database_url.startswith("sqlite"):
        path = settings.database_url.split("sqlite:///", 1)[-1]
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            from ingestion.seed_synthetic_data import seed
            seed()
    yield


@pytest.fixture(scope="session", autouse=True)
def _clean_identity_test_rows():
    """Keep test-created keys/tenants/audit rows out of the local database."""
    yield

    from app.core.identity import (
        ApiKey,
        AuditEvent,
        Tenant,
        User,
        get_identity_session,
    )

    with get_identity_session() as session:
        session.query(ApiKey).delete()
        session.query(User).delete()
        session.query(Tenant).filter(Tenant.id != settings.default_tenant_id).delete()
        session.query(AuditEvent).delete()
        session.commit()
