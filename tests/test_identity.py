import uuid

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.core.config import settings
from sqlalchemy import select

from app.core.identity import (
    ROLE_RANK,
    ApiKey,
    authenticate,
    create_api_key,
    create_tenant,
    get_identity_session,
    hash_api_key,
    init_identity,
    revoke_api_key,
)
from app.main import app
from app.rag.pipeline import retrieve
from app.rag.store import collection_name_for_tenant, reset_collection, upsert_chunks

init_identity()
client = TestClient(app)


def unique_tenant_id() -> str:
    return f"test-{uuid.uuid4().hex[:10]}"


def test_db_api_key_resolves_tenant_user_and_role():
    tenant_id = unique_tenant_id()
    create_tenant(tenant_id, "Identity Test", sql_enabled=False)
    token = create_api_key(tenant_id, "viewer@example.com", "viewer")

    context = authenticate(f"Bearer {token}")

    assert context.tenant_id == tenant_id
    assert context.email == "viewer@example.com"
    assert context.role == "viewer"
    assert context.method == "db_key"
    assert context.sql_enabled is False


def test_invalid_db_api_key_is_rejected():
    with pytest.raises(HTTPException) as exc:
        authenticate("Bearer not-a-real-token")
    assert exc.value.status_code == 401


def test_unknown_tenant_cannot_issue_a_key():
    with pytest.raises(ValueError):
        create_api_key("missing-tenant", "user@example.com", "viewer")


def test_non_default_tenant_cannot_enable_shared_sql():
    with pytest.raises(ValueError):
        create_tenant(unique_tenant_id(), "Shared SQL Attempt", sql_enabled=True)


def test_api_key_revocation_takes_effect():
    tenant_id = unique_tenant_id()
    create_tenant(tenant_id, "Revocation Test")
    token = create_api_key(tenant_id, "revoke@example.com", "viewer")

    with get_identity_session() as session:
        key_id = session.scalar(
            select(ApiKey.id).where(ApiKey.key_hash == hash_api_key(token))
        )
    assert key_id is not None
    assert revoke_api_key(key_id)

    with pytest.raises(HTTPException):
        authenticate(f"Bearer {token}")


def test_role_hierarchy_is_ordered():
    assert ROLE_RANK["viewer"] < ROLE_RANK["analyst"]
    assert ROLE_RANK["analyst"] < ROLE_RANK["controller"]
    assert ROLE_RANK["controller"] < ROLE_RANK["admin"]


def test_tenant_collections_are_separate():
    assert collection_name_for_tenant("alpha") != collection_name_for_tenant("beta")
    assert collection_name_for_tenant("alpha") == "ledgermind_docs_alpha"


def test_viewer_key_cannot_call_admin_endpoint(monkeypatch):
    monkeypatch.setattr(settings, "api_key", None)
    monkeypatch.setattr(settings, "auth_required", True)

    tenant_id = unique_tenant_id()
    create_tenant(tenant_id, "Viewer Tenant")
    token = create_api_key(tenant_id, "viewer@example.com", "viewer")

    response = client.post(
        "/admin/tenants",
        headers={"Authorization": f"Bearer {token}"},
        json={"tenant_id": unique_tenant_id(), "name": "Should Not Exist"},
    )

    assert response.status_code == 403


def test_retrieval_is_tenant_scoped():
    tenant_a = unique_tenant_id()
    tenant_b = unique_tenant_id()
    reset_collection(tenant_a)
    reset_collection(tenant_b)
    try:
        upsert_chunks(
            ids=[f"{tenant_a}::1"],
            documents=["Alpha unique reimbursement policy for tenant A."],
            metadatas=[{"source": "a.txt", "chunk_index": 0}],
            tenant_id=tenant_a,
        )
        upsert_chunks(
            ids=[f"{tenant_b}::1"],
            documents=["Beta unrelated vendor contract for tenant B."],
            metadatas=[{"source": "b.txt", "chunk_index": 0}],
            tenant_id=tenant_b,
        )

        hits_a = retrieve("Alpha reimbursement policy", top_k=3, tenant_id=tenant_a)
        hits_b = retrieve("Alpha reimbursement policy", top_k=3, tenant_id=tenant_b)

        assert hits_a
        assert any("Alpha" in hit.text for hit in hits_a)
        assert all("Alpha" not in hit.text for hit in hits_b)
    finally:
        reset_collection(tenant_a)
        reset_collection(tenant_b)


def test_admin_key_can_create_a_tenant(monkeypatch):
    monkeypatch.setattr(settings, "api_key", None)
    monkeypatch.setattr(settings, "auth_required", True)

    tenant_id = unique_tenant_id()
    create_tenant(tenant_id, "Admin Tenant")
    token = create_api_key(tenant_id, "admin@example.com", "admin")
    new_tenant_id = unique_tenant_id()

    response = client.post(
        "/admin/tenants",
        headers={"Authorization": f"Bearer {token}"},
        json={"tenant_id": new_tenant_id, "name": "Created Through API"},
    )

    assert response.status_code == 200
    assert response.json()["tenant_id"] == new_tenant_id
