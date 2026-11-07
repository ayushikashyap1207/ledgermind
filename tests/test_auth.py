import logging

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.core.auth import require_api_key
from app.core.config import settings
from app.main import app

client = TestClient(app)


def test_auth_is_disabled_when_no_key_is_configured(monkeypatch):
    monkeypatch.setattr(settings, "api_key", None)

    assert require_api_key(None) is None


def test_auth_rejects_missing_or_wrong_bearer_token(monkeypatch):
    monkeypatch.setattr(settings, "api_key", "secret")

    with pytest.raises(HTTPException) as missing:
        require_api_key(None)
    assert missing.value.status_code == 401

    with pytest.raises(HTTPException) as wrong:
        require_api_key("Bearer wrong")
    assert wrong.value.status_code == 401

    assert require_api_key("Bearer secret") is None


def test_protected_endpoint_returns_401_without_a_key(monkeypatch):
    monkeypatch.setattr(settings, "api_key", "secret")

    response = client.post("/query", json={"question": "test question"})

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_audit_middleware_logs_request_metadata(caplog):
    with caplog.at_level(logging.INFO, logger="ledgermind.audit"):
        response = client.get("/health")

    assert response.status_code == 200
    assert response.headers["x-request-id"]
    assert any(
        '"path": "/health"' in record.message and '"method": "GET"' in record.message
        for record in caplog.records
    )
