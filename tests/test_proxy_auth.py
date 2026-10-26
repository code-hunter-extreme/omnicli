"""Proxy auth: every key-touching endpoint enforces the operator token."""

import pytest
from fastapi.testclient import TestClient

from omnicli import proxy


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(proxy.cfg, "data", {**proxy.cfg.data, "providers": {}})
    return TestClient(proxy.app, raise_server_exceptions=False)


def test_models_requires_token_when_set(client, monkeypatch):
    monkeypatch.setenv("OMNICLI_PROXY_TOKEN", "sekrit")
    assert client.get("/v1/models").status_code == 401
    ok = client.get("/v1/models", headers={"Authorization": "Bearer sekrit"})
    assert ok.status_code == 200


def test_health_requires_token_when_set(client, monkeypatch):
    monkeypatch.setenv("OMNICLI_PROXY_TOKEN", "sekrit")
    assert client.get("/health").status_code == 401
    ok = client.get("/health", headers={"Authorization": "Bearer sekrit"})
    assert ok.status_code == 200


def test_health_open_on_tokenless_loopback_default(client, monkeypatch):
    monkeypatch.delenv("OMNICLI_PROXY_TOKEN", raising=False)
    assert client.get("/health").status_code == 200
