from fastapi.testclient import TestClient
from core.platform.playground import create_playground, DEFAULT_MODEL
from tests.test_platform_service import setup
from tests.test_platform_workflow import request
from types import SimpleNamespace
from unittest.mock import AsyncMock


def test_invalid_model_output_is_inspectable_and_tenant_scoped(postgres_dsn):
    raw = '```json\n{"content": "unfinished <script>alert(1)</script>"\n```'
    invoke = AsyncMock(return_value=SimpleNamespace(content=raw, usage_metadata={"input_tokens": 4, "output_tokens": 9}))
    service, _, _ = setup(postgres_dsn, invoke)
    client = TestClient(create_playground(service=service))
    a, b = client.get("/playground/config").json()["tenants"]
    headers = lambda tenant: {"Authorization": "Bearer " + tenant["key"], "X-User-ID": "legacy-researcher-owner"}
    response = client.post("/v1/turns", json=request(tenantId=a["id"]).model_dump(), headers=headers(a))
    assert response.status_code == 502
    assert response.json()["error"]["rawModelOutput"] == raw
    receipt = client.get("/v1/requests/r", headers=headers(a))
    assert receipt.json()["error"]["rawModelOutput"] == raw
    assert client.get("/v1/requests/r", headers=headers(b)).status_code == 404
    assert service.store.usage_summary(a["id"])[0]["knownOutputTokens"] == 9
    assert 'aria-label="Raw model output"' in client.get("/").text


def test_playground_uses_real_api_and_separate_tenants(postgres_dsn):
    service, production, invoke = setup(postgres_dsn)
    assert production.get("/playground/config").status_code == 404
    client = TestClient(create_playground(service=service))
    assert client.get("/").status_code == 200
    assert "Researcher lab" in client.get("/").text
    config = client.get("/playground/config").json()
    assert config["provider"] == "anthropic"
    a, b = config["tenants"]
    headers = lambda tenant: {"Authorization": "Bearer " + tenant["key"], "X-User-ID": "legacy-researcher-owner"}
    response = client.post("/v1/turns", json=request(tenantId=a["id"]).model_dump(), headers=headers(a))
    assert response.status_code == 200
    assert response.json()["usage"]["calls"] == 1
    assert client.get("/v1/requests/r", headers=headers(b)).status_code == 404
    assert client.get("/v1/usage/summary", headers=headers(b)).json()["totals"]["calls"] == 0
    assert client.get("/v1/usage/summary", headers=headers(a)).json()["totals"]["calls"] == 1
    invoke.assert_awaited_once()


def test_default_playground_backend_selects_claude(postgres_dsn, monkeypatch):
    monkeypatch.setenv("RESEARCHER_DATABASE_URL", postgres_dsn)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-not-a-real-provider-key")
    monkeypatch.setenv("RESEARCHER_LLM_PROVIDER", "openai")
    monkeypatch.delenv("RESEARCHER_LLM_MODEL", raising=False)
    app = create_playground()
    assert app.state.service.workflow.profile["llm"]["provider"] == "anthropic"
    assert app.state.service.workflow.profile["llm"]["default_model"] == DEFAULT_MODEL
    assert "test-not-a-real-provider-key" not in TestClient(app).get("/playground/config").text
