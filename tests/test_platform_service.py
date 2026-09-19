import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest
from fastapi.testclient import TestClient

from core.llm.factory import LoggedChatModel
from core.platform.app import create_app
from core.platform.service import PlatformService
from core.platform.store import PlatformStore, ServiceError
from core.platform.workflow import PlatformWorkflow
from tests.test_platform_workflow import request

KEY_A, KEY_B = "a" * 32, "b" * 32


def setup(postgres_dsn, invoke=None, **kwargs):
    invoke = invoke or AsyncMock(return_value=SimpleNamespace(content='{"content":"answer"}',
        usage_metadata={"input_tokens": 11, "output_tokens": 7}, response_metadata={"model_name": "actual"}))
    engine = PlatformWorkflow(model_factory=lambda step, profile: LoggedChatModel(
        SimpleNamespace(ainvoke=invoke), step_name=step, provider="test", model_name="alias"))
    service = PlatformService(PlatformStore(postgres_dsn), engine, **kwargs)
    app = create_app(service=service, tenant_keys={KEY_A: "a", KEY_B: "b"})
    return service, TestClient(app), invoke


def headers(key=KEY_A):
    return {"Authorization": "Bearer " + key, "X-User-ID": "legacy-researcher-owner"}


def test_users_in_same_tenant_are_isolated(postgres_dsn):
    _, client, invoke = setup(postgres_dsn)
    first = {**headers(), "X-User-ID": "user-1"}
    second = {**headers(), "X-User-ID": "user-2"}
    body = request().model_dump()
    assert client.post('/v1/turns', json=body, headers={'Authorization': 'Bearer ' + KEY_A}).status_code == 422
    result = client.post('/v1/turns', json=body, headers=first)
    assert result.status_code == 200
    assert result.json()['userId'] == 'user-1'
    assert result.json()['usage']['steps'][0]['userId'] == 'user-1'
    for path in ('/v1/requests/r', '/v1/requests/r/events'):
        assert client.get(path, headers=second).status_code == 404
    assert client.post('/v1/requests/r/stop', headers=second).status_code == 404
    assert client.post('/v1/turns', json=body, headers=second).status_code == 409
    assert invoke.await_count == 1
    assert client.get('/v1/usage/events', headers=second).json()['items'] == []
    assert client.get('/v1/usage/summary', headers=second).json()['totals']['calls'] == 0
    assert client.post('/v1/turns', json={**body, 'requestId': 'r2'}, headers=second).status_code == 200
    for owner in (first, second):
        page = client.get('/v1/usage/events', headers=owner).json()
        assert len(page['items']) == 1
        assert page['userId'] == page['items'][0]['userId'] == owner['X-User-ID']
    stream = client.get('/v1/requests/r/events', headers=first)
    assert stream.status_code == 200
    assert 'user-1' in stream.text and 'user-2' not in stream.text


def test_progress_is_durable_ordered_and_fenced(postgres_dsn):
    store = PlatformStore(postgres_dsn)
    store.claim("a", "r", "s", "hash", 30)
    store.progress("a", "r", "research", "Retrieving articles")
    store.progress("a", "r", "research", "Retrieved an article")
    receipt = PlatformStore(postgres_dsn).get_request("a", "r")
    assert [event["sequence"] for event in receipt["progress"]] == [1, 2]
    assert receipt["progress"][1]["message"] == "Retrieved an article"
    with pytest.raises(ServiceError):
        store.progress("b", "r", "code", "Wrong tenant")
    store.stop("a", "r")
    with pytest.raises(ServiceError):
        store.progress("a", "r", "code", "Too late")
    assert store.get_request("a", "r")["progress"] == receipt["progress"]


def test_auth_tenant_isolation_replay_and_usage(postgres_dsn):
    service, client, invoke = setup(postgres_dsn)
    body = request().model_dump()
    assert client.post("/v1/turns", json=body).status_code == 401
    assert client.post("/v1/turns", json=body, headers=headers(KEY_B)).status_code == 403
    assert client.post("/v1/turns", json=body, headers=headers(KEY_B)).json()["tenantId"] == "b"
    first = client.post("/v1/turns", json=body, headers=headers())
    assert first.status_code == 200
    assert client.post("/v1/turns", json=body, headers=headers()).json() == first.json()
    assert invoke.await_count == 1
    assert client.post("/v1/turns", json={**body, "message": "different"}, headers=headers()).status_code == 409
    assert client.get("/v1/requests/r", headers=headers(KEY_B)).status_code == 404
    assert client.post("/v1/requests/r/stop", headers=headers(KEY_B)).status_code == 404
    assert client.get("/v1/usage/events", headers=headers(KEY_B)).json()["items"] == []
    report = client.get("/v1/usage/summary", headers=headers()).json()
    assert report["totals"] == dict(calls=1, unknownUsageCalls=0, knownInputTokens=11, knownOutputTokens=7)
    assert report["models"][0]["model"] == "actual"
    event = client.get("/v1/usage/events", headers=headers()).json()["items"][0]
    assert event["requestedModel"] == "alias" and event["tenantId"] == "a"
    body["tenantId"] = "b"
    assert client.post("/v1/turns", json=body, headers=headers(KEY_B)).status_code == 200
    reopened = PlatformStore(service.store.dsn)
    assert reopened.get_request("a", "r")["result"] == first.json()


def test_concurrent_duplicate_only_one_owner_and_capacity_is_tenant_scoped(postgres_dsn):
    store = PlatformStore(postgres_dsn)
    def claim(_):
        return store.claim("a", "same", "s", "hash", 10)[0]
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert sum(pool.map(claim, range(4))) == 1
    with pytest.raises(ServiceError) as exc:
        store.claim("a", "other", "s", "hash", 10, tenant_limit=1)
    assert exc.value.status == 429
    assert store.claim("b", "same", "s", "hash", 10, tenant_limit=1)[0]


def test_stop_fences_late_publication_and_cancels_model(postgres_dsn):
    async def scenario():
        started, cancelled = asyncio.Event(), asyncio.Event()
        async def invoke(_):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        service, _, _ = setup(postgres_dsn, invoke)
        task = asyncio.create_task(service.execute("a", request()))
        await started.wait()
        service.store.stop("a", "r")
        with pytest.raises(ServiceError):
            await task
        assert cancelled.is_set()
        assert service.store.list_events("a")[0]["status"] == "interrupted"
        assert service.store.get_request("a", "r")["status"] == "stopped"
        assert not service.store.finish("a", "r", "succeeded", result={"late": True})
    asyncio.run(scenario())


def test_timeout_failed_output_and_meter_survive(postgres_dsn):
    async def slow(_):
        await asyncio.sleep(10)
    service, client, _ = setup(postgres_dsn, slow, timeouts={"chat": 0.01})
    response = client.post("/v1/turns", json=request().model_dump(), headers=headers())
    assert response.status_code == 504
    assert service.store.get_request("a", "r")["result"] is None
    assert service.store.list_events("a")[0]["usageAvailable"] is False
    assert client.post("/v1/turns", json=request().model_dump(), headers=headers()).status_code == 409


def test_malformed_output_keeps_real_usage_and_redacts_errors(postgres_dsn):
    invoke = AsyncMock(return_value=SimpleNamespace(content="not json: private source", usage_metadata={"input_tokens": 5, "output_tokens": 3}))
    service, client, _ = setup(postgres_dsn, invoke)
    result = client.post("/v1/turns", json=request().model_dump(), headers=headers())
    assert result.status_code == 502
    assert result.json()["usage"]["completionTokens"] == 3
    assert service.store.get_request("a", "r")["usage"]["promptTokens"] == 5
    assert "private source" not in result.text
    assert service.store.usage_summary("a")[0]["knownOutputTokens"] == 3
    assert service.store.get_request("a", "r")["status"] == "failed"


def test_contract_limits_usage_filters_and_fail_closed_config(postgres_dsn):
    _, client, _ = setup(postgres_dsn)
    assert client.post("/v1/turns", content=b"x" * 512001, headers=headers()).status_code == 413
    assert client.get("/v1/usage/events?pageSize=1001", headers=headers()).status_code == 422
    assert client.get("/v1/usage/summary?since=2026-01-01", headers=headers()).status_code == 422
    for keys in ({}, {"weak": "a"}, {KEY_A: "../tenant"}):
        with pytest.raises(ValueError):
            create_app(tenant_keys=keys)
    assert client.get("/openapi.json").status_code == 200


def test_expired_request_cannot_publish_or_silently_retry(postgres_dsn):
    store = PlatformStore(postgres_dsn)
    store.claim("a", "r", "s", "hash", -10)
    assert not store.finish("a", "r", "succeeded", result={})
    assert store.get_request("a", "r")["status"] == "interrupted"
    owned, receipt = store.claim("a", "r", "s", "hash", 10)
    assert not owned and receipt["status"] == "interrupted"


def test_concurrent_service_tenants_have_separate_accounting(postgres_dsn):
    async def scenario():
        async def invoke(messages):
            context = json.loads(messages[1].content)
            await asyncio.sleep(0.01)
            count = 7 if context["message"] == "tenant a" else 19
            return SimpleNamespace(content='{"content":"answer"}', usage_metadata={"input_tokens": count, "output_tokens": 1})
        service, _, _ = setup(postgres_dsn, invoke)
        a, b = await asyncio.gather(service.execute("a", request(message="tenant a")),
            service.execute("b", request(tenantId="b", message="tenant b")))
        assert a["usage"]["promptTokens"] == 7
        assert b["usage"]["promptTokens"] == 19
        assert service.store.usage_summary("a")[0]["calls"] == 1
        assert service.store.usage_summary("b")[0]["calls"] == 1
    asyncio.run(scenario())


def test_storage_failure_does_not_invoke_provider(postgres_dsn, monkeypatch):
    import psycopg
    service, client, invoke = setup(postgres_dsn)
    def fail(_):
        raise psycopg.OperationalError("private path")
    monkeypatch.setattr(service.store, "record", fail)
    result = client.post("/v1/turns", json=request().model_dump(), headers=headers())
    assert result.status_code == 502
    invoke.assert_not_awaited()
    assert "private path" not in result.text
