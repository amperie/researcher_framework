import asyncio
import httpx
import pytest
from core.platform.app import create_app
from core.platform.client import ResearcherClient, ResearcherError
from tests.test_platform_service import setup, KEY_A
from tests.test_platform_workflow import request


def test_client_integrates_with_service_and_replays_without_new_llm_call(postgres_dsn):
    service, _, invoke = setup(postgres_dsn)
    app = create_app(service=service, tenant_keys={KEY_A: "a"})
    async def scenario():
        async with ResearcherClient("http://researcher", KEY_A, "a", transport=httpx.ASGITransport(app)) as client:
            result = await client.turn(request())
            assert result.content == "answer"
            assert (await client.turn(request())) == result
            assert (await client.get_request("r"))["status"] == "succeeded"
            assert (await client.usage_summary())["totals"]["knownInputTokens"] == 11
            with pytest.raises(ValueError):
                await client.turn(request(tenantId="b"))
            with pytest.raises(ValueError):
                await client.get_request("../usage/summary")
            with pytest.raises(ResearcherError) as exc:
                await client.get_request("missing")
            assert exc.value.status_code == 404
    asyncio.run(scenario())
    assert invoke.await_count == 1


def test_client_rejects_wrong_tenant_response():
    async def scenario():
        transport = httpx.MockTransport(lambda _: httpx.Response(200, json={"tenantId": "b"}))
        async with ResearcherClient("http://researcher", KEY_A, "a", transport=transport) as client:
            with pytest.raises(ResearcherError):
                await client.get_request("r")
    asyncio.run(scenario())


@pytest.mark.parametrize("response", [httpx.Response(503, text="private proxy error"),
    httpx.Response(502, json={"error": None}), httpx.Response(200, json=["bad"]), httpx.Response(200, text="bad")])
def test_client_handles_malformed_gateway_responses(response):
    async def scenario():
        async with ResearcherClient("http://researcher", KEY_A, "a", transport=httpx.MockTransport(lambda _: response)) as client:
            with pytest.raises(ResearcherError) as exc:
                await client.get_request("r")
            assert exc.value.status_code == (response.status_code if response.is_error else 502)
            assert "private" not in str(exc.value)
    asyncio.run(scenario())


@pytest.mark.parametrize("method", ["get_request", "stop"])
def test_client_rejects_receipt_for_another_request(method):
    async def scenario():
        transport = httpx.MockTransport(lambda _: httpx.Response(200, json={"tenantId": "a", "userId": "legacy-researcher-owner", "requestId": "other"}))
        async with ResearcherClient("http://researcher", KEY_A, "a", transport=transport) as client:
            with pytest.raises(ResearcherError, match="different request"):
                await getattr(client, method)("r")
    asyncio.run(scenario())


def test_checked_in_openapi_matches_service():
    import json
    from pathlib import Path
    from core.platform.export_openapi import schema
    checked_in = Path(__file__).resolve().parents[1] / "docs" / "platform-openapi.json"
    assert json.loads(checked_in.read_text(encoding="utf-8")) == schema()
