import asyncio
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessageChunk

from core.llm.factory import LoggedChatModel
from core.llm.usage import UsageLedger, get_usage_report, usage_scope


def response(n=3):
    return SimpleNamespace(usage_metadata={"input_tokens": n, "output_tokens": 2},
                           response_metadata={"model_name": "actual-model"})


def test_concurrent_tenants_and_durable_ledger(postgres_dsn):
    ledger = UsageLedger(postgres_dsn)
    def call(tenant):
        with usage_scope(tenant, "same-request", sink=ledger.record):
            model = LoggedChatModel(SimpleNamespace(invoke=lambda _: response()), model_name="alias", provider="test")
            model.invoke([])
            report = get_usage_report()
            assert report["calls"] == 1
            return report
    with ThreadPoolExecutor(max_workers=2) as pool:
        reports = list(pool.map(call, ["a", "b"]))
    for tenant, report in zip(["a", "b"], reports):
        event = UsageLedger(ledger.dsn).list_events(tenant)[0]
        assert event["tenantId"] == tenant
        assert event["requestedModel"] == "alias"
        assert event["model"] == "actual-model"
        assert report["totalTokens"] == 5
    assert ledger.list_events("other") == []


def test_failed_and_unknown_usage_are_not_zero(postgres_dsn):
    ledger = UsageLedger(postgres_dsn)
    def fail(_):
        raise RuntimeError("provider failed")
    with usage_scope("a", "r", sink=ledger.record):
        with pytest.raises(RuntimeError):
            LoggedChatModel(SimpleNamespace(invoke=fail)).invoke([])
        LoggedChatModel(SimpleNamespace(invoke=lambda _: SimpleNamespace())).invoke([])
    events = ledger.list_events("a")
    assert [e["status"] for e in events] == ["failed", "succeeded"]
    assert all(e["promptTokens"] is None and not e["usageAvailable"] for e in events)


def test_async_tasks_do_not_share_totals():
    async def run():
        async def call(tenant):
            async def invoke(_):
                await asyncio.sleep(0)
                return response()
            with usage_scope(tenant, "r"):
                await LoggedChatModel(SimpleNamespace(ainvoke=invoke)).ainvoke([])
                return get_usage_report()
        return await asyncio.gather(call("a"), call("b"))
    assert [r["calls"] for r in asyncio.run(run())] == [1, 1]


def test_stream_accounts_once_and_close_is_interrupted():
    chunks = [AIMessageChunk(content="hello"), AIMessageChunk(content="", usage_metadata={
        "input_tokens": 3, "output_tokens": 2, "total_tokens": 5})]
    model = LoggedChatModel(SimpleNamespace(stream=lambda _: iter(chunks)))
    with usage_scope("a", "r"):
        assert len(list(model.stream([]))) == 2
        assert get_usage_report()["totalTokens"] == 5
        stream = model.stream([])
        next(stream)
        stream.close()
        events = get_usage_report()["steps"]
        assert len(events) == 2
        assert events[-1]["status"] == "interrupted"
        assert events[-1]["totalTokens"] is None


def test_scope_is_restored_and_ledger_rejects_unscoped_event(postgres_dsn):
    with usage_scope("a", "outer"):
        LoggedChatModel(SimpleNamespace(invoke=lambda _: response())).invoke([])
        with usage_scope("b", "inner"):
            assert get_usage_report()["calls"] == 0
        assert get_usage_report()["calls"] == 1
    with pytest.raises(ValueError):
        UsageLedger(postgres_dsn).record({})
