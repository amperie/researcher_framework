"""Request-local accounting with an optional durable, tenant-scoped ledger."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable
from uuid import uuid4

from psycopg.types.json import Jsonb
from core.platform.database import Database
from core.platform.identity import current_user


@dataclass(frozen=True)
class UsageScope:
    tenant_id: str
    request_id: str
    session_id: str | None = None
    sink: Callable[[dict], None] | None = None


_scope: ContextVar[UsageScope | None] = ContextVar("llm_usage_scope", default=None)
_events: ContextVar[tuple[dict, ...]] = ContextVar("llm_usage_events", default=())


@contextmanager
def usage_scope(tenant_id: str, request_id: str, session_id: str | None = None, sink=None):
    if not tenant_id or not request_id:
        raise ValueError("Tenant and request identity are required")
    scope_token = _scope.set(UsageScope(tenant_id, request_id, session_id, sink))
    events_token = _events.set(())
    try:
        yield
    finally:
        _events.reset(events_token)
        _scope.reset(scope_token)


def reset_usage() -> None:
    _events.set(())


def _publish(event: dict) -> None:
    scope = _scope.get()
    if scope and scope.sink:
        scope.sink(event)
    _events.set((*[item for item in _events.get() if item["callId"] != event["callId"]], event))


def start_call(step: str, provider: str, model: str) -> dict:
    scope = _scope.get()
    event = {
        "callId": str(uuid4()), "tenantId": scope.tenant_id if scope else None,
        "userId": current_user(),
        "requestId": scope.request_id if scope else None,
        "sessionId": scope.session_id if scope else None,
        "step": step, "provider": provider, "requestedModel": model, "model": model,
        "reportedModel": None, "status": "started", "usageAvailable": False,
        "promptTokens": None, "completionTokens": None, "totalTokens": None,
        "startedAt": datetime.now(timezone.utc).isoformat(), "finishedAt": None,
    }
    _publish(event)
    return event


def finish_call(event: dict, response=None, status: str = "succeeded") -> dict:
    metadata = getattr(response, "response_metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}
    usage = getattr(response, "usage_metadata", None)
    usage = usage if isinstance(usage, dict) and usage else metadata.get("token_usage") or metadata.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    incoming = usage.get("input_tokens", usage.get("prompt_tokens"))
    outgoing = usage.get("output_tokens", usage.get("completion_tokens"))
    available = isinstance(incoming, int) and isinstance(outgoing, int) and incoming >= 0 and outgoing >= 0
    reported = metadata.get("model_name") or metadata.get("model")
    reported = reported if isinstance(reported, str) else None
    result = {
        **event, "status": status, "reportedModel": reported,
        "model": reported or event["requestedModel"], "usageAvailable": available,
        "promptTokens": incoming if available else None,
        "completionTokens": outgoing if available else None,
        "totalTokens": incoming + outgoing if available else None,
        "finishedAt": datetime.now(timezone.utc).isoformat(),
    }
    _publish(result)
    return result


def get_usage_report(events=None) -> dict:
    events = _events.get() if events is None else events
    providers = sorted({e["provider"] for e in events if e["provider"]})
    models = sorted({e["model"] for e in events if e["model"]})
    return {
        "provider": providers[0] if len(providers) == 1 else None,
        "model": models[0] if len(models) == 1 else None,
        **{key: sum(e[key] or 0 for e in events) for key in ("promptTokens", "completionTokens", "totalTokens")},
        "calls": len(events), "steps": [dict(e) for e in events],
    }


class UsageLedger(Database):
    """PostgreSQL usage ledger; each event commits independently of LLM work."""

    def record(self, event: dict) -> None:
        event = {**event, 'userId': event.get('userId', current_user())}
        if event['userId'] != current_user():
            raise ValueError('Usage user context mismatch')
        if not event.get("tenantId") or not event.get("requestId"):
            raise ValueError("Durable usage requires tenant and request identity")
        with self.connect(event["tenantId"]) as conn:
            self.ensure_user(conn, event['tenantId'])
            changed = conn.execute("""INSERT INTO researcher.llm_usage (tenant_id,call_id,request_id,started_at,event,user_id)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT(tenant_id, call_id) DO UPDATE SET event=excluded.event
                WHERE llm_usage.request_id=excluded.request_id AND llm_usage.event->>'status'='started'""",
                (event["tenantId"], event["callId"], event["requestId"], event["startedAt"], Jsonb(event), event['userId'])).rowcount
            if not changed:
                old = conn.execute("SELECT request_id FROM researcher.llm_usage WHERE tenant_id=%s AND call_id=%s",
                    (event["tenantId"], event["callId"])).fetchone()
                if not old or old[0] != event["requestId"]:
                    raise ValueError("callId already belongs to another request")

    def list_events(self, tenant_id: str, *, request_id: str | None = None, limit=100, offset=0) -> list[dict]:
        if not tenant_id:
            raise ValueError("Tenant identity is required")
        if not 1 <= limit <= 1000 or offset < 0:
            raise ValueError("Invalid pagination")
        with self.connect(tenant_id) as conn:
            rows = conn.execute("""SELECT event FROM researcher.llm_usage WHERE tenant_id=%s
                AND (%s::text IS NULL OR request_id=%s) ORDER BY started_at, call_id LIMIT %s OFFSET %s""",
                (tenant_id, request_id, request_id, limit, offset)).fetchall()
        return [row[0] for row in rows]
