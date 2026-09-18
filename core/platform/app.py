"""Internal HTTP transport. QC authenticates users and sends tenant-bound credentials."""
from datetime import datetime, timezone
import hashlib
import hmac
import json
import os
import psycopg
from typing import Annotated

from fastapi import Depends, FastAPI, Header, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import TypeAdapter

from core.platform.models import Identity, TurnRequest, TurnResult, RequestReceipt, UsagePage, UsageSummary, ErrorEnvelope
from core.platform.service import PlatformService
from core.platform.store import PlatformStore, ServiceError
from core.platform.progress import stream_progress


class BodyLimit:
    def __init__(self, app, limit=512000):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > self.limit:
                return await JSONResponse(status_code=413, content={"tenantId": None, "diagnostics": [], "error": {
                    "code": "request_too_large", "message": "Request exceeds 512000 bytes", "retryable": False}})(scope, receive, send)
            if not message.get("more_body"):
                break
        delivered = False
        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()
        await self.app(scope, replay, send)


def create_app(*, service: PlatformService | None = None, tenant_keys: dict[str, str] | None = None) -> FastAPI:
    keys = tenant_keys if tenant_keys is not None else json.loads(os.environ.get("RESEARCHER_TENANT_KEYS", "{}"))
    if not isinstance(keys, dict) or not keys:
        raise ValueError("RESEARCHER_TENANT_KEYS must map nonempty service credentials to tenants")
    keyring = []
    for key, tenant in keys.items():
        if not isinstance(key, str) or len(key) < 32:
            raise ValueError("Each tenant credential must contain at least 32 characters")
        TypeAdapter(Identity).validate_python(tenant)
        keyring.append((hashlib.sha256(key.encode()).digest(), tenant))
    if service is None:
        store = PlatformStore()
        store.check_ready()
        service = PlatformService(store)
    app = FastAPI(title="QC Researcher Service", version="1.0.0",
        responses={status: {"model": ErrorEnvelope} for status in (401, 403, 404, 409, 413, 422, 429, 502, 503, 504)}, description=(
        "Internal tenant-bound authoring API. Turns execute inline; no background queue is created. "
        "QC owns sessions, draft application, runtime validation and immutable version publication. "
        "Longer workflows may be invoked by QC's executor using the same contract."))
    app.add_middleware(BodyLimit)
    app.state.service = service
    bearer = HTTPBearer(auto_error=False)

    def tenant(request: Request, credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)]):
        if credentials is None:
            raise ServiceError(401, "unauthorized", "A tenant-bound service credential is required")
        digest = hashlib.sha256(credentials.credentials.encode()).digest()
        matched = None
        for candidate, identity in keyring:
            if hmac.compare_digest(candidate, digest):
                matched = identity
        if matched is None:
            raise ServiceError(401, "unauthorized", "Invalid service credential")
        request.state.authorized_tenant = matched
        return matched

    @app.exception_handler(ServiceError)
    async def service_error(request: Request, exc: ServiceError):
        return JSONResponse(status_code=exc.status, content={"tenantId": getattr(request.state, "authorized_tenant", None),
            "error": exc.error, "usage": getattr(exc, "usage", None), "diagnostics": []},
            headers={"WWW-Authenticate": "Bearer"} if exc.status == 401 else None)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        return JSONResponse(status_code=422, content={"tenantId": getattr(request.state, "authorized_tenant", None), "error": {"code": "invalid_request",
            "message": "Request does not satisfy the service contract", "retryable": False},
            "diagnostics": [{"path": list(e["loc"]), "message": e["msg"]} for e in exc.errors()]})

    @app.exception_handler(psycopg.Error)
    async def storage_error(request: Request, exc: psycopg.Error):
        return JSONResponse(status_code=503, content={"tenantId": getattr(request.state, "authorized_tenant", None), "diagnostics": [], "error": {"code": "storage_unavailable",
            "message": "Durable storage is unavailable; inspect the request before retrying", "retryable": True}})

    @app.get("/health")
    def health():
        return {"status": "ok", "schemaVersion": "1"}

    @app.get("/ready")
    def ready():
        service.store.check_ready()
        return {"status": "ready", "schemaVersion": "1"}

    @app.post("/v1/turns", response_model=TurnResult)
    async def turn(body: TurnRequest, identity: str = Depends(tenant)):
        return await service.execute(identity, body)

    @app.get("/v1/requests/{request_id}", response_model=RequestReceipt)
    def get_request(request_id: Identity, identity: str = Depends(tenant)):
        return service.store.get_request(identity, request_id)

    @app.post("/v1/requests/{request_id}/stop", response_model=RequestReceipt)
    def stop(request_id: Identity, identity: str = Depends(tenant)):
        return service.store.stop(identity, request_id)

    @app.get("/v1/requests/{request_id}/events", response_class=StreamingResponse,
        responses={200: {"content": {"text/event-stream": {"schema": {"type": "string"}}}}})
    def progress_events(request_id: Identity, identity: str = Depends(tenant),
                        after: int = Query(0, ge=0), last_event_id: int | None = Header(None, ge=0)):
        """Replay progress after a sequence, then stream until terminal status. Never starts work."""
        receipt = service.store.get_request(identity, request_id)
        return StreamingResponse(stream_progress(service.store, identity, request_id, receipt,
            max(after, last_event_id or 0)), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache, no-store", "X-Accel-Buffering": "no"})

    @app.get("/v1/usage/events", response_model=UsagePage)
    def events(identity: str = Depends(tenant), requestId: Identity | None = None,
               page: int = Query(1, ge=1), pageSize: int = Query(100, ge=1, le=500)):
        rows = service.store.list_events(identity, request_id=requestId, limit=pageSize + 1, offset=(page - 1) * pageSize)
        return {"tenantId": identity, "items": rows[:pageSize],
                "pagination": {"page": page, "pageSize": pageSize, "hasNextPage": len(rows) > pageSize}}

    @app.get("/v1/usage/summary", response_model=UsageSummary)
    def summary(identity: str = Depends(tenant), since: datetime | None = None, until: datetime | None = None):
        if any(value is not None and value.tzinfo is None for value in (since, until)):
            raise ServiceError(422, "invalid_time_range", "Usage timestamps must include a timezone")
        if since and until and since >= until:
            raise ServiceError(422, "invalid_time_range", "since must precede until")
        fmt = lambda value: value.astimezone(timezone.utc).isoformat() if value else None
        rows = service.store.usage_summary(identity, fmt(since), fmt(until))
        return {"tenantId": identity, "models": rows, "totals": {
            key: sum(row[key] for row in rows) for key in ("calls", "unknownUsageCalls", "knownInputTokens", "knownOutputTokens")}}

    return app
