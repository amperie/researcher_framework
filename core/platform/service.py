"""Authoring request lifecycle, shared by HTTP callers and executor handlers."""
import asyncio
from contextlib import suppress
import logging
from core.llm.usage import usage_scope
from core.platform.models import TurnRequest, content_hash
from core.platform.store import PlatformStore, ServiceError
from core.platform.workflow import InvalidModelOutput, PlatformWorkflow

log = logging.getLogger(__name__)
TIMEOUTS = {"chat": 45, "research": 90, "ideate": 60, "code": 120, "build": 130, "validate": 10, "workflow": 240}


class PlatformService:
    def __init__(self, store: PlatformStore, workflow: PlatformWorkflow | None = None, *, timeouts=None, tenant_limit=2, total_limit=32, capture_model_output=False):
        self.store = store
        self.workflow = workflow or PlatformWorkflow()
        self.timeouts = dict(TIMEOUTS if timeouts is None else timeouts)
        self.tenant_limit, self.total_limit = tenant_limit, total_limit
        self.capture_model_output = capture_model_output

    async def execute(self, tenant: str, request: TurnRequest) -> dict:
        if tenant != request.tenantId:
            raise ServiceError(403, "tenant_mismatch", "Request tenant does not match the authenticated tenant")
        fingerprint = content_hash(request.model_dump_json())
        timeout = self.timeouts[request.action]
        owned, receipt = self.store.claim(tenant, request.requestId, request.sessionId, fingerprint, timeout,
            tenant_limit=self.tenant_limit, total_limit=self.total_limit)
        if not owned:
            if receipt["status"] == "succeeded":
                return receipt["result"]
            failure = ServiceError(409, "request_" + receipt["status"],
                "Inspect the existing request. A deliberate new attempt requires a new requestId.")
            failure.usage = self.store.get_request(tenant, request.requestId)["usage"]
            raise failure
        task = None
        try:
            with usage_scope(tenant, request.requestId, request.sessionId, self.store.record):
                task = asyncio.create_task(self.workflow.run(request,
                    progress=lambda stage, message=None: self.store.progress(tenant, request.requestId, stage, message)))
                async with asyncio.timeout(timeout):
                    while not task.done():
                        await asyncio.wait({task}, timeout=0.2)
                        if self.store.get_request(tenant, request.requestId)["status"] != "running":
                            raise ServiceError(409, "request_inactive", "Generation stopped or expired")
                    result = task.result().model_dump(mode="json")
                if not self.store.finish(tenant, request.requestId, "succeeded", result=result):
                    raise ServiceError(409, "request_inactive", "Generation no longer owns publication")
                return result
        except asyncio.CancelledError:
            self.store.finish(tenant, request.requestId, "interrupted", error={"code": "interrupted", "message": "Caller or service interrupted generation", "retryable": False})
            raise
        except Exception as exc:
            if isinstance(exc, ServiceError):
                failure = exc
            elif isinstance(exc, TimeoutError):
                failure = ServiceError(504, "generation_timeout", "Generation exceeded its time limit")
            elif isinstance(exc, InvalidModelOutput):
                failure = ServiceError(502, "invalid_model_output", str(exc))
                if self.capture_model_output:
                    failure.error["rawModelOutput"] = exc.raw_output
            else:
                # Do not log provider exception text: it can contain source or credentials.
                log.warning("Authoring failure tenant=%s request=%s type=%s", tenant, request.requestId, type(exc).__name__)
                failure = ServiceError(502, "generation_failed", "Generation failed; inspect usage before retrying")
            self.store.finish(tenant, request.requestId, "failed", error=failure.error)
            if task is not None and not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await task
            failure.usage = self.store.get_request(tenant, request.requestId)["usage"]
            raise failure from None
        finally:
            if task is not None:
                if not task.done():
                    task.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await task
