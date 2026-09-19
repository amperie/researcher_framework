"""Small async client for QC. No implicit retries of billable requests."""
import httpx
from core.platform.models import TurnRequest, TurnResult
from core.platform.identity import LEGACY_USER


class ResearcherError(RuntimeError):
    def __init__(self, status_code, error):
        super().__init__(error.get("message", "Researcher request failed"))
        self.status_code, self.error = status_code, error


class ResearcherClient:
    def __init__(self, base_url: str, credential: str, tenant_id: str, *, user_id=LEGACY_USER, transport=None):
        self.tenant_id = tenant_id
        self.user_id = self._id(user_id)
        self.http = httpx.AsyncClient(base_url=base_url.rstrip("/"), transport=transport,
            headers={"Authorization": "Bearer " + credential, 'X-User-ID': self.user_id}, timeout=httpx.Timeout(270, connect=5), trust_env=False)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.http.aclose()

    async def _request(self, method, path, **kwargs):
        # A transport timeout has an uncertain outcome: callers inspect requestId.
        response = await self.http.request(method, path, **kwargs)
        fallback = {"code": "invalid_response", "message": "Researcher returned an invalid response"}
        try:
            data = response.json()
        except ValueError:
            raise ResearcherError(response.status_code if response.is_error else 502, fallback) from None
        if not isinstance(data, dict):
            raise ResearcherError(response.status_code if response.is_error else 502, fallback)
        if response.is_error:
            error = data.get("error")
            raise ResearcherError(response.status_code, error if isinstance(error, dict) else fallback)
        if data.get("tenantId") != self.tenant_id:
            raise ResearcherError(502, {"code": "tenant_mismatch", "message": "Researcher returned a different tenant"})
        if data.get('userId') != self.user_id:
            raise ResearcherError(502, {'code': 'user_mismatch', 'message': 'Researcher returned a different user'})
        return data

    async def turn(self, request: TurnRequest) -> TurnResult:
        if request.tenantId != self.tenant_id:
            raise ValueError("Request tenant does not match this client")
        data = await self._request("POST", "/v1/turns", json=request.model_dump(mode="json"))
        result = TurnResult.model_validate(data)
        if result.requestId != request.requestId or result.sessionId != request.sessionId:
            raise ResearcherError(502, {"code": "context_mismatch", "message": "Researcher returned different request context"})
        return result

    async def get_request(self, request_id: str):
        return await self._receipt("GET", request_id)

    async def stop(self, request_id: str):
        return await self._receipt("POST", request_id, "/stop")

    async def _receipt(self, method, request_id, suffix=""):
        data = await self._request(method, "/v1/requests/" + self._id(request_id) + suffix)
        if data.get("requestId") != request_id:
            raise ResearcherError(502, {"code": "context_mismatch", "message": "Researcher returned a different request"})
        return data

    async def usage_summary(self, *, since=None, until=None):
        return await self._request("GET", "/v1/usage/summary", params={k: v for k, v in {"since": since, "until": until}.items() if v is not None})

    @staticmethod
    def _id(value):
        from pydantic import TypeAdapter
        from core.platform.models import Identity
        return TypeAdapter(Identity).validate_python(value)
