"""Tenant-scoped PostgreSQL receipts, cancellation fences, and usage summaries."""
from psycopg.types.json import Jsonb
from core.llm.usage import UsageLedger, get_usage_report


class ServiceError(Exception):
    def __init__(self, status: int, code: str, message: str, retryable: bool = False):
        super().__init__(message)
        self.status = status
        self.error = {"code": code, "message": message, "retryable": retryable}


class PlatformStore(UsageLedger):
    @staticmethod
    def _expire(conn, tenant):
        error = {"code": "outcome_unknown", "message": "The request expired without a published result. Inspect usage before deliberately retrying with a new requestId.", "retryable": False}
        conn.execute("""UPDATE researcher.platform_requests SET status='interrupted', finished_at=clock_timestamp(), error=%s
            WHERE tenant_id=%s AND status='running' AND expires_at<=clock_timestamp()""", (Jsonb(error), tenant))

    @staticmethod
    def _receipt(row):
        stamp = lambda value: value.isoformat() if value is not None else None
        return dict(tenantId=row[0], requestId=row[1], sessionId=row[2], status=row[4], stage=row[5],
            startedAt=stamp(row[6]), expiresAt=stamp(row[7]), finishedAt=stamp(row[8]),
            result=row[9], error=row[10], progress=row[11])

    def claim(self, tenant, request_id, session_id, fingerprint, timeout, *, tenant_limit=2, total_limit=32):
        if not tenant or not request_id:
            raise ValueError("Tenant and request identity required")
        with self.connect(tenant) as conn:
            # Serialize admission across replicas, never the LLM work itself.
            conn.execute("SELECT pg_advisory_xact_lock(71824002)")
            self._expire(conn, tenant)
            row = conn.execute("SELECT * FROM researcher.platform_requests WHERE tenant_id=%s AND request_id=%s", (tenant, request_id)).fetchone()
            if row:
                if row[3] != fingerprint:
                    raise ServiceError(409, "idempotency_conflict", "requestId already belongs to different input")
                return False, self._receipt(row)
            active = conn.execute("""SELECT count(*) FROM researcher.platform_requests
                WHERE tenant_id=%s AND status='running' AND expires_at>clock_timestamp()""", (tenant,)).fetchone()[0]
            total = conn.execute("SELECT researcher.active_request_count()").fetchone()[0]
            if active >= tenant_limit or total >= total_limit:
                raise ServiceError(429, "capacity_exceeded", "Researcher capacity is temporarily exhausted", True)
            row = conn.execute("""INSERT INTO researcher.platform_requests
                (tenant_id, request_id, session_id, input_hash, status, stage, started_at, expires_at)
                VALUES (%s,%s,%s,%s,'running','starting',clock_timestamp(),clock_timestamp() + %s * interval '1 second')
                RETURNING *""", (tenant, request_id, session_id, fingerprint, timeout + 5)).fetchone()
        return True, self._receipt(row)

    def get_request(self, tenant, request_id):
        with self.connect(tenant) as conn:
            self._expire(conn, tenant)
            row = conn.execute("SELECT * FROM researcher.platform_requests WHERE tenant_id=%s AND request_id=%s", (tenant, request_id)).fetchone()
            events = conn.execute("SELECT event FROM researcher.llm_usage WHERE tenant_id=%s AND request_id=%s ORDER BY started_at,call_id", (tenant, request_id)).fetchall() if row else []
        if row is None:
            raise ServiceError(404, "not_found", "Request not found")
        return {**self._receipt(row), "usage": get_usage_report([event[0] for event in events])}

    def progress(self, tenant, request_id, stage, message=None):
        with self.connect(tenant) as conn:
            changed = conn.execute("""UPDATE researcher.platform_requests SET stage=%s,
                progress=progress || jsonb_build_array(jsonb_build_object(
                    'sequence', jsonb_array_length(progress)+1, 'stage', %s::text,
                    'message', %s::text, 'at', clock_timestamp()))
                WHERE tenant_id=%s AND request_id=%s AND status='running' AND expires_at>clock_timestamp()""",
                (stage, stage, message or stage, tenant, request_id)).rowcount
        if not changed:
            raise ServiceError(409, "request_inactive", "Request no longer owns publication")

    def finish(self, tenant, request_id, status, *, result=None, error=None):
        with self.connect(tenant) as conn:
            return bool(conn.execute("""UPDATE researcher.platform_requests SET status=%s, result=%s, error=%s, finished_at=clock_timestamp()
                WHERE tenant_id=%s AND request_id=%s AND status='running' AND expires_at>clock_timestamp()""",
                (status, Jsonb(result) if result is not None else None, Jsonb(error) if error else None, tenant, request_id)).rowcount)

    def stop(self, tenant, request_id):
        self.get_request(tenant, request_id)
        self.finish(tenant, request_id, "stopped", error={"code": "stopped", "message": "Generation was stopped", "retryable": False})
        return self.get_request(tenant, request_id)

    def usage_summary(self, tenant, since=None, until=None):
        with self.connect(tenant) as conn:
            rows = conn.execute("""SELECT event->>'provider', event->>'model', count(*),
                count(*) FILTER (WHERE NOT (event->>'usageAvailable')::boolean),
                coalesce(sum((event->>'promptTokens')::bigint), 0),
                coalesce(sum((event->>'completionTokens')::bigint), 0)
                FROM researcher.llm_usage WHERE tenant_id=%s
                  AND (%s::timestamptz IS NULL OR started_at>=%s::timestamptz)
                  AND (%s::timestamptz IS NULL OR started_at<%s::timestamptz)
                GROUP BY event->>'provider', event->>'model'
                ORDER BY event->>'provider', event->>'model'""",
                (tenant, since, since, until, until)).fetchall()
        return [{"provider": r[0], "model": r[1], "calls": r[2], "unknownUsageCalls": r[3],
                 "knownInputTokens": int(r[4]), "knownOutputTokens": int(r[5])} for r in rows]
