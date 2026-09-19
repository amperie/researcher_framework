"""These tests use a real PostgreSQL server and a non-owner runtime role."""
import pytest
import psycopg
from concurrent.futures import ThreadPoolExecutor
from psycopg.types.json import Jsonb
from core.platform.database import Database
from core.platform.identity import user_scope
from database.migrations import migrate
from core.platform.store import PlatformStore, ServiceError
from core.llm.usage import start_call, finish_call, usage_scope
from types import SimpleNamespace


def test_migrations_are_idempotent_and_runtime_cannot_migrate(postgres):
    migrate(postgres["owner"])
    migrate(postgres["owner"])
    db = Database(postgres["runtime"])
    db.check_ready()
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        migrate(postgres["runtime"])
    with pytest.raises(ValueError, match="non-owner"):
        Database(postgres["owner"]).check_ready()


def test_rls_applies_even_without_tenant_filters(postgres):
    db = Database(postgres["runtime"])
    with db.connect("a") as conn:
        db.ensure_user(conn, "a")
        conn.execute("""INSERT INTO researcher.platform_requests (tenant_id,request_id,session_id,input_hash,status,stage,started_at,expires_at,finished_at,result,error,user_id) VALUES
            ('a','r','s','hash','running','starting',now(),now()+interval '1 minute',NULL,NULL,NULL,'legacy-researcher-owner')""")
    with db.connect("b") as conn:
        assert conn.execute("SELECT * FROM researcher.platform_requests").fetchall() == []
        assert conn.execute("UPDATE researcher.platform_requests SET stage='stolen'").rowcount == 0
        assert conn.execute("SELECT researcher.active_request_count()").fetchone()[0] == 1
    with db.connect() as conn:
        assert conn.execute("SELECT * FROM researcher.platform_requests").fetchall() == []
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        with db.connect("b") as conn:
            conn.execute("""INSERT INTO researcher.platform_requests (tenant_id,request_id,session_id,input_hash,status,stage,started_at,expires_at,finished_at,result,error,user_id) VALUES
                ('a','forged','s','hash','running','starting',now(),now(),NULL,NULL,NULL,'legacy-researcher-owner')""")
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        with db.connect("a") as conn:
            db.ensure_user(conn, "a")
            conn.execute("ALTER TABLE researcher.platform_requests DISABLE ROW LEVEL SECURITY")


def test_database_target_is_explicit():
    with pytest.raises(ValueError, match="dedicated"):
        Database("postgresql://localhost/qc")


def test_same_tenant_user_rls_and_shared_capacity(postgres_dsn):
    store = PlatformStore(postgres_dsn)
    with user_scope('first'):
        store.claim('a', 'r', 's', 'hash', 60)
        with usage_scope('a', 'r', sink=store.record):
            event = start_call('chat', 'provider', 'model')
    with user_scope('second'):
        with store.connect('a') as conn:
            assert conn.execute('SELECT * FROM researcher.platform_requests').fetchall() == []
            assert conn.execute('SELECT * FROM researcher.llm_usage').fetchall() == []
            assert conn.execute("UPDATE researcher.platform_requests SET stage='stolen'").rowcount == 0
        with pytest.raises(ValueError):
            store.record(event)
        with pytest.raises(ServiceError) as exc:
            store.claim('a', 'r2', 's', 'hash', 60, tenant_limit=1)
        assert exc.value.status == 429
    with user_scope('first'):
        assert store.get_request('a', 'r')['stage'] == 'starting'


def test_runtime_rejects_role_administration_privilege(postgres):
    from psycopg import sql
    from psycopg.conninfo import conninfo_to_dict
    role = sql.Identifier(conninfo_to_dict(postgres["runtime"])["user"])
    try:
        with psycopg.connect(postgres["owner"]) as conn:
            conn.execute(sql.SQL("ALTER ROLE {} CREATEROLE").format(role))
        with pytest.raises(ValueError, match="administrative"):
            Database(postgres["runtime"]).check_ready()
    finally:
        with psycopg.connect(postgres["owner"]) as conn:
            conn.execute(sql.SQL("ALTER ROLE {} NOCREATEROLE").format(role))


def test_global_capacity_across_tenants_and_store_instances(postgres_dsn):
    def claim(i):
        try:
            return PlatformStore(postgres_dsn).claim(str(i), "r", "s", "hash", 60, total_limit=3)[0]
        except ServiceError as exc:
            assert exc.status == 429
            return False
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(claim, range(8))) == 3
    store = PlatformStore(postgres_dsn)
    with store.connect() as conn:
        assert conn.execute("SELECT researcher.active_request_count()").fetchone()[0] == 3
        assert conn.execute("SELECT * FROM researcher.platform_requests").fetchall() == []


def test_publication_and_stop_have_one_winner(postgres_dsn):
    store = PlatformStore(postgres_dsn)
    store.claim("a", "r", "s", "hash", 60)
    def finish(status):
        return PlatformStore(postgres_dsn).finish("a", "r", status, result={"winner": status})
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sum(pool.map(finish, ["succeeded", "stopped"])) == 1
    receipt = store.get_request("a", "r")
    assert receipt["result"]["winner"] == receipt["status"]
    assert store.claim("b", "r", "s", "hash", 60, total_limit=1)[0]


def test_usage_rls_identity_and_terminal_event_are_preserved(postgres_dsn):
    store = PlatformStore(postgres_dsn)
    with usage_scope("a", "r", sink=store.record):
        event = start_call("chat", "provider", "alias")
        completed = finish_call(event, SimpleNamespace(usage_metadata={"input_tokens": 11, "output_tokens": 7}))
    store.record(event)  # A delayed initial write must not erase final token counts.
    assert store.list_events("a") == [completed]
    with pytest.raises(ValueError, match="another request"):
        store.record({**event, "requestId": "other"})
    store.record({**completed, "tenantId": "b"})  # Same call ID is independent per tenant.
    with store.connect("b") as conn:
        assert [row[0] for row in conn.execute("SELECT tenant_id FROM researcher.llm_usage")] == ["b"]
        assert conn.execute("UPDATE researcher.llm_usage SET request_id='stolen' WHERE tenant_id='a'").rowcount == 0
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        with store.connect("b") as conn:
            conn.execute("INSERT INTO researcher.llm_usage (tenant_id,call_id,request_id,started_at,event,user_id) VALUES ('a','forged','r',now(),%s,'legacy-researcher-owner')",
                (Jsonb({**completed, "callId": "forged"}),))
    with pytest.raises(psycopg.errors.CheckViolation):
        with store.connect("b") as conn:
            conn.execute("INSERT INTO researcher.llm_usage (tenant_id,call_id,request_id,started_at,event,user_id) VALUES ('b','forged','r',now(),%s,'legacy-researcher-owner')",
                (Jsonb({**completed, "callId": "forged"}),))
    assert store.usage_summary("a", since=event["startedAt"])[0]["knownInputTokens"] == 11
    assert store.usage_summary("a", until=event["startedAt"]) == []


def test_migration_drift_is_rejected(postgres):
    with psycopg.connect(postgres["owner"]) as conn:
        conn.execute("UPDATE researcher.schema_migrations SET sha256='changed'")
    with pytest.raises(ValueError, match="Applied migration changed"):
        migrate(postgres["owner"])


def test_startup_and_readiness_require_restricted_role_and_rls(postgres, monkeypatch):
    from fastapi.testclient import TestClient
    from core.platform.app import create_app
    monkeypatch.setenv("RESEARCHER_DATABASE_URL", postgres["runtime"])
    client = TestClient(create_app(tenant_keys={"x" * 32: "a"}))
    assert client.get("/ready").status_code == 200
    monkeypatch.setenv("RESEARCHER_DATABASE_URL", postgres["owner"])
    with pytest.raises(ValueError, match="non-owner"):
        create_app(tenant_keys={"x" * 32: "a"})
    with psycopg.connect(postgres["owner"]) as conn:
        conn.execute("ALTER TABLE researcher.llm_usage NO FORCE ROW LEVEL SECURITY")
    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "storage_unavailable"
    assert "postgres" not in response.text


def test_transaction_identity_resets_and_rollback_does_not_publish(postgres_dsn):
    db = Database(postgres_dsn)
    with pytest.raises(RuntimeError, match="abort"):
        with db.connect("a") as conn:
            db.ensure_user(conn, "a")
            conn.execute("""INSERT INTO researcher.platform_requests (tenant_id,request_id,session_id,input_hash,status,stage,started_at,expires_at,finished_at,result,error,user_id) VALUES
                ('a','r','s','hash','running','starting',now(),now(),NULL,NULL,NULL,'legacy-researcher-owner')""")
            raise RuntimeError("abort")
    with db.connect() as conn:
        assert conn.execute("SELECT current_setting('app.tenant_id')").fetchone()[0] == ""
        assert conn.execute("SHOW transaction_isolation").fetchone()[0] == "read committed"
        assert conn.execute("SELECT researcher.active_request_count()").fetchone()[0] == 0
    with db.connect("a") as conn:
        assert conn.execute("SELECT * FROM researcher.platform_requests").fetchall() == []
