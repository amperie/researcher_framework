"""Short PostgreSQL transactions with transaction-local tenant identity."""
from contextlib import contextmanager
import os

import psycopg
from database.target import validate_target


class Database:
    def __init__(self, dsn=None):
        self.dsn = dsn or os.environ.get("RESEARCHER_DATABASE_URL")
        if not self.dsn:
            raise ValueError("RESEARCHER_DATABASE_URL must point to the qc-researcher PostgreSQL database")
        validate_target(self.dsn)

    @contextmanager
    def connect(self, tenant=""):
        with psycopg.connect(self.dsn, connect_timeout=5) as conn:
            conn.execute("SET TRANSACTION ISOLATION LEVEL READ COMMITTED")
            conn.execute("SET LOCAL search_path TO pg_catalog")
            conn.execute("SET LOCAL TIME ZONE 'UTC'")
            conn.execute("SET LOCAL statement_timeout = '5s'")
            conn.execute("SET LOCAL lock_timeout = '3s'")
            role = conn.execute("""SELECT EXISTS (
                SELECT 1 FROM pg_roles WHERE pg_has_role(current_user, oid, 'MEMBER')
                AND (rolsuper OR rolbypassrls OR rolcreaterole OR rolcreatedb OR rolreplication))
                OR pg_has_role(current_user, 'qc_researcher_owner', 'MEMBER')""").fetchone()
            if role[0]:
                raise ValueError("Researcher runtime requires a non-owner role without administrative privileges or privileged memberships")
            conn.execute("SELECT set_config('app.tenant_id', %s, true)", (tenant,))
            yield conn

    def check_ready(self):
        with self.connect() as conn:
            secured = conn.execute("""SELECT count(*)=2 AND bool_and(relrowsecurity AND relforcerowsecurity)
                FROM pg_class WHERE oid IN ('researcher.platform_requests'::regclass, 'researcher.llm_usage'::regclass)""").fetchone()[0]
            if not secured:
                raise psycopg.OperationalError("Researcher tables require forced row-level security")
            conn.execute("SELECT tenant_id, progress FROM researcher.platform_requests LIMIT 0")
            conn.execute("SELECT tenant_id FROM researcher.llm_usage LIMIT 0")
