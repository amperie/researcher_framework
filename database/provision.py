"""One-time database/role provisioning, separate from schema migrations."""
import os
import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo


def provision(admin_dsn, password):
    if len(password) < 32:
        raise ValueError("Use a generated runtime password of at least 32 characters")
    with psycopg.connect(make_conninfo(admin_dsn, dbname="postgres"), autocommit=True, connect_timeout=5) as conn:
        if not conn.execute("SELECT 1 FROM pg_roles WHERE rolname='qc_researcher_owner'").fetchone():
            conn.execute("CREATE ROLE qc_researcher_owner NOLOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE")
        if not conn.execute("SELECT 1 FROM pg_roles WHERE rolname='qc_researcher'").fetchone():
            conn.execute(sql.SQL("CREATE ROLE qc_researcher LOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE PASSWORD {}").format(sql.Literal(password)))
        if not conn.execute("SELECT 1 FROM pg_database WHERE datname='qc-researcher'").fetchone():
            conn.execute('CREATE DATABASE "qc-researcher" OWNER qc_researcher_owner')
        conn.execute('REVOKE ALL ON DATABASE "qc-researcher" FROM PUBLIC')
        conn.execute('GRANT CONNECT ON DATABASE "qc-researcher" TO qc_researcher')
    with psycopg.connect(make_conninfo(admin_dsn, dbname="qc-researcher"), connect_timeout=5) as conn:
        conn.execute("REVOKE CREATE ON SCHEMA public FROM PUBLIC")


if __name__ == "__main__":
    provision(os.environ["RESEARCHER_ADMIN_DATABASE_URL"], os.environ["RESEARCHER_DATABASE_PASSWORD"])
