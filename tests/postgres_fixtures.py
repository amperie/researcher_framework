"""Disposable databases only; never use the application's configured database."""
import os
from uuid import uuid4
import pytest
import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo
from database.migrations import migrate


@pytest.fixture(scope="session")
def postgres_cluster():
    admin = os.environ.get("RESEARCHER_TEST_ADMIN_URL")
    if not admin:
        pytest.skip("Set RESEARCHER_TEST_ADMIN_URL to an isolated PostgreSQL test cluster")
    password, login = uuid4().hex, "researcher_test_" + uuid4().hex
    with psycopg.connect(admin, autocommit=True) as conn:
        for role in ("qc_researcher_owner", "qc_researcher"):
            if not conn.execute("SELECT 1 FROM pg_roles WHERE rolname=%s", (role,)).fetchone():
                conn.execute(sql.SQL("CREATE ROLE {} NOLOGIN").format(sql.Identifier(role)))
        conn.execute(sql.SQL("CREATE ROLE {} LOGIN PASSWORD {} IN ROLE qc_researcher").format(sql.Identifier(login), sql.Literal(password)))
    try:
        yield admin, make_conninfo(admin, user=login, password=password)
    finally:
        with psycopg.connect(admin, autocommit=True) as conn:
            conn.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(login)))


@pytest.fixture
def postgres(postgres_cluster):
    admin, runtime = postgres_cluster
    name = "qc-researcher-test-" + uuid4().hex
    with psycopg.connect(admin, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE DATABASE {} OWNER qc_researcher_owner").format(sql.Identifier(name)))
    owner = make_conninfo(admin, dbname=name)
    try:
        migrate(owner)
        yield {"owner": owner, "runtime": make_conninfo(runtime, dbname=name)}
    finally:
        with psycopg.connect(admin, autocommit=True) as conn:
            conn.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))


@pytest.fixture
def postgres_dsn(postgres):
    return postgres["runtime"]
