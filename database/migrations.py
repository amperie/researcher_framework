"""Forward-only, checksummed SQL migrations; independent of service and LLM code."""
import hashlib
import os
from pathlib import Path
import re
import psycopg
import sqlparse
from sqlparse import tokens
from database.target import validate_target

VERSIONS = Path(__file__).with_name("versions")
NAME = re.compile(r"([0-9]{3})_([a-z][a-z0-9_]*)\.sql")


def validate_sql(name, contents):
    # Dollar-quoted function bodies and quoted strings are single tokens, so a
    # procedure's BEGIN/END is not mistaken for top-level transaction control.
    for statement in sqlparse.parse(contents):
        words = [token.value.upper() for token in statement.flatten()
                 if not token.is_whitespace and token.ttype not in tokens.Comment]
        if not words:
            continue
        if words[0] in {"BEGIN", "START", "COMMIT", "END", "ROLLBACK", "ABORT", "SAVEPOINT", "RELEASE"} or words[:2] == ["PREPARE", "TRANSACTION"]:
            raise ValueError(f"Transaction control is not allowed in migration {name}")


def files(directory=VERSIONS):
    paths = sorted(Path(directory).glob("*.sql"))
    versions = set()
    for path in paths:
        match = NAME.fullmatch(path.name)
        if not match or match[1] in versions:
            raise ValueError(f"Invalid or duplicate migration version: {path.name}")
        versions.add(match[1])
    if not paths:
        raise ValueError("No SQL migrations found")
    return [(path.name, text, hashlib.sha256(text.encode()).hexdigest())
            for path in paths for text in [path.read_text(encoding="utf-8")]]


def history(conn):
    if not conn.execute("SELECT to_regclass('researcher.schema_migrations')").fetchone()[0]:
        return []
    return conn.execute("SELECT name,sha256 FROM researcher.schema_migrations ORDER BY name").fetchall()


def verify(local, applied):
    if [row[0] for row in applied] != [row[0] for row in local[:len(applied)]]:
        raise ValueError("Migration history differs: missing, renamed, or out-of-order migration")
    for (_, _, digest), (name, old) in zip(local, applied):
        if digest != old:
            raise ValueError(f"Applied migration changed: {name}; add a new migration")


def status(dsn, directory=VERSIONS):
    validate_target(dsn)
    local = files(directory)
    for name, contents, _ in local:
        validate_sql(name, contents)
    with psycopg.connect(dsn, connect_timeout=5) as conn:
        conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
        conn.execute("SET LOCAL ROLE qc_researcher_owner")
        conn.execute("SET LOCAL search_path TO pg_catalog")
        applied = history(conn)
        verify(local, applied)
    return [{"name": name, "status": "applied" if i < len(applied) else "pending"}
            for i, (name, _, _) in enumerate(local)]


def migrate(dsn, directory=VERSIONS):
    validate_target(dsn)
    local = files(directory)
    for name, contents, _ in local:
        validate_sql(name, contents)
    with psycopg.connect(dsn, connect_timeout=5) as conn:
        conn.execute("SET TRANSACTION ISOLATION LEVEL READ COMMITTED")
        conn.execute("SET LOCAL ROLE qc_researcher_owner")
        conn.execute("SET LOCAL search_path TO pg_catalog")
        conn.execute("SET LOCAL lock_timeout = '10s'")
        legacy_user = os.environ.get('RESEARCHER_LEGACY_USER_ID', 'legacy-researcher-owner')
        if not re.fullmatch(r'[A-Za-z0-9_.:-]{1,160}', legacy_user):
            raise ValueError('Invalid RESEARCHER_LEGACY_USER_ID')
        conn.execute("SELECT set_config('app.legacy_user_id',%s,true)", (legacy_user,))
        conn.execute("SELECT pg_advisory_xact_lock(71824001)")
        applied = history(conn)
        verify(local, applied)
        conn.execute("CREATE SCHEMA IF NOT EXISTS researcher AUTHORIZATION qc_researcher_owner")
        conn.execute("""CREATE TABLE IF NOT EXISTS researcher.schema_migrations (
            name text PRIMARY KEY, sha256 text NOT NULL, applied_at timestamptz NOT NULL DEFAULT now())""")
        pending = local[len(applied):]
        for name, contents, digest in pending:
            conn.execute(contents)
            conn.execute("INSERT INTO researcher.schema_migrations(name,sha256) VALUES (%s,%s)", (name, digest))
    return [name for name, _, _ in pending]


def new(name, directory=VERSIONS):
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        raise ValueError("Use a lowercase migration name with letters, digits and underscores")
    local = files(directory)
    version = int(local[-1][0][:3]) + 1
    if version > 999:
        raise ValueError("Migration version range exhausted")
    path = Path(directory) / f"{version:03}_{name}.sql"
    with path.open("x", encoding="utf-8") as output:
        output.write("-- Runs inside the migration transaction. Use schema-qualified names.\n")
    return path
