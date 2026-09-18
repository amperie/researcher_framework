# Researcher database migrations

This package owns the structure of `qc-researcher`, independently of the QC API's
migrations and the researcher HTTP/LLM runtime. SQL versions live in `versions/`.
History stays in `researcher.schema_migrations`; moving the initial migration here
preserves its filename and checksum, so existing installations need no rebuild.

From the repository root, install just the migration dependencies in a dedicated
environment, or use the existing development environment:

```powershell
$env:UV_PROJECT_ENVIRONMENT = '.venv-migrations'
uv sync --frozen --only-group migrations --no-install-project
```

Set `RESEARCHER_MIGRATION_DATABASE_URL` through your secret manager or shell to a
connection for **qc-researcher** authorized to assume `qc_researcher_owner`. Do not
use this credential for the running service. The migration runner refuses other
database names except disposable `qc-researcher-test-*` databases used by tests.

```powershell
uv run --no-sync python -m database status
uv run --no-sync python -m database new add_research_index
# Edit database/versions/002_add_research_index.sql, review it, and test it.
uv run --no-sync python -m database up
uv run --no-sync python -m database check
```

- `status`: read-only list of applied and pending migrations; it never initializes tables.
- `new <name>`: creates the next numbered SQL file locally; no database credential needed.
- `up`: verifies history and applies all pending migrations in one transaction.
- `check`: exits 0 only when every migration is applied and unchanged; otherwise nonzero.

Files use `NNN_lowercase_name.sql`. Commit reviewed migrations with the application
changes that need them. Never edit, rename, remove or insert a migration before an
already-applied version. Missing files, duplicate versions, changed checksums and
out-of-order history fail verification. Concurrent runners serialize using a
PostgreSQL advisory lock. A failed migration rolls back the entire pending batch,
including its history entries. Existing applied migrations remain intact.

Write schema-qualified, transactional SQL. The runner rejects top-level transaction
control before executing any migrations. Do not put transaction-control commands,
`CREATE DATABASE`, or operations such as `CREATE INDEX CONCURRENTLY` in these files.
Migrations are trusted operator code, not user-submitted SQL. There is intentionally
no automatic down command: use a reviewed forward correction, or restore a backup
when data recovery is needed. Use backward-compatible schema changes when rolling
out multiple service versions.

Provisioning is a separate one-time administrator operation:

```powershell
# Set RESEARCHER_ADMIN_DATABASE_URL and a generated RESEARCHER_DATABASE_PASSWORD.
uv run --no-sync python -m database.provision
```

For deployment, `Dockerfile.migrations` builds a standalone non-root image containing
only this package and PostgreSQL client dependencies. Inject the migration URL at
runtime and run `up` as a release step, then `check`, before starting the service.
Neither the service nor its image automatically migrates the database. The previous
`core.platform.migrate` and `core.platform.provision` commands remain thin compatibility
wrappers around this package.

Test using `RESEARCHER_TEST_ADMIN_URL` targeting `postgres` on a test-capable server:

```powershell
.venv\Scripts\python.exe scripts/test-platform.py tests/test_database_migrations.py -q
```

Tests use disposable databases; they verify fresh creation, upgrades, rollback,
concurrency, history integrity, and read-only status against real PostgreSQL.
