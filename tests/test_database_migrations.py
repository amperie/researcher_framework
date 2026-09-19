from concurrent.futures import ThreadPoolExecutor
import shutil
import psycopg
import pytest
from database.migrations import VERSIONS, files, migrate, new, status, validate_sql
from database.__main__ import main


@pytest.fixture
def versions(tmp_path):
    shutil.copytree(VERSIONS, tmp_path, dirs_exist_ok=True)
    return tmp_path


def test_new_validates_names_and_versions(versions):
    assert new("add_index", versions).name == "004_add_index.sql"
    with pytest.raises(ValueError):
        new("../bad", versions)
    (versions / "004_duplicate.sql").write_text("SELECT 1;")
    with pytest.raises(ValueError, match="duplicate"):
        files(versions)


@pytest.mark.parametrize("control", ["COMMIT", "END WORK", "ROLLBACK", "ABORT", "BEGIN", "START TRANSACTION", "SAVEPOINT a", "RELEASE SAVEPOINT a", "PREPARE TRANSACTION 'x'"])
def test_migrations_reject_transaction_control(control):
    with pytest.raises(ValueError, match="Transaction control"):
        validate_sql("002_bad.sql", "SELECT ';COMMIT'; /* comment */ " + control + ";")


def test_transaction_words_in_function_bodies_are_allowed():
    validate_sql("002_valid.sql", """-- COMMIT;
        CREATE FUNCTION researcher.example() RETURNS void LANGUAGE plpgsql AS $$
        BEGIN RAISE NOTICE 'ROLLBACK;'; END $$;
        SELECT 'BEGIN; COMMIT';""")


def test_commit_cannot_leave_partial_schema(postgres, versions):
    (versions / "004_partial.sql").write_text(
        "CREATE TABLE researcher.partial_commit (id integer); COMMIT; SELECT missing_function();")
    with pytest.raises(ValueError, match="Transaction control"):
        migrate(postgres["owner"], versions)
    with psycopg.connect(postgres["owner"]) as conn:
        assert conn.execute("SELECT to_regclass('researcher.partial_commit')").fetchone()[0] is None
        assert conn.execute("SELECT count(*) FROM researcher.schema_migrations").fetchone()[0] == len(files())


def test_pending_apply_and_concurrent_runners(postgres, versions):
    (versions / "004_marker.sql").write_text("CREATE TABLE researcher.migration_marker (id integer);")
    assert status(postgres["owner"], versions)[-1]["status"] == "pending"
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: migrate(postgres["owner"], versions), range(2)))
    assert sorted(results, key=len) == [[], ["004_marker.sql"]]
    assert all(row["status"] == "applied" for row in status(postgres["owner"], versions))


def test_failure_rolls_back_schema_and_history(postgres, versions):
    (versions / "004_marker.sql").write_text("CREATE TABLE researcher.migration_marker (id integer);")
    (versions / "005_broken.sql").write_text("SELECT nonexistent_migration_function();")
    with pytest.raises(psycopg.Error):
        migrate(postgres["owner"], versions)
    with psycopg.connect(postgres["owner"]) as conn:
        assert conn.execute("SELECT to_regclass('researcher.migration_marker')").fetchone()[0] is None
        assert conn.execute("SELECT count(*) FROM researcher.schema_migrations").fetchone()[0] == len(files())


def test_missing_or_retroactive_migration_rejected(postgres, versions):
    (versions / "000_retroactive.sql").write_text("SELECT 1;")
    with pytest.raises(ValueError, match="history differs"):
        migrate(postgres["owner"], versions)
    (versions / "000_retroactive.sql").unlink()
    (versions / "001_platform.sql").rename(versions / "001_renamed.sql")
    with pytest.raises(ValueError, match="history differs"):
        status(postgres["owner"], versions)


def test_status_does_not_initialize_schema(postgres):
    with psycopg.connect(postgres["owner"]) as conn:
        conn.execute("DROP SCHEMA researcher CASCADE")
    assert status(postgres["owner"])[0]["status"] == "pending"
    with psycopg.connect(postgres["owner"]) as conn:
        assert conn.execute("SELECT to_regnamespace('researcher')").fetchone()[0] is None
    assert migrate(postgres["owner"]) == [name for name, _, _ in files()]


def test_user_migration_backfills_existing_assets(postgres, tmp_path, monkeypatch):
    from psycopg.types.json import Jsonb
    for path in sorted(VERSIONS.glob('*.sql'))[:2]:
        shutil.copy(path, tmp_path)
    with psycopg.connect(postgres['owner']) as conn:
        conn.execute('DROP SCHEMA researcher CASCADE')
    migrate(postgres['owner'], tmp_path)
    with psycopg.connect(postgres['owner']) as conn:
        result = {'proposal': {'source': 'saved'}, 'validation': {'valid': True}, 'usage': {'steps': [{'model': 'actual'}]}}
        conn.execute("""INSERT INTO researcher.platform_requests
            (tenant_id,request_id,session_id,input_hash,status,stage,started_at,expires_at,result)
            VALUES ('a','r','s','hash','succeeded','complete',now(),now(),%s)""", (Jsonb(result),))
        conn.execute("""INSERT INTO researcher.llm_usage VALUES ('a','c','r',now(),%s)""",
            (Jsonb({'tenantId': 'a', 'callId': 'c', 'requestId': 'r', 'model': 'actual'}),))
    monkeypatch.setenv('RESEARCHER_LEGACY_USER_ID', 'seeded-owner')
    assert migrate(postgres['owner']) == ['003_user_ownership.sql']
    assert migrate(postgres['owner']) == []
    with psycopg.connect(postgres['owner']) as conn:
        user, result = conn.execute('SELECT user_id,result FROM researcher.platform_requests').fetchone()
        assert user == result['userId'] == result['proposal']['userId'] == result['validation']['userId'] == 'seeded-owner'
        assert result['proposal']['source'] == 'saved'
        assert result['usage']['steps'][0]['userId'] == 'seeded-owner'
        assert conn.execute("SELECT user_id,event->>'userId' FROM researcher.llm_usage").fetchone() == ('seeded-owner', 'seeded-owner')


def test_cli_checks_without_logging_credentials(postgres, monkeypatch, capsys):
    monkeypatch.setenv("RESEARCHER_MIGRATION_DATABASE_URL", postgres["owner"])
    assert main(["check"]) == 0
    assert main(["up"]) == 0
    with psycopg.connect(postgres["owner"]) as conn:
        conn.execute("DROP SCHEMA researcher CASCADE")
    assert main(["check"]) == 1
    assert "pending" in capsys.readouterr().out
    monkeypatch.delenv("RESEARCHER_MIGRATION_DATABASE_URL")
    with pytest.raises(SystemExit) as exc:
        main(["up"])
    assert exc.value.code == 1
