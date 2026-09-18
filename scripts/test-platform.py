"""Run tests with PostgreSQL; --local uses scripts/test-postgres.sh's cluster."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from psycopg.conninfo import make_conninfo

parser = argparse.ArgumentParser()
parser.add_argument("--local", action="store_true")
args, pytest_args = parser.parse_known_args()
root = Path(__file__).resolve().parents[1]
if args.local:
    os.environ["RESEARCHER_TEST_ADMIN_URL"] = make_conninfo(
        host="127.0.0.1", port=55439, dbname="postgres", user="postgres", connect_timeout=5,
        password=(root / ".tmp/postgres-test.password").read_text().strip())
if not os.environ.get("RESEARCHER_TEST_ADMIN_URL"):
    parser.error("Set RESEARCHER_TEST_ADMIN_URL or start the local cluster and use --local")
raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", *(pytest_args or ["-q"])], cwd=root))
