#!/bin/sh
# Local Docker test cluster; never connects to the configured platform database.
set -eu
cd "$(dirname "$0")/.."
mkdir -p .tmp
if [ ! -f .tmp/postgres-test.password ]; then
    umask 077
    openssl rand -hex 24 > .tmp/postgres-test.password
fi
docker compose -f compose.test-postgres.yaml up -d --wait
printf 'Test PostgreSQL ready on localhost:55439; password stored in .tmp/postgres-test.password\n'
