"""Generate the checked-in integration contract without loading real credentials."""
import argparse
import json
from pathlib import Path
from core.platform.app import create_app
from core.platform.service import PlatformService
from core.platform.store import PlatformStore


def schema():
    # Store construction is lazy; schema generation never connects to a database.
    service = PlatformService(PlatformStore("dbname=qc-researcher"))
    app = create_app(service=service, tenant_keys={"schema-export-only-not-a-real-key-0000": "schema"})
    return app.openapi()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="docs/platform-openapi.json")
    args = parser.parse_args()
    Path(args.output).write_text(json.dumps(schema(), indent=2) + "\n", encoding="utf-8")
