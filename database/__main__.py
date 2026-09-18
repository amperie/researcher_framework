"""Run from the repository root: python -m database {status,check,up,new}."""
import argparse
import os
import psycopg
from database.migrations import migrate, new, status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("status", "check", "up"):
        commands.add_parser(name)
    commands.add_parser("new").add_argument("name")
    args = parser.parse_args(argv)
    try:
        if args.command == "new":
            print(new(args.name))
            return 0
        dsn = os.environ.get("RESEARCHER_MIGRATION_DATABASE_URL")
        if not dsn:
            raise ValueError("Set RESEARCHER_MIGRATION_DATABASE_URL to an owner-authorized connection")
        if args.command == "up":
            applied = migrate(dsn)
            print("Applied: " + ", ".join(applied) if applied else "Database is up to date")
            return 0
        rows = status(dsn)
        for row in rows:
            print(f"{row['status']:7} {row['name']}")
        return int(args.command == "check" and any(row["status"] == "pending" for row in rows))
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Migration error: {exc}\n")
    except psycopg.Error as exc:
        parser.exit(1, f"Database migration failed (SQLSTATE {exc.sqlstate or 'connection error'}).\n")


if __name__ == "__main__":
    raise SystemExit(main())
