"""Compatibility entry point; prefer python -m database up."""
import os
from database.migrations import migrate

if __name__ == "__main__":
    migrate(os.environ["RESEARCHER_MIGRATION_DATABASE_URL"])
