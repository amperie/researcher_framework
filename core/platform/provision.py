"""Compatibility entry point; prefer python -m database.provision."""
import os
from database.provision import provision

if __name__ == "__main__":
    provision(os.environ["RESEARCHER_ADMIN_DATABASE_URL"], os.environ["RESEARCHER_DATABASE_PASSWORD"])
