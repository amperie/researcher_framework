"""Run with python -m core.platform; bind privately behind QC/TLS in deployment."""
import logging
import os
import uvicorn

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run("core.platform.app:create_app", factory=True,
        host=os.environ.get("RESEARCHER_HOST", "127.0.0.1"),
        port=int(os.environ.get("RESEARCHER_PORT", "8091")),
        timeout_graceful_shutdown=15)
