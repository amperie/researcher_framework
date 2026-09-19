"""Local developer playground; intentionally exposes its test service credentials."""
import os
from pathlib import Path
import secrets

from fastapi.responses import FileResponse, JSONResponse
from configs.config import _load_dotenv, get_config
from core.platform.app import create_app
from core.platform.identity import LEGACY_USER

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "claude-haiku-4-5-20251001"


def create_playground(*, service=None):
    if service is None:
        _load_dotenv(ROOT / "configs/platform.env")
        _load_dotenv(ROOT / "configs/.env")
        os.environ["RESEARCHER_LLM_PROVIDER"] = "anthropic"
        os.environ.setdefault("RESEARCHER_LLM_MODEL", DEFAULT_MODEL)
        get_config.cache_clear()
        if not get_config().anthropic_api_key:
            raise ValueError("Set ANTHROPIC_API_KEY in your environment or configs/.env")
    tenants = [{"id": "playground-a", "key": secrets.token_urlsafe(32)},
               {"id": "playground-b", "key": secrets.token_urlsafe(32)}]
    app = create_app(service=service, tenant_keys={item["key"]: item["id"] for item in tenants})
    app.state.service.capture_model_output = True

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(Path(__file__).with_name("playground.html"), headers={"Cache-Control": "no-store"})

    @app.get("/playground/config", include_in_schema=False)
    def config():
        return JSONResponse({"tenants": tenants, "userId": os.environ.get('RESEARCHER_PLAYGROUND_USER_ID', LEGACY_USER), "provider": "anthropic",
                "model": os.environ.get("RESEARCHER_LLM_MODEL", DEFAULT_MODEL)}, headers={"Cache-Control": "no-store"})

    @app.get("/activity.js", include_in_schema=False)
    def activity_script():
        return FileResponse(Path(__file__).with_name("activity.js"), media_type="text/javascript",
            headers={"Cache-Control": "no-store"})

    return app
