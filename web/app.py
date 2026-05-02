from __future__ import annotations

from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from web.service import delete_orphans, diagnostics, get_run_bundle, list_run_summaries, scan_orphans


APP_ROOT = Path(__file__).resolve().parent
STATIC_ROOT = APP_ROOT / "static"

app = FastAPI(title="Research Run Inspector", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(STATIC_ROOT)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_ROOT / "index.html")


@app.get("/api/runs")
def api_runs(
    profile: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
) -> dict:
    return {"runs": list_run_summaries(profile_name=profile, limit=limit)}


@app.get("/api/runs/{profile_name}/{record_id}")
def api_run_detail(profile_name: str, record_id: str) -> dict:
    try:
        return get_run_bundle(profile_name, record_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/diagnostics")
def api_diagnostics() -> dict:
    return diagnostics()


@app.post("/api/admin/orphans/scan")
def api_scan_orphans(profile: str | None = Query(default=None)) -> dict:
    return scan_orphans(profile_name=profile)


@app.post("/api/admin/orphans/delete")
def api_delete_orphans(
    payload: dict | None = Body(default=None),
    profile: str | None = Query(default=None),
) -> dict:
    if not bool((payload or {}).get("confirm")):
        raise HTTPException(status_code=400, detail="Deletion requires confirm=true in the request body.")
    return delete_orphans(profile_name=profile)
