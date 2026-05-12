from __future__ import annotations

from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from web.service import (
    command_brainstorm_session,
    create_brainstorm_session,
    create_proposal_seed,
    create_run_handoff,
    delete_orphans,
    diagnostics,
    execute_brainstorm_session,
    get_brainstorm_session,
    get_run_bundle,
    list_run_summaries,
    scan_orphans,
)


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


@app.post("/api/runs/{profile_name}/{record_id}/handoffs")
def api_run_handoff(profile_name: str, record_id: str, payload: dict | None = Body(default=None)) -> dict:
    try:
        return create_run_handoff(profile_name, record_id, payload or {})
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/runs/{profile_name}/{record_id}/proposal-seeds")
def api_proposal_seed(profile_name: str, record_id: str, payload: dict | None = Body(default=None)) -> dict:
    try:
        return create_proposal_seed(profile_name, record_id, payload or {})
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/diagnostics")
def api_diagnostics() -> dict:
    return diagnostics()


@app.post("/api/brainstorm/sessions")
def api_create_brainstorm_session(payload: dict | None = Body(default=None)) -> dict:
    data = payload or {}
    profile_name = str(data.get("profile_name") or "").strip()
    if not profile_name:
        raise HTTPException(status_code=400, detail="profile_name is required")
    try:
        return create_brainstorm_session(profile_name, data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/brainstorm/sessions/{profile_name}/{session_id}")
def api_get_brainstorm_session(profile_name: str, session_id: str) -> dict:
    try:
        return get_brainstorm_session(profile_name, session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/brainstorm/sessions/{profile_name}/{session_id}/commands")
def api_command_brainstorm_session(profile_name: str, session_id: str, payload: dict | None = Body(default=None)) -> dict:
    try:
        return command_brainstorm_session(profile_name, session_id, payload or {})
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/brainstorm/sessions/{profile_name}/{session_id}/execute")
def api_execute_brainstorm_session(profile_name: str, session_id: str, payload: dict | None = Body(default=None)) -> dict:
    try:
        return execute_brainstorm_session(profile_name, session_id, payload or {})
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


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
