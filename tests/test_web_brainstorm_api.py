from __future__ import annotations

from fastapi.testclient import TestClient

from web.app import app


def test_create_brainstorm_session_endpoint_returns_payload(monkeypatch):
    monkeypatch.setattr(
        "web.app.create_brainstorm_session",
        lambda profile_name, payload: {"profile_name": profile_name, "session_id": "brainstorm-1", "status": "awaiting_user"},
    )
    client = TestClient(app)

    response = client.post("/api/brainstorm/sessions", json={"profile_name": "neuralsignal", "direction": "test"})

    assert response.status_code == 200
    assert response.json()["session_id"] == "brainstorm-1"


def test_get_brainstorm_session_endpoint_returns_payload(monkeypatch):
    monkeypatch.setattr(
        "web.app.get_brainstorm_session",
        lambda profile_name, session_id: {"profile_name": profile_name, "session_id": session_id, "status": "awaiting_user"},
    )
    client = TestClient(app)

    response = client.get("/api/brainstorm/sessions/neuralsignal/brainstorm-1")

    assert response.status_code == 200
    assert response.json()["session_id"] == "brainstorm-1"


def test_command_brainstorm_session_endpoint_returns_payload(monkeypatch):
    monkeypatch.setattr(
        "web.app.command_brainstorm_session",
        lambda profile_name, session_id, payload: {"profile_name": profile_name, "session_id": session_id, "status": "running"},
    )
    client = TestClient(app)

    response = client.post("/api/brainstorm/sessions/neuralsignal/brainstorm-1/commands", json={"command": "continue"})

    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_execute_brainstorm_session_endpoint_returns_payload(monkeypatch):
    monkeypatch.setattr(
        "web.app.execute_brainstorm_session",
        lambda profile_name, session_id, payload: {"profile_name": profile_name, "session_id": session_id, "start_node": "implement", "result": {}},
    )
    client = TestClient(app)

    response = client.post("/api/brainstorm/sessions/neuralsignal/brainstorm-1/execute", json={})

    assert response.status_code == 200
    assert response.json()["start_node"] == "implement"
