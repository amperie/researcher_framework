"""Tests for admin orphan cleanup API endpoints."""
from __future__ import annotations

from fastapi.testclient import TestClient

from web.app import app


def test_admin_orphan_scan_endpoint_returns_payload(monkeypatch):
    monkeypatch.setattr(
        "web.app.scan_orphans",
        lambda profile_name=None: {"mode": "scan", "profiles": [], "totals": {"orphan_chroma_records": 0}},
    )
    client = TestClient(app)

    response = client.post("/api/admin/orphans/scan")

    assert response.status_code == 200
    assert response.json()["mode"] == "scan"


def test_admin_orphan_delete_endpoint_requires_confirmation(monkeypatch):
    called = {"value": False}

    def _delete(profile_name=None):
        called["value"] = True
        return {"mode": "delete", "profiles": [], "totals": {}, "deleted": {}, "errors": []}

    monkeypatch.setattr("web.app.delete_orphans", _delete)
    client = TestClient(app)

    response = client.post("/api/admin/orphans/delete", json={})

    assert response.status_code == 400
    assert called["value"] is False

    response = client.post("/api/admin/orphans/delete", json={"confirm": True})

    assert response.status_code == 200
    assert response.json()["mode"] == "delete"
    assert called["value"] is True
