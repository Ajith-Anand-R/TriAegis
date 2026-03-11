from __future__ import annotations

from fastapi.testclient import TestClient

from api import app


client = TestClient(app)


def _login(username: str = "doctor", password: str = "doctor123") -> str:
    response = client.post("/api/auth/login", json={"username": username, "password": password})
    assert response.status_code == 200
    data = response.json()
    return str(data["access_token"])


def test_auth_me_requires_token() -> None:
    response = client.get("/api/auth/me")
    assert response.status_code == 401


def test_security_config_requires_admin_role() -> None:
    token = _login("doctor", "doctor123")
    response = client.get("/api/security/config", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 403


def test_security_config_admin_ok() -> None:
    token = _login("admin", "admin123")
    response = client.get("/api/security/config", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    payload = response.json()
    assert "secure" in payload
    assert "using_default_secret" in payload


def test_phi_safe_csv_export_redacts_patient_id_and_notes() -> None:
    token = _login("doctor", "doctor123")
    response = client.post(
        "/api/export/csv/phi-safe",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "records": [
                {
                    "patient_id": "PAT-123456",
                    "risk_level": "High",
                    "notes": "private-note",
                    "department": "Emergency Department",
                }
            ]
        },
    )
    assert response.status_code == 200
    body = response.text
    assert "PAT-123456" not in body
    assert "PA***56" in body
    assert "[REDACTED]" in body


def test_medication_screen_endpoint_contract() -> None:
    token = _login("doctor", "doctor123")
    response = client.post(
        "/api/safety/medication-screen",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "medications": "warfarin,aspirin",
            "allergies": "aspirin",
            "conditions": "heart disease",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["screening"]["overall_risk"] in {"medium", "high"}
    assert "interaction_alerts" in payload["screening"]
