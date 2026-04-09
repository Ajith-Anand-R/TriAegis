from __future__ import annotations

import uuid

from fastapi.testclient import TestClient

import api as api_module
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


def test_admin_clear_all_requires_admin_role() -> None:
    token = _login("doctor", "doctor123")
    response = client.delete("/api/admin/data/all", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 403


def test_admin_clear_all_post_requires_admin_role() -> None:
    token = _login("doctor", "doctor123")
    response = client.post("/api/admin/data/all", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 403


def test_admin_delete_recent_post_requires_admin_role() -> None:
    token = _login("doctor", "doctor123")
    response = client.post(
        "/api/admin/data/recent?days=30&scope=all",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 403


def test_admin_delete_specific_post_requires_admin_role() -> None:
    token = _login("doctor", "doctor123")
    response = client.post(
        "/api/admin/data/specific",
        headers={"Authorization": f"Bearer {token}"},
        json={"patient_id": "PAT-POST-REQ-001"},
    )
    assert response.status_code == 403


def test_admin_delete_specific_post_deletes_prediction() -> None:
    token = _login("admin", "admin123")
    patient_id = f"DEL-SPEC-{uuid.uuid4().hex[:8]}"
    create_response = client.post(
        "/api/predict/save",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "Patient_ID": patient_id,
            "Patient Name": "Delete Specific",
            "Age": 39,
            "Gender": "Male",
            "Symptoms": "cough",
            "Blood Pressure": "120/80",
            "Heart Rate": 75,
            "Temperature": 98.6,
            "Pre-Existing Conditions": "none",
        },
    )
    assert create_response.status_code == 200

    prediction_id = int(create_response.json().get("prediction_id", 0))
    assert prediction_id > 0

    delete_response = client.post(
        "/api/admin/data/specific",
        headers={"Authorization": f"Bearer {token}"},
        json={"prediction_id": prediction_id},
    )
    assert delete_response.status_code == 200

    payload = delete_response.json()
    assert int(payload.get("deleted_prediction_rows", 0)) >= 1


def test_closed_loop_status_visible_to_clinical_roles() -> None:
    token = _login("doctor", "doctor123")
    response = client.get("/api/system/loop-status", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    payload = response.json()
    assert "state" in payload
    assert "config" in payload


def test_closed_loop_run_once_requires_admin() -> None:
    token = _login("doctor", "doctor123")
    response = client.post(
        "/api/system/loop/run-once",
        headers={"Authorization": f"Bearer {token}"},
        json={"queue_monitoring": False, "overflow_rebalance": False},
    )
    assert response.status_code == 403


def test_closed_loop_run_once_admin_contract() -> None:
    token = _login("admin", "admin123")
    response = client.post(
        "/api/system/loop/run-once",
        headers={"Authorization": f"Bearer {token}"},
        json={"queue_monitoring": False, "overflow_rebalance": False},
    )
    assert response.status_code == 200
    payload = response.json()
    assert "ran_at" in payload
    assert "state" in payload
    assert payload["queue_monitoring"] is None
    assert payload["overflow_rebalance"] is None


def test_closed_loop_control_requires_admin() -> None:
    token = _login("doctor", "doctor123")
    response = client.post(
        "/api/system/loop/control",
        headers={"Authorization": f"Bearer {token}"},
        json={"action": "pause"},
    )
    assert response.status_code == 403


def test_closed_loop_control_pause_admin_contract() -> None:
    token = _login("admin", "admin123")
    response = client.post(
        "/api/system/loop/control",
        headers={"Authorization": f"Bearer {token}"},
        json={"action": "pause"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["action"] == "pause"
    assert payload["state"]["enabled"] is False
    assert payload["state"]["running"] is False


def test_closed_loop_control_restart_invokes_stop_then_start(monkeypatch) -> None:
    token = _login("admin", "admin123")
    calls: list[str] = []

    def _fake_stop() -> None:
        calls.append("stop")
        api_module._set_closed_loop_state({"enabled": False, "running": False})

    def _fake_start() -> None:
        calls.append("start")
        api_module._set_closed_loop_state({"enabled": True, "running": True})

    monkeypatch.setattr(api_module, "_stop_closed_loop_worker", _fake_stop)
    monkeypatch.setattr(api_module, "_start_closed_loop_worker", _fake_start)

    response = client.post(
        "/api/system/loop/control",
        headers={"Authorization": f"Bearer {token}"},
        json={"action": "restart"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert calls == ["stop", "start"]
    assert payload["action"] == "restart"
    assert payload["state"]["enabled"] is True
    assert payload["state"]["running"] is True


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


def test_history_defaults_to_latest_record_per_patient() -> None:
    token = _login("doctor", "doctor123")
    patient_id = f"HIST-DEDUP-{uuid.uuid4().hex[:8]}"

    payload = {
        "Patient_ID": patient_id,
        "Patient Name": "Dedup History",
        "Age": 44,
        "Gender": "Male",
        "Symptoms": "cough",
        "Blood Pressure": "122/80",
        "Heart Rate": 76,
        "Temperature": 98.7,
        "Pre-Existing Conditions": "none",
    }

    first = client.post(
        "/api/predict/save",
        headers={"Authorization": f"Bearer {token}"},
        json=payload,
    )
    assert first.status_code == 200

    second_payload = dict(payload)
    second_payload["Symptoms"] = "cough,fatigue"
    second = client.post(
        "/api/predict/save",
        headers={"Authorization": f"Bearer {token}"},
        json=second_payload,
    )
    assert second.status_code == 200

    response = client.get("/api/history", params={"patient_id": patient_id})
    assert response.status_code == 200
    records = response.json().get("records", [])
    assert len(records) == 1
    assert str(records[0].get("patient_id")) == patient_id
