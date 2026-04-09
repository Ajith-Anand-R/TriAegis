from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import api
from utils.occupancy import OccupancyEngine


client = TestClient(api.app)


@pytest.fixture(autouse=True)
def isolated_occupancy_engine(tmp_path) -> None:
    api._occupancy = OccupancyEngine(
        db_path=tmp_path / "occupancy_api.db",
        seed_path=api.ROOT / "data" / "occupancy_seed.json",
    )
    api._occupancy.seed_from_file(reset=True)
    yield
    api._occupancy = None


def _login(username: str = "doctor", password: str = "doctor123") -> str:
    response = client.post("/api/auth/login", json={"username": username, "password": password})
    assert response.status_code == 200
    return str(response.json()["access_token"])


def test_occupancy_seed_and_list_endpoints() -> None:
    admin_token = _login("admin", "admin123")

    seed_response = client.post(
        "/api/occupancy/seed",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"reset": True},
    )
    assert seed_response.status_code == 200
    seed_payload = seed_response.json()
    assert int(seed_payload["seeded"]["hospital_count"]) >= 1
    assert int(seed_payload["seeded"]["ward_count"]) >= 1

    doctor_token = _login("doctor", "doctor123")
    hospitals_response = client.get(
        "/api/hospitals",
        headers={"Authorization": f"Bearer {doctor_token}"},
    )
    assert hospitals_response.status_code == 200
    hospitals_payload = hospitals_response.json()
    assert int(hospitals_payload["summary"]["hospital_count"]) >= 1
    assert len(hospitals_payload["hospitals"]) >= 1

    wards_response = client.get(
        "/api/wards",
        headers={"Authorization": f"Bearer {doctor_token}"},
    )
    assert wards_response.status_code == 200
    wards_payload = wards_response.json()
    assert int(wards_payload["count"]) >= 1
    assert len(wards_payload["wards"]) >= 1


def test_admit_then_discharge_cycle() -> None:
    admin_token = _login("admin", "admin123")
    seed_response = client.post(
        "/api/occupancy/seed",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"reset": True},
    )
    assert seed_response.status_code == 200

    doctor_token = _login("doctor", "doctor123")
    wards_response = client.get(
        "/api/wards",
        headers={"Authorization": f"Bearer {doctor_token}"},
    )
    assert wards_response.status_code == 200
    wards = wards_response.json()["wards"]

    target = next((ward for ward in wards if int(ward["available_beds"]) > 0), None)
    assert target is not None

    admit_response = client.post(
        "/api/admit",
        headers={"Authorization": f"Bearer {doctor_token}"},
        json={
            "ward_id": str(target["ward_id"]),
            "patient_id": "TEST-PAT-9001",
        },
    )
    assert admit_response.status_code == 200
    admit_payload = admit_response.json()
    assert admit_payload["admission"]["action"] == "reserved"
    bed_id = str(admit_payload["admission"]["bed_id"])

    discharge_response = client.post(
        "/api/discharge",
        headers={"Authorization": f"Bearer {doctor_token}"},
        json={
            "bed_id": bed_id,
            "patient_id": "TEST-PAT-9001",
        },
    )
    assert discharge_response.status_code == 200
    discharge_payload = discharge_response.json()
    assert discharge_payload["discharge"]["action"] == "released"
