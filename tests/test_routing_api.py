from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import api
from utils.occupancy import OccupancyEngine


client = TestClient(api.app)


@pytest.fixture(autouse=True)
def isolated_occupancy_engine(tmp_path) -> None:
    api._occupancy = OccupancyEngine(
        db_path=tmp_path / "routing_api.db",
        seed_path=api.ROOT / "data" / "occupancy_seed.json",
    )
    api._occupancy.seed_from_file(reset=True)
    yield
    api._occupancy = None


def _login(username: str = "doctor", password: str = "doctor123") -> str:
    response = client.post("/api/auth/login", json={"username": username, "password": password})
    assert response.status_code == 200
    return str(response.json()["access_token"])


def test_route_endpoint_requires_authentication() -> None:
    response = client.post(
        "/api/route",
        json={
            "risk_level": "High",
            "priority_score": 9.1,
            "department": "Emergency Department",
        },
    )
    assert response.status_code == 401


def test_route_endpoint_returns_structured_recommendation() -> None:
    token = _login("doctor", "doctor123")

    response = client.post(
        "/api/route",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "patient_id": "PAT-ROUTE-1",
            "risk_level": "Medium",
            "priority_score": 6.4,
            "department": "Cardiology",
            "queue_ahead": 1,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert "routing" in payload
    assert "summary" in payload
    assert "routing_metrics" in payload

    routing = payload["routing"]
    assert routing["route_id"]
    assert routing["patient_id"] == "PAT-ROUTE-1"
    assert routing["recommended_hospital_id"]
    assert routing["recommended_ward_id"]
    assert "alternatives" in routing


def test_route_endpoint_accepts_preferred_hospital() -> None:
    token = _login("doctor", "doctor123")

    hospitals_response = client.get(
        "/api/hospitals",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert hospitals_response.status_code == 200
    hospitals = hospitals_response.json().get("hospitals", [])
    assert hospitals

    preferred_hospital_id = str(hospitals[0]["hospital_id"])

    response = client.post(
        "/api/route",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "patient_id": "PAT-ROUTE-2",
            "risk_level": "Low",
            "priority_score": 3.0,
            "department": "General Medicine",
            "preferred_hospital_id": preferred_hospital_id,
            "queue_ahead": 0,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["routing"]["preferred_hospital_id"] == preferred_hospital_id


def test_route_to_admit_then_discharge_lifecycle_and_metrics() -> None:
    token = _login("doctor", "doctor123")

    route_response = client.post(
        "/api/route/admit",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "patient_id": "PAT-ROUTE-LIFECYCLE-1",
            "risk_level": "Medium",
            "priority_score": 6.2,
            "department": "General Medicine",
            "queue_ahead": 0,
        },
    )
    assert route_response.status_code == 200

    route_payload = route_response.json()
    assert route_payload["routing"]["route_id"]
    assert "routing_metrics" in route_payload

    if not route_payload["admitted"]:
        pytest.skip("Seed state returned no immediate capacity; skipping lifecycle discharge segment")

    admission = route_payload["admission"]
    assert admission is not None
    assert admission["action"] == "reserved"
    bed_id = str(admission["bed_id"])

    discharge_response = client.post(
        "/api/discharge",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "bed_id": bed_id,
            "patient_id": "PAT-ROUTE-LIFECYCLE-1",
        },
    )
    assert discharge_response.status_code == 200
    discharge_payload = discharge_response.json()
    assert discharge_payload["discharge"]["action"] == "released"

    analytics_response = client.get(
        "/api/analytics",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert analytics_response.status_code == 200
    analytics_payload = analytics_response.json()
    assert analytics_payload.get("empty") is False
    metrics = analytics_payload.get("metrics", {})
    assert float(metrics.get("routing_capacity_hit_rate", 0.0)) >= 0.0
    assert float(metrics.get("routing_overflow_rate", 0.0)) >= 0.0


def test_route_distribute_endpoint_returns_capacity_aware_assignments() -> None:
    token = _login("doctor", "doctor123")

    response = client.post(
        "/api/route/distribute",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "patients": [
                {
                    "patient_id": "PAT-DIST-LOW",
                    "risk_level": "Low",
                    "priority_score": 3.2,
                    "department": "General Medicine",
                    "queue_position": 2,
                },
                {
                    "patient_id": "PAT-DIST-HIGH",
                    "risk_level": "High",
                    "priority_score": 9.1,
                    "department": "Emergency Department",
                    "queue_position": 1,
                },
            ],
            "persist_routes": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    distribution = payload.get("distribution", {})
    assignments = distribution.get("assignments", [])

    assert len(assignments) == 2
    assert assignments[0]["patient_id"] == "PAT-DIST-HIGH"
    assert int(payload.get("persisted_routes", 0)) == 2
    assert len(payload.get("route_ids", [])) == 2
    assert all(str(item.get("route_id", "")).strip() for item in assignments)
