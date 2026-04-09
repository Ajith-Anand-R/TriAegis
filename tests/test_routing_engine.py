from __future__ import annotations

from utils.occupancy import OccupancyEngine
from utils.routing_engine import RoutingEngine


def _seed_payload() -> dict:
    return {
        "hospitals": [
            {
                "hospital_id": "H1",
                "name": "Hospital One",
                "region": "North",
                "wards": [
                    {
                        "ward_id": "H1-CARD",
                        "name": "Cardiology Ward",
                        "specialty": "Cardiology",
                        "capacity": 10,
                        "occupied": 4,
                    },
                    {
                        "ward_id": "H1-GEN",
                        "name": "General Ward",
                        "specialty": "General Medicine",
                        "capacity": 10,
                        "occupied": 2,
                    },
                ],
            },
            {
                "hospital_id": "H2",
                "name": "Hospital Two",
                "region": "South",
                "wards": [
                    {
                        "ward_id": "H2-CARD",
                        "name": "Cardiology Ward",
                        "specialty": "Cardiology",
                        "capacity": 10,
                        "occupied": 4,
                    }
                ],
            },
        ]
    }


def test_routing_prefers_specialty_match_when_capacity_exists(tmp_path) -> None:
    occupancy = OccupancyEngine(db_path=tmp_path / "routing.db")
    occupancy.seed_from_payload(_seed_payload(), reset=True)
    routing = RoutingEngine(occupancy)

    result = routing.route_patient(
        patient_id="PAT-1",
        risk_level="Medium",
        priority_score=6.8,
        department="Cardiology",
        queue_ahead=1,
    )

    assert result["recommended_ward_id"] in {"H1-CARD", "H2-CARD"}
    assert result["has_capacity"] is True
    assert result["recommended_bed_id"] is not None


def test_routing_honors_preferred_hospital_in_tie(tmp_path) -> None:
    occupancy = OccupancyEngine(db_path=tmp_path / "routing.db")
    occupancy.seed_from_payload(_seed_payload(), reset=True)
    routing = RoutingEngine(occupancy)

    result = routing.route_patient(
        patient_id="PAT-2",
        risk_level="Low",
        priority_score=3.5,
        department="Cardiology",
        preferred_hospital_id="H2",
        queue_ahead=0,
    )

    assert result["recommended_hospital_id"] == "H2"


def test_routing_marks_overflow_when_no_beds_anywhere(tmp_path) -> None:
    payload = {
        "hospitals": [
            {
                "hospital_id": "H-FULL",
                "name": "Full Hospital",
                "region": "West",
                "wards": [
                    {
                        "ward_id": "FULL-ED",
                        "name": "Emergency Unit",
                        "specialty": "Emergency Department",
                        "capacity": 6,
                        "occupied": 6,
                    },
                    {
                        "ward_id": "FULL-GEN",
                        "name": "General Ward",
                        "specialty": "General Medicine",
                        "capacity": 6,
                        "occupied": 6,
                    },
                ],
            }
        ]
    }

    occupancy = OccupancyEngine(db_path=tmp_path / "routing.db")
    occupancy.seed_from_payload(payload, reset=True)
    routing = RoutingEngine(occupancy)

    result = routing.route_patient(
        patient_id="PAT-3",
        risk_level="High",
        priority_score=9.4,
        department="Emergency Department",
        queue_ahead=2,
    )

    assert result["has_capacity"] is False
    assert result["recommended_bed_id"] is None
    assert result["overflow_risk"] in {"warning", "critical"}
    assert result["estimated_wait_minutes"] >= 0


def test_routing_prefers_emergency_fallback_when_specialty_missing(tmp_path) -> None:
    payload = {
        "hospitals": [
            {
                "hospital_id": "H-FALLBACK",
                "name": "Fallback Hospital",
                "region": "North",
                "wards": [
                    {
                        "ward_id": "H-FALLBACK-ED",
                        "name": "Emergency Unit",
                        "specialty": "Emergency Department",
                        "capacity": 8,
                        "occupied": 3,
                    },
                    {
                        "ward_id": "H-FALLBACK-GEN",
                        "name": "General Ward",
                        "specialty": "General Medicine",
                        "capacity": 8,
                        "occupied": 2,
                    },
                ],
            }
        ]
    }

    occupancy = OccupancyEngine(db_path=tmp_path / "routing.db")
    occupancy.seed_from_payload(payload, reset=True)
    routing = RoutingEngine(occupancy)

    result = routing.route_patient(
        patient_id="PAT-4",
        risk_level="Medium",
        priority_score=6.1,
        department="Cardiology",
        queue_ahead=0,
    )

    assert result["recommended_ward_id"] == "H-FALLBACK-ED"
    assert result["specialty_match_tier"] in {2, 3}


def test_low_acuity_balancing_avoids_overloaded_hospital(tmp_path) -> None:
    payload = {
        "hospitals": [
            {
                "hospital_id": "H-OVER",
                "name": "Overloaded Hospital",
                "region": "Central",
                "wards": [
                    {
                        "ward_id": "H-OVER-GEN",
                        "name": "General Ward",
                        "specialty": "General Medicine",
                        "capacity": 10,
                        "occupied": 9,
                    }
                ],
            },
            {
                "hospital_id": "H-BALANCED",
                "name": "Balanced Hospital",
                "region": "Central",
                "wards": [
                    {
                        "ward_id": "H-BALANCED-GEN",
                        "name": "General Ward",
                        "specialty": "General Medicine",
                        "capacity": 10,
                        "occupied": 3,
                    }
                ],
            },
        ]
    }

    occupancy = OccupancyEngine(db_path=tmp_path / "routing.db")
    occupancy.seed_from_payload(payload, reset=True)
    routing = RoutingEngine(occupancy)

    result = routing.route_patient(
        patient_id="PAT-5",
        risk_level="Low",
        priority_score=3.2,
        department="General Medicine",
        queue_ahead=0,
    )

    assert result["recommended_hospital_id"] == "H-BALANCED"
    assert result["overflow_risk"] in {"none", "low"}


def test_distribute_inflow_orders_by_priority_and_risk(tmp_path) -> None:
    payload = {
        "hospitals": [
            {
                "hospital_id": "H-ORDER",
                "name": "Ordering Hospital",
                "region": "Central",
                "wards": [
                    {
                        "ward_id": "H-ORDER-ED",
                        "name": "Emergency",
                        "specialty": "Emergency Department",
                        "capacity": 5,
                        "occupied": 1,
                    }
                ],
            }
        ]
    }

    occupancy = OccupancyEngine(db_path=tmp_path / "routing.db")
    occupancy.seed_from_payload(payload, reset=True)
    routing = RoutingEngine(occupancy)

    result = routing.distribute_patient_inflow(
        [
            {
                "patient_id": "PAT-LOW",
                "risk_level": "Low",
                "priority_score": 3.2,
                "department": "General Medicine",
                "queue_position": 2,
            },
            {
                "patient_id": "PAT-HIGH",
                "risk_level": "High",
                "priority_score": 9.4,
                "department": "Emergency Department",
                "queue_position": 1,
            },
        ]
    )

    assignments = result["assignments"]
    assert len(assignments) == 2
    assert assignments[0]["patient_id"] == "PAT-HIGH"
    assert assignments[0]["inflow_rank"] == 1
    assert assignments[1]["patient_id"] == "PAT-LOW"


def test_distribute_inflow_spreads_patients_when_capacity_is_limited(tmp_path) -> None:
    payload = {
        "hospitals": [
            {
                "hospital_id": "H-A",
                "name": "Hospital A",
                "region": "North",
                "wards": [
                    {
                        "ward_id": "H-A-GEN",
                        "name": "General Ward",
                        "specialty": "General Medicine",
                        "capacity": 1,
                        "occupied": 0,
                    }
                ],
            },
            {
                "hospital_id": "H-B",
                "name": "Hospital B",
                "region": "South",
                "wards": [
                    {
                        "ward_id": "H-B-GEN",
                        "name": "General Ward",
                        "specialty": "General Medicine",
                        "capacity": 1,
                        "occupied": 0,
                    }
                ],
            },
        ]
    }

    occupancy = OccupancyEngine(db_path=tmp_path / "routing.db")
    occupancy.seed_from_payload(payload, reset=True)
    routing = RoutingEngine(occupancy)

    result = routing.distribute_patient_inflow(
        [
            {
                "patient_id": "PAT-1",
                "risk_level": "Medium",
                "priority_score": 6.0,
                "department": "General Medicine",
                "queue_position": 1,
            },
            {
                "patient_id": "PAT-2",
                "risk_level": "Medium",
                "priority_score": 5.8,
                "department": "General Medicine",
                "queue_position": 2,
            },
        ]
    )

    assignments = result["assignments"]
    assert len(assignments) == 2
    assert all(bool(item["has_capacity"]) for item in assignments)
    assert {item["recommended_hospital_id"] for item in assignments} == {"H-A", "H-B"}
    assert int(result["served_with_capacity"]) == 2
