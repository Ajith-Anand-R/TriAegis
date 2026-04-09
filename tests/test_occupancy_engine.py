from __future__ import annotations

import pytest

from utils.occupancy import OccupancyEngine


def _seed_payload() -> dict:
    return {
        "hospitals": [
            {
                "hospital_id": "HOSP-TEST",
                "name": "Test Hospital",
                "region": "Test",
                "wards": [
                    {
                        "ward_id": "WARD-OK",
                        "name": "General Ward",
                        "specialty": "General Medicine",
                        "capacity": 10,
                        "occupied": 5,
                    },
                    {
                        "ward_id": "WARD-WARNING",
                        "name": "Intermediate Ward",
                        "specialty": "Cardiology",
                        "capacity": 10,
                        "occupied": 8,
                    },
                    {
                        "ward_id": "WARD-CRITICAL",
                        "name": "Critical Ward",
                        "specialty": "Emergency Department",
                        "capacity": 10,
                        "occupied": 10,
                    },
                ],
            }
        ]
    }


def test_seed_and_load_status_thresholds(tmp_path) -> None:
    engine = OccupancyEngine(db_path=tmp_path / "occupancy.db")
    summary = engine.seed_from_payload(_seed_payload(), reset=True)

    assert summary["hospital_count"] == 1
    assert summary["ward_count"] == 3
    assert summary["bed_count"] == 30

    wards = {row["ward_id"]: row for row in engine.get_ward_occupancy()}
    assert wards["WARD-OK"]["load_status"] == "OK"
    assert wards["WARD-WARNING"]["load_status"] == "Warning"
    assert wards["WARD-CRITICAL"]["load_status"] == "Critical"


def test_reserve_then_release_updates_ward_snapshot(tmp_path) -> None:
    engine = OccupancyEngine(db_path=tmp_path / "occupancy.db")
    engine.seed_from_payload(_seed_payload(), reset=True)

    before = engine.get_ward_snapshot("WARD-OK")
    reservation = engine.reserve_bed("WARD-OK", "PAT-1001")
    after_reservation = engine.get_ward_snapshot("WARD-OK")

    assert reservation["action"] == "reserved"
    assert after_reservation["occupied_beds"] == before["occupied_beds"] + 1

    release = engine.release_bed(str(reservation["bed_id"]), patient_id="PAT-1001")
    after_release = engine.get_ward_snapshot("WARD-OK")

    assert release["action"] == "released"
    assert after_release["occupied_beds"] == before["occupied_beds"]


def test_reserve_bed_raises_when_ward_is_full(tmp_path) -> None:
    engine = OccupancyEngine(db_path=tmp_path / "occupancy.db")
    engine.seed_from_payload(_seed_payload(), reset=True)

    with pytest.raises(ValueError, match="No available bed"):
        engine.reserve_bed("WARD-CRITICAL", "PAT-OVERFLOW")


def test_wait_estimation_is_higher_for_full_ward(tmp_path) -> None:
    engine = OccupancyEngine(db_path=tmp_path / "occupancy.db")
    engine.seed_from_payload(_seed_payload(), reset=True)

    low_pressure_wait = engine.estimate_wait_time("WARD-OK", queue_ahead=0, average_turnover_minutes=30)
    high_pressure_wait = engine.estimate_wait_time("WARD-CRITICAL", queue_ahead=3, average_turnover_minutes=30)

    assert low_pressure_wait["estimated_wait_minutes"] == 0
    assert high_pressure_wait["estimated_wait_minutes"] > low_pressure_wait["estimated_wait_minutes"]
    assert high_pressure_wait["wait_band"] in {"Moderate", "Extended"}
