from __future__ import annotations

import time

import api as api_module


class _FakeOccupancy:
    def __init__(self) -> None:
        self.recorded_routes: list[dict] = []

    def get_hospital_occupancy(self):
        return [{"hospital_id": "h-1"}]

    def seed_from_file(self, reset: bool = False) -> None:
        return None

    def current_load(self):
        return {
            "wards": [{"ward_id": "w-1", "load_ratio": 0.99}],
            "summary": {"critical_ward_count": 1, "warning_ward_count": 0},
        }

    def record_route_decision(self, routing, queue_ahead: int, source: str, metadata: dict):
        route_id = f"route-{len(self.recorded_routes) + 1}"
        self.recorded_routes.append(
            {
                "route_id": route_id,
                "routing": dict(routing),
                "queue_ahead": int(queue_ahead),
                "source": source,
                "metadata": dict(metadata),
            }
        )
        return route_id


class _FakeDatabase:
    def get_priority_queue(self, status: str | None = None):
        if status != "waiting":
            return []
        return [
            {
                "patient_id": "PAT-CL-001",
                "risk_level": "High",
                "priority_score": 9.6,
                "department": "Emergency Department",
                "queue_position": 1,
                "prediction_id": 101,
            },
            {
                "patient_id": "PAT-CL-002",
                "risk_level": "Medium",
                "priority_score": 7.2,
                "department": "General Medicine",
                "queue_position": 2,
                "prediction_id": 102,
            },
        ]


class _FakeRoutingEngine:
    def __init__(self, occupancy) -> None:
        self.occupancy = occupancy

    def distribute_patient_inflow(self, payloads):
        assignments = []
        for idx, payload in enumerate(payloads, start=1):
            assignments.append(
                {
                    "patient_id": payload["patient_id"],
                    "queue_ahead": payload.get("queue_ahead", 0),
                    "inflow_rank": idx,
                    "assigned_hospital_id": "h-1",
                    "assigned_ward_id": "w-1",
                    "overflow_risk": "low",
                }
            )
        return {
            "assignments": assignments,
            "total_incoming_requests": len(payloads),
            "served_with_capacity": len(payloads),
            "overflow_recommended": 0,
        }


def _reset_rebalance_state() -> None:
    api_module._set_closed_loop_state(
        {
            "last_rebalance_signature": "",
            "last_rebalance_epoch": 0.0,
            "last_rebalance_at": None,
            "rebalance_runs": 0,
        }
    )


def test_overflow_rebalance_cycle_persists_routes_when_overflow(monkeypatch) -> None:
    fake_occupancy = _FakeOccupancy()
    fake_db = _FakeDatabase()

    monkeypatch.setattr(api_module, "get_occupancy", lambda: fake_occupancy)
    monkeypatch.setattr(api_module, "get_db", lambda: fake_db)
    monkeypatch.setattr(api_module, "RoutingEngine", _FakeRoutingEngine)

    _reset_rebalance_state()

    result = api_module._run_overflow_rebalance_cycle(
        enforce_cooldown=False,
        max_patients=10,
    )

    assert result["overflow_detected"] is True
    assert int(result["waiting_considered"]) == 2
    assert int(result["persisted_routes"]) == 2
    assert result["skipped_reason"] is None
    assert len(fake_occupancy.recorded_routes) == 2


def test_overflow_rebalance_cycle_honors_cooldown_for_same_signature(monkeypatch) -> None:
    fake_occupancy = _FakeOccupancy()
    fake_db = _FakeDatabase()

    monkeypatch.setattr(api_module, "get_occupancy", lambda: fake_occupancy)
    monkeypatch.setattr(api_module, "get_db", lambda: fake_db)
    monkeypatch.setattr(api_module, "RoutingEngine", _FakeRoutingEngine)

    api_module._set_closed_loop_state(
        {
            "last_rebalance_signature": "PAT-CL-001:1|PAT-CL-002:2",
            "last_rebalance_epoch": time.time(),
            "rebalance_runs": 0,
        }
    )

    result = api_module._run_overflow_rebalance_cycle(
        enforce_cooldown=True,
        max_patients=10,
    )

    assert result["overflow_detected"] is True
    assert str(result["skipped_reason"]) == "cooldown"
    assert int(result["persisted_routes"]) == 0
    assert result["distribution"] is None
    assert fake_occupancy.recorded_routes == []
