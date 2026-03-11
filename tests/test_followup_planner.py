from __future__ import annotations

from utils.followup_planner import build_followup_plan


def test_followup_planner_high_risk_path() -> None:
    patient = {
        "Symptoms": "chest pain,severe shortness of breath",
        "Temperature": 101.4,
    }
    prediction = {
        "risk_level": "High",
        "priority_category": "Critical",
    }
    safety = {
        "escalation_level": "immediate",
    }

    result = build_followup_plan(patient, prediction, safety)

    assert "Immediate" in result["review_window"]
    assert "Do not discharge" in result["disposition_guidance"]


def test_followup_planner_routine_path() -> None:
    patient = {
        "Symptoms": "mild headache",
        "Temperature": 98.6,
    }
    prediction = {
        "risk_level": "Low",
        "priority_category": "Low",
    }
    safety = {
        "escalation_level": "routine",
    }

    result = build_followup_plan(patient, prediction, safety)

    assert "Routine" in result["review_window"]
    assert "Outpatient" in result["disposition_guidance"]
