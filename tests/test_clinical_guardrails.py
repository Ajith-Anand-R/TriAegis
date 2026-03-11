from __future__ import annotations

from utils.clinical_guardrails import evaluate_red_flags


def test_red_flag_detects_critical_cluster() -> None:
    patient = {
        "Patient_ID": "rf-001",
        "Age": 67,
        "Gender": "Male",
        "Symptoms": "chest pain,severe shortness of breath",
        "Blood Pressure": "186/116",
        "Heart Rate": 145,
        "Temperature": 101.2,
        "Pre-Existing Conditions": "heart disease,hypertension",
    }

    result = evaluate_red_flags(patient)

    assert result["has_red_flags"] is True
    assert result["escalation_level"] == "immediate"
    assert result["trigger_count"] >= 3


def test_red_flag_all_clear_returns_routine() -> None:
    patient = {
        "Patient_ID": "rf-002",
        "Age": 28,
        "Gender": "Female",
        "Symptoms": "mild headache,runny nose",
        "Blood Pressure": "118/76",
        "Heart Rate": 74,
        "Temperature": 98.4,
        "Pre-Existing Conditions": "none",
    }

    result = evaluate_red_flags(patient)

    assert result["has_red_flags"] is False
    assert result["escalation_level"] == "routine"
    assert result["trigger_count"] == 0
