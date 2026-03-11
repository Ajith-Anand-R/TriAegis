from __future__ import annotations

from utils.clinical_handoff import build_handoff_summary


def test_handoff_summary_builds_sbar() -> None:
    patient = {
        "Patient_ID": "handoff-001",
        "Age": 61,
        "Gender": "Male",
        "Symptoms": "chest pain,confusion,dizziness",
        "Blood Pressure": "172/104",
        "Heart Rate": 118,
        "Temperature": 100.8,
        "Pre-Existing Conditions": "diabetes,hypertension",
    }
    prediction = {
        "risk_level": "High",
        "priority_score": 9.1,
        "priority_category": "Critical",
        "department": "Emergency Department",
        "estimated_wait_time": "8 minutes",
    }
    safety = {
        "escalation_level": "immediate",
        "trigger_count": 2,
        "triggers": [
            {"detail": "Emergency symptoms detected: chest pain"},
            {"detail": "Blood pressure 172/104 exceeds emergency threshold"},
        ],
    }

    result = build_handoff_summary(
        patient=patient,
        prediction=prediction,
        safety_screen=safety,
        clinical_explanation="High risk due to cardio-neuro profile.",
    )

    assert result["format"] == "SBAR"
    assert "Patient handoff-001" in result["situation"]
    assert "Route to Emergency Department" in result["recommendation"]
    assert result["structured"]["red_flag_count"] == 2


def test_handoff_summary_routine_path() -> None:
    patient = {
        "Patient_ID": "handoff-002",
        "Age": 31,
        "Gender": "Female",
        "Symptoms": "mild headache,runny nose",
        "Blood Pressure": "118/76",
        "Heart Rate": 72,
        "Temperature": 98.4,
        "Pre-Existing Conditions": "none",
    }
    prediction = {
        "risk_level": "Low",
        "priority_score": 2.8,
        "priority_category": "Low",
        "department": "General Medicine",
        "estimated_wait_time": "55 minutes",
    }
    safety = {
        "escalation_level": "routine",
        "trigger_count": 0,
        "triggers": [],
    }

    result = build_handoff_summary(
        patient=patient,
        prediction=prediction,
        safety_screen=safety,
        clinical_explanation="Low-risk profile.",
    )

    assert result["structured"]["escalation_level"] == "routine"
    assert "Continue standard monitoring" in result["recommendation"]
