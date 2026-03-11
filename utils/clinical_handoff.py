from __future__ import annotations

from typing import Dict, List, TypedDict, cast


class RedFlagItem(TypedDict, total=False):
    code: str
    severity: str
    detail: str


class HandoffStructured(TypedDict):
    risk_level: str
    priority_score: float
    priority_category: str
    department: str
    estimated_wait_time: str
    escalation_level: str
    red_flag_count: int


class HandoffSummary(TypedDict):
    patient_id: str
    format: str
    situation: str
    background: str
    assessment: str
    recommendation: str
    clinical_explanation_excerpt: str
    structured: HandoffStructured


def _as_int(value: object, default: int) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _as_float(value: object, default: float) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _vitals_line(patient: Dict[str, object]) -> str:
    bp = str(patient.get("Blood Pressure", "120/80"))
    hr = _as_int(patient.get("Heart Rate", 70), 70)
    temp = _as_float(patient.get("Temperature", 98.6), 98.6)
    return f"BP {bp}, HR {hr} bpm, Temp {temp:.1f} F"


def _top_symptoms(patient: Dict[str, object], max_items: int = 3) -> List[str]:
    symptoms = [item.strip() for item in str(patient.get("Symptoms", "")).split(",") if item.strip()]
    return symptoms[:max_items]


def build_handoff_summary(
    patient: Dict[str, object],
    prediction: Dict[str, object],
    safety_screen: Dict[str, object],
    clinical_explanation: str,
) -> HandoffSummary:
    patient_id = str(patient.get("Patient_ID", "unknown"))
    age = _as_int(patient.get("Age", 0), 0)
    gender = str(patient.get("Gender", "Unknown"))

    risk = str(prediction.get("risk_level", "Unknown"))
    priority_score = _as_float(prediction.get("priority_score", 0.0), 0.0)
    priority_category = str(prediction.get("priority_category", "Unknown"))
    department = str(prediction.get("department", "General Medicine"))
    wait_time = str(prediction.get("estimated_wait_time", "unknown"))

    escalation_level = str(safety_screen.get("escalation_level", "routine"))
    red_flags = cast(List[RedFlagItem], safety_screen.get("triggers", []))
    red_flag_text = "; ".join(str(item.get("detail", "")) for item in red_flags[:3]) if red_flags else "No acute red flags detected"

    symptoms = _top_symptoms(patient)
    symptom_text = ", ".join(symptoms) if symptoms else "non-specific symptoms"

    situation = (
        f"Patient {patient_id} ({age}y {gender}) presents with {symptom_text}. "
        f"Model triage risk: {risk} ({priority_category}, score {priority_score:.1f}/10)."
    )

    background = (
        f"Vitals: {_vitals_line(patient)}. "
        f"Known conditions: {str(patient.get('Pre-Existing Conditions', 'none'))}."
    )

    assessment = (
        f"Safety escalation level: {escalation_level}. "
        f"Red-flag summary: {red_flag_text}."
    )

    recommendation = (
        f"Route to {department}. Target wait: {wait_time}. "
        "Immediate clinician reassessment required if vitals worsen or new neurological/cardiac symptoms emerge."
        if escalation_level in {"immediate", "urgent"}
        else f"Route to {department}. Target wait: {wait_time}. Continue standard monitoring and reassess if condition changes."
    )

    return {
        "patient_id": patient_id,
        "format": "SBAR",
        "situation": situation,
        "background": background,
        "assessment": assessment,
        "recommendation": recommendation,
        "clinical_explanation_excerpt": clinical_explanation[:500],
        "structured": {
            "risk_level": risk,
            "priority_score": priority_score,
            "priority_category": priority_category,
            "department": department,
            "estimated_wait_time": wait_time,
            "escalation_level": escalation_level,
            "red_flag_count": _as_int(safety_screen.get("trigger_count", 0), 0),
        },
    }
