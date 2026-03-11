from __future__ import annotations

from typing import Dict, List


def _as_float(value: object, default: float) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _symptoms(raw: str) -> List[str]:
    return [item.strip().lower() for item in str(raw).split(",") if item.strip()]


def build_followup_plan(
    patient: Dict[str, object],
    prediction: Dict[str, object],
    safety_screen: Dict[str, object],
) -> Dict[str, object]:
    risk = str(prediction.get("risk_level", "Low"))
    category = str(prediction.get("priority_category", "Low"))
    escalation = str(safety_screen.get("escalation_level", "routine"))
    symptoms = _symptoms(str(patient.get("Symptoms", "")))

    warning_signs: List[str] = [
        "New chest pain, severe shortness of breath, or loss of consciousness",
        "Persistent confusion, seizures, or sudden neurological changes",
        "Uncontrolled bleeding or rapidly worsening vital signs",
    ]

    if "high fever" in symptoms or _as_float(patient.get("Temperature", 98.6), 98.6) >= 102.0:
        warning_signs.append("Fever persists above 102 F despite initial treatment")

    if risk == "High" or category in {"Critical", "Urgent"} or escalation == "immediate":
        review_window = "Immediate bedside reassessment and continuous monitoring"
        followup_window = "Reassess every 15-30 minutes until stabilized"
        disposition = "Do not discharge without senior clinician sign-off"
    elif risk == "Medium" or category == "High" or escalation == "urgent":
        review_window = "Urgent clinical reassessment within 1 hour"
        followup_window = "Repeat vitals every 1-2 hours"
        disposition = "Discharge only if improving and clinically stable"
    else:
        review_window = "Routine clinical follow-up"
        followup_window = "Repeat vitals per ward protocol"
        disposition = "Outpatient follow-up in 24-72 hours if symptoms persist"

    patient_instructions = [
        "Hydrate adequately unless fluid restriction is advised",
        "Take medications exactly as prescribed",
        "Return immediately if any warning sign appears",
    ]

    return {
        "risk_level": risk,
        "priority_category": category,
        "review_window": review_window,
        "followup_window": followup_window,
        "disposition_guidance": disposition,
        "warning_signs": warning_signs,
        "patient_instructions": patient_instructions,
    }
