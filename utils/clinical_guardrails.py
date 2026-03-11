from __future__ import annotations

from typing import Dict, List


EMERGENCY_SYMPTOMS = {
    "chest pain",
    "severe shortness of breath",
    "loss of consciousness",
    "stroke symptoms",
    "severe bleeding",
    "uncontrolled bleeding",
    "seizure",
    "severe trauma",
}


def _parse_multi(raw: str | None) -> List[str]:
    if not raw:
        return []
    return [item.strip().lower() for item in str(raw).split(",") if item.strip()]


def _parse_bp(raw: str) -> tuple[int, int]:
    if "/" not in raw:
        return 120, 80
    left, right = raw.split("/", 1)
    try:
        return int(left), int(right)
    except ValueError:
        return 120, 80


def evaluate_red_flags(patient: Dict[str, object]) -> Dict[str, object]:
    symptoms = _parse_multi(str(patient.get("Symptoms", "")))
    bp_s, bp_d = _parse_bp(str(patient.get("Blood Pressure", "120/80")))
    heart_rate = int(patient.get("Heart Rate", 70) or 70)
    temperature = float(patient.get("Temperature", 98.6) or 98.6)
    age = int(patient.get("Age", 0) or 0)

    triggers: List[Dict[str, str]] = []

    emergency_hits = sorted([symptom for symptom in symptoms if symptom in EMERGENCY_SYMPTOMS])
    if emergency_hits:
        triggers.append(
            {
                "code": "emergency_symptom_cluster",
                "severity": "critical",
                "detail": f"Emergency symptoms detected: {', '.join(emergency_hits)}",
            }
        )

    if bp_s >= 180 or bp_d >= 110:
        triggers.append(
            {
                "code": "hypertensive_crisis_pattern",
                "severity": "critical",
                "detail": f"Blood pressure {bp_s}/{bp_d} exceeds emergency threshold",
            }
        )

    if heart_rate >= 140 or heart_rate <= 40:
        triggers.append(
            {
                "code": "critical_heart_rate",
                "severity": "critical",
                "detail": f"Heart rate {heart_rate} bpm is in a critical range",
            }
        )

    if temperature >= 103.5:
        triggers.append(
            {
                "code": "high_fever_instability",
                "severity": "high",
                "detail": f"Temperature {temperature:.1f} F indicates severe febrile risk",
            }
        )

    if age >= 75 and ("confusion" in symptoms or "stroke symptoms" in symptoms):
        triggers.append(
            {
                "code": "elderly_neuro_red_flag",
                "severity": "high",
                "detail": "Elderly patient with neurological red flags requires urgent clinician review",
            }
        )

    critical_count = sum(1 for trigger in triggers if trigger["severity"] == "critical")
    high_count = sum(1 for trigger in triggers if trigger["severity"] == "high")

    if critical_count > 0:
        escalation_level = "immediate"
        recommendation = "Immediate emergency clinician intervention required"
    elif high_count > 0:
        escalation_level = "urgent"
        recommendation = "Urgent in-person clinical review is recommended"
    else:
        escalation_level = "routine"
        recommendation = "No immediate red-flag trigger detected; continue standard triage"

    return {
        "has_red_flags": bool(triggers),
        "escalation_level": escalation_level,
        "recommendation": recommendation,
        "trigger_count": len(triggers),
        "triggers": triggers,
    }
