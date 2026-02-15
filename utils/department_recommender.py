from __future__ import annotations

from datetime import UTC, datetime
from typing import Dict, List, Sequence


EMERGENCY_SYMPTOMS = {
    "severe bleeding",
    "loss of consciousness",
    "stroke symptoms",
    "severe allergic reaction",
    "uncontrolled bleeding",
    "severe trauma",
}

CARDIOLOGY_SYMPTOMS = {"chest pain", "palpitations", "irregular heartbeat"}
NEUROLOGY_SYMPTOMS = {
    "severe headache",
    "confusion",
    "dizziness",
    "seizure",
    "stroke symptoms",
    "loss of consciousness",
    "mild dizziness",
}
PULMONOLOGY_SYMPTOMS = {
    "severe shortness of breath",
    "moderate shortness of breath",
    "difficulty breathing",
    "cough",
}
ORTHOPEDICS_SYMPTOMS = {"joint pain", "back pain", "minor injury", "sprain"}
GASTRO_SYMPTOMS = {"abdominal pain", "severe abdominal pain", "persistent vomiting", "nausea", "constipation"}

BASE_WAIT_BY_DEPARTMENT = {
    "Emergency Department": 5,
    "Cardiology": 30,
    "Neurology": 35,
    "Pulmonology": 40,
    "Orthopedics": 45,
    "Gastroenterology": 40,
    "Pediatrics": 25,
    "General Medicine": 50,
}


def _parse_multi(raw_value: str | None) -> List[str]:
    if not raw_value:
        return []
    values = [token.strip() for token in str(raw_value).split(",") if token.strip()]
    return [token for token in values if token.lower() != "none"]


def _bp_parts(blood_pressure: str) -> tuple[int, int]:
    parts = blood_pressure.split("/")
    if len(parts) != 2:
        raise ValueError(f"Invalid blood pressure format: {blood_pressure}")
    return int(parts[0]), int(parts[1])


def _vitals_urgency(bp_s: int, bp_d: int, heart_rate: int, temperature: float) -> float:
    bp_component = max(0.0, (bp_s - 120) / 60) + max(0.0, (bp_d - 80) / 35)
    hr_component = abs(heart_rate - 75) / 45
    temp_component = abs(temperature - 98.6) / 2.2
    return max(0.0, bp_component + hr_component + temp_component)


def _normalize_to_1_10(raw_score: float, min_raw: float = 2.0, max_raw: float = 25.0) -> float:
    clipped = max(min_raw, min(max_raw, raw_score))
    normalized = 1 + ((clipped - min_raw) / (max_raw - min_raw)) * 9
    return round(float(normalized), 1)


def priority_category(score: float) -> str:
    if score >= 9:
        return "Critical"
    if score >= 7:
        return "Urgent"
    if score >= 5:
        return "High"
    if score >= 3:
        return "Standard"
    return "Low"


def calculate_priority(
    risk_level: str,
    age: int,
    symptoms: Sequence[str],
    bp_s: int,
    bp_d: int,
    heart_rate: int,
    temperature: float,
    condition_count: int,
    ehr_history_score: float = 0.0,
) -> Dict[str, float | str]:
    risk_weight_map = {"Low": 1, "Medium": 3, "High": 5}
    risk_weight = risk_weight_map.get(risk_level, 1)

    age_factor = (age / 100) * 2
    vitals_component = _vitals_urgency(bp_s, bp_d, heart_rate, temperature)
    symptom_severity = sum(1 for symptom in symptoms if symptom in EMERGENCY_SYMPTOMS or symptom in CARDIOLOGY_SYMPTOMS) * 0.5
    comorbidity_factor = condition_count * 0.3

    ehr_component = max(0.0, min(ehr_history_score, 3.0))

    raw_priority = (
        (risk_weight * 3.0)
        + (age_factor * 1.5)
        + (vitals_component * 2.0)
        + (symptom_severity * 2.0)
        + (comorbidity_factor * 1.5)
        + (ehr_component * 2.0)
    )

    score = _normalize_to_1_10(raw_priority)

    if risk_level == "High" or any(symptom in EMERGENCY_SYMPTOMS for symptom in symptoms):
        score = max(score, 9.0)
    elif risk_level == "Medium":
        score = max(score, 5.0)

    if bp_s > 180 or bp_d > 110 or heart_rate > 140 or heart_rate < 45 or temperature > 103.5:
        score = max(score, 9.0)

    max_by_risk = {
        "Low": 4.9,
        "Medium": 8.9,
        "High": 10.0,
    }
    score = min(score, max_by_risk.get(risk_level, 10.0))

    category = priority_category(score)
    return {
        "priority_score": round(score, 1),
        "priority_category": category,
        "raw_priority": round(raw_priority, 3),
    }


def recommend_department(
    age: int,
    symptoms_raw: str,
    conditions_raw: str,
    risk_level: str,
    blood_pressure: str,
    heart_rate: int,
    temperature: float,
    priority_score: float,
) -> Dict[str, str]:
    symptoms = _parse_multi(symptoms_raw)
    conditions = _parse_multi(conditions_raw)
    bp_s, bp_d = _bp_parts(blood_pressure)

    if age < 18:
        return {
            "department": "Pediatrics",
            "reason": "Patient age is below 18; pediatric-first routing policy",
            "urgency": priority_category(priority_score),
            "alternative_department": "Emergency Department if condition worsens",
            "routing_logic": "rule_based",
        }

    emergency_flags = [
        risk_level == "High",
        priority_score >= 9.0,
        any(symptom in EMERGENCY_SYMPTOMS for symptom in symptoms),
        bp_s > 180 or bp_d > 110,
        heart_rate > 140 or heart_rate < 45,
        temperature > 103.5,
        ("confusion" in symptoms and len(symptoms) >= 2),
        ("chest pain" in symptoms and (bp_s >= 140 or heart_rate >= 100) and age > 60),
    ]
    if any(emergency_flags):
        return {
            "department": "Emergency Department",
            "reason": "Critical condition requiring immediate emergency attention",
            "urgency": "Critical",
            "alternative_department": "Cardiology or Neurology after stabilization",
            "routing_logic": "rule_based",
        }

    if CARDIOLOGY_SYMPTOMS.intersection(symptoms) or "heart disease" in conditions:
        return {
            "department": "Cardiology",
            "reason": "Cardiac symptom profile and/or cardiac comorbidity",
            "urgency": priority_category(priority_score),
            "alternative_department": "Emergency Department if symptoms escalate",
            "routing_logic": "rule_based",
        }

    if NEUROLOGY_SYMPTOMS.intersection(symptoms) or "stroke history" in conditions:
        return {
            "department": "Neurology",
            "reason": "Neurological symptom pattern suggests specialist evaluation",
            "urgency": priority_category(priority_score),
            "alternative_department": "Emergency Department for acute deterioration",
            "routing_logic": "rule_based",
        }

    if PULMONOLOGY_SYMPTOMS.intersection(symptoms) or "asthma" in conditions or "COPD" in conditions:
        return {
            "department": "Pulmonology",
            "reason": "Respiratory symptoms or respiratory comorbidity indicate pulmonology routing",
            "urgency": priority_category(priority_score),
            "alternative_department": "Emergency Department for severe breathing distress",
            "routing_logic": "rule_based",
        }

    if ORTHOPEDICS_SYMPTOMS.intersection(symptoms):
        return {
            "department": "Orthopedics",
            "reason": "Musculoskeletal complaint cluster routed to orthopedics",
            "urgency": priority_category(priority_score),
            "alternative_department": "General Medicine if generalized symptoms emerge",
            "routing_logic": "rule_based",
        }

    if GASTRO_SYMPTOMS.intersection(symptoms):
        return {
            "department": "Gastroenterology",
            "reason": "Gastrointestinal symptom profile indicates GI specialist review",
            "urgency": priority_category(priority_score),
            "alternative_department": "Emergency Department if severe dehydration or instability",
            "routing_logic": "rule_based",
        }

    return {
        "department": "General Medicine",
        "reason": "No clear specialist dominance; default safe routing",
        "urgency": priority_category(priority_score),
        "alternative_department": "Specialty referral after physician screening",
        "routing_logic": "rule_based",
    }


def estimate_wait_time(department: str, queue_position: int, priority_score: float) -> str:
    base_wait = BASE_WAIT_BY_DEPARTMENT.get(department, 50)
    score = max(priority_score, 1.0)
    computed_wait = int(round(base_wait * (queue_position / score)))

    category = priority_category(score)
    if category == "Critical":
        computed_wait = min(computed_wait, 10)
    elif category == "Urgent":
        computed_wait = min(max(computed_wait, 10), 20)
    elif category == "High":
        computed_wait = min(max(computed_wait, 20), 40)
    elif category == "Standard":
        computed_wait = min(max(computed_wait, 40), 60)
    else:
        computed_wait = max(computed_wait, 60)

    return f"{computed_wait} minutes"


def build_triage_decision(
    patient: Dict[str, object],
    queue_position: int = 1,
    ehr_history_score: float = 0.0,
) -> Dict[str, object]:
    age = int(patient["Age"])
    symptoms_raw = str(patient.get("Symptoms", ""))
    conditions_raw = str(patient.get("Pre-Existing Conditions", ""))
    risk_level = str(patient["Risk_Level"])
    blood_pressure = str(patient["Blood Pressure"])
    heart_rate = int(patient["Heart Rate"])
    temperature = float(patient["Temperature"])
    condition_count = len(_parse_multi(conditions_raw))
    symptoms = _parse_multi(symptoms_raw)
    bp_s, bp_d = _bp_parts(blood_pressure)

    priority = calculate_priority(
        risk_level=risk_level,
        age=age,
        symptoms=symptoms,
        bp_s=bp_s,
        bp_d=bp_d,
        heart_rate=heart_rate,
        temperature=temperature,
        condition_count=condition_count,
        ehr_history_score=ehr_history_score,
    )

    recommendation = recommend_department(
        age=age,
        symptoms_raw=symptoms_raw,
        conditions_raw=conditions_raw,
        risk_level=risk_level,
        blood_pressure=blood_pressure,
        heart_rate=heart_rate,
        temperature=temperature,
        priority_score=float(priority["priority_score"]),
    )

    wait_time = estimate_wait_time(
        department=recommendation["department"],
        queue_position=queue_position,
        priority_score=float(priority["priority_score"]),
    )

    reasoning = (
        f"{risk_level} risk profile with {len(symptoms)} symptom(s), "
        f"priority {priority['priority_score']}/10, routed to {recommendation['department']}"
    )

    return {
        "priority_score": priority["priority_score"],
        "priority_category": priority["priority_category"],
        "queue_position": queue_position,
        "estimated_wait_time": wait_time,
        "reasoning": reasoning,
        "department": recommendation["department"],
        "department_reason": recommendation["reason"],
        "urgency": recommendation["urgency"],
        "alternative_department": recommendation["alternative_department"],
        "routing_logic": recommendation["routing_logic"],
        "arrival_time": datetime.now(UTC).isoformat(),
    }


def sort_priority_queue(patients: List[Dict[str, object]]) -> List[Dict[str, object]]:
    return sorted(
        patients,
        key=lambda x: (
            -float(x.get("priority_score", 0)),
            -int(x.get("Age", 0)),
            str(x.get("arrival_time", "")),
        ),
    )
