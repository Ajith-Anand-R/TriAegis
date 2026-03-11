from __future__ import annotations

from typing import Dict, List, Set


INTERACTION_RULES = {
    frozenset({"warfarin", "aspirin"}): {
        "severity": "high",
        "message": "Increased bleeding risk when warfarin is combined with aspirin",
    },
    frozenset({"warfarin", "ibuprofen"}): {
        "severity": "high",
        "message": "Major bleeding risk with warfarin and ibuprofen",
    },
    frozenset({"lisinopril", "spironolactone"}): {
        "severity": "medium",
        "message": "Hyperkalemia risk with lisinopril and spironolactone",
    },
    frozenset({"metformin", "contrast dye"}): {
        "severity": "medium",
        "message": "Potential lactic acidosis risk around contrast exposure with metformin",
    },
}

CONDITION_CONTRAINDICATIONS = {
    "kidney disease": {
        "nsaids": "Avoid routine NSAID use in chronic kidney disease unless clinician-approved",
    },
    "asthma": {
        "ibuprofen": "NSAID sensitivity may worsen asthma in susceptible patients",
    },
    "heart disease": {
        "diclofenac": "Certain NSAIDs can increase cardiovascular risk in heart disease",
    },
}


def _parse_tokens(raw: str | None) -> List[str]:
    if not raw:
        return []
    tokens = [item.strip().lower() for item in str(raw).split(",") if item.strip()]
    return sorted(set(tokens))


def _classify_medication_classes(meds: Set[str]) -> Set[str]:
    classes: Set[str] = set()
    nsaids = {"ibuprofen", "diclofenac", "naproxen"}
    if meds.intersection(nsaids):
        classes.add("nsaids")
    return classes


def evaluate_medication_safety(
    medications_raw: str,
    allergies_raw: str,
    conditions_raw: str = "",
) -> Dict[str, object]:
    medications = _parse_tokens(medications_raw)
    allergies = _parse_tokens(allergies_raw)
    conditions = _parse_tokens(conditions_raw)

    med_set = set(medications)
    allergy_set = set(allergies)

    interaction_alerts: List[Dict[str, str]] = []
    allergy_alerts: List[Dict[str, str]] = []
    contraindication_alerts: List[Dict[str, str]] = []

    med_list = sorted(med_set)
    for i in range(len(med_list)):
        for j in range(i + 1, len(med_list)):
            pair = frozenset({med_list[i], med_list[j]})
            rule = INTERACTION_RULES.get(pair)
            if rule:
                interaction_alerts.append(
                    {
                        "type": "interaction",
                        "severity": str(rule["severity"]),
                        "medications": ", ".join(sorted(pair)),
                        "message": str(rule["message"]),
                    }
                )

    for med in med_set:
        if med in allergy_set:
            allergy_alerts.append(
                {
                    "type": "allergy",
                    "severity": "high",
                    "medication": med,
                    "message": f"Recorded allergy conflict for medication: {med}",
                }
            )

    med_classes = _classify_medication_classes(med_set)
    for condition in conditions:
        rules = CONDITION_CONTRAINDICATIONS.get(condition, {})
        for token, message in rules.items():
            if token in med_set or token in med_classes:
                contraindication_alerts.append(
                    {
                        "type": "contraindication",
                        "severity": "medium",
                        "condition": condition,
                        "trigger": token,
                        "message": message,
                    }
                )

    high_severity = any(alert.get("severity") == "high" for alert in (interaction_alerts + allergy_alerts))
    medium_severity = any(alert.get("severity") == "medium" for alert in (interaction_alerts + allergy_alerts + contraindication_alerts))

    if high_severity:
        overall = "high"
        recommendation = "Medication plan requires immediate clinician and pharmacy review before administration"
    elif medium_severity:
        overall = "medium"
        recommendation = "Medication plan should be reviewed by a clinician before routine administration"
    else:
        overall = "low"
        recommendation = "No known high-risk conflicts detected from configured screening rules"

    return {
        "overall_risk": overall,
        "recommendation": recommendation,
        "medication_count": len(med_set),
        "allergy_count": len(allergy_set),
        "condition_count": len(conditions),
        "interaction_alerts": interaction_alerts,
        "allergy_alerts": allergy_alerts,
        "contraindication_alerts": contraindication_alerts,
    }
