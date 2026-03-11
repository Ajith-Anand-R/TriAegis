from __future__ import annotations

import re
from typing import Dict, List, Tuple


def validate_age(age: int) -> Tuple[bool, str]:
    if not isinstance(age, int):
        return False, "Age must be an integer"
    if age < 0 or age > 120:
        return False, "Age must be between 0 and 120"
    return True, ""


def validate_gender(gender: str) -> Tuple[bool, str]:
    allowed = {"Male", "Female", "Other", "Unknown"}
    if gender not in allowed:
        return False, f"Gender must be one of: {', '.join(sorted(allowed))}"
    return True, ""


def validate_blood_pressure(value: str) -> Tuple[bool, str]:
    if not re.match(r"^\d{2,3}/\d{2,3}$", str(value).strip()):
        return False, "Blood Pressure must be in systolic/diastolic format (e.g., 120/80)"
    systolic, diastolic = [int(part) for part in value.split("/")]
    if not (70 <= systolic <= 260):
        return False, "Systolic BP out of acceptable range (70-260)"
    if not (40 <= diastolic <= 160):
        return False, "Diastolic BP out of acceptable range (40-160)"
    return True, ""


def validate_heart_rate(hr: int) -> Tuple[bool, str]:
    if not isinstance(hr, int):
        return False, "Heart Rate must be an integer"
    if hr < 30 or hr > 220:
        return False, "Heart Rate must be between 30 and 220 bpm"
    return True, ""


def validate_temperature(temp: float) -> Tuple[bool, str]:
    if temp < 90.0 or temp > 112.0:
        return False, "Temperature must be between 90.0°F and 112.0°F"
    return True, ""


def sanitize_text_list(value: str) -> str:
    parts = [token.strip() for token in str(value).split(",") if token.strip()]
    clean_parts = [re.sub(r"[^a-zA-Z0-9\- ]", "", token) for token in parts]
    if not clean_parts:
        return "none"
    return ",".join(clean_parts)


def validate_patient_payload(payload: Dict[str, object]) -> Tuple[bool, List[str]]:
    errors: List[str] = []

    age_ok, age_msg = validate_age(int(payload.get("Age", -1)))
    if not age_ok:
        errors.append(age_msg)

    gender_ok, gender_msg = validate_gender(str(payload.get("Gender", "Unknown")))
    if not gender_ok:
        errors.append(gender_msg)

    bp_ok, bp_msg = validate_blood_pressure(str(payload.get("Blood Pressure", "")))
    if not bp_ok:
        errors.append(bp_msg)

    hr_ok, hr_msg = validate_heart_rate(int(payload.get("Heart Rate", -1)))
    if not hr_ok:
        errors.append(hr_msg)

    temp_ok, temp_msg = validate_temperature(float(payload.get("Temperature", -1.0)))
    if not temp_ok:
        errors.append(temp_msg)

    return len(errors) == 0, errors
