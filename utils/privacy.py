from __future__ import annotations

from typing import Dict, Iterable, List


SENSITIVE_KEYS = {
    "patient_id",
    "Patient_ID",
    "notes",
    "clinician_notes",
    "address",
    "phone",
    "email",
}


def mask_patient_identifier(value: str) -> str:
    token = str(value or "")
    if len(token) <= 4:
        return "***"
    return f"{token[:2]}***{token[-2:]}"


def redact_record(record: Dict[str, object], extra_sensitive_keys: Iterable[str] | None = None) -> Dict[str, object]:
    sensitive = set(SENSITIVE_KEYS)
    if extra_sensitive_keys:
        sensitive.update(extra_sensitive_keys)

    redacted: Dict[str, object] = {}
    for key, value in record.items():
        if key in {"patient_id", "Patient_ID"}:
            redacted[key] = mask_patient_identifier(str(value))
        elif key in sensitive:
            redacted[key] = "[REDACTED]"
        else:
            redacted[key] = value
    return redacted


def redact_records(records: List[Dict[str, object]], extra_sensitive_keys: Iterable[str] | None = None) -> List[Dict[str, object]]:
    return [redact_record(row, extra_sensitive_keys=extra_sensitive_keys) for row in records]
