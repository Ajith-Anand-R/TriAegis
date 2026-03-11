from __future__ import annotations

from utils.privacy import mask_patient_identifier, redact_record, redact_records


def test_mask_patient_identifier() -> None:
    assert mask_patient_identifier("PAT-123456") == "PA***56"
    assert mask_patient_identifier("AB") == "***"


def test_redact_record_masks_sensitive_fields() -> None:
    row = {
        "patient_id": "P001234",
        "risk_level": "High",
        "notes": "private",
        "department": "Emergency Department",
    }

    redacted = redact_record(row)

    assert redacted["patient_id"] != row["patient_id"]
    assert redacted["notes"] == "[REDACTED]"
    assert redacted["risk_level"] == "High"


def test_redact_records_bulk() -> None:
    rows = [
        {"Patient_ID": "P12345", "value": 1},
        {"Patient_ID": "P67890", "value": 2},
    ]

    output = redact_records(rows)

    assert len(output) == 2
    assert output[0]["Patient_ID"] != "P12345"
    assert output[1]["Patient_ID"] != "P67890"
