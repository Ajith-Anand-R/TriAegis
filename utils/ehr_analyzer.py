"""EHR / EMR history analyser.

Parses uploaded EHR CSV files and computes a **history risk score**
(0.0 – 3.0) that can be added to the existing triage priority formula,
giving patients with concerning medical histories a higher priority.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Sequence


# ── chronic‑condition keywords matched against Diagnosis_Code descriptions ──
CHRONIC_CONDITION_KEYWORDS = {
    "diabetes", "hypertension", "asthma", "copd", "heart disease",
    "coronary artery disease", "chronic kidney disease", "cancer",
    "heart failure", "stroke", "obesity", "depression",
    "hypothyroidism", "hyperthyroidism", "arthritis", "osteoporosis",
    "osteopenia", "sleep apnea", "fatty liver", "epilepsy",
    "chronic obstructive", "congestive", "pre-diabetes",
    "neuropathy", "atrial fibrillation",
}


@dataclass
class EHRHistorySummary:
    """Compact summary of a patient's historical health record."""

    patient_file: str = ""
    total_records: int = 0
    total_visits: int = 0
    hospitalizations: int = 0
    er_visits: int = 0
    procedures: int = 0
    chronic_conditions: List[str] = field(default_factory=list)
    active_medications: int = 0
    abnormal_labs: int = 0
    total_labs: int = 0
    immunizations: int = 0
    diagnoses: List[str] = field(default_factory=list)
    history_risk_score: float = 0.0
    risk_factors: List[str] = field(default_factory=list)
    score_breakdown: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "patient_file": self.patient_file,
            "total_records": self.total_records,
            "total_visits": self.total_visits,
            "hospitalizations": self.hospitalizations,
            "er_visits": self.er_visits,
            "procedures": self.procedures,
            "chronic_conditions": self.chronic_conditions,
            "active_medications": self.active_medications,
            "abnormal_labs": self.abnormal_labs,
            "total_labs": self.total_labs,
            "immunizations": self.immunizations,
            "diagnoses": self.diagnoses,
            "history_risk_score": round(self.history_risk_score, 2),
            "risk_factors": self.risk_factors,
            "score_breakdown": {k: round(v, 2) for k, v in self.score_breakdown.items()},
        }


# ──────────────────────────────────────────────────────────────────────
#  Parsing
# ──────────────────────────────────────────────────────────────────────

REQUIRED_EHR_COLUMNS = {"Date", "Record_Type"}


def validate_ehr_csv(records: Sequence[Dict[str, str]], fieldnames: Sequence[str] | None = None) -> None:
    if fieldnames is not None:
        missing_columns = [column for column in sorted(REQUIRED_EHR_COLUMNS) if column not in fieldnames]
        if missing_columns:
            raise ValueError(f"Missing required column: {missing_columns[0]}")

    if not records:
        raise ValueError("File is empty")

    for required_column in sorted(REQUIRED_EHR_COLUMNS):
        if required_column not in records[0]:
            raise ValueError(f"Missing required column: {required_column}")

    for row_index, row in enumerate(records, start=1):
        date_value = (row.get("Date") or "").strip()
        if not date_value:
            raise ValueError(f"Missing Date value at row {row_index}")
        try:
            datetime.strptime(date_value, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError(
                f"Invalid date format at row {row_index} in Date column: {date_value}. Use YYYY-MM-DD"
            ) from exc

def parse_ehr_csv(file_path: str | Path) -> List[Dict[str, str]]:
    """Read an EHR CSV and return a list of row‑dicts."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"EHR file not found: {path}")
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a .csv file, got: {path.suffix}")

    records: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            records.append(dict(row))
        validate_ehr_csv(records, fieldnames=reader.fieldnames)
    return records


def parse_ehr_csv_from_bytes(raw_bytes: bytes, filename: str = "upload.csv") -> List[Dict[str, str]]:
    """Parse EHR CSV content directly from bytes (e.g. API upload)."""
    import io
    if not raw_bytes:
        raise ValueError("File is empty")
    try:
        text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Unable to decode {filename}. Please upload a UTF-8 encoded CSV file") from exc
    reader = csv.DictReader(io.StringIO(text))
    rows = [dict(row) for row in reader]
    try:
        validate_ehr_csv(rows, fieldnames=reader.fieldnames)
    except ValueError as exc:
        raise ValueError(f"Invalid EHR CSV ({filename}): {exc}") from exc
    return rows


# ──────────────────────────────────────────────────────────────────────
#  Lab‑result helpers
# ──────────────────────────────────────────────────────────────────────

def _is_abnormal_lab(value_str: str, normal_range: str) -> bool:
    """Heuristic check whether a lab value is outside its normal range."""
    if not value_str or not normal_range:
        return False
    try:
        value = float(value_str)
    except (ValueError, TypeError):
        return False

    rng = normal_range.strip()

    # Handles "<200", ">40" style
    if rng.startswith("<"):
        try:
            return value >= float(rng[1:])
        except ValueError:
            return False
    if rng.startswith(">"):
        try:
            return value <= float(rng[1:])
        except ValueError:
            return False

    # Handles "70-100" style
    if "-" in rng:
        parts = rng.split("-")
        if len(parts) == 2:
            try:
                lo, hi = float(parts[0]), float(parts[1])
                return value < lo or value > hi
            except ValueError:
                pass
    return False


def _extract_chronic_conditions(records: List[Dict[str, str]]) -> List[str]:
    """Extract unique chronic conditions from diagnosis rows."""
    conditions: set[str] = set()
    for row in records:
        record_type = (row.get("Record_Type") or "").strip().lower()
        if record_type != "diagnosis":
            continue
        description = (row.get("Description") or "").lower()
        for keyword in CHRONIC_CONDITION_KEYWORDS:
            if keyword in description:
                conditions.add(keyword.title())
    return sorted(conditions)


def _count_active_medications(records: List[Dict[str, str]]) -> int:
    """Approximate the number of active medications from the latest visit."""
    latest_meds = ""
    latest_date = ""
    for row in records:
        med_field = (row.get("Medication") or "").strip()
        date_field = (row.get("Date") or "").strip()
        if med_field and date_field >= latest_date:
            latest_date = date_field
            latest_meds = med_field
    if not latest_meds:
        return 0
    return len([m.strip() for m in latest_meds.split(",") if m.strip()])


def _count_recent(records: List[Dict[str, str]], record_type: str, months: int = 24) -> int:
    """Count records of a given type within the last *months* months."""
    cutoff = datetime.now() - timedelta(days=months * 30)
    count = 0
    for row in records:
        if (row.get("Record_Type") or "").strip().lower() != record_type:
            continue
        date_str = (row.get("Date") or "").strip()
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            if dt >= cutoff:
                count += 1
        except ValueError:
            count += 1  # be conservative – count if date is unparseable
    return count


# ──────────────────────────────────────────────────────────────────────
#  Scoring
# ──────────────────────────────────────────────────────────────────────

def compute_history_risk_score(summary: EHRHistorySummary) -> float:
    """
    Compute a 0.0–3.0 EHR history risk bonus from the summary.

    Breakdown:
        Hospitalizations (last 2yr):  0.50 each, max 1.50
        ER visits       (last 2yr):  0.30 each, max 0.90
        Chronic conditions:           0.20 each, max 1.00
        Polypharmacy (meds > 3):      0.10 each over 3, max 0.50
        Abnormal labs:                 0.15 each, max 0.60
    """
    hosp_score = min(summary.hospitalizations * 0.50, 1.50)
    er_score = min(summary.er_visits * 0.30, 0.90)
    cond_score = min(len(summary.chronic_conditions) * 0.20, 1.00)
    poly_score = min(max(summary.active_medications - 3, 0) * 0.10, 0.50)
    lab_score = min(summary.abnormal_labs * 0.15, 0.60)

    raw = hosp_score + er_score + cond_score + poly_score + lab_score
    capped = round(min(raw, 3.0), 2)

    summary.score_breakdown = {
        "hospitalizations": hosp_score,
        "er_visits": er_score,
        "chronic_conditions": cond_score,
        "polypharmacy": poly_score,
        "abnormal_labs": lab_score,
    }
    return capped


# ──────────────────────────────────────────────────────────────────────
#  Main analyser
# ──────────────────────────────────────────────────────────────────────

def analyze_ehr_history(records: List[Dict[str, str]], filename: str = "") -> EHRHistorySummary:
    """Analyse a list of EHR record dicts and produce a scored summary."""
    summary = EHRHistorySummary(patient_file=filename)
    summary.total_records = len(records)

    # ── gather counts ──
    for row in records:
        rt = (row.get("Record_Type") or "").strip().lower()
        if rt == "visit":
            summary.total_visits += 1
        elif rt == "hospitalization":
            summary.hospitalizations += 1
        elif rt == "er_visit":
            summary.er_visits += 1
        elif rt == "procedure":
            summary.procedures += 1
        elif rt == "immunization":
            summary.immunizations += 1
        elif rt == "diagnosis":
            desc = (row.get("Description") or "").strip()
            if desc and desc not in summary.diagnoses:
                summary.diagnoses.append(desc)
        elif rt == "lab":
            summary.total_labs += 1
            val = (row.get("Lab_Value") or "").strip()
            rng = (row.get("Lab_Normal_Range") or "").strip()
            if val and rng and _is_abnormal_lab(val, rng):
                summary.abnormal_labs += 1

    summary.chronic_conditions = _extract_chronic_conditions(records)
    summary.active_medications = _count_active_medications(records)

    # ── compute the risk score ──
    summary.history_risk_score = compute_history_risk_score(summary)

    # ── human‑readable risk factors ──
    if summary.hospitalizations >= 2:
        summary.risk_factors.append(
            f"{summary.hospitalizations} hospitalizations indicate frequent acute episodes"
        )
    elif summary.hospitalizations == 1:
        summary.risk_factors.append("1 prior hospitalization on record")

    if summary.er_visits >= 2:
        summary.risk_factors.append(
            f"{summary.er_visits} ER visits suggest recurrent urgent presentations"
        )

    if len(summary.chronic_conditions) >= 3:
        summary.risk_factors.append(
            f"Multi-morbidity: {', '.join(summary.chronic_conditions[:5])}"
        )
    elif summary.chronic_conditions:
        summary.risk_factors.append(
            f"Chronic conditions: {', '.join(summary.chronic_conditions)}"
        )

    if summary.active_medications >= 5:
        summary.risk_factors.append(
            f"Polypharmacy risk: {summary.active_medications} active medications"
        )

    if summary.abnormal_labs >= 3:
        summary.risk_factors.append(
            f"{summary.abnormal_labs} of {summary.total_labs} lab results out of normal range"
        )

    if not summary.risk_factors:
        summary.risk_factors.append("No significant historical risk factors identified")

    return summary
