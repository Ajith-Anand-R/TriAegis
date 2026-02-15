from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path
from typing import Dict, List

import pandas as pd
from pandas.errors import ParserError
from docx import Document
from PyPDF2 import PdfReader


LOGGER = logging.getLogger(__name__)


STANDARD_FIELDS = [
    "Patient_ID",
    "Age",
    "Gender",
    "Symptoms",
    "Blood Pressure",
    "Heart Rate",
    "Temperature",
    "Pre-Existing Conditions",
]

FIELD_ALIASES = {
    "patient_id": "Patient_ID",
    "patientid": "Patient_ID",
    "patient": "Patient_ID",
    "id": "Patient_ID",
    "age": "Age",
    "gender": "Gender",
    "sex": "Gender",
    "symptoms": "Symptoms",
    "symptom": "Symptoms",
    "bp": "Blood Pressure",
    "blood_pressure": "Blood Pressure",
    "bloodpressure": "Blood Pressure",
    "blood pressure": "Blood Pressure",
    "heart_rate": "Heart Rate",
    "heartrate": "Heart Rate",
    "heart rate": "Heart Rate",
    "hr": "Heart Rate",
    "temperature": "Temperature",
    "temp": "Temperature",
    "pre_existing_conditions": "Pre-Existing Conditions",
    "pre-existing conditions": "Pre-Existing Conditions",
    "preexistingconditions": "Pre-Existing Conditions",
    "medical_history": "Pre-Existing Conditions",
    "conditions": "Pre-Existing Conditions",
}

DEFAULTS: Dict[str, str | int | float] = {
    "Patient_ID": "",
    "Age": 40,
    "Gender": "Unknown",
    "Symptoms": "",
    "Blood Pressure": "120/80",
    "Heart Rate": 70,
    "Temperature": 98.6,
    "Pre-Existing Conditions": "none",
}

PATTERNS = {
    "Age": re.compile(r"age[:\s]+(\d{1,3})", flags=re.IGNORECASE),
    "Gender": re.compile(r"gender[:\s]+(male|female|other|m|f)", flags=re.IGNORECASE),
    "Blood Pressure": re.compile(r"b\.?p\.?[:\s]+(\d{2,3})\s*/\s*(\d{2,3})", flags=re.IGNORECASE),
    "Heart Rate": re.compile(
        r"(?:heart\s*rate|hr)[:\s]+(\d{2,3})",
        flags=re.IGNORECASE,
    ),
    "Temperature": re.compile(r"temp(?:erature)?[:\s]+([\d.]+)", flags=re.IGNORECASE),
    "Symptoms": re.compile(
        r"symptoms?[:\s]+(.+?)(?:\n\n|\nVital|\n[A-Z][A-Za-z\s]+:|$)",
        flags=re.IGNORECASE | re.DOTALL,
    ),
    "Pre-Existing Conditions": re.compile(
        r"(?:medical\s+history|pre[-\s]existing|conditions?)[:\s]+(.+?)(?:\n\n|\n[A-Z][A-Za-z\s]+:|$)",
        flags=re.IGNORECASE | re.DOTALL,
    ),
}


def _normalize_text_block(value: str) -> str:
    compact = re.sub(r"\s+", " ", value).strip(" ,.;\n\t")
    return compact


def _normalize_gender(value: str) -> str:
    lower = value.lower().strip()
    if lower == "m":
        return "Male"
    if lower == "f":
        return "Female"
    if lower in {"male", "female", "other"}:
        return lower.capitalize()
    return "Unknown"


def _normalize_bp(value: str) -> str:
    match = re.search(r"(\d{2,3})\s*/\s*(\d{2,3})", value)
    if not match:
        return str(DEFAULTS["Blood Pressure"])
    systolic = max(70, min(260, int(match.group(1))))
    diastolic = max(40, min(160, int(match.group(2))))
    return f"{systolic}/{diastolic}"


def _to_int(value: object, default: int) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def _to_float(value: object, default: float) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _validate_and_fill(record: Dict[str, object]) -> Dict[str, object]:
    patient_id = str(record.get("Patient_ID", "")).strip() or str(uuid.uuid4())

    age_default = int(DEFAULTS["Age"])
    age_value = max(0, min(120, _to_int(record.get("Age", age_default), age_default)))

    gender_value = _normalize_gender(str(record.get("Gender", DEFAULTS["Gender"])))
    bp_value = _normalize_bp(str(record.get("Blood Pressure", DEFAULTS["Blood Pressure"])))

    heart_rate_default = int(DEFAULTS["Heart Rate"])
    heart_rate_value = max(
        30,
        min(220, _to_int(record.get("Heart Rate", heart_rate_default), heart_rate_default)),
    )

    temperature_default = float(DEFAULTS["Temperature"])
    temperature_value = round(
        max(90.0, min(112.0, _to_float(record.get("Temperature", temperature_default), temperature_default))),
        1,
    )

    symptoms_value = _normalize_text_block(str(record.get("Symptoms", "")))
    conditions = _normalize_text_block(str(record.get("Pre-Existing Conditions", "")))
    conditions_value = conditions if conditions else "none"

    return {
        "Patient_ID": patient_id,
        "Age": age_value,
        "Gender": gender_value,
        "Symptoms": symptoms_value,
        "Blood Pressure": bp_value,
        "Heart Rate": heart_rate_value,
        "Temperature": temperature_value,
        "Pre-Existing Conditions": conditions_value,
    }


def _canonical_column(column_name: str) -> str | None:
    key = re.sub(r"[^a-z0-9_ ]", "", column_name.strip().lower())
    key = key.replace("-", " ").replace("__", "_").strip()
    key_no_space = key.replace(" ", "")
    if key in FIELD_ALIASES:
        return FIELD_ALIASES[key]
    if key_no_space in FIELD_ALIASES:
        return FIELD_ALIASES[key_no_space]
    return None


def parse_csv(file_path: str | Path) -> List[Dict[str, object]]:
    csv_path = Path(file_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    try:
        dataframe = pd.read_csv(csv_path)
    except ParserError as exc:
        message = str(exc)
        line_match = re.search(r"line\s+(\d+)", message, flags=re.IGNORECASE)
        line_hint = f" on line {line_match.group(1)}" if line_match else ""
        expected_columns = ", ".join(STANDARD_FIELDS)
        raise ValueError(
            "Malformed CSV detected"
            f"{line_hint}. If a field contains commas (for example symptoms or conditions), "
            "wrap the field in double quotes. "
            f"Expected columns: {expected_columns}"
        ) from exc
    except UnicodeDecodeError as exc:
        raise ValueError("Unable to decode CSV. Please upload a UTF-8 encoded file") from exc

    if dataframe.empty:
        return []

    rename_map = {}
    for col in dataframe.columns:
        canonical = _canonical_column(col)
        if canonical:
            rename_map[col] = canonical
    dataframe = dataframe.rename(columns=rename_map)

    records: List[Dict[str, object]] = []
    for _, row in dataframe.iterrows():
        raw_record: Dict[str, object] = {}
        for field in STANDARD_FIELDS:
            if field in dataframe.columns and pd.notna(row.get(field)):
                raw_record[field] = row[field]
        records.append(_validate_and_fill(raw_record))
    return records


def parse_spreadsheet(file_path: str | Path) -> List[Dict[str, object]]:
    spreadsheet_path = Path(file_path)
    if not spreadsheet_path.exists():
        raise FileNotFoundError(f"Spreadsheet file not found: {spreadsheet_path}")

    try:
        dataframe = pd.read_excel(spreadsheet_path)
    except Exception as exc:
        raise ValueError(f"Failed to parse spreadsheet file: {spreadsheet_path}. Error: {exc}") from exc

    if dataframe.empty:
        return []

    rename_map = {}
    for col in dataframe.columns:
        canonical = _canonical_column(str(col))
        if canonical:
            rename_map[col] = canonical
    dataframe = dataframe.rename(columns=rename_map)

    records: List[Dict[str, object]] = []
    for _, row in dataframe.iterrows():
        raw_record: Dict[str, object] = {}
        for field in STANDARD_FIELDS:
            if field in dataframe.columns and pd.notna(row.get(field)):
                raw_record[field] = row[field]
        records.append(_validate_and_fill(raw_record))
    return records


def _extract_pdf_text(file_path: str | Path) -> str:
    pdf_path = Path(file_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    try:
        reader = PdfReader(str(pdf_path))
        page_texts = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:
        raise ValueError(f"Failed to parse PDF file: {pdf_path}. Error: {exc}") from exc

    text = "\n".join(page_texts).strip()
    if not text:
        raise ValueError("PDF appears empty or unreadable")
    return text


def _extract_docx_text(file_path: str | Path) -> str:
    docx_path = Path(file_path)
    if not docx_path.exists():
        raise FileNotFoundError(f"DOCX file not found: {docx_path}")

    try:
        document = Document(str(docx_path))
    except Exception as exc:
        raise ValueError(f"Failed to parse DOCX file: {docx_path}. Error: {exc}") from exc

    text = "\n".join(paragraph.text for paragraph in document.paragraphs).strip()
    if not text:
        raise ValueError("DOCX appears empty or unreadable")
    return text


def _extract_record_from_text(text: str) -> Dict[str, object]:
    extracted: Dict[str, object] = {}

    age_match = PATTERNS["Age"].search(text)
    if age_match:
        extracted["Age"] = age_match.group(1)

    gender_match = PATTERNS["Gender"].search(text)
    if gender_match:
        extracted["Gender"] = gender_match.group(1)

    bp_match = PATTERNS["Blood Pressure"].search(text)
    if bp_match:
        extracted["Blood Pressure"] = f"{bp_match.group(1)}/{bp_match.group(2)}"

    hr_match = PATTERNS["Heart Rate"].search(text)
    if hr_match:
        extracted["Heart Rate"] = hr_match.group(1)

    temp_match = PATTERNS["Temperature"].search(text)
    if temp_match:
        extracted["Temperature"] = temp_match.group(1)

    symptoms_match = PATTERNS["Symptoms"].search(text)
    if symptoms_match:
        extracted["Symptoms"] = _normalize_text_block(symptoms_match.group(1).replace("\n", " "))

    conditions_match = PATTERNS["Pre-Existing Conditions"].search(text)
    if conditions_match:
        extracted["Pre-Existing Conditions"] = _normalize_text_block(conditions_match.group(1).replace("\n", " "))

    extracted["Patient_ID"] = str(uuid.uuid4())
    return _validate_and_fill(extracted)


def parse_pdf(file_path: str | Path) -> Dict[str, object]:
    text = _extract_pdf_text(file_path)
    return _extract_record_from_text(text)


def parse_docx(file_path: str | Path) -> Dict[str, object]:
    text = _extract_docx_text(file_path)
    return _extract_record_from_text(text)


def parse_document(file_path: str | Path) -> List[Dict[str, object]]:
    path_obj = Path(file_path)
    suffix = path_obj.suffix.lower()

    if suffix == ".csv":
        return parse_csv(path_obj)
    if suffix in {".xlsx", ".xls"}:
        return parse_spreadsheet(path_obj)
    if suffix == ".pdf":
        return [parse_pdf(path_obj)]
    if suffix == ".docx":
        return [parse_docx(path_obj)]

    raise ValueError(f"Unsupported file format: {suffix}. Supported formats: .csv, .xlsx, .xls, .pdf, .docx")
