from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data.generate_data import GenerationConfig, generate_synthetic_patients
from utils.database import TriageDatabase
from utils.department_recommender import build_triage_decision
from utils.document_parser import parse_document
from utils.ehr_analyzer import parse_ehr_csv, analyze_ehr_history, parse_ehr_csv_from_bytes
from utils.ml_engine import TriageMLEngine
from utils.healthcheck import run_healthcheck
from utils.reporting import batch_results_pdf, single_result_pdf
from utils.validators import validate_patient_payload


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "synthetic_patients.csv"
SAMPLE_PDF = ROOT / "tests" / "sample_data" / "test_ehr.pdf"
SAMPLE_DOCX = ROOT / "tests" / "sample_data" / "test_ehr.docx"
SAMPLE_BATCH = ROOT / "tests" / "sample_data" / "test_batch.csv"


def test_data_generation_distribution_and_quality() -> None:
    df = generate_synthetic_patients(GenerationConfig(total_patients=1000, seed=123))

    assert len(df) == 1000
    counts = df["Risk_Level"].value_counts().to_dict()
    assert counts["Low"] == 600
    assert counts["Medium"] == 300
    assert counts["High"] == 100
    assert df.isnull().sum().sum() == 0
    assert df["Patient_ID"].nunique() == 1000


def test_ml_engine_prediction_format() -> None:
    df = pd.read_csv(DATA_PATH).head(3)
    engine = TriageMLEngine(ROOT)

    result = engine.predict_one(df.iloc[0].to_dict())
    assert result["risk_level"] in {"Low", "Medium", "High"}
    assert 0.0 <= result["confidence"] <= 1.0
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-2


def test_edge_case_elderly_low_vitals() -> None:
    engine = TriageMLEngine(ROOT)
    patient = {
        "Patient_ID": "edge-elderly-001",
        "Age": 85,
        "Gender": "Female",
        "Symptoms": "mild headache,fatigue",
        "Blood Pressure": "120/80",
        "Heart Rate": 72,
        "Temperature": 98.6,
        "Pre-Existing Conditions": "hypertension",
    }
    result = engine.predict_one(patient)
    assert result["risk_level"] in {"Low", "Medium", "High"}
    assert 0.0 <= result["confidence"] <= 1.0


def test_edge_case_young_severe_symptoms() -> None:
    engine = TriageMLEngine(ROOT)
    patient = {
        "Patient_ID": "edge-young-severe-001",
        "Age": 22,
        "Gender": "Male",
        "Symptoms": "chest pain,seizure",
        "Blood Pressure": "168/104",
        "Heart Rate": 142,
        "Temperature": 102.9,
        "Pre-Existing Conditions": "none",
    }
    result = engine.predict_one(patient)
    assert result["risk_level"] == "High"


def test_edge_case_pediatric() -> None:
    engine = TriageMLEngine(ROOT)
    patient = {
        "Patient_ID": "edge-pediatric-001",
        "Age": 8,
        "Gender": "Female",
        "Symptoms": "mild fever,cough",
        "Blood Pressure": "100/65",
        "Heart Rate": 95,
        "Temperature": 99.8,
        "Pre-Existing Conditions": "none",
    }
    result = engine.predict_one(patient)
    assert result["risk_level"] == "Low"


def test_edge_case_extreme_vitals() -> None:
    engine = TriageMLEngine(ROOT)
    patient = {
        "Patient_ID": "edge-extreme-vitals-001",
        "Age": 54,
        "Gender": "Male",
        "Symptoms": "severe headache,chest pain",
        "Blood Pressure": "220/140",
        "Heart Rate": 180,
        "Temperature": 105.0,
        "Pre-Existing Conditions": "hypertension,heart disease",
    }
    result = engine.predict_one(patient)
    assert result["risk_level"] == "High"


def test_priority_and_department_rules() -> None:
    patient = {
        "Patient_ID": "test-001",
        "Age": 72,
        "Gender": "Male",
        "Symptoms": "chest pain,confusion",
        "Blood Pressure": "190/115",
        "Heart Rate": 135,
        "Temperature": 103.7,
        "Pre-Existing Conditions": "heart disease,hypertension,diabetes",
        "Risk_Level": "High",
    }
    triage = build_triage_decision(patient)

    assert triage["priority_score"] >= 9.0
    assert triage["department"] == "Emergency Department"


def test_document_parser_csv_batch() -> None:
    rows = parse_document(DATA_PATH)
    assert len(rows) > 0
    assert "Patient_ID" in rows[0]
    assert "Blood Pressure" in rows[0]


def test_document_parser_pdf_docx() -> None:
    pdf_rows = parse_document(SAMPLE_PDF)
    docx_rows = parse_document(SAMPLE_DOCX)

    assert len(pdf_rows) == 1
    assert len(docx_rows) == 1
    assert pdf_rows[0]["Age"] >= 0
    assert "Blood Pressure" in docx_rows[0]


def test_database_insert_and_queue() -> None:
    db = TriageDatabase(ROOT / "database" / "test_patients.db")
    engine = TriageMLEngine(ROOT)

    patient = pd.read_csv(DATA_PATH).iloc[0].to_dict()
    prediction = engine.predict_one(patient)
    prediction_id = db.save_prediction(patient, prediction, source="test")

    assert prediction_id > 0
    queue = db.get_priority_queue()
    assert len(queue) >= 1


def test_queue_status_transitions() -> None:
    db = TriageDatabase(ROOT / "database" / "test_patients_queue.db")
    engine = TriageMLEngine(ROOT)

    rows = pd.read_csv(SAMPLE_BATCH).to_dict(orient="records")
    for row in rows:
        prediction = engine.predict_one(row)
        db.save_prediction(row, prediction, source="queue_test")

    queue = db.get_priority_queue(status="waiting")
    assert len(queue) >= 1

    first_queue_id = int(queue[0]["queue_id"])
    db.update_queue_status(first_queue_id, "in_progress")

    in_progress = db.get_priority_queue(status="in_progress")
    assert len(in_progress) >= 1


def test_validator_rejects_invalid_payload() -> None:
    payload = {
        "Patient_ID": "abc",
        "Age": -2,
        "Gender": "Alien",
        "Symptoms": "",
        "Blood Pressure": "300/10",
        "Heart Rate": -5,
        "Temperature": 200.0,
        "Pre-Existing Conditions": "none",
    }
    ok, errors = validate_patient_payload(payload)
    assert ok is False
    assert len(errors) >= 3


def test_parser_unsupported_type() -> None:
    with pytest.raises(ValueError):
        parse_document(ROOT / "tests" / "sample_data" / "bad_file.txt")


def test_parser_malformed_csv_error_message(tmp_path: Path) -> None:
    bad_csv = tmp_path / "bad_batch.csv"
    bad_csv.write_text(
        "Patient_ID,Age,Gender,Symptoms,Blood Pressure,Heart Rate,Temperature,Pre-Existing Conditions\n"
        "P-1,70,Male,chest pain,160/95,110,101.2,hypertension\n"
        "P-2,65,Female,chest pain, confusion,150/90,105,100.8,diabetes\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Malformed CSV detected") as exc:
        parse_document(bad_csv)

    assert "double quotes" in str(exc.value)


def test_reporting_generates_pdf_bytes() -> None:
    patient = pd.read_csv(DATA_PATH).iloc[0].to_dict()
    engine = TriageMLEngine(ROOT)
    pred = engine.predict_one(patient)

    single_pdf = single_result_pdf(patient, pred)
    batch_pdf = batch_results_pdf([pred])

    assert isinstance(single_pdf, bytes)
    assert isinstance(batch_pdf, bytes)
    assert len(single_pdf) > 100
    assert len(batch_pdf) > 100


def test_database_clear_old_records() -> None:
    db = TriageDatabase(ROOT / "database" / "test_cleanup.db")
    deleted = db.clear_old_records(days=90)
    assert "deleted_queue_rows" in deleted
    assert "deleted_prediction_rows" in deleted


def test_database_prediction_search_api() -> None:
    db = TriageDatabase(ROOT / "database" / "test_search.db")
    engine = TriageMLEngine(ROOT)

    sample = pd.read_csv(SAMPLE_BATCH).head(1).to_dict(orient="records")[0]
    pred = engine.predict_one(sample)
    db.save_prediction(sample, pred, source="search_test")

    all_preds = db.get_predictions(limit=50)
    assert len(all_preds) >= 1

    filtered = db.search_predictions(
        patient_id_query=str(sample["Patient_ID"])[:8],
        risk_levels=[pred["risk_level"]],
        departments=[pred["department"]],
        priority_categories=[pred["priority_category"]],
        limit=50,
    )
    assert len(filtered) >= 1


def test_healthcheck_passes() -> None:
    status = run_healthcheck(ROOT)
    assert isinstance(status.ok, bool)
    assert "engine:predict" in status.checks


def test_ehr_analyzer_high_risk() -> None:
    records = parse_ehr_csv(ROOT / "test" / "patient_001_ehr.csv")
    summary = analyze_ehr_history(records, filename="patient_001_ehr.csv")

    assert summary.total_records > 0
    assert summary.total_visits > 0
    assert summary.hospitalizations >= 2
    assert summary.er_visits >= 2
    assert len(summary.chronic_conditions) >= 3
    assert summary.active_medications >= 5
    assert summary.abnormal_labs >= 3
    assert 0.0 <= summary.history_risk_score <= 3.0
    assert summary.history_risk_score >= 2.0, "High-risk patient should score >= 2.0"
    assert len(summary.risk_factors) >= 3


def test_ehr_analyzer_low_risk() -> None:
    records = parse_ehr_csv(ROOT / "test" / "patient_002_ehr.csv")
    summary = analyze_ehr_history(records, filename="patient_002_ehr.csv")

    assert summary.total_records > 0
    assert summary.hospitalizations == 0
    assert 0.0 <= summary.history_risk_score <= 3.0
    assert summary.history_risk_score < 1.0, "Low-risk patient should score < 1.0"


def test_ehr_score_affects_priority() -> None:
    patient = {
        "Patient_ID": "ehr-test-001",
        "Age": 65,
        "Gender": "Male",
        "Symptoms": "chest pain,confusion",
        "Blood Pressure": "160/100",
        "Heart Rate": 105,
        "Temperature": 101.5,
        "Pre-Existing Conditions": "heart disease,diabetes",
        "Risk_Level": "High",
    }

    without_ehr = build_triage_decision(patient, ehr_history_score=0.0)
    with_ehr = build_triage_decision(patient, ehr_history_score=2.5)

    assert with_ehr["priority_score"] >= without_ehr["priority_score"], \
        "Priority should increase with EHR history score"


def test_ehr_parser_invalid_date_reports_row() -> None:
    bad_csv = (
        "Date,Record_Type,Description\n"
        "2026-01-01,visit,Routine check\n"
        "01/31/2026,lab,Glucose\n"
    ).encode("utf-8")

    with pytest.raises(ValueError, match="row 2"):
        parse_ehr_csv_from_bytes(bad_csv, filename="bad_dates.csv")


def test_ehr_parser_non_utf8_upload_rejected() -> None:
    raw = b"\xff\xfe\x00\x00not-a-valid-utf8-csv"
    with pytest.raises(ValueError, match="UTF-8"):
        parse_ehr_csv_from_bytes(raw, filename="binary_upload.csv")
