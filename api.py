"""FastAPI backend for TriAegis – AI Patient Triage System.

Exposes the platform functionality via REST endpoints consumed by the Next.js frontend.
All existing utils/ modules are reused directly.
"""

from __future__ import annotations

import io
import json
import inspect
import uuid
import urllib.error
import urllib.request
import numpy as np
from functools import wraps
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from auth import authenticate_user, create_access_token, get_current_user, require_role, security_configuration_status
from utils.database import TriageDatabase
from utils.document_parser import parse_document
from utils.explainer import ShapExplainer
from utils.healthcheck import run_healthcheck
from utils.clinical_guardrails import evaluate_red_flags
from utils.clinical_handoff import build_handoff_summary
from utils.medication_safety import evaluate_medication_safety
from utils.followup_planner import build_followup_plan
from utils.privacy import redact_records
from utils.monitoring import build_drift_report
from utils.ml_engine import TriageMLEngine
from utils.reporting import batch_results_pdf, dataframe_to_csv_bytes, single_result_pdf
from utils.validators import sanitize_text_list, validate_patient_payload

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Singleton helpers
# ---------------------------------------------------------------------------
_engine: TriageMLEngine | None = None
_db: TriageDatabase | None = None
_explainer: ShapExplainer | None = None
_sim_state: Dict[str, object] = {
    "current_minute": 0,
    "queue": [],
    "timeline": [],
    "arrived": 0,
    "completed": 0,
}


def get_engine() -> TriageMLEngine:
    global _engine
    if _engine is None:
        _engine = TriageMLEngine(ROOT)
    return _engine


def get_db() -> TriageDatabase:
    global _db
    if _db is None:
        _db = TriageDatabase(ROOT / "database" / "patients.db")
    return _db


def get_explainer() -> ShapExplainer:
    global _explainer
    if _explainer is None:
        _explainer = ShapExplainer.from_project_root(ROOT)
    return _explainer


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class PatientPayload(BaseModel):
    Patient_ID: str = Field(default_factory=lambda: str(uuid.uuid4()))
    Age: int = 40
    Gender: str = "Male"
    Symptoms: str = ""
    Blood_Pressure: str = Field("120/80", alias="Blood Pressure")
    Heart_Rate: int = Field(70, alias="Heart Rate")
    Temperature: float = 98.6
    Pre_Existing_Conditions: str = Field("none", alias="Pre-Existing Conditions")

    model_config = {"populate_by_name": True}

    def to_engine_dict(self) -> Dict[str, object]:
        return {
            "Patient_ID": self.Patient_ID,
            "Age": self.Age,
            "Gender": self.Gender,
            "Symptoms": self.Symptoms,
            "Blood Pressure": self.Blood_Pressure,
            "Heart Rate": self.Heart_Rate,
            "Temperature": self.Temperature,
            "Pre-Existing Conditions": self.Pre_Existing_Conditions,
        }


class QueueStatusUpdate(BaseModel):
    status: str


class HistoryQuery(BaseModel):
    patient_id_query: Optional[str] = None
    risk_levels: Optional[List[str]] = None
    departments: Optional[List[str]] = None
    priority_categories: Optional[List[str]] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    limit: int = 2000


class LoginRequest(BaseModel):
    username: str
    password: str


class SimulationStepRequest(BaseModel):
    minutes: int = 15
    lambda_rate: float = 1.5
    seed: int = 42


class AdminDeleteRequest(BaseModel):
    patient_id: Optional[str] = None
    prediction_id: Optional[int] = None
    queue_id: Optional[int] = None


class OutcomeFeedbackRequest(BaseModel):
    prediction_id: int
    actual_risk_level: str
    final_department: str
    outcome_status: str = "discharged"
    clinician_role: str = "Doctor"
    notes: Optional[str] = None


class DialogueTurn(BaseModel):
    role: str
    content: str


class SymptomDialogueRequest(BaseModel):
    presenting_complaint: str
    transcript: List[DialogueTurn] = Field(default_factory=list)
    known_patient: Dict[str, object] = Field(default_factory=dict)
    model: str = "phi3:mini"
    max_followup_questions: int = Field(default=3, ge=1, le=6)


class SafetyScreenRequest(BaseModel):
    Patient_ID: str = Field(default_factory=lambda: str(uuid.uuid4()))
    Age: int = 40
    Gender: str = "Male"
    Symptoms: str = ""
    Blood_Pressure: str = Field("120/80", alias="Blood Pressure")
    Heart_Rate: int = Field(70, alias="Heart Rate")
    Temperature: float = 98.6
    Pre_Existing_Conditions: str = Field("none", alias="Pre-Existing Conditions")

    model_config = {"populate_by_name": True}

    def to_dict(self) -> Dict[str, object]:
        return {
            "Patient_ID": self.Patient_ID,
            "Age": self.Age,
            "Gender": self.Gender,
            "Symptoms": self.Symptoms,
            "Blood Pressure": self.Blood_Pressure,
            "Heart Rate": self.Heart_Rate,
            "Temperature": self.Temperature,
            "Pre-Existing Conditions": self.Pre_Existing_Conditions,
        }


class MedicationSafetyRequest(BaseModel):
    medications: str = ""
    allergies: str = ""
    conditions: str = ""


class ExportRecordsRequest(BaseModel):
    records: List[dict]


# ---------------------------------------------------------------------------
# Constants used by the frontend
# ---------------------------------------------------------------------------
SYMPTOM_OPTIONS = [
    "chest pain", "severe shortness of breath", "confusion",
    "severe bleeding", "loss of consciousness", "stroke symptoms",
    "severe abdominal pain", "difficulty breathing",
    "severe allergic reaction", "uncontrolled bleeding", "seizure",
    "severe trauma", "moderate shortness of breath", "high fever",
    "persistent vomiting", "severe headache", "palpitations",
    "moderate bleeding", "severe pain", "dizziness", "fainting",
    "dehydration", "abdominal pain", "irregular heartbeat",
    "mild headache", "cough", "cold", "minor pain", "fatigue",
    "nausea", "mild fever", "sore throat", "runny nose", "muscle ache",
    "rash", "minor injury", "constipation", "mild dizziness",
    "back pain", "joint pain", "insomnia", "anxiety", "minor cut", "sprain",
]

CONDITION_OPTIONS = [
    "diabetes", "hypertension", "asthma", "heart disease", "COPD",
    "kidney disease", "obesity", "cancer", "stroke history", "arthritis",
    "high cholesterol", "thyroid disorder", "anxiety", "depression",
]

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="TriAegis API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _error_response(status_code: int, error: str, detail: object) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": error, "detail": detail})


def safe_endpoint(func):
    if inspect.iscoroutinefunction(func):
        @wraps(func)
        async def _async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except HTTPException as exc:
                return _error_response(exc.status_code, "RequestError", exc.detail)
            except Exception as exc:
                return _error_response(500, "InternalServerError", str(exc))
        return _async_wrapper

    @wraps(func)
    def _sync_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except HTTPException as exc:
            return _error_response(exc.status_code, "RequestError", exc.detail)
        except Exception as exc:
            return _error_response(500, "InternalServerError", str(exc))
    return _sync_wrapper


def sanitize_predict_payload(payload: Dict[str, object]) -> Dict[str, object]:
    clean_payload = dict(payload)
    clean_payload["Symptoms"] = sanitize_text_list(str(payload.get("Symptoms", "")))
    clean_payload["Pre-Existing Conditions"] = sanitize_text_list(str(payload.get("Pre-Existing Conditions", "none")))
    return clean_payload


def _age_band(age: int) -> str:
    if age < 18:
        return "0-17"
    if age < 30:
        return "18-29"
    if age < 45:
        return "30-44"
    if age < 60:
        return "45-59"
    if age < 75:
        return "60-74"
    return "75+"


def _positive_rate(frame: pd.DataFrame, group_col: str) -> pd.DataFrame:
    grouped = frame.groupby(group_col, as_index=False).agg(
        positive_rate=("predicted_high", "mean"),
        sample_size=("predicted_high", "count"),
    )
    grouped["positive_rate"] = grouped["positive_rate"].fillna(0.0)
    return grouped


def _equal_opportunity_rate(frame: pd.DataFrame, group_col: str) -> pd.DataFrame:
    positive_truth = frame[frame["actual_high"] == 1]
    if positive_truth.empty:
        return pd.DataFrame(columns=[group_col, "tpr", "sample_size"])
    grouped = positive_truth.groupby(group_col, as_index=False).agg(
        tpr=("predicted_high", "mean"),
        sample_size=("predicted_high", "count"),
    )
    grouped["tpr"] = grouped["tpr"].fillna(0.0)
    return grouped


def _parity_diff(metric_df: pd.DataFrame, metric_col: str) -> float:
    if metric_df.empty:
        return 0.0
    return float(metric_df[metric_col].max() - metric_df[metric_col].min())


def _sort_sim_queue(queue_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    sorted_rows = sorted(
        queue_rows,
        key=lambda row: (
            -float(row["priority_score"]),
            -int(row["age"]),
            int(row["arrival_minute"]),
        ),
    )
    for idx, row in enumerate(sorted_rows, start=1):
        row["queue_position"] = idx
    return sorted_rows


def _build_clinical_explanation(
    patient_payload: Dict[str, object],
    prediction: Dict[str, object],
    explanation: Dict[str, object],
) -> Dict[str, object]:
    probabilities = prediction.get("probabilities", {})
    confidence = float(prediction.get("confidence", 0.0))
    sorted_probs = sorted(
        [float(probabilities.get("Low", 0.0)), float(probabilities.get("Medium", 0.0)), float(probabilities.get("High", 0.0))],
        reverse=True,
    )
    probability_gap = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) >= 2 else sorted_probs[0]

    out_of_distribution = bool(prediction.get("out_of_distribution", False))
    manual_review_recommended = out_of_distribution or confidence < 0.60 or probability_gap < 0.15
    confidence_band = "High" if confidence >= 0.85 else "Moderate" if confidence >= 0.60 else "Low"

    age = int(patient_payload.get("Age", 0))
    heart_rate = int(patient_payload.get("Heart Rate", 70))
    temperature = float(patient_payload.get("Temperature", 98.6))
    symptoms = [item.strip() for item in str(patient_payload.get("Symptoms", "")).split(",") if item.strip()]
    bp_raw = str(patient_payload.get("Blood Pressure", "120/80"))
    bp_s, bp_d = 120, 80
    if "/" in bp_raw:
        part_a, part_b = bp_raw.split("/", 1)
        try:
            bp_s, bp_d = int(part_a), int(part_b)
        except ValueError:
            pass

    top_factors = explanation.get("explanation", {}).get("top_contributors", [])[:3]
    factor_lines: List[str] = []
    for factor in top_factors:
        feature = str(factor.get("feature", "Unknown factor"))
        interpretation = str(factor.get("interpretation", "Contributed to risk"))
        factor_lines.append(f"- {feature}: {interpretation}")

    if not factor_lines:
        factor_lines.append("- Multi-factor physiological risk pattern detected")

    vitals_note = (
        f"Heart rate {heart_rate} bpm, BP {bp_s}/{bp_d} mmHg, Temp {temperature}°F"
    )
    symptom_note = ", ".join(symptoms[:3]) if symptoms else "no severe symptom cluster documented"

    summary = (
        f"{prediction.get('risk_level', 'Unknown')} risk triage recommendation. "
        f"Primary signals: {symptom_note}. Vitals review: {vitals_note}."
    )

    explanation_text = "\n".join([
        summary,
        "Top clinical contributors:",
        *factor_lines,
    ])

    if manual_review_recommended:
        if out_of_distribution:
            explanation_text += "\n⚠️ Presentation appears outside learned distribution; immediate clinician verification is required."
        else:
            explanation_text += "\n⚠️ Confidence is borderline; manual clinical review is recommended."

    return {
        "clinical_explanation": explanation_text,
        "confidence_band": confidence_band,
        "manual_review_recommended": manual_review_recommended,
    }


def _fallback_followup_questions(complaint: str, max_questions: int) -> List[str]:
    text = complaint.lower()
    if "chest" in text or "pain" in text:
        questions = [
            "When did the pain start, and is it constant or intermittent?",
            "Does the pain radiate to your arm, jaw, or back?",
            "Any shortness of breath, sweating, or nausea with the pain?",
        ]
    elif "breath" in text or "cough" in text:
        questions = [
            "How long have you had breathing difficulty?",
            "Is it worse at rest, with activity, or when lying flat?",
            "Any wheezing, fever, chest tightness, or bluish lips?",
        ]
    elif "head" in text or "dizzy" in text or "confusion" in text:
        questions = [
            "When did the neurological symptoms begin?",
            "Any weakness, facial droop, speech change, or vision change?",
            "Do you have a history of stroke, seizures, or head injury?",
        ]
    else:
        questions = [
            "When did the symptoms start and how have they changed?",
            "On a scale of 1 to 10, how severe are the symptoms now?",
            "Do you have associated red-flag symptoms such as fainting, severe pain, or breathing difficulty?",
        ]
    return questions[:max_questions]


def _query_ollama_symptom_dialogue(body: SymptomDialogueRequest) -> Dict[str, object]:
    transcript_block = "\n".join([f"{turn.role}: {turn.content}" for turn in body.transcript]) or "none"
    known_patient = json.dumps(body.known_patient, ensure_ascii=True)

    prompt = (
        "You are an emergency triage assistant. Produce strict JSON only with keys: "
        "follow_up_questions (array of strings), extracted (object), red_flags (array of strings), "
        "urgency_hint (string). "
        f"Presenting complaint: {body.presenting_complaint}. "
        f"Known patient fields: {known_patient}. "
        f"Transcript so far: {transcript_block}. "
        f"Limit follow_up_questions to at most {body.max_followup_questions}."
    )

    payload = {
        "model": body.model,
        "stream": False,
        "format": "json",
        "messages": [
            {"role": "system", "content": "You are concise, clinically safe, and JSON-only."},
            {"role": "user", "content": prompt},
        ],
    }

    req = urllib.request.Request(
        url="http://localhost:11434/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    content = data.get("message", {}).get("content", "{}")
    parsed = json.loads(content)
    questions = parsed.get("follow_up_questions", [])
    if not isinstance(questions, list):
        questions = []

    return {
        "source": "ollama",
        "model": body.model,
        "follow_up_questions": [str(item) for item in questions][: body.max_followup_questions],
        "structured": {
            "extracted": parsed.get("extracted", {}),
            "red_flags": parsed.get("red_flags", []),
            "urgency_hint": parsed.get("urgency_hint", "unknown"),
        },
    }


def _predict_wait_minutes(queue_rows: List[Dict[str, object]], target_row: Dict[str, object]) -> int:
    department_staff = {
        "Emergency Department": 3,
        "Cardiology": 2,
        "Neurology": 2,
        "Pulmonology": 2,
        "Orthopedics": 2,
        "Gastroenterology": 2,
        "Pediatrics": 2,
        "General Medicine": 3,
    }
    avg_treatment_minutes = 18
    department = str(target_row.get("department", "General Medicine"))
    priority = float(target_row.get("priority_score", 1.0))
    queue_position = int(target_row.get("queue_position", 1))

    patients_ahead = 0
    for row in queue_rows:
        if str(row.get("status", "waiting")) != "waiting":
            continue
        if str(row.get("department", "")) != department:
            continue
        if int(row.get("queue_position", 9999)) < queue_position:
            patients_ahead += 1

    staff = department_staff.get(department, 2)
    raw_wait = (patients_ahead * avg_treatment_minutes) / max(staff, 1)
    priority_factor = max(priority / 5.0, 0.6)
    adjusted_wait = int(round(max(2.0, raw_wait / priority_factor)))
    return adjusted_wait


def _monitoring_note(queue_row: Dict[str, object]) -> Dict[str, object]:
    status = str(queue_row.get("status", "waiting"))
    if status != "waiting":
        return {"deteriorating": False, "monitoring_note": ""}

    arrival_raw = str(queue_row.get("arrival_time", ""))
    elapsed_minutes = 0
    try:
        arrival = datetime.fromisoformat(arrival_raw.replace(" ", "T")).replace(tzinfo=UTC)
        elapsed_minutes = max(0, int((datetime.now(UTC) - arrival).total_seconds() // 60))
    except ValueError:
        pass

    bp_s = int(queue_row.get("bp_systolic") or 120)
    bp_d = int(queue_row.get("bp_diastolic") or 80)
    heart_rate = int(queue_row.get("heart_rate") or 70)
    temperature = float(queue_row.get("temperature") or 98.6)

    worsening_signals = 0
    if bp_s < 90 or bp_s > 180:
        worsening_signals += 1
    if bp_d > 110:
        worsening_signals += 1
    if heart_rate < 45 or heart_rate > 130:
        worsening_signals += 1
    if temperature >= 102.5:
        worsening_signals += 1

    deteriorating = elapsed_minutes >= 10 and worsening_signals >= 2
    if deteriorating:
        return {
            "deteriorating": True,
            "monitoring_note": (
                f"Deterioration watch: {worsening_signals} warning signals after {elapsed_minutes} min wait"
            ),
        }
    return {
        "deteriorating": False,
        "monitoring_note": f"Monitored {elapsed_minutes} min in queue",
    }


def _build_fairness_artifacts(sample_size: int) -> Dict[str, object]:
    dataset_path = ROOT / "data" / "synthetic_patients.csv"
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found for fairness analysis")

    frame = pd.read_csv(dataset_path).head(sample_size).copy()
    required_cols = {
        "Patient_ID",
        "Age",
        "Gender",
        "Symptoms",
        "Blood Pressure",
        "Heart Rate",
        "Temperature",
        "Pre-Existing Conditions",
        "Risk_Level",
    }
    if not required_cols.issubset(set(frame.columns)):
        missing = sorted(required_cols.difference(frame.columns))
        raise HTTPException(status_code=422, detail=f"Dataset missing required fairness columns: {', '.join(missing)}")

    engine = get_engine()
    predictions = engine.predict_batch(frame.to_dict(orient="records"))
    pred_map = {row["patient_id"]: row["risk_level"] for row in predictions}

    frame["predicted_risk"] = frame["Patient_ID"].astype(str).map(pred_map)
    frame["age_band"] = frame["Age"].astype(int).apply(_age_band)
    frame["predicted_high"] = (frame["predicted_risk"] == "High").astype(int)
    frame["actual_high"] = (frame["Risk_Level"] == "High").astype(int)

    gender_dp = _positive_rate(frame, "Gender")
    age_dp = _positive_rate(frame, "age_band")
    gender_eo = _equal_opportunity_rate(frame, "Gender")
    age_eo = _equal_opportunity_rate(frame, "age_band")

    dist_gender = (
        frame.groupby(["Gender", "predicted_risk"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    dist_age = (
        frame.groupby(["age_band", "predicted_risk"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )

    return {
        "sample_size": int(len(frame)),
        "metrics": {
            "demographic_parity_gender": round(_parity_diff(gender_dp, "positive_rate"), 6),
            "demographic_parity_age": round(_parity_diff(age_dp, "positive_rate"), 6),
            "equal_opportunity_gender": round(_parity_diff(gender_eo, "tpr"), 6),
            "equal_opportunity_age": round(_parity_diff(age_eo, "tpr"), 6),
        },
        "dist_gender": dist_gender,
        "dist_age": dist_age,
        "gender_dp": gender_dp.rename(columns={"positive_rate": "high_risk_rate"}),
        "age_dp": age_dp.rename(columns={"positive_rate": "high_risk_rate"}),
        "gender_eo": gender_eo,
        "age_eo": age_eo,
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return _error_response(500, "InternalServerError", str(exc))


@app.exception_handler(HTTPException)
async def global_http_exception_handler(request: Request, exc: HTTPException):
    return _error_response(exc.status_code, "RequestError", exc.detail)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return _error_response(422, "ValidationError", exc.errors())


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/constants")
@safe_endpoint
def constants():
    return {
        "symptoms": SYMPTOM_OPTIONS,
        "conditions": CONDITION_OPTIONS,
    }


@app.post("/api/auth/login")
@safe_endpoint
def login(body: LoginRequest):
    user = authenticate_user(body.username, body.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = create_access_token({"sub": user["username"], "role": user["role"]})
    return {"access_token": token, "token_type": "bearer"}


@app.get("/api/auth/me")
@safe_endpoint
def auth_me(current_user: Dict[str, str] = Depends(get_current_user)):
    return current_user


@app.get("/api/fairness")
@safe_endpoint
def fairness_analysis(
    sample_size: int = Query(default=2000, ge=100, le=5000),
    current_user: Dict[str, str] = Depends(require_role("Nurse", "Doctor", "Admin")),
):
    artifacts = _build_fairness_artifacts(sample_size)

    return {
        "sample_size": artifacts["sample_size"],
        "metrics": artifacts["metrics"],
        "distributions": {
            "gender": artifacts["dist_gender"].to_dict(orient="records"),
            "age_band": artifacts["dist_age"].to_dict(orient="records"),
        },
        "tables": {
            "demographic_parity_gender": artifacts["gender_dp"].to_dict(orient="records"),
            "demographic_parity_age": artifacts["age_dp"].to_dict(orient="records"),
            "equal_opportunity_gender": artifacts["gender_eo"].to_dict(orient="records"),
            "equal_opportunity_age": artifacts["age_eo"].to_dict(orient="records"),
        },
    }


@app.get("/api/fairness/export")
@safe_endpoint
def fairness_export_csv(
    report_type: str = Query(...),
    sample_size: int = Query(default=2000, ge=100, le=5000),
    current_user: Dict[str, str] = Depends(require_role("Nurse", "Doctor", "Admin")),
):
    artifacts = _build_fairness_artifacts(sample_size)
    report_map = {
        "demographic_parity_gender": artifacts["gender_dp"],
        "demographic_parity_age": artifacts["age_dp"],
        "equal_opportunity_gender": artifacts["gender_eo"],
        "equal_opportunity_age": artifacts["age_eo"],
        "prediction_distribution_gender": artifacts["dist_gender"],
        "prediction_distribution_age": artifacts["dist_age"],
    }
    if report_type not in report_map:
        raise HTTPException(status_code=422, detail="Unknown report_type")

    df = report_map[report_type]
    csv_bytes = dataframe_to_csv_bytes(df)
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={report_type}.csv"},
    )


@app.get("/api/simulation/state")
@safe_endpoint
def simulation_state(current_user: Dict[str, str] = Depends(require_role("Nurse", "Doctor", "Admin"))):
    return _sim_state


@app.post("/api/simulation/reset")
@safe_endpoint
def simulation_reset(current_user: Dict[str, str] = Depends(require_role("Nurse", "Doctor", "Admin"))):
    global _sim_state
    _sim_state = {
        "current_minute": 0,
        "queue": [],
        "timeline": [],
        "arrived": 0,
        "completed": 0,
    }
    return _sim_state


@app.get("/api/simulation/export")
@safe_endpoint
def simulation_export_csv(
    report_type: str = Query(...),
    current_user: Dict[str, str] = Depends(require_role("Nurse", "Doctor", "Admin")),
):
    if report_type == "queue":
        df = pd.DataFrame(_sim_state["queue"])
    elif report_type == "timeline":
        df = pd.DataFrame(_sim_state["timeline"])
    else:
        raise HTTPException(status_code=422, detail="Unknown report_type")

    if df.empty:
        df = pd.DataFrame([{"status": "empty"}])

    csv_bytes = dataframe_to_csv_bytes(df)
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=simulation_{report_type}.csv"},
    )


@app.post("/api/simulation/step")
@safe_endpoint
def simulation_step(
    body: SimulationStepRequest,
    current_user: Dict[str, str] = Depends(require_role("Nurse", "Doctor", "Admin")),
):
    global _sim_state
    dataset_path = ROOT / "data" / "synthetic_patients.csv"
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found for simulation")

    pool_df = pd.read_csv(dataset_path)
    if pool_df.empty:
        raise HTTPException(status_code=422, detail="Simulation dataset is empty")

    rng = np.random.default_rng(body.seed + int(_sim_state["current_minute"]))
    engine = get_engine()

    department_capacity = {
        "Emergency Department": 2,
        "Cardiology": 1,
        "Neurology": 1,
        "Pulmonology": 1,
        "Orthopedics": 1,
        "Gastroenterology": 1,
        "Pediatrics": 1,
        "General Medicine": 2,
    }

    queue_rows = list(_sim_state["queue"])
    timeline_rows = list(_sim_state["timeline"])
    current_minute = int(_sim_state["current_minute"])
    arrived = int(_sim_state["arrived"])
    completed = int(_sim_state["completed"])

    for _ in range(body.minutes):
        current_minute += 1
        arrivals_count = int(rng.poisson(body.lambda_rate))

        if arrivals_count > 0:
            sampled = pool_df.sample(n=arrivals_count, replace=True, random_state=int(rng.integers(0, 10_000_000)))
            for _, row in sampled.iterrows():
                payload = row.to_dict()
                payload["Patient_ID"] = f"SIM-{current_minute}-{arrived + 1}"
                prediction = engine.predict_one(payload)

                queue_rows.append(
                    {
                        "patient_id": prediction["patient_id"],
                        "age": int(payload["Age"]),
                        "risk_level": prediction["risk_level"],
                        "priority_score": float(prediction["priority_score"]),
                        "department": prediction["department"],
                        "arrival_minute": current_minute,
                        "queue_position": 0,
                    }
                )
                arrived += 1

        queue_rows = _sort_sim_queue(queue_rows)

        for department, capacity in department_capacity.items():
            served = 0
            retained_rows: List[Dict[str, object]] = []
            for item in queue_rows:
                if item["department"] == department and served < capacity:
                    served += 1
                    completed += 1
                    continue
                retained_rows.append(item)
            queue_rows = retained_rows

        queue_rows = _sort_sim_queue(queue_rows)
        if queue_rows:
            load_snapshot = pd.DataFrame(queue_rows)["department"].value_counts().to_dict()
        else:
            load_snapshot = {}

        for department in department_capacity:
            timeline_rows.append(
                {
                    "minute": current_minute,
                    "department": department,
                    "waiting": int(load_snapshot.get(department, 0)),
                }
            )

    _sim_state = {
        "current_minute": current_minute,
        "queue": queue_rows,
        "timeline": timeline_rows,
        "arrived": arrived,
        "completed": completed,
    }
    return _sim_state


@app.post("/api/predict")
@safe_endpoint
def predict_single(
    patient: PatientPayload,
    current_user: Dict[str, str] = Depends(require_role("Nurse", "Doctor", "Admin")),
):
    payload = sanitize_predict_payload(patient.to_engine_dict())
    valid, errors = validate_patient_payload(payload)
    if not valid:
        raise HTTPException(status_code=422, detail=errors)

    engine = get_engine()
    result = engine.predict_one(payload)

    explainer = get_explainer()
    explanation = explainer.explain(payload)
    clinical = _build_clinical_explanation(payload, result, explanation)

    return {
        "patient": payload,
        "prediction": result,
        "explanation": explanation,
        **clinical,
    }


@app.post("/api/predict/save")
@safe_endpoint
def predict_and_save(
    patient: PatientPayload,
    current_user: Dict[str, str] = Depends(require_role("Doctor", "Admin")),
):
    payload = sanitize_predict_payload(patient.to_engine_dict())
    valid, errors = validate_patient_payload(payload)
    if not valid:
        raise HTTPException(status_code=422, detail=errors)

    engine = get_engine()
    result = engine.predict_one(payload)

    explainer = get_explainer()
    explanation = explainer.explain(payload)
    clinical = _build_clinical_explanation(payload, result, explanation)

    db = get_db()
    shap_rows = explanation["explanation"]["top_contributors"] if explanation else []
    prediction_id = db.save_prediction(payload, result, shap_top_contributors=shap_rows, source="manual")

    return {
        "patient": payload,
        "prediction": result,
        "explanation": explanation,
        **clinical,
        "saved": True,
        "prediction_id": prediction_id,
    }


@app.post("/api/predict/batch")
@safe_endpoint
async def predict_batch(
    file: UploadFile = File(...),
    current_user: Dict[str, str] = Depends(require_role("Nurse", "Doctor", "Admin")),
):
    allowed_suffixes = (".csv", ".xlsx", ".xls")
    if not file.filename or not file.filename.lower().endswith(allowed_suffixes):
        raise HTTPException(status_code=400, detail="Only CSV, XLSX, and XLS files are supported for batch processing")

    temp_path = ROOT / "data" / f"tmp_batch_{file.filename}"
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    content = await file.read()
    temp_path.write_bytes(content)

    try:
        rows = parse_document(temp_path)
        sanitized_rows = [sanitize_predict_payload(row) for row in rows]
        engine = get_engine()
        results = engine.predict_batch(sanitized_rows)
        result_df = pd.DataFrame(results)

        risk_counts = {
            "total": len(result_df),
            "high": int((result_df["risk_level"] == "High").sum()) if not result_df.empty else 0,
            "medium": int((result_df["risk_level"] == "Medium").sum()) if not result_df.empty else 0,
            "low": int((result_df["risk_level"] == "Low").sum()) if not result_df.empty else 0,
        }

        return {
            "source_rows": sanitized_rows[:10],
            "source_count": len(sanitized_rows),
            "results": results,
            "risk_counts": risk_counts,
        }
    finally:
        temp_path.unlink(missing_ok=True)


@app.post("/api/predict/batch/save")
@safe_endpoint
async def batch_save(
    results: List[dict],
    patients: List[dict],
    current_user: Dict[str, str] = Depends(require_role("Nurse", "Doctor", "Admin")),
):
    db = get_db()
    saved = 0
    for idx, result in enumerate(results):
        patient = patients[idx] if idx < len(patients) else patients[-1]
        patient = sanitize_predict_payload(patient)
        db.save_prediction(patient, result, source="batch")
        saved += 1
    return {"saved": saved}


@app.post("/api/triage/symptom-dialogue")
@safe_endpoint
def symptom_dialogue(
    body: SymptomDialogueRequest,
    current_user: Dict[str, str] = Depends(require_role("Nurse", "Doctor", "Admin")),
):
    try:
        return _query_ollama_symptom_dialogue(body)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, ValueError):
        return {
            "source": "fallback",
            "model": "rule_based",
            "follow_up_questions": _fallback_followup_questions(body.presenting_complaint, body.max_followup_questions),
            "structured": {
                "extracted": {
                    "chief_complaint": body.presenting_complaint,
                    "reported_duration": "unknown",
                    "reported_severity": "unknown",
                },
                "red_flags": [],
                "urgency_hint": "requires_clinical_assessment",
            },
        }


@app.post("/api/safety/red-flags")
@safe_endpoint
def safety_red_flags(
    body: SafetyScreenRequest,
    current_user: Dict[str, str] = Depends(require_role("Nurse", "Doctor", "Admin")),
):
    payload = sanitize_predict_payload(body.to_dict())
    valid, errors = validate_patient_payload(payload)
    if not valid:
        raise HTTPException(status_code=422, detail=errors)

    return {
        "patient": payload,
        "safety_screen": evaluate_red_flags(payload),
    }


@app.post("/api/safety/handoff-summary")
@safe_endpoint
def safety_handoff_summary(
    body: SafetyScreenRequest,
    current_user: Dict[str, str] = Depends(require_role("Nurse", "Doctor", "Admin")),
):
    payload = sanitize_predict_payload(body.to_dict())
    valid, errors = validate_patient_payload(payload)
    if not valid:
        raise HTTPException(status_code=422, detail=errors)

    prediction = get_engine().predict_one(payload)
    explanation = get_explainer().explain(payload)
    clinical = _build_clinical_explanation(payload, prediction, explanation)
    safety_screen = evaluate_red_flags(payload)
    handoff = build_handoff_summary(
        patient=payload,
        prediction=prediction,
        safety_screen=safety_screen,
        clinical_explanation=str(clinical.get("clinical_explanation", "")),
    )

    return {
        "patient": payload,
        "prediction": prediction,
        "safety_screen": safety_screen,
        "handoff_summary": handoff,
    }


@app.post("/api/safety/medication-screen")
@safe_endpoint
def safety_medication_screen(
    body: MedicationSafetyRequest,
    current_user: Dict[str, str] = Depends(require_role("Nurse", "Doctor", "Admin")),
):
    return {
        "screening": evaluate_medication_safety(
            medications_raw=body.medications,
            allergies_raw=body.allergies,
            conditions_raw=body.conditions,
        )
    }


@app.post("/api/safety/followup-plan")
@safe_endpoint
def safety_followup_plan(
    body: SafetyScreenRequest,
    current_user: Dict[str, str] = Depends(require_role("Nurse", "Doctor", "Admin")),
):
    payload = sanitize_predict_payload(body.to_dict())
    valid, errors = validate_patient_payload(payload)
    if not valid:
        raise HTTPException(status_code=422, detail=errors)

    prediction = get_engine().predict_one(payload)
    safety_screen = evaluate_red_flags(payload)
    plan = build_followup_plan(payload, prediction, safety_screen)

    return {
        "patient": payload,
        "prediction": prediction,
        "safety_screen": safety_screen,
        "followup_plan": plan,
    }


@app.post("/api/upload")
@safe_endpoint
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    temp_path = ROOT / "data" / f"tmp_upload_{file.filename}"
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    content = await file.read()
    temp_path.write_bytes(content)

    try:
        parsed = parse_document(temp_path)
        return {"parsed": parsed[:3]}
    finally:
        temp_path.unlink(missing_ok=True)


# ---------- Queue ----------

@app.get("/api/queue")
@safe_endpoint
def get_queue(status: Optional[str] = None):
    db = get_db()
    monitoring_events = db.apply_continuous_monitoring(check_interval_minutes=5)
    queue_data = db.get_priority_queue(status=status if status != "all" else None)
    queue_df = pd.DataFrame(queue_data)

    dept_load = pd.DataFrame(db.get_department_load())
    alerts = [event["message"] for event in monitoring_events]

    if not queue_df.empty:
        waiting = queue_df[queue_df["status"] == "waiting"] if "status" in queue_df.columns else pd.DataFrame()
        critical_waiting = waiting[waiting["priority_score"] >= 9.0] if not waiting.empty else pd.DataFrame()
        if not critical_waiting.empty:
            alerts.append(f"Priority alert: {len(critical_waiting)} critical patients are waiting")

        if not dept_load.empty:
            overload = dept_load[dept_load["waiting_count"] >= 10]
            if not overload.empty:
                alerts.append("Department overload: " + ", ".join(overload["department"].tolist()))

        enriched_queue: List[Dict[str, object]] = []
        for row in queue_data:
            note = _monitoring_note(row)
            dynamic_wait_minutes = _predict_wait_minutes(queue_data, row)
            dynamic_wait = f"{dynamic_wait_minutes} minutes"
            trend = db.get_vitals_trend(str(row.get("patient_id", "")), window_points=4)

            priority_score = float(row.get("priority_score", 0.0))
            if priority_score >= 9.0:
                esi_level = 1
            elif priority_score >= 7.5:
                esi_level = 2
            elif priority_score >= 5.0:
                esi_level = 3
            elif priority_score >= 3.0:
                esi_level = 4
            else:
                esi_level = 5

            enriched = {
                **row,
                **note,
                "dynamic_estimated_wait_minutes": dynamic_wait_minutes,
                "dynamic_estimated_wait": dynamic_wait,
                "esi_level": esi_level,
                "vitals_trend": trend,
            }
            if note["deteriorating"]:
                alerts.append(f"Re-triage recommended: {row.get('patient_id')} ({note['monitoring_note']})")
            if bool(trend.get("is_deteriorating")):
                alerts.append(f"Trend alert: {row.get('patient_id')} shows progressive instability ({', '.join(trend.get('signals', []))})")
            enriched_queue.append(enriched)

        return {
            "queue": enriched_queue,
            "waiting_count": int(len(waiting)),
            "critical_count": int((waiting["priority_score"] >= 9.0).sum()) if not waiting.empty else 0,
            "alerts": alerts,
            "departments": sorted(queue_df["department"].dropna().unique().tolist()) if "department" in queue_df.columns else [],
        }
    return {"queue": [], "waiting_count": 0, "critical_count": 0, "alerts": alerts, "departments": []}


@app.post("/api/queue/next")
@safe_endpoint
def call_next_patient():
    db = get_db()
    queue_data = db.get_priority_queue(status="waiting")
    waiting = pd.DataFrame(queue_data)
    if waiting.empty:
        raise HTTPException(status_code=404, detail="No patients waiting")
    top = waiting.sort_values("queue_position").head(1)
    queue_id = int(top.iloc[0]["queue_id"])
    db.update_queue_status(queue_id, "in_progress")
    return {"message": "Top queue patient moved to in progress", "queue_id": queue_id}


@app.patch("/api/queue/{queue_id}/status")
@safe_endpoint
def update_queue_status(queue_id: int, body: QueueStatusUpdate):
    db = get_db()
    db.update_queue_status(queue_id, body.status)
    return {"message": f"Queue item {queue_id} → {body.status}"}


@app.delete("/api/queue/completed")
@safe_endpoint
def clear_completed(current_user: Dict[str, str] = Depends(require_role("Doctor", "Admin"))):
    db = get_db()
    deleted = db.clear_completed()
    return {"deleted": deleted}


@app.post("/api/feedback/outcome")
@safe_endpoint
def submit_outcome_feedback(
    body: OutcomeFeedbackRequest,
    current_user: Dict[str, str] = Depends(require_role("Doctor", "Admin")),
):
    db = get_db()
    try:
        feedback_id = db.save_outcome_feedback(
            prediction_id=body.prediction_id,
            actual_risk_level=body.actual_risk_level,
            final_department=body.final_department,
            outcome_status=body.outcome_status,
            clinician_role=body.clinician_role,
            notes=body.notes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return {
        "saved": True,
        "feedback_id": feedback_id,
    }


@app.get("/api/feedback/outcome")
@safe_endpoint
def list_outcome_feedback(
    limit: int = Query(default=200, ge=1, le=2000),
    current_user: Dict[str, str] = Depends(require_role("Doctor", "Admin")),
):
    db = get_db()
    rows = db.get_outcome_feedback(limit=limit)
    frame = pd.DataFrame(rows)
    agreement_rate = 0.0
    if not frame.empty and {"actual_risk_level", "predicted_risk_level"}.issubset(frame.columns):
        agreement_rate = float((frame["actual_risk_level"] == frame["predicted_risk_level"]).mean() * 100)

    return {
        "count": int(len(rows)),
        "agreement_rate": round(agreement_rate, 1),
        "rows": rows,
    }


# ---------- Analytics ----------

@app.get("/api/analytics")
@safe_endpoint
def analytics(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    risk_levels: Optional[str] = None,
    departments: Optional[str] = None,
):
    db = get_db()
    queue = pd.DataFrame(db.get_priority_queue())
    high_risk = pd.DataFrame(db.get_high_risk_patients(limit=500))
    predictions = pd.DataFrame(db.get_predictions(limit=5000))

    if predictions.empty and queue.empty:
        return {"empty": True}

    # Parse dates
    if not predictions.empty:
        predictions["timestamp"] = pd.to_datetime(predictions["timestamp"], errors="coerce")
        predictions = predictions.dropna(subset=["timestamp"])
        predictions["date"] = predictions["timestamp"].dt.date

    # Apply filters
    filtered = predictions.copy()
    if not filtered.empty:
        if start_date:
            filtered = filtered[filtered["date"] >= date.fromisoformat(start_date)]
        if end_date:
            filtered = filtered[filtered["date"] <= date.fromisoformat(end_date)]
        if risk_levels:
            levels = risk_levels.split(",")
            filtered = filtered[filtered["risk_level"].isin(levels)]
        if departments:
            depts = departments.split(",")
            filtered = filtered[filtered["recommended_department"].isin(depts)]

    # Metrics
    total = len(filtered)
    high_risk_rate = float((filtered["risk_level"] == "High").mean() * 100) if total > 0 and "risk_level" in filtered.columns else 0.0
    avg_priority = float(filtered["priority_score"].mean()) if total > 0 and "priority_score" in filtered.columns else 0.0
    avg_confidence = float(filtered["model_confidence"].mean() * 100) if total > 0 and "model_confidence" in filtered.columns else 0.0

    waiting_queue = queue[queue["status"] == "waiting"] if not queue.empty and "status" in queue.columns else pd.DataFrame()

    # Chart data
    risk_counts = filtered["risk_level"].value_counts().to_dict() if "risk_level" in filtered.columns else {}
    dept_counts = filtered["recommended_department"].value_counts().head(10).to_dict() if "recommended_department" in filtered.columns else {}

    # Trend data
    trend_data = []
    if not filtered.empty and "risk_level" in filtered.columns:
        trend = filtered.groupby([filtered["date"].astype(str), "risk_level"], as_index=False).size().rename(columns={"size": "count"})
        trend_data = trend.to_dict(orient="records")

    # Priority category
    priority_cat_data = filtered["priority_category"].value_counts().to_dict() if "priority_category" in filtered.columns else {}

    # Symptom data
    symptom_data = {}
    if "symptoms" in filtered.columns:
        symptom_series = filtered["symptoms"].fillna("").str.split(",").explode().str.strip().replace("", pd.NA).dropna()
        symptom_data = symptom_series.value_counts().head(12).to_dict()

    # Recent activity
    activity_cols = ["timestamp", "patient_id", "risk_level", "priority_score", "priority_category", "recommended_department", "model_confidence"]
    available_cols = [c for c in activity_cols if c in filtered.columns]
    recent = filtered.sort_values("timestamp", ascending=False)[available_cols].head(30) if available_cols else pd.DataFrame()
    if "timestamp" in recent.columns:
        recent["timestamp"] = recent["timestamp"].astype(str)

    # Filter options
    all_risk = sorted(predictions["risk_level"].dropna().unique().tolist()) if "risk_level" in predictions.columns else []
    all_depts = sorted(predictions["recommended_department"].dropna().unique().tolist()) if "recommended_department" in predictions.columns else []
    date_range = {}
    if not predictions.empty and "date" in predictions.columns:
        date_range = {"min": str(predictions["date"].min()), "max": str(predictions["date"].max())}

    return {
        "empty": False,
        "metrics": {
            "total_predictions": total,
            "high_risk_rate": round(high_risk_rate, 1),
            "avg_priority": round(avg_priority, 2),
            "avg_confidence": round(avg_confidence, 1),
            "queue_waiting": int(len(waiting_queue)),
            "critical_waiting": int((waiting_queue["priority_score"] >= 9.0).sum()) if not waiting_queue.empty and "priority_score" in waiting_queue.columns else 0,
            "recent_high_risk": int(len(high_risk)),
        },
        "charts": {
            "risk_counts": risk_counts,
            "dept_counts": dept_counts,
            "trend": trend_data,
            "priority_categories": priority_cat_data,
            "symptoms": symptom_data,
        },
        "recent_activity": recent.to_dict(orient="records") if not recent.empty else [],
        "filter_options": {
            "risk_levels": all_risk,
            "departments": all_depts,
            "date_range": date_range,
        },
    }


# ---------- History ----------

@app.get("/api/history")
@safe_endpoint
def history(
    patient_id: Optional[str] = None,
    risk_levels: Optional[str] = None,
    departments: Optional[str] = None,
    priority_categories: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 2000,
):
    db = get_db()
    all_predictions = pd.DataFrame(db.get_predictions(limit=5000))

    if all_predictions.empty:
        return {"empty": True, "records": [], "filter_options": {}}

    records = db.search_predictions(
        patient_id_query=patient_id or None,
        risk_levels=risk_levels.split(",") if risk_levels else None,
        departments=departments.split(",") if departments else None,
        priority_categories=priority_categories.split(",") if priority_categories else None,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
    )

    dept_options = sorted(all_predictions["recommended_department"].dropna().unique().tolist()) if "recommended_department" in all_predictions.columns else []

    return {
        "empty": False,
        "records": records,
        "filter_options": {
            "departments": dept_options,
        },
    }


@app.get("/api/history/phi-safe")
@safe_endpoint
def history_phi_safe(
    patient_id: Optional[str] = None,
    risk_levels: Optional[str] = None,
    departments: Optional[str] = None,
    priority_categories: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 2000,
    current_user: Dict[str, str] = Depends(require_role("Nurse", "Doctor", "Admin")),
):
    db = get_db()
    records = db.search_predictions(
        patient_id_query=patient_id or None,
        risk_levels=risk_levels.split(",") if risk_levels else None,
        departments=departments.split(",") if departments else None,
        priority_categories=priority_categories.split(",") if priority_categories else None,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
    )
    redacted = redact_records(records, extra_sensitive_keys={"notes"})
    return {
        "empty": len(redacted) == 0,
        "records": redacted,
        "phi_safe": True,
    }


@app.delete("/api/history/old")
@safe_endpoint
def clear_old_records(
    days: int = 90,
    current_user: Dict[str, str] = Depends(require_role("Doctor", "Admin")),
):
    db = get_db()
    deleted = db.clear_old_records(days=days)
    return deleted


# ---------- Admin Data Controls ----------

@app.delete("/api/admin/data/specific")
@safe_endpoint
def admin_delete_specific(
    body: AdminDeleteRequest,
    current_user: Dict[str, str] = Depends(require_role("Admin")),
):
    if not any([body.patient_id, body.prediction_id is not None, body.queue_id is not None]):
        raise HTTPException(status_code=422, detail="Provide at least one target: patient_id, prediction_id, or queue_id")

    db = get_db()
    deleted = db.delete_specific_records(
        patient_id=body.patient_id,
        prediction_id=body.prediction_id,
        queue_id=body.queue_id,
    )

    if body.prediction_id is not None and int(deleted.get("deleted_prediction_rows", 0)) == 0:
        raise HTTPException(status_code=404, detail=f"Prediction {body.prediction_id} not found")
    if body.queue_id is not None and int(deleted.get("deleted_queue_rows", 0)) == 0 and body.patient_id is None and body.prediction_id is None:
        raise HTTPException(status_code=404, detail=f"Queue item {body.queue_id} not found")
    if body.patient_id is not None:
        affected = int(deleted.get("deleted_prediction_rows", 0)) + int(deleted.get("deleted_patient_rows", 0))
        if affected == 0:
            raise HTTPException(status_code=404, detail=f"Patient {body.patient_id} not found")

    return {
        "message": "Specific deletion complete",
        **deleted,
    }


@app.delete("/api/admin/data/recent")
@safe_endpoint
def admin_delete_recent(
    days: int = Query(default=30, ge=1, le=365),
    scope: str = Query(default="all", pattern="^(all|queue|predictions|patients)$"),
    current_user: Dict[str, str] = Depends(require_role("Admin")),
):
    db = get_db()
    deleted = db.delete_recent_records(days=days, scope=scope)
    return {
        "message": f"Recent deletion complete for last {days} days",
        "scope": scope,
        "days": days,
        **deleted,
    }


# ---------- Exports ----------

@app.post("/api/export/pdf/single")
@safe_endpoint
def export_single_pdf(patient: dict, prediction: dict):
    pdf_bytes = single_result_pdf(patient, prediction)
    return Response(content=pdf_bytes, media_type="application/pdf",
                    headers={"Content-Disposition": f"attachment; filename=triage_{prediction.get('patient_id', 'report')}.pdf"})


@app.post("/api/export/pdf/batch")
@safe_endpoint
def export_batch_pdf(results: List[dict]):
    pdf_bytes = batch_results_pdf(results)
    return Response(content=pdf_bytes, media_type="application/pdf",
                    headers={"Content-Disposition": "attachment; filename=batch_report.pdf"})


@app.post("/api/export/csv")
@safe_endpoint
def export_csv(records: List[dict]):
    df = pd.DataFrame(records)
    csv_bytes = dataframe_to_csv_bytes(df)
    return Response(content=csv_bytes, media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=export.csv"})


@app.post("/api/export/csv/phi-safe")
@safe_endpoint
def export_csv_phi_safe(
    body: ExportRecordsRequest,
    current_user: Dict[str, str] = Depends(require_role("Nurse", "Doctor", "Admin")),
):
    redacted = redact_records(body.records, extra_sensitive_keys={"notes", "clinician_notes"})
    df = pd.DataFrame(redacted)
    csv_bytes = dataframe_to_csv_bytes(df)
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=export_phi_safe.csv"},
    )


# ---------- Health ----------

@app.get("/api/healthcheck")
@safe_endpoint
def healthcheck():
    status = run_healthcheck(ROOT)
    return status.to_dict()


@app.get("/api/security/config")
@safe_endpoint
def security_config(current_user: Dict[str, str] = Depends(require_role("Admin"))):
    return security_configuration_status()


# ---------- Dashboard summary ----------

@app.get("/api/dashboard")
@safe_endpoint
def dashboard_summary():
    db = get_db()
    queue = pd.DataFrame(db.get_priority_queue())
    predictions = pd.DataFrame(db.get_predictions(limit=5000))

    waiting = queue[queue["status"] == "waiting"] if not queue.empty and "status" in queue.columns else pd.DataFrame()
    active_queue = (
        queue[queue["status"] != "completed"]
        if not queue.empty and "status" in queue.columns
        else queue
    )

    if not active_queue.empty and "priority_score" in active_queue.columns:
        high_count = int((active_queue["priority_score"] >= 9.0).sum())
    elif not predictions.empty and "risk_level" in predictions.columns:
        high_count = int((predictions["risk_level"] == "High").sum())
    else:
        high_count = 0

    if not active_queue.empty and "department" in active_queue.columns:
        dept_count = int(active_queue["department"].nunique())
    elif not predictions.empty and "recommended_department" in predictions.columns:
        dept_count = int(predictions["recommended_department"].nunique())
    else:
        dept_count = 0

    return {
        "total_predictions": int(len(predictions)),
        "queue_waiting": int(len(waiting)),
        "high_risk_cases": high_count,
        "active_departments": dept_count,
    }


@app.get("/api/monitoring/drift")
@safe_endpoint
def monitoring_drift(
    current_user: Dict[str, str] = Depends(require_role("Doctor", "Admin")),
):
    db = get_db()
    predictions = db.get_predictions(limit=1000)
    feedback_rows = db.get_outcome_feedback(limit=500)
    waiting_queue = db.get_priority_queue(status="waiting")

    report = build_drift_report(
        predictions=predictions,
        feedback_rows=feedback_rows,
        waiting_queue_count=len(waiting_queue),
    )
    return report
