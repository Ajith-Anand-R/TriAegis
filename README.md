# AI Patient Triage System

Production-style local triage system built with Python, XGBoost, FastAPI, Next.js, and SQLite.

## Features

- Synthetic patient data generation (`data/generate_data.py`) with exact risk distribution (60/30/10).
- Feature engineering + model training pipeline (`models/train_model.py`).
- Inference engine for single and batch predictions (`utils/ml_engine.py`).
- Rule-based department routing + priority queue logic (`utils/department_recommender.py`).
- SHAP explainability module (`utils/explainer.py`).
- Document ingestion for CSV/PDF/DOCX (`utils/document_parser.py`).
- Report export utilities for PDF/CSV outputs (`utils/reporting.py`).
- SQLite persistence and queue management (`utils/database.py`).
- FastAPI backend (`api.py`) with JWT-based authentication and role-based access control.
- Next.js frontend (`frontend/`) as the single production UI.
- Enhanced analytics + history filtering + queue alerts in the web UI.
- Integration tests (`tests/test_system.py`).

## Project Structure

```text
TriAegis/
├── api.py
├── auth.py
├── requirements.txt
├── README.md
├── run.ps1
├── run.sh
├── run-dev.ps1
├── data/
│   ├── generate_data.py
│   ├── generate_sample_documents.py
│   └── synthetic_patients.csv
├── models/
│   ├── train_model.py
│   └── saved_models/
├── utils/
│   ├── ml_engine.py
│   ├── document_parser.py
│   ├── explainer.py
│   ├── healthcheck.py
│   ├── department_recommender.py
│   ├── database.py
│   ├── reporting.py
│   └── validators.py
├── scripts/
│   └── system_self_check.py
├── database/
│   └── patients.db
├── frontend/
│   └── src/
└── tests/
        └── test_system.py
```

## Must-Do Coverage Status

- Generate and document model metrics: ✅
    - Metrics are generated during training and saved to `models/saved_models/metrics.json`.
    - Visual artifacts are generated and saved to:
        - `models/saved_models/confusion_matrix.png`
        - `models/saved_models/roc_curve.png`
        - `models/saved_models/feature_importance.png`
- Add basic authentication to API: ✅
    - Implemented in `auth.py` and integrated in `api.py`.
    - JWT login endpoint: `POST /api/auth/login`
    - Current user endpoint: `GET /api/auth/me`
    - Role-protected endpoints via `require_role(...)`.
- Improve error handling in EHR parser: ✅
    - Row-level validation errors for malformed dates.
    - Friendly UTF-8 decode errors for uploaded CSV bytes.
    - Validation wrappers include filename context.
- Test edge cases: ✅
    - Existing model/triage edge tests are present.
    - Added EHR parser edge tests (invalid dates + non-UTF8 uploads).
- Simplify to ONE frontend (Next.js): ✅
    - Startup scripts and docs now target FastAPI + Next.js as the production path.

## Advanced Differentiators Implemented

- Continuous risk monitoring for waiting queue patients:
    - Re-checks waiting patients at 5-minute intervals.
    - Detects deterioration signals from vitals and escalates queue priority automatically.
    - Emits queue alerts for re-triage recommendations.

- Dynamic wait-time prediction:
    - Computes wait estimate from patients ahead, average treatment duration, staffing, and priority.
    - Exposed in queue responses as `dynamic_estimated_wait`.

- Natural-language clinical explanations:
    - Prediction endpoints now return `clinical_explanation`, `confidence_band`, and `manual_review_recommended`.
    - Supports safer triage by flagging low-confidence/ambiguous cases for manual review.

## Quick Start

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

1. Generate synthetic data:

```bash
python data/generate_data.py
```

1. Generate sample PDF/DOCX/CSV documents for testing/demo:

```bash
python data/generate_sample_documents.py
```

1. Train the model and save artifacts:

```bash
python models/train_model.py
```

1. Run tests:

```bash
pytest tests/test_system.py -q
```

1. Start backend + frontend:

```powershell
./run.ps1
```

1. Run production self-check:

```bash
python scripts/system_self_check.py
```

1. Linux/macOS one-command startup:

```bash
./run.sh
```

## API Authentication

- Login request:

```http
POST /api/auth/login
Content-Type: application/json

{
    "username": "doctor",
    "password": "doctor123"
}
```

- Include bearer token for protected routes:

```http
Authorization: Bearer <access_token>
```

- Demo users:
    - `doctor / doctor123`
    - `nurse / nurse123`
    - `admin / admin123`

## Release Smoke Test

Run the backend smoke test after starting FastAPI (`uvicorn api:app --port 8000`):

```powershell
./scripts/smoke_test.ps1
```

Optional custom base URL:

```powershell
./scripts/smoke_test.ps1 -BaseUrl "http://127.0.0.1:8000"
```

The script validates:
- `/api/healthcheck`
- `/api/auth/login`
- `/api/auth/me`
- `/api/predict`
- `/api/queue?status=waiting`

## EHR Parser Error Handling

- Empty files return clear `File is empty` validation errors.
- Missing required columns are reported explicitly.
- Invalid date formats include row number and expected format (`YYYY-MM-DD`).
- Non-UTF8 uploads return a friendly decode error telling users to upload UTF-8 CSV.

## Metrics Artifacts

Run:

```bash
python models/train_model.py
```

Outputs:

- `models/saved_models/metrics.json`
- `models/saved_models/confusion_matrix.png`
- `models/saved_models/roc_curve.png`
- `models/saved_models/feature_importance.png`

## Notes

- All storage is local using SQLite (`database/patients.db`).
- Model artifacts are saved under `models/saved_models/`.
- The app assumes training artifacts already exist before launch.
- Frontend is Next.js (`frontend/`) and backend is FastAPI (`api.py`).
