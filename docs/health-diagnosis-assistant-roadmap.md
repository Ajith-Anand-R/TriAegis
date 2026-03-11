# TriAegis Health Diagnosis Assistant Audit and Roadmap

## 1) Current Feature Inventory (Implemented)

### Core backend and ML
- Risk prediction engine (`utils/ml_engine.py`) with:
  - `predict_one`, `predict_batch`
  - Probability output (`Low/Medium/High`), confidence, OOD flag, ESI mapping
  - Differential diagnosis suggestion list
- Triage logic (`utils/department_recommender.py`) with:
  - Priority scoring (1-10)
  - Department routing
  - Wait-time estimation
- EHR history scoring (`utils/ehr_analyzer.py`) with:
  - CSV validation/parsing
  - Historical risk bonus (0.0-3.0)
  - Clinical risk-factor summary
- Explainability (`utils/explainer.py`): SHAP contributors + protective factors
- Data/document ingestion (`utils/document_parser.py`): CSV/XLSX/XLS/PDF/DOCX
- Persistence and queueing (`utils/database.py`):
  - Patient records, predictions, queue status, vitals snapshots
  - Outcome feedback storage
  - Continuous monitoring + trend-based escalation

### API and workflows
- Auth + role checks (`auth.py`, `api.py`): JWT auth, role-based endpoint guards
- Prediction endpoints (`/api/predict`, `/api/predict/save`, batch variants)
- Queue operations (`/api/queue`, status transitions, next patient)
- Analytics/history/fairness endpoints
- Outcome feedback endpoint for predicted-vs-actual tracking
- Simulation endpoints for synthetic queue load testing
- Export endpoints (PDF/CSV)

### Frontend
- Next.js app with pages:
  - Dashboard, analysis, diagnosis, queue, history, analytics, fairness, simulation, health, login
- Analysis and diagnosis flows already support:
  - Symptom and vital input
  - Batch upload
  - Differential diagnosis display
  - Manual review alerts
- Operational pages for queue and history with admin controls

### Testing and scripts
- Broad integration/unit coverage in `tests/test_system.py`
- Self-check script (`scripts/system_self_check.py`)
- Train/evaluate pipeline in `models/train_model.py`

## 2) Key Gaps to Become a Strong Health Diagnosis Assistant

## Clinical safety gaps
- No explicit red-flag protocol endpoint for immediate emergency triggers independent of model output.
- No standardized escalation/handoff summary format (SBAR-style) for clinician workflows.
- No medication interaction/allergy safety screening module.
- No explicit contraindication checks for age/pregnancy/comorbidity-sensitive recommendations.

## Data and governance gaps
- No audit trail for who viewed/updated sensitive records.
- JWT secret has a permissive default fallback in code (`auth.py`).
- No PHI masking/redaction mode in analytics/history exports.
- No consent and retention policy enforcement by patient-level flags.

## Model reliability gaps
- No model version pinning in prediction responses and persisted rows.
- No drift monitoring against live outcome feedback.
- No threshold calibration endpoint for high-risk sensitivity tuning.

## Product workflow gaps
- No structured intake protocol (chief complaint -> red flags -> focused questions -> recommendation).
- No follow-up plan generation (home care warning signs, re-visit windows, monitoring checklist).
- No specialist handoff packet auto-generation for cross-department transfers.

## Engineering/ops gaps
- Missing CI checks for backend + frontend test matrix.
- Sparse endpoint-level tests for API auth/authorization and failure modes.
- Legacy/unused structure present: empty `TriAegis/frontend/` folder.
- Potential stack overlap: Streamlit app in `app.py` + Next.js frontend in `frontend/`.

## 3) Prioritized Feature Additions

## P0 (High impact, safety first)
1. Red-flag emergency protocol service.
2. Clinician handoff summary generator (SBAR-like output).
3. Medication/allergy interaction screening.
4. Mandatory safety disclaimer + manual-review gate for uncertain/OOD outputs.
5. Secure secrets policy (fail startup if JWT secret uses insecure default in non-dev).

## P1 (Care quality and reliability)
1. Follow-up care plan generation with warning signs and revisit instructions.
2. Model-versioning and inference metadata logging.
3. Outcome-feedback powered calibration dashboard.
4. PHI-safe export mode and role-based field masking.
5. Endpoint-level API tests (authz, validation, error contracts).

## P2 (Scale and product maturity)
1. Alerting and monitoring (queue overload, inference failures, drift signals).
2. Human-in-the-loop triage review queue.
3. Cohort analytics (comorbidity and age-specific pathways).
4. Cleanup of legacy/unused folders and docs consolidation.
5. Optional FHIR-compatible data mapping layer.

## 4) Step-by-Step To-Do (Additive, no existing behavior changes)

1. Add a dedicated clinical safety module (`utils/clinical_guardrails.py`) for red-flag checks.
2. Add new safety API endpoint(s) under `/api/safety/*` without altering existing endpoints.
3. Add tests for red-flag protocol outcomes and edge cases.
4. Add handoff summary module and endpoint.
5. Add medication/allergy checker module and endpoint.
6. Add follow-up plan generator module and endpoint.
7. Add model metadata/version fields to new storage tables (keep old tables intact).
8. Add PHI-safe export endpoint variants.
9. Add drift-monitor script and dashboard endpoint.
10. Add CI workflow for backend/frontend tests and linting.

## 5) Immediate Quick Wins (1-2 days)

1. Implement red-flag protocol API + tests.
2. Add clinician handoff summary endpoint using existing prediction + explanation output.
3. Add security hardening check for JWT secret configuration.
4. Add documentation updates and endpoint examples.

## 7) Implemented in This Iteration

- Added red-flag safety rules module: `utils/clinical_guardrails.py`.
- Added clinician handoff summary module: `utils/clinical_handoff.py`.
- Added medication safety screening module: `utils/medication_safety.py`.
- Added follow-up care planning module: `utils/followup_planner.py`.
- Added PHI/privacy masking utility: `utils/privacy.py`.
- Added safety endpoints in `api.py`:
  - `POST /api/safety/red-flags`
  - `POST /api/safety/handoff-summary`
  - `POST /api/safety/medication-screen`
  - `POST /api/safety/followup-plan`
- Added security/privacy endpoints in `api.py`:
  - `GET /api/security/config` (admin-only security configuration check)
  - `GET /api/history/phi-safe` (redacted records view)
  - `POST /api/export/csv/phi-safe` (PHI-safe CSV export)
- Added focused tests:
  - `tests/test_clinical_guardrails.py`
  - `tests/test_clinical_handoff.py`
  - `tests/test_medication_safety.py`
  - `tests/test_followup_planner.py`
  - `tests/test_privacy_controls.py`
  - `tests/test_api_security_and_contracts.py`
- Added CI workflow for backend and frontend checks:
  - `.github/workflows/ci.yml`
- Added `httpx` to `requirements.txt` for FastAPI TestClient support in API contract tests.
- Added monitoring/drift utility: `utils/monitoring.py`.
- Added drift endpoint:
  - `GET /api/monitoring/drift`
- Added monitoring tests:
  - `tests/test_monitoring.py`

## 6) Notes About Existing Structure

- `TriAegis/frontend/` is currently empty and appears unused.
- `app.py` (Streamlit) and `frontend/` (Next.js) both exist; this can confuse deployment and should be documented as legacy vs primary UI.
- Production path appears to be FastAPI + Next.js per root `README.md` and scripts.
