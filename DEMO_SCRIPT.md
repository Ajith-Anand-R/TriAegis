# 5-Minute Demo Script

## 0:00 - 0:40 | Problem + Solution

- Hospitals need fast triage and explainable prioritization.
- This system predicts risk (Low/Medium/High), recommends department routing, and assigns queue priority (1-10).

## 0:40 - 2:10 | Single Patient Flow

1. Open `Single Patient Analysis` tab.
2. Enter a high-risk profile (e.g., chest pain + elevated BP).
3. Click `Analyze Patient`.
4. Show:
   - Risk prediction + confidence
   - Priority score/category + estimated wait
   - Recommended department with reasoning
   - SHAP top contributors (why model predicted this risk)
5. Click `Save to Database` and `Export as PDF`.

## 2:10 - 3:10 | Batch Flow

1. Open `Batch Processing` tab.
2. Upload `tests/sample_data/test_batch.csv`.
3. Click `Process All Patients`.
4. Show queue-sorted output and risk distribution.
5. Click `Save All to Database`, `Download Results CSV`, and `Download Priority Queue PDF`.

## 3:10 - 4:10 | Queue + History

1. Open `Priority Queue Management` tab.
2. Show filters by status/department.
3. Click `Call Next Patient`, then `Mark as Completed`.
4. Open `Patient History` tab.
5. Filter by risk/department and export CSV/PDF report.

## 4:10 - 5:00 | Technical Highlights + Close

- Data pipeline: synthetic data generation with controlled class ratios.
- ML stack: XGBoost multi-class classifier with engineered clinical features.
- Explainability: SHAP contributor breakdown for transparent decisions.
- Persistence: SQLite schema for predictions, explanations, and queue state.
- Testing: `11 passed` integration/edge tests.

## Backup if Demo Fails

- Run tests live: `pytest tests/test_system.py -q`
- Show generated artifacts:
  - `models/saved_models/xgb_risk_classifier.pkl`
  - `models/saved_models/metrics.json`
  - `data/synthetic_patients.csv`
