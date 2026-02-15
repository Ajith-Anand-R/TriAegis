from __future__ import annotations

import json
from datetime import datetime, UTC
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd

from utils.database import TriageDatabase
from utils.document_parser import parse_document
from utils.ml_engine import TriageMLEngine


@dataclass
class HealthStatus:
    ok: bool
    checks: Dict[str, str]
    details: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        return {
            "ok": self.ok,
            "checks": self.checks,
            "details": self.details,
        }


def run_healthcheck(root: str | Path | None = None) -> HealthStatus:
    base = Path(root) if root else Path(__file__).resolve().parents[1]

    checks: Dict[str, str] = {}
    details: Dict[str, object] = {}

    required_paths = {
        "dataset": base / "data" / "synthetic_patients.csv",
        "model": base / "models" / "saved_models" / "xgb_risk_classifier.pkl",
        "preprocessor": base / "models" / "saved_models" / "preprocessor.pkl",
        "metrics": base / "models" / "saved_models" / "metrics.json",
        "db": base / "database" / "patients.db",
    }

    for name, path in required_paths.items():
        exists = path.exists()
        checks[f"file:{name}"] = "pass" if exists else "fail"
        details[f"path:{name}"] = str(path)
        if exists and path.is_file():
            stat = path.stat()
            details[f"{name}_file_size_bytes"] = int(stat.st_size)
            details[f"{name}_last_modified_utc"] = datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat()

    model_path = required_paths["model"]
    if model_path.exists():
        try:
            model_obj = joblib.load(model_path)
            checks["model:integrity"] = "pass" if hasattr(model_obj, "predict") else "fail"
            if not hasattr(model_obj, "predict"):
                details["model_integrity_error"] = "Loaded model does not expose predict()"
        except Exception as exc:
            checks["model:integrity"] = "fail"
            details["model_integrity_error"] = str(exc)

    metrics_path = required_paths["metrics"]
    if metrics_path.exists():
        try:
            metrics_data = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics_version = metrics_data.get("metrics_version")
            details["metrics_version"] = metrics_version
            checks["metrics:version"] = "pass" if metrics_version == "1.0" else "fail"
            if metrics_version != "1.0":
                details["metrics_version_error"] = "metrics.json version mismatch; expected 1.0"
        except Exception as exc:
            checks["metrics:version"] = "fail"
            details["metrics_version_error"] = str(exc)

    try:
        parsed_rows = parse_document(base / "data" / "synthetic_patients.csv")
        checks["parser:csv"] = "pass"
        details["parser_rows"] = len(parsed_rows)
    except Exception as exc:
        checks["parser:csv"] = "fail"
        details["parser_error"] = str(exc)
        parsed_rows = []

    try:
        engine = TriageMLEngine(base)
        checks["engine:load"] = "pass"
    except Exception as exc:
        checks["engine:load"] = "fail"
        details["engine_error"] = str(exc)
        engine = None

    if engine is not None and parsed_rows:
        try:
            sample = parsed_rows[0]
            prediction = engine.predict_one(sample)
            checks["engine:predict"] = "pass"
            details["sample_prediction"] = {
                "risk_level": prediction.get("risk_level"),
                "priority_score": prediction.get("priority_score"),
                "department": prediction.get("department"),
            }
        except Exception as exc:
            checks["engine:predict"] = "fail"
            details["predict_error"] = str(exc)

    try:
        db = TriageDatabase(base / "database" / "patients.db")
        queue_size = len(db.get_priority_queue())
        checks["database:connect"] = "pass"
        details["queue_size"] = queue_size
    except Exception as exc:
        checks["database:connect"] = "fail"
        details["database_error"] = str(exc)

    ok = all(status == "pass" for status in checks.values())
    return HealthStatus(ok=ok, checks=checks, details=details)


def healthcheck_dataframe(root: str | Path | None = None) -> pd.DataFrame:
    status = run_healthcheck(root)
    rows = [
        {"check": check_name, "status": check_status}
        for check_name, check_status in status.checks.items()
    ]
    return pd.DataFrame(rows)
