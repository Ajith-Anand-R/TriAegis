from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

from utils.department_recommender import build_triage_decision
from models.train_model import RISK_MAPPING, build_features


RISK_LABELS = {value: key for key, value in RISK_MAPPING.items()}


@dataclass
class PredictionResult:
    patient_id: str
    risk_level: str
    confidence: float
    probabilities: Dict[str, float]
    priority_score: float
    priority_category: str
    department: str
    department_reason: str
    urgency: str
    estimated_wait_time: str
    routing_logic: str
    queue_position: int
    processing_time_ms: int
    ehr_history_score: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "patient_id": self.patient_id,
            "risk_level": self.risk_level,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "priority_score": self.priority_score,
            "priority_category": self.priority_category,
            "department": self.department,
            "department_reason": self.department_reason,
            "urgency": self.urgency,
            "estimated_wait_time": self.estimated_wait_time,
            "routing_logic": self.routing_logic,
            "queue_position": self.queue_position,
            "processing_time_ms": self.processing_time_ms,
            "ehr_history_score": self.ehr_history_score,
        }


class TriageMLEngine:
    def __init__(self, root: str | Path | None = None) -> None:
        base = Path(root) if root else Path(__file__).resolve().parents[1]
        self.root = base
        self.saved_models = self.root / "models" / "saved_models"

        self.model = joblib.load(self.saved_models / "xgb_risk_classifier.pkl")
        self.preprocessor = joblib.load(self.saved_models / "preprocessor.pkl")

        self.feature_columns: List[str] = list(self.preprocessor["feature_columns"])
        self.symptom_mlb = self.preprocessor["symptom_mlb"]

    def _prepare_features(self, input_df: pd.DataFrame) -> pd.DataFrame:
        features, _ = build_features(input_df, mlb=self.symptom_mlb)
        features = features.reindex(columns=self.feature_columns, fill_value=0)
        return features

    def _risk_from_prediction(self, probabilities: np.ndarray) -> tuple[str, Dict[str, float], float]:
        pred_index = int(np.argmax(probabilities))
        risk_level = RISK_LABELS[pred_index]
        probs = {
            "Low": round(float(probabilities[0]), 4),
            "Medium": round(float(probabilities[1]), 4),
            "High": round(float(probabilities[2]), 4),
        }
        confidence = round(float(np.max(probabilities)), 4)
        return risk_level, probs, confidence

    def predict_one(
        self,
        patient: Dict[str, object],
        queue_position: int = 1,
        ehr_history_score: float = 0.0,
    ) -> Dict[str, object]:
        start = time.perf_counter()

        patient_row = pd.DataFrame([patient]).copy()
        if "Risk_Level" not in patient_row.columns:
            patient_row["Risk_Level"] = "Low"

        features = self._prepare_features(patient_row)
        probabilities = self.model.predict_proba(features)[0]
        risk_level, probs, confidence = self._risk_from_prediction(probabilities)

        enriched_patient = dict(patient)
        enriched_patient["Risk_Level"] = risk_level

        triage_decision = build_triage_decision(
            enriched_patient,
            queue_position=queue_position,
            ehr_history_score=ehr_history_score,
        )

        duration_ms = int((time.perf_counter() - start) * 1000)
        result = PredictionResult(
            patient_id=str(patient.get("Patient_ID", "unknown")),
            risk_level=risk_level,
            confidence=confidence,
            probabilities=probs,
            priority_score=float(triage_decision["priority_score"]),
            priority_category=str(triage_decision["priority_category"]),
            department=str(triage_decision["department"]),
            department_reason=str(triage_decision["department_reason"]),
            urgency=str(triage_decision["urgency"]),
            estimated_wait_time=str(triage_decision["estimated_wait_time"]),
            routing_logic=str(triage_decision["routing_logic"]),
            queue_position=queue_position,
            processing_time_ms=duration_ms,
            ehr_history_score=ehr_history_score,
        )
        return result.to_dict()

    def predict_batch(
        self,
        patients: List[Dict[str, object]],
        ehr_scores: Dict[str, float] | None = None,
    ) -> List[Dict[str, object]]:
        if not patients:
            return []

        start = time.perf_counter()
        input_df = pd.DataFrame(patients).copy()
        if "Risk_Level" not in input_df.columns:
            input_df["Risk_Level"] = "Low"

        features = self._prepare_features(input_df)
        probabilities_batch = self.model.predict_proba(features)
        elapsed_total_ms = int((time.perf_counter() - start) * 1000)

        results: List[Dict[str, object]] = []
        for idx, probs in enumerate(probabilities_batch):
            risk_level, prob_map, confidence = self._risk_from_prediction(probs)
            patient = dict(patients[idx])
            patient["Risk_Level"] = risk_level

            queue_position = idx + 1
            pid = str(patient.get("Patient_ID", f"patient_{idx + 1}"))
            score = (ehr_scores or {}).get(pid, 0.0)
            triage_decision = build_triage_decision(
                patient,
                queue_position=queue_position,
                ehr_history_score=score,
            )

            result = {
                "patient_id": pid,
                "risk_level": risk_level,
                "confidence": confidence,
                "probabilities": prob_map,
                "priority_score": float(triage_decision["priority_score"]),
                "priority_category": str(triage_decision["priority_category"]),
                "department": str(triage_decision["department"]),
                "department_reason": str(triage_decision["department_reason"]),
                "urgency": str(triage_decision["urgency"]),
                "estimated_wait_time": str(triage_decision["estimated_wait_time"]),
                "routing_logic": str(triage_decision["routing_logic"]),
                "queue_position": queue_position,
                "processing_time_ms": max(1, elapsed_total_ms // len(patients)),
                "ehr_history_score": score,
            }
            results.append(result)

        results.sort(key=lambda x: (-x["priority_score"], -int(next((p.get("Age", 0) for p in patients if str(p.get("Patient_ID", "")) == x["patient_id"]), 0)), x["queue_position"]))

        for new_position, result in enumerate(results, start=1):
            result["queue_position"] = new_position

        return results
