from __future__ import annotations

from typing import Dict, List

import pandas as pd


def _safe_mean(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return float(series.mean())


def build_drift_report(
    predictions: List[Dict[str, object]],
    feedback_rows: List[Dict[str, object]],
    waiting_queue_count: int,
) -> Dict[str, object]:
    pred_df = pd.DataFrame(predictions)
    fb_df = pd.DataFrame(feedback_rows)

    alerts: List[Dict[str, str]] = []

    if pred_df.empty:
        return {
            "status": "insufficient_data",
            "alerts": [
                {
                    "type": "data",
                    "severity": "low",
                    "message": "No predictions available for drift analysis",
                }
            ],
            "metrics": {},
        }

    recent = pred_df.head(100).copy()
    reference = pred_df.iloc[100:300].copy() if len(pred_df) > 100 else pd.DataFrame()

    recent_high_rate = float((recent.get("risk_level", pd.Series(dtype=str)) == "High").mean()) if not recent.empty else 0.0
    reference_high_rate = float((reference.get("risk_level", pd.Series(dtype=str)) == "High").mean()) if not reference.empty else recent_high_rate
    high_risk_shift = recent_high_rate - reference_high_rate

    recent_confidence = _safe_mean(pd.to_numeric(recent.get("model_confidence", pd.Series(dtype=float)), errors="coerce").dropna())
    reference_confidence = _safe_mean(pd.to_numeric(reference.get("model_confidence", pd.Series(dtype=float)), errors="coerce").dropna()) if not reference.empty else recent_confidence
    confidence_shift = recent_confidence - reference_confidence

    if abs(high_risk_shift) >= 0.15:
        alerts.append(
            {
                "type": "distribution_shift",
                "severity": "medium",
                "message": f"High-risk prediction rate shifted by {high_risk_shift:.2f} (recent {recent_high_rate:.2f} vs reference {reference_high_rate:.2f})",
            }
        )

    if confidence_shift <= -0.10:
        alerts.append(
            {
                "type": "confidence_drop",
                "severity": "medium",
                "message": f"Average model confidence dropped by {abs(confidence_shift):.2f}",
            }
        )

    agreement_rate = 0.0
    if not fb_df.empty and {"actual_risk_level", "predicted_risk_level"}.issubset(fb_df.columns):
        agreement_rate = float((fb_df["actual_risk_level"] == fb_df["predicted_risk_level"]).mean())
        if agreement_rate < 0.70:
            alerts.append(
                {
                    "type": "outcome_mismatch",
                    "severity": "high",
                    "message": f"Outcome agreement is low at {agreement_rate:.2f}; review calibration and triage thresholds",
                }
            )

    if waiting_queue_count >= 15:
        alerts.append(
            {
                "type": "queue_overload",
                "severity": "medium",
                "message": f"Waiting queue is high ({waiting_queue_count}); consider surge protocol",
            }
        )

    status = "alert" if alerts else "stable"

    return {
        "status": status,
        "alerts": alerts,
        "metrics": {
            "recent_high_risk_rate": round(recent_high_rate, 4),
            "reference_high_risk_rate": round(reference_high_rate, 4),
            "high_risk_shift": round(high_risk_shift, 4),
            "recent_confidence": round(recent_confidence, 4),
            "reference_confidence": round(reference_confidence, 4),
            "confidence_shift": round(confidence_shift, 4),
            "outcome_agreement_rate": round(agreement_rate, 4),
            "waiting_queue_count": int(waiting_queue_count),
            "prediction_sample_size": int(len(pred_df)),
            "feedback_sample_size": int(len(fb_df)),
        },
    }
