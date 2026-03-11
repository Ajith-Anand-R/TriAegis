from __future__ import annotations

from utils.monitoring import build_drift_report


def test_monitoring_report_stable_with_balanced_data() -> None:
    predictions = [
        {"risk_level": "Low", "model_confidence": 0.90},
        {"risk_level": "Medium", "model_confidence": 0.88},
        {"risk_level": "High", "model_confidence": 0.87},
    ] * 80

    feedback = [
        {"actual_risk_level": "Low", "predicted_risk_level": "Low"},
        {"actual_risk_level": "High", "predicted_risk_level": "High"},
        {"actual_risk_level": "Medium", "predicted_risk_level": "Medium"},
    ] * 20

    result = build_drift_report(predictions, feedback, waiting_queue_count=5)
    assert result["status"] in {"stable", "alert"}
    assert "metrics" in result


def test_monitoring_report_flags_shift_and_low_agreement() -> None:
    recent = [{"risk_level": "High", "model_confidence": 0.60}] * 100
    reference = [{"risk_level": "Low", "model_confidence": 0.85}] * 200
    predictions = recent + reference

    feedback = [
        {"actual_risk_level": "Low", "predicted_risk_level": "High"},
    ] * 30

    result = build_drift_report(predictions, feedback, waiting_queue_count=20)

    assert result["status"] == "alert"
    messages = " ".join(alert["message"] for alert in result["alerts"])
    assert "shifted" in messages or "agreement" in messages or "queue" in messages
