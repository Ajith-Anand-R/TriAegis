from __future__ import annotations

from utils.medication_safety import evaluate_medication_safety


def test_medication_safety_detects_interaction_and_allergy() -> None:
    result = evaluate_medication_safety(
        medications_raw="warfarin, aspirin, ibuprofen",
        allergies_raw="aspirin",
        conditions_raw="heart disease",
    )

    assert result["overall_risk"] == "high"
    assert len(result["interaction_alerts"]) >= 2
    assert len(result["allergy_alerts"]) == 1


def test_medication_safety_detects_condition_contraindication() -> None:
    result = evaluate_medication_safety(
        medications_raw="ibuprofen",
        allergies_raw="",
        conditions_raw="kidney disease,asthma",
    )

    assert result["overall_risk"] in {"medium", "high"}
    assert len(result["contraindication_alerts"]) >= 1


def test_medication_safety_low_risk_path() -> None:
    result = evaluate_medication_safety(
        medications_raw="metformin",
        allergies_raw="penicillin",
        conditions_raw="",
    )

    assert result["overall_risk"] == "low"
    assert result["interaction_alerts"] == []
    assert result["allergy_alerts"] == []
