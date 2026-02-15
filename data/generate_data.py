from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from faker import Faker


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)


HIGH_RISK_SYMPTOMS: List[str] = [
    "chest pain",
    "severe shortness of breath",
    "confusion",
    "severe bleeding",
    "loss of consciousness",
    "stroke symptoms",
    "severe abdominal pain",
    "difficulty breathing",
    "severe allergic reaction",
    "uncontrolled bleeding",
    "seizure",
    "severe trauma",
]

MEDIUM_RISK_SYMPTOMS: List[str] = [
    "moderate shortness of breath",
    "high fever",
    "persistent vomiting",
    "severe headache",
    "palpitations",
    "moderate bleeding",
    "severe pain",
    "dizziness",
    "fainting",
    "dehydration",
    "abdominal pain",
    "irregular heartbeat",
]

LOW_RISK_SYMPTOMS: List[str] = [
    "mild headache",
    "cough",
    "cold",
    "minor pain",
    "fatigue",
    "nausea",
    "mild fever",
    "sore throat",
    "runny nose",
    "muscle ache",
    "rash",
    "minor injury",
    "constipation",
    "mild dizziness",
    "back pain",
    "joint pain",
    "insomnia",
    "anxiety",
    "minor cut",
    "sprain",
]

ALL_CONDITIONS: List[str] = [
    "diabetes",
    "hypertension",
    "asthma",
    "heart disease",
    "COPD",
    "kidney disease",
    "obesity",
    "cancer",
    "stroke history",
    "arthritis",
    "high cholesterol",
    "thyroid disorder",
    "anxiety",
    "depression",
]

INFECTION_SYMPTOMS = {
    "high fever",
    "persistent vomiting",
    "dehydration",
    "cough",
    "cold",
    "mild fever",
    "sore throat",
    "runny nose",
}

CARDIAC_SYMPTOMS = {"chest pain", "palpitations", "irregular heartbeat"}


@dataclass(frozen=True)
class GenerationConfig:
    total_patients: int = 10_000
    low_ratio: float = 0.60
    medium_ratio: float = 0.30
    high_ratio: float = 0.10
    seed: int = 42


def _risk_counts(total_patients: int, low_ratio: float, medium_ratio: float, high_ratio: float) -> Dict[str, int]:
    if not np.isclose(low_ratio + medium_ratio + high_ratio, 1.0):
        raise ValueError("Risk ratios must sum to 1.0")

    low_count = int(round(total_patients * low_ratio))
    medium_count = int(round(total_patients * medium_ratio))
    high_count = total_patients - low_count - medium_count

    return {"Low": low_count, "Medium": medium_count, "High": high_count}


def _generate_patient_ids(count: int, faker: Faker) -> List[str]:
    ids: List[str] = []
    seen = set()
    while len(ids) < count:
        patient_id = faker.uuid4()
        if patient_id not in seen:
            seen.add(patient_id)
            ids.append(patient_id)
    return ids


def _generate_ages_by_risk(risk_levels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    ages = np.zeros_like(risk_levels, dtype=np.int32)

    low_idx = np.where(risk_levels == "Low")[0]
    med_idx = np.where(risk_levels == "Medium")[0]
    high_idx = np.where(risk_levels == "High")[0]

    ages[low_idx] = np.clip(rng.normal(loc=45, scale=16, size=len(low_idx)).round().astype(int), 18, 90)
    ages[med_idx] = np.clip(rng.normal(loc=55, scale=14, size=len(med_idx)).round().astype(int), 18, 90)
    ages[high_idx] = np.clip(rng.normal(loc=67, scale=11, size=len(high_idx)).round().astype(int), 18, 90)

    return ages


def _generate_genders(count: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(["Male", "Female", "Other"], size=count, p=[0.48, 0.48, 0.04])


def _condition_count_for_age(age: int, rng: np.random.Generator) -> int:
    if age <= 40:
        if rng.random() > 0.20:
            return 0
        return int(rng.integers(1, 2))
    if age <= 60:
        if rng.random() > 0.60:
            return 0
        return int(rng.integers(1, 3))
    if age <= 80:
        if rng.random() > 0.85:
            return int(rng.integers(0, 2))
        return int(rng.integers(2, 4))
    if rng.random() > 0.95:
        return int(rng.integers(1, 3))
    return int(rng.integers(2, 5))


def _generate_conditions(ages: np.ndarray, rng: np.random.Generator) -> List[str]:
    output: List[str] = []
    for age in ages:
        count = _condition_count_for_age(int(age), rng)
        if count == 0:
            output.append("none")
            continue
        choices = rng.choice(ALL_CONDITIONS, size=count, replace=False)
        output.append(",".join(sorted(choices.tolist())))
    return output


def _parse_multi_value_field(raw_value: str) -> List[str]:
    if not raw_value:
        return []
    parts = [token.strip() for token in raw_value.split(",") if token.strip()]
    return [token for token in parts if token.lower() != "none"]


def _build_symptoms(risk_level: str, rng: np.random.Generator) -> List[str]:
    if risk_level == "High":
        total = int(rng.integers(2, 5))
        symptoms = [rng.choice(HIGH_RISK_SYMPTOMS)]
        pool = list((set(HIGH_RISK_SYMPTOMS) | set(MEDIUM_RISK_SYMPTOMS)) - set(symptoms))
        extra = rng.choice(pool, size=max(0, total - 1), replace=False).tolist()
        symptoms.extend(extra)
        return symptoms

    if risk_level == "Medium":
        total = int(rng.integers(2, 5))
        symptoms = [rng.choice(MEDIUM_RISK_SYMPTOMS)]
        pool = list((set(MEDIUM_RISK_SYMPTOMS) | set(LOW_RISK_SYMPTOMS)) - set(symptoms))
        extra = rng.choice(pool, size=max(0, total - 1), replace=False).tolist()
        symptoms.extend(extra)
        return symptoms

    total = int(rng.integers(1, 4))
    symptoms = rng.choice(LOW_RISK_SYMPTOMS, size=total, replace=False).tolist()
    return symptoms


def _generate_symptoms(risk_levels: np.ndarray, rng: np.random.Generator) -> List[str]:
    symptom_strings: List[str] = []
    for risk_level in risk_levels:
        symptom_strings.append(",".join(_build_symptoms(str(risk_level), rng)))
    return symptom_strings


def _parse_bp_range(risk_level: str) -> Tuple[int, int, int, int]:
    if risk_level == "High":
        return (160, 200, 100, 120)
    if risk_level == "Medium":
        return (135, 179, 88, 109)
    return (108, 139, 68, 89)


def _generate_vitals(
    risk_levels: np.ndarray,
    ages: np.ndarray,
    symptoms: Sequence[str],
    conditions: Sequence[str],
    rng: np.random.Generator,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    bp_values: List[str] = []
    hr_values = np.zeros(len(risk_levels), dtype=np.int32)
    temp_values = np.zeros(len(risk_levels), dtype=np.float32)

    for idx, risk_level in enumerate(risk_levels):
        symptom_list = _parse_multi_value_field(symptoms[idx])
        condition_list = _parse_multi_value_field(conditions[idx])

        syst_min, syst_max, dias_min, dias_max = _parse_bp_range(str(risk_level))
        age_adjust = max(0, int((int(ages[idx]) - 50) * 0.35))

        systolic = int(rng.integers(syst_min, syst_max + 1) + age_adjust)
        diastolic = int(rng.integers(dias_min, dias_max + 1) + max(0, int(age_adjust * 0.4)))

        if risk_level == "High" and rng.random() < 0.42:
            systolic = int(rng.integers(180, 201))
            diastolic = int(max(diastolic, rng.integers(110, 121)))

        systolic = int(np.clip(systolic, 85, 220))
        diastolic = int(np.clip(diastolic, 50, 130))
        bp_values.append(f"{systolic}/{diastolic}")

        if risk_level == "High":
            hr = int(rng.integers(112, 151))
            if rng.random() < 0.08:
                hr = int(rng.integers(42, 50))
        elif risk_level == "Medium":
            hr = int(rng.integers(95, 131))
        else:
            hr = int(rng.integers(58, 98))

        if CARDIAC_SYMPTOMS.intersection(symptom_list):
            hr += int(rng.integers(8, 21))
        hr = int(np.clip(hr, 40, 170))
        hr_values[idx] = hr

        if risk_level == "High":
            temp = float(rng.uniform(101.6, 104.0))
        elif risk_level == "Medium":
            temp = float(rng.uniform(99.8, 102.9))
        else:
            temp = float(rng.uniform(97.5, 100.3))

        if INFECTION_SYMPTOMS.intersection(symptom_list):
            temp += float(rng.uniform(0.2, 1.3))
        if "COPD" in condition_list or "asthma" in condition_list:
            temp += float(rng.uniform(-0.2, 0.4))

        temp_values[idx] = float(np.clip(temp, 95.0, 105.5))

    return bp_values, hr_values, np.round(temp_values, 1)


def _enforce_high_risk_criteria(
    dataframe: pd.DataFrame,
    rng: np.random.Generator,
) -> None:
    high_mask = dataframe["Risk_Level"] == "High"

    for idx in dataframe[high_mask].index:
        symptoms = _parse_multi_value_field(dataframe.at[idx, "Symptoms"])
        bp_s, bp_d = [int(part) for part in dataframe.at[idx, "Blood Pressure"].split("/")]
        hr = int(dataframe.at[idx, "Heart Rate"])
        temp = float(dataframe.at[idx, "Temperature"])
        age = int(dataframe.at[idx, "Age"])
        condition_count = len(_parse_multi_value_field(dataframe.at[idx, "Pre-Existing Conditions"]))

        criteria = [
            any(symptom in HIGH_RISK_SYMPTOMS for symptom in symptoms),
            bp_s >= 180 or bp_d >= 110,
            hr > 130 or hr < 50,
            temp >= 103.0,
            age > 75 and condition_count >= 3,
            "chest pain" in symptoms and (bp_s >= 140 or hr >= 100),
            "confusion" in symptoms and len(symptoms) >= 2,
        ]

        if any(criteria):
            continue

        if rng.random() < 0.5:
            symptoms.append("chest pain")
            dataframe.at[idx, "Symptoms"] = ",".join(sorted(set(symptoms)))
        else:
            new_bp_s = max(bp_s, int(rng.integers(180, 201)))
            new_bp_d = max(bp_d, int(rng.integers(110, 121)))
            dataframe.at[idx, "Blood Pressure"] = f"{new_bp_s}/{new_bp_d}"


def _validate_dataset(dataframe: pd.DataFrame, expected_counts: Dict[str, int]) -> None:
    required_columns = [
        "Patient_ID",
        "Age",
        "Gender",
        "Symptoms",
        "Blood Pressure",
        "Heart Rate",
        "Temperature",
        "Pre-Existing Conditions",
        "Risk_Level",
    ]

    missing = [column for column in required_columns if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if dataframe[required_columns].isnull().any().any():
        raise ValueError("Dataset contains null values")

    if not dataframe["Patient_ID"].is_unique:
        raise ValueError("Patient_ID values are not unique")

    actual_counts = dataframe["Risk_Level"].value_counts().to_dict()
    for risk in ("Low", "Medium", "High"):
        if actual_counts.get(risk, 0) != expected_counts[risk]:
            raise ValueError(
                f"Risk distribution mismatch for {risk}: expected {expected_counts[risk]}, got {actual_counts.get(risk, 0)}"
            )

    bp_parts = dataframe["Blood Pressure"].str.split("/", expand=True).astype(int)
    if (bp_parts[0] < 85).any() or (bp_parts[0] > 220).any():
        raise ValueError("Systolic BP out of medically valid bounds")
    if (bp_parts[1] < 50).any() or (bp_parts[1] > 130).any():
        raise ValueError("Diastolic BP out of medically valid bounds")
    if (dataframe["Heart Rate"] < 40).any() or (dataframe["Heart Rate"] > 170).any():
        raise ValueError("Heart rate out of medically valid bounds")
    if (dataframe["Temperature"] < 95.0).any() or (dataframe["Temperature"] > 105.5).any():
        raise ValueError("Temperature out of medically valid bounds")


def generate_synthetic_patients(config: GenerationConfig) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    faker = Faker()
    Faker.seed(config.seed)

    counts = _risk_counts(config.total_patients, config.low_ratio, config.medium_ratio, config.high_ratio)
    risk_levels = np.array(
        ["Low"] * counts["Low"] + ["Medium"] * counts["Medium"] + ["High"] * counts["High"],
        dtype=object,
    )
    rng.shuffle(risk_levels)

    patient_ids = _generate_patient_ids(config.total_patients, faker)
    ages = _generate_ages_by_risk(risk_levels, rng)
    genders = _generate_genders(config.total_patients, rng)
    conditions = _generate_conditions(ages, rng)
    symptoms = _generate_symptoms(risk_levels, rng)
    blood_pressure, heart_rate, temperature = _generate_vitals(risk_levels, ages, symptoms, conditions, rng)

    dataframe = pd.DataFrame(
        {
            "Patient_ID": patient_ids,
            "Age": ages,
            "Gender": genders,
            "Symptoms": symptoms,
            "Blood Pressure": blood_pressure,
            "Heart Rate": heart_rate,
            "Temperature": temperature,
            "Pre-Existing Conditions": conditions,
            "Risk_Level": risk_levels,
        }
    )

    _enforce_high_risk_criteria(dataframe, rng)
    _validate_dataset(dataframe, counts)

    return dataframe


def save_dataset(dataframe: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)
    LOGGER.info("Saved dataset to %s", output_path)


def _print_summary(dataframe: pd.DataFrame) -> None:
    counts = dataframe["Risk_Level"].value_counts().reindex(["Low", "Medium", "High"])
    LOGGER.info("Risk distribution: %s", counts.to_dict())
    LOGGER.info(
        "Age stats | mean=%.2f std=%.2f min=%d max=%d",
        dataframe["Age"].mean(),
        dataframe["Age"].std(),
        int(dataframe["Age"].min()),
        int(dataframe["Age"].max()),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic patient triage data")
    parser.add_argument("--total-patients", type=int, default=10_000, help="Number of synthetic patients to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible generation")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "synthetic_patients.csv",
        help="Output CSV file path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = GenerationConfig(total_patients=args.total_patients, seed=args.seed)

    LOGGER.info("Generating %d synthetic patients...", config.total_patients)
    dataframe = generate_synthetic_patients(config)
    save_dataset(dataframe, args.output)
    _print_summary(dataframe)


if __name__ == "__main__":
    main()
