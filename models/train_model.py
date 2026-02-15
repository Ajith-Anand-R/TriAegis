from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, label_binarize
from xgboost import XGBClassifier


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)


RISK_MAPPING = {"Low": 0, "Medium": 1, "High": 2}

HIGH_RISK_SYMPTOMS = {
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
}

CARDIAC_SYMPTOMS = {"chest pain", "palpitations", "irregular heartbeat"}
RESPIRATORY_SYMPTOMS = {
    "severe shortness of breath",
    "moderate shortness of breath",
    "difficulty breathing",
    "cough",
}
NEUROLOGICAL_SYMPTOMS = {
    "confusion",
    "seizure",
    "stroke symptoms",
    "loss of consciousness",
    "severe headache",
    "dizziness",
    "fainting",
    "mild dizziness",
}


@dataclass
class Paths:
    root: Path

    @property
    def data_csv(self) -> Path:
        return self.root / "data" / "synthetic_patients.csv"

    @property
    def saved_models(self) -> Path:
        return self.root / "models" / "saved_models"

    @property
    def model_path(self) -> Path:
        return self.saved_models / "xgb_risk_classifier.pkl"

    @property
    def preprocessor_path(self) -> Path:
        return self.saved_models / "preprocessor.pkl"

    @property
    def label_encoders_path(self) -> Path:
        return self.saved_models / "label_encoders.pkl"

    @property
    def feature_names_path(self) -> Path:
        return self.saved_models / "feature_names.json"

    @property
    def metrics_path(self) -> Path:
        return self.saved_models / "metrics.json"

    @property
    def confusion_matrix_path(self) -> Path:
        return self.saved_models / "confusion_matrix.png"

    @property
    def roc_curve_path(self) -> Path:
        return self.saved_models / "roc_curve.png"

    @property
    def feature_importance_path(self) -> Path:
        return self.saved_models / "feature_importance.png"


def parse_bp_column(bp_series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    bp_split = bp_series.str.split("/", expand=True)
    systolic = pd.to_numeric(bp_split[0], errors="coerce")
    diastolic = pd.to_numeric(bp_split[1], errors="coerce")
    if systolic.isna().any() or diastolic.isna().any():
        raise ValueError("Invalid Blood Pressure values detected")
    return systolic.astype(int), diastolic.astype(int)


def parse_multi(raw: str) -> List[str]:
    if not isinstance(raw, str) or not raw.strip():
        return []
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return [item for item in values if item.lower() != "none"]


def age_group(age: int) -> str:
    if age <= 12:
        return "child"
    if age <= 30:
        return "young_adult"
    if age <= 50:
        return "adult"
    if age <= 70:
        return "senior"
    return "elderly"


def build_features(dataframe: pd.DataFrame, mlb: MultiLabelBinarizer | None = None) -> Tuple[pd.DataFrame, MultiLabelBinarizer]:
    df = dataframe.copy()

    df["bp_systolic"], df["bp_diastolic"] = parse_bp_column(df["Blood Pressure"])
    df["symptom_list"] = df["Symptoms"].apply(parse_multi)
    df["condition_list"] = df["Pre-Existing Conditions"].apply(parse_multi)

    df["bp_risk_score"] = ((df["bp_systolic"] - 120) / 20) + ((df["bp_diastolic"] - 80) / 10)
    df["hr_risk_score"] = (df["Heart Rate"] - 70).abs() / 30
    df["temp_risk_score"] = (df["Temperature"] - 98.6).abs() / 2

    df["age_group"] = df["Age"].astype(int).apply(age_group)

    if mlb is None:
        mlb = MultiLabelBinarizer()
        symptom_matrix = mlb.fit_transform(df["symptom_list"])
    else:
        symptom_matrix = mlb.transform(df["symptom_list"])

    symptom_columns = [f"symptom_{col.replace(' ', '_')}" for col in mlb.classes_]
    symptom_df = pd.DataFrame(symptom_matrix, columns=symptom_columns, index=df.index)

    df["symptom_count"] = df["symptom_list"].apply(len)
    df["has_high_risk_symptom"] = df["symptom_list"].apply(lambda x: int(any(s in HIGH_RISK_SYMPTOMS for s in x)))
    df["has_cardiac_symptom"] = df["symptom_list"].apply(lambda x: int(any(s in CARDIAC_SYMPTOMS for s in x)))
    df["has_respiratory_symptom"] = df["symptom_list"].apply(lambda x: int(any(s in RESPIRATORY_SYMPTOMS for s in x)))
    df["has_neurological_symptom"] = df["symptom_list"].apply(lambda x: int(any(s in NEUROLOGICAL_SYMPTOMS for s in x)))

    df["condition_count"] = df["condition_list"].apply(len)
    df["has_diabetes"] = df["condition_list"].apply(lambda x: int("diabetes" in x))
    df["has_hypertension"] = df["condition_list"].apply(lambda x: int("hypertension" in x))
    df["has_heart_disease"] = df["condition_list"].apply(lambda x: int("heart disease" in x))
    df["has_respiratory_disease"] = df["condition_list"].apply(lambda x: int("asthma" in x or "COPD" in x))

    def comorbidity_score(items: List[str]) -> int:
        score = 0
        for condition in items:
            if condition == "heart disease":
                score += 3
            elif condition == "diabetes":
                score += 2
            else:
                score += 1
        return score

    df["comorbidity_score"] = df["condition_list"].apply(comorbidity_score)

    df["age_bp_interaction"] = df["Age"] * df["bp_risk_score"]
    df["symptom_condition_interaction"] = df["has_cardiac_symptom"] * df["has_heart_disease"]
    df["age_condition_interaction"] = (df["Age"] / 10) * df["condition_count"]
    df["vitals_severity"] = df["bp_risk_score"] + df["hr_risk_score"] + df["temp_risk_score"]

    base_features = [
        "Age",
        "Heart Rate",
        "Temperature",
        "bp_systolic",
        "bp_diastolic",
        "bp_risk_score",
        "hr_risk_score",
        "temp_risk_score",
        "symptom_count",
        "has_high_risk_symptom",
        "has_cardiac_symptom",
        "has_respiratory_symptom",
        "has_neurological_symptom",
        "condition_count",
        "has_diabetes",
        "has_hypertension",
        "has_heart_disease",
        "has_respiratory_disease",
        "comorbidity_score",
        "age_bp_interaction",
        "symptom_condition_interaction",
        "age_condition_interaction",
        "vitals_severity",
    ]

    categorical = pd.get_dummies(df[["Gender", "age_group"]], prefix=["gender", "age_group"], dtype=int)

    feature_df = pd.concat([df[base_features], categorical, symptom_df], axis=1)
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return feature_df, mlb


def evaluate_model(
    model: XGBClassifier,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, object]:
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)

    accuracy = float(accuracy_score(y_test, y_pred))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test,
        y_pred,
        labels=[0, 1, 2],
        zero_division=0,
    )
    report = classification_report(
        y_test,
        y_pred,
        target_names=["Low", "Medium", "High"],
        output_dict=True,
        zero_division=0,
    )
    conf = confusion_matrix(y_test, y_pred, labels=[0, 1, 2]).tolist()
    roc_auc_ovr = float(roc_auc_score(y_test, y_prob, multi_class="ovr"))

    metrics: Dict[str, object] = {
        "accuracy": accuracy,
        "per_class": {
            "Low": {
                "precision": float(precision[0]),
                "recall": float(recall[0]),
                "f1": float(f1[0]),
                "support": int(support[0]),
            },
            "Medium": {
                "precision": float(precision[1]),
                "recall": float(recall[1]),
                "f1": float(f1[1]),
                "support": int(support[1]),
            },
            "High": {
                "precision": float(precision[2]),
                "recall": float(recall[2]),
                "f1": float(f1[2]),
                "support": int(support[2]),
            },
        },
        "confusion_matrix": conf,
        "classification_report": report,
        "roc_auc_ovr": roc_auc_ovr,
    }
    return metrics


def cross_validate_score(x: pd.DataFrame, y: pd.Series, sample_weights: np.ndarray, seed: int = 42) -> Dict[str, float]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_scores: List[float] = []

    for train_idx, valid_idx in cv.split(x, y):
        x_train = x.iloc[train_idx]
        y_train = y.iloc[train_idx]
        x_valid = x.iloc[valid_idx]
        y_valid = y.iloc[valid_idx]

        fold_model = XGBClassifier(
            n_estimators=300,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            early_stopping_rounds=50,
            random_state=seed,
            n_jobs=-1,
        )
        fold_model.fit(
            x_train,
            y_train,
            sample_weight=sample_weights[train_idx],
            eval_set=[(x_valid, y_valid)],
            verbose=False,
        )
        pred = fold_model.predict(x_valid)
        fold_scores.append(float(accuracy_score(y_valid, pred)))

    return {
        "cv_accuracy_mean": float(np.mean(fold_scores)),
        "cv_accuracy_std": float(np.std(fold_scores)),
    }


def generate_charts(
    paths: Paths,
    model: XGBClassifier,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    feature_columns: List[str],
) -> None:
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)

    conf = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Low", "Medium", "High"],
        yticklabels=["Low", "Medium", "High"],
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(paths.confusion_matrix_path, dpi=180)
    plt.close()

    y_test_binarized = label_binarize(y_test, classes=[0, 1, 2])
    plt.figure(figsize=(8, 6))
    class_labels = {0: "Low", 1: "Medium", 2: "High"}
    plotted = False
    for idx in range(3):
        if len(np.unique(y_test_binarized[:, idx])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_test_binarized[:, idx], y_prob[:, idx])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=2, label=f"{class_labels[idx]} (AUC = {roc_auc:.3f})")
        plotted = True

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (One-vs-Rest)")
    if plotted:
        plt.legend(loc="lower right")
    else:
        plt.text(0.5, 0.5, "ROC unavailable: insufficient class variation in test split", ha="center", va="center")
    plt.tight_layout()
    plt.savefig(paths.roc_curve_path, dpi=180)
    plt.close()

    importance_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False).head(20)

    plt.figure(figsize=(10, 8))
    sns.barplot(data=importance_df, x="importance", y="feature", palette="viridis")
    plt.title("Top Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(paths.feature_importance_path, dpi=180)
    plt.close()


def train(paths: Paths, seed: int = 42) -> Dict[str, object]:
    paths.saved_models.mkdir(parents=True, exist_ok=True)

    if not paths.data_csv.exists():
        raise FileNotFoundError(f"Dataset not found at {paths.data_csv}")

    LOGGER.info("Loading data from %s", paths.data_csv)
    raw_df = pd.read_csv(paths.data_csv)
    if raw_df.isnull().any().any():
        raise ValueError("Input dataset contains null values")

    y = raw_df["Risk_Level"].map(RISK_MAPPING)
    if y.isna().any():
        raise ValueError("Unexpected Risk_Level values in dataset")
    y = y.astype(int)

    feature_df, symptom_mlb = build_features(raw_df)
    LOGGER.info("Prepared %d engineered features", feature_df.shape[1])

    x_train, x_temp, y_train, y_temp = train_test_split(
        feature_df,
        y,
        test_size=0.30,
        random_state=seed,
        stratify=y,
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.50,
        random_state=seed,
        stratify=y_temp,
    )

    all_columns = feature_df.columns.tolist()
    x_train = x_train.reindex(columns=all_columns, fill_value=0)
    x_val = x_val.reindex(columns=all_columns, fill_value=0)
    x_test = x_test.reindex(columns=all_columns, fill_value=0)

    class_weights = {0: 1.0, 1: 2.0, 2: 6.0}
    sample_weights = y_train.map(class_weights).to_numpy(dtype=float)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        early_stopping_rounds=50,
        random_state=seed,
        n_jobs=-1,
    )

    LOGGER.info("Training XGBoost model...")
    model.fit(
        x_train,
        y_train,
        sample_weight=sample_weights,
        eval_set=[(x_val, y_val)],
        verbose=False,
    )

    LOGGER.info("Evaluating model on holdout test set...")
    metrics = evaluate_model(model, x_test, y_test)
    cv_metrics = cross_validate_score(
        feature_df,
        y,
        y.map(class_weights).to_numpy(dtype=float),
        seed=seed,
    )
    metrics.update(cv_metrics)
    metrics["metrics_version"] = "1.0"

    feature_importance = model.feature_importances_
    importance_df = pd.DataFrame(
        {
            "feature": all_columns,
            "importance": feature_importance,
        }
    ).sort_values("importance", ascending=False)
    top_features = importance_df.head(15).to_dict(orient="records")
    metrics["top_feature_importance"] = top_features

    LOGGER.info("Saving model and artifacts...")
    joblib.dump(model, paths.model_path)

    preprocessor_artifact = {
        "symptom_mlb": symptom_mlb,
        "feature_columns": all_columns,
        "gender_levels": sorted(raw_df["Gender"].dropna().unique().tolist()),
        "age_group_levels": ["child", "young_adult", "adult", "senior", "elderly"],
        "risk_mapping": RISK_MAPPING,
    }
    joblib.dump(preprocessor_artifact, paths.preprocessor_path)
    joblib.dump({"risk_mapping": RISK_MAPPING}, paths.label_encoders_path)

    with paths.feature_names_path.open("w", encoding="utf-8") as feature_file:
        json.dump(all_columns, feature_file, indent=2)

    with paths.metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    # ── Generate visual charts ──
    generate_charts(paths, model, x_test, y_test, all_columns)

    LOGGER.info("Model path: %s", paths.model_path)
    LOGGER.info("Preprocessor path: %s", paths.preprocessor_path)
    LOGGER.info("Accuracy: %.4f", metrics["accuracy"])
    LOGGER.info("High-risk recall: %.4f", metrics["per_class"]["High"]["recall"])
    LOGGER.info(
        "CV accuracy: %.4f ± %.4f",
        metrics["cv_accuracy_mean"],
        metrics["cv_accuracy_std"],
    )

    return metrics


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = Paths(root=root)
    metrics = train(paths=paths, seed=42)

    LOGGER.info("Training completed successfully.")
    LOGGER.info("Summary: %s", json.dumps({
        "accuracy": metrics["accuracy"],
        "high_recall": metrics["per_class"]["High"]["recall"],
        "cv_mean": metrics["cv_accuracy_mean"],
        "cv_std": metrics["cv_accuracy_std"],
    }, indent=2))


if __name__ == "__main__":
    main()
