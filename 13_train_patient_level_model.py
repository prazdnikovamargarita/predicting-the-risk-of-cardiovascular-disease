from __future__ import annotations

from pathlib import Path

from pipeline_common import INPUT_DIR, OUTPUT_DIR

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None


# ============================== CONFIG ==============================
# ============================== INPUT ==============================
INPUT_POSITIVE_CSV = INPUT_DIR / "modeling" / "records_positive.csv"
INPUT_NEGATIVE_CSV = INPUT_DIR / "modeling" / "records_negative.csv"
# ============================== OUTPUT =============================
OUTPUT_MODEL_DIR = OUTPUT_DIR / "modeling"
# ====================================================================

ID_COL = "ID_PT"
LABEL_COL = "label"

# One file replaces all old `7*` scripts.
MODEL_TYPE = "random_forest"  # "random_forest" or "xgboost"
FEATURE_MODE = "auto_numeric"  # "auto_numeric", "blood_pressure", or "ecg_selected"

N_UNIQUE_IDS_PER_CLASS = 2800
N_SPLITS = 5
RANDOM_STATE = 42
THRESHOLD = 0.50
PATIENT_PROBABILITY_AGGREGATION = "mean"  # "mean" or "max"

BLOOD_PRESSURE_FEATURES = [
    "upper_at",
    "down_at",
    "upper_at_minus_down_at",
    "mean_arterial_pressure",
]

ECG_SELECTED_FEATURES = [
    "RR_Interval_ms",
    "PRratio",
    "QTnorm",
    "Delta_QT",
    "TQRSratio",
    "Delta_QRS_T",
    "Rmean",
    "STmean",
    "RSratio",
    "TRratio",
    "Delta_R",
]

SERVICE_COLUMNS = {
    ID_COL,
    LABEL_COL,
    "Prompt",
    "Unnamed: 0",
    "Date",
    "DATE_F",
    "match_text",
    "source_file",
}
# ====================================================================


def add_standard_blood_pressure_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if {"upper_at", "down_at"}.issubset(df.columns):
        df["upper_at"] = pd.to_numeric(df["upper_at"], errors="coerce")
        df["down_at"] = pd.to_numeric(df["down_at"], errors="coerce")
        df["upper_at_minus_down_at"] = df["upper_at"] - df["down_at"]
        df["mean_arterial_pressure"] = df["down_at"] + (df["upper_at_minus_down_at"] / 3.0)
    return df


def load_and_prepare_data() -> pd.DataFrame:
    df_positive = pd.read_csv(INPUT_POSITIVE_CSV)
    df_negative = pd.read_csv(INPUT_NEGATIVE_CSV)

    df_positive[LABEL_COL] = 1
    df_negative[LABEL_COL] = 0

    if ID_COL not in df_positive.columns or ID_COL not in df_negative.columns:
        raise KeyError(f"Both input files must contain ID column: {ID_COL}")

    df_positive[ID_COL] = df_positive[ID_COL].astype(str).str.strip()
    df_negative[ID_COL] = df_negative[ID_COL].astype(str).str.strip()

    ids_positive = set(df_positive[ID_COL].unique())
    ids_negative = set(df_negative[ID_COL].unique())
    overlapping_ids = ids_positive & ids_negative

    if overlapping_ids:
        # Negative class loses overlapping patients to avoid mixed labels.
        df_negative = df_negative[~df_negative[ID_COL].isin(overlapping_ids)].copy()

    rng = np.random.default_rng(RANDOM_STATE)

    ids_positive = np.array(list(df_positive[ID_COL].unique()))
    ids_negative = np.array(list(df_negative[ID_COL].unique()))

    if N_UNIQUE_IDS_PER_CLASS:
        positive_n = min(N_UNIQUE_IDS_PER_CLASS, len(ids_positive))
        negative_n = min(N_UNIQUE_IDS_PER_CLASS, len(ids_negative))

        sampled_positive = rng.choice(ids_positive, size=positive_n, replace=False)
        sampled_negative = rng.choice(ids_negative, size=negative_n, replace=False)

        df_positive = df_positive[df_positive[ID_COL].isin(sampled_positive)].copy()
        df_negative = df_negative[df_negative[ID_COL].isin(sampled_negative)].copy()

    df = pd.concat([df_negative, df_positive], ignore_index=True)
    df = df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    df = add_standard_blood_pressure_features(df)

    df[LABEL_COL] = df[LABEL_COL].astype(int)
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    if FEATURE_MODE == "blood_pressure":
        feature_cols = [col for col in BLOOD_PRESSURE_FEATURES if col in df.columns]
    elif FEATURE_MODE == "ecg_selected":
        feature_cols = [col for col in ECG_SELECTED_FEATURES if col in df.columns]
    elif FEATURE_MODE == "auto_numeric":
        feature_cols = (
            df.drop(columns=list(SERVICE_COLUMNS & set(df.columns)), errors="ignore")
            .select_dtypes(include="number")
            .columns.tolist()
        )
    else:
        raise ValueError(f"Unknown FEATURE_MODE: {FEATURE_MODE}")

    if not feature_cols:
        raise ValueError("No feature columns were selected. Check FEATURE_MODE and input columns.")

    return feature_cols


def assert_one_label_per_patient(df: pd.DataFrame) -> None:
    label_counts = df.groupby(ID_COL)[LABEL_COL].nunique()
    bad = label_counts[label_counts != 1]
    if len(bad) > 0:
        raise ValueError(
            "Some patients have more than one label. "
            "Patient-level cross-validation requires one class per patient."
        )


def build_model(y_train: np.ndarray):
    if MODEL_TYPE == "random_forest":
        return RandomForestClassifier(
            n_estimators=400,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
            min_samples_leaf=5,
            min_samples_split=10,
        )

    if MODEL_TYPE == "xgboost":
        if XGBClassifier is None:
            raise ImportError("xgboost is not installed. Install it or use MODEL_TYPE='random_forest'.")

        n_positive = np.sum(y_train == 1)
        n_negative = np.sum(y_train == 0)
        scale_pos_weight = (n_negative / n_positive) if n_positive > 0 else 1.0

        return XGBClassifier(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            importance_type="gain",
        )

    raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")


def aggregate_patient_probabilities(row_level_predictions: pd.DataFrame) -> pd.DataFrame:
    if PATIENT_PROBABILITY_AGGREGATION == "max":
        agg_func = "max"
    elif PATIENT_PROBABILITY_AGGREGATION == "mean":
        agg_func = "mean"
    else:
        raise ValueError("PATIENT_PROBABILITY_AGGREGATION must be 'mean' or 'max'.")

    return (
        row_level_predictions
        .groupby(ID_COL)
        .agg(
            label=(LABEL_COL, "max"),
            probability=("probability", agg_func),
            fold=("fold", "first"),
        )
        .reset_index()
    )


def run_patient_level_cross_validation(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    assert_one_label_per_patient(df)

    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[ID_COL, LABEL_COL]).reset_index(drop=True)

    patient_labels = df.groupby(ID_COL)[LABEL_COL].max()
    patient_ids = patient_labels.index.to_numpy()
    patient_y = patient_labels.to_numpy()

    min_class_size = min(np.sum(patient_y == 0), np.sum(patient_y == 1))
    n_splits = min(N_SPLITS, int(min_class_size))
    if n_splits < 2:
        raise ValueError("Not enough patients in each class for cross-validation.")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    patient_prediction_parts = []
    feature_importances = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(patient_ids, patient_y), start=1):
        train_ids = set(patient_ids[train_idx])
        test_ids = set(patient_ids[test_idx])

        train_df = df[df[ID_COL].isin(train_ids)].copy()
        test_df = df[df[ID_COL].isin(test_ids)].copy()

        X_train = train_df[feature_cols].to_numpy()
        y_train = train_df[LABEL_COL].astype(int).to_numpy()
        X_test = test_df[feature_cols].to_numpy()

        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        model = build_model(y_train)
        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_test)[:, 1]

        row_predictions = test_df[[ID_COL, LABEL_COL]].copy()
        row_predictions["probability"] = probabilities
        row_predictions["fold"] = fold

        patient_predictions = aggregate_patient_probabilities(row_predictions)
        patient_prediction_parts.append(patient_predictions)

        if hasattr(model, "feature_importances_"):
            feature_importances.append(model.feature_importances_)

    predictions = pd.concat(patient_prediction_parts, ignore_index=True)

    if feature_importances:
        importance_values = np.mean(np.vstack(feature_importances), axis=0)
        importance_df = pd.DataFrame(
            {
                "feature": feature_cols,
                "importance": importance_values,
            }
        ).sort_values("importance", ascending=False)
    else:
        importance_df = pd.DataFrame(columns=["feature", "importance"])

    return predictions, importance_df


def evaluate_predictions(predictions: pd.DataFrame) -> dict:
    threshold = THRESHOLD / 100.0 if THRESHOLD > 1.0 else THRESHOLD

    y_true = predictions["label"].astype(int).to_numpy()
    y_prob = predictions["probability"].to_numpy()
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "model_type": MODEL_TYPE,
        "feature_mode": FEATURE_MODE,
        "threshold": threshold,
        "patient_probability_aggregation": PATIENT_PROBABILITY_AGGREGATION,
        "n_patients": int(len(predictions)),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    except Exception:
        metrics["roc_auc"] = np.nan

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics["tn"] = int(cm[0, 0])
    metrics["fp"] = int(cm[0, 1])
    metrics["fn"] = int(cm[1, 0])
    metrics["tp"] = int(cm[1, 1])

    predictions = predictions.copy()
    predictions["prediction"] = y_pred

    return metrics, predictions


def main() -> None:
    df = load_and_prepare_data()
    feature_cols = get_feature_columns(df)

    predictions, importance_df = run_patient_level_cross_validation(df, feature_cols)
    metrics, predictions = evaluate_predictions(predictions)

    OUTPUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    predictions.to_csv(OUTPUT_MODEL_DIR / "patient_level_predictions.csv", index=False, encoding="utf-8-sig")
    importance_df.to_csv(OUTPUT_MODEL_DIR / "feature_importance.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([metrics]).to_csv(OUTPUT_MODEL_DIR / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"feature": feature_cols}).to_csv(OUTPUT_MODEL_DIR / "features_used.csv", index=False, encoding="utf-8-sig")

    print("Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print(f"Saved output to: {OUTPUT_MODEL_DIR}")


if __name__ == "__main__":
    main()
