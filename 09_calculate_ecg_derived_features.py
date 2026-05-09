from __future__ import annotations

import re
from pathlib import Path

from pipeline_common import INPUT_DIR, OUTPUT_DIR

import numpy as np
import pandas as pd


# ============================== CONFIG ==============================
# ============================== INPUT ==============================
INPUT_CSV_PATH = INPUT_DIR / "ecg_by_group" / "target_diagnosis" / "ecg_before_diagnosis.csv"
# ============================== OUTPUT =============================
OUTPUT_CSV_PATH = OUTPUT_DIR / "ecg_features" / "target_diagnosis" / "ecg_before_diagnosis_with_derived_features.csv"
# ====================================================================

LEAD_VALUE_SUFFIXES = ("PA", "QA", "RA", "SA", "STM", "TA")
LIMB_LEADS = {"1", "2", "3", "I", "II", "III", "AVR", "AVL", "AVF", "aVR", "aVL", "aVF"}
CHEST_LEAD_PATTERN = re.compile(r"^V[1-6]$", re.IGNORECASE)
# ====================================================================


def safe_div(numerator, denominator):
    numerator = pd.Series(numerator) if not isinstance(numerator, pd.Series) else numerator
    denominator = pd.Series(denominator) if not isinstance(denominator, pd.Series) else denominator
    denominator = denominator.mask(denominator == 0, np.nan)
    return numerator / denominator


def angular_diff_deg(a, b):
    return ((a - b + 180) % 360 - 180).abs()


def infer_available_leads(df: pd.DataFrame) -> list[str]:
    leads = set()
    pattern = re.compile(rf"^(.+?)_(?:{'|'.join(LEAD_VALUE_SUFFIXES)})$")
    for col in df.columns:
        match = pattern.match(col)
        if match:
            leads.add(match.group(1))
    return sorted(leads)


def classify_leads(leads: list[str]) -> tuple[list[str], list[str]]:
    limb = [lead for lead in leads if lead in LIMB_LEADS]
    chest = [lead for lead in leads if CHEST_LEAD_PATTERN.match(lead)]
    return limb, chest


def cols_for(df: pd.DataFrame, suffix: str, leads: list[str]) -> list[str]:
    return [f"{lead}_{suffix}" for lead in leads if f"{lead}_{suffix}" in df.columns]


def clean_numeric_columns(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    if not numeric_cols:
        return df

    cleaned = (
        df[numeric_cols]
        .astype(str)
        .replace({",": "."}, regex=True)
        .replace(r"[^0-9\-\.]", "", regex=True)
    )

    for col in numeric_cols:
        df[col] = pd.to_numeric(cleaned[col], errors="coerce")
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    base_cols = [
        "Frequency",
        "Sample_Time",
        "HR",
        "P_Interval",
        "QRS_Interval",
        "T_Interval",
        "PR_Interval",
        "QT_Interval",
        "QTc_Interval",
        "P_Axis",
        "QRS_Axis",
        "T_Axis",
    ]

    available_leads = infer_available_leads(df)
    amplitude_cols = [
        col
        for col in df.columns
        if re.match(rf"^(.+?)_(?:{'|'.join(LEAD_VALUE_SUFFIXES)})$", col)
    ]

    numeric_cols = [col for col in base_cols + amplitude_cols if col in df.columns]
    df = clean_numeric_columns(df, numeric_cols)

    if "HR" in df.columns:
        df["RR_Interval_ms"] = safe_div(60000.0, df["HR"])

    if {"PR_Interval", "QRS_Interval"}.issubset(df.columns):
        df["PRratio"] = safe_div(df["PR_Interval"], df["QRS_Interval"])

    if {"QT_Interval", "RR_Interval_ms"}.issubset(df.columns):
        df["QTnorm"] = safe_div(df["QT_Interval"], df["RR_Interval_ms"])

    if {"QT_Interval", "QTc_Interval"}.issubset(df.columns):
        df["Delta_QT"] = df["QT_Interval"] - df["QTc_Interval"]
        df["QT_minus_QTc"] = df["Delta_QT"]

    if {"T_Interval", "QRS_Interval"}.issubset(df.columns):
        df["TQRSratio"] = safe_div(df["T_Interval"], df["QRS_Interval"])

    if {"QT_Interval", "HR"}.issubset(df.columns):
        df["QT_over_HR"] = safe_div(df["QT_Interval"], df["HR"])

    if {"P_Axis", "QRS_Axis"}.issubset(df.columns):
        df["Delta_P_QRS"] = angular_diff_deg(df["P_Axis"], df["QRS_Axis"])

    if {"QRS_Axis", "T_Axis"}.issubset(df.columns):
        df["Delta_QRS_T"] = angular_diff_deg(df["QRS_Axis"], df["T_Axis"])

    if "QRS_Axis" in df.columns:
        df["QRSaxisnorm"] = df["QRS_Axis"] / 180.0

    if {"Delta_P_QRS", "Delta_QRS_T"}.issubset(df.columns):
        df["Composite_Axis_Score"] = df["Delta_P_QRS"] + df["Delta_QRS_T"]

    ra_cols = cols_for(df, "RA", available_leads)
    sa_cols = cols_for(df, "SA", available_leads)
    ta_cols = cols_for(df, "TA", available_leads)
    stm_cols = cols_for(df, "STM", available_leads)

    if ra_cols:
        r_values = df[ra_cols]
        df["Rmax"] = r_values.max(axis=1)
        df["Rmean"] = r_values.mean(axis=1)
        df["Rstd"] = r_values.std(axis=1, ddof=0)

    if sa_cols:
        df["Smin"] = df[sa_cols].min(axis=1)

    if stm_cols:
        df["STmean"] = df[stm_cols].mean(axis=1)

    if ta_cols:
        df["Tmean"] = df[ta_cols].mean(axis=1)

    if {"Rmax", "Smin"}.issubset(df.columns):
        df["RSratio"] = safe_div(df["Rmax"], df["Smin"].abs())

    if {"Tmean", "Rmean"}.issubset(df.columns):
        df["TRratio"] = safe_div(df["Tmean"], df["Rmean"])

    limb_leads, chest_leads = classify_leads(available_leads)
    ra_limb = cols_for(df, "RA", limb_leads)
    ra_chest = cols_for(df, "RA", chest_leads)

    if ra_limb:
        df["Rmean_limb"] = df[ra_limb].mean(axis=1)

    if ra_chest:
        df["Rmean_chest"] = df[ra_chest].mean(axis=1)

    if {"Rmean_chest", "Rmean_limb"}.issubset(df.columns):
        df["Delta_R"] = df["Rmean_chest"] - df["Rmean_limb"]
        df["Rdiff_chest_minus_limb"] = df["Delta_R"]

    return df


def main() -> None:
    df = pd.read_csv(INPUT_CSV_PATH)
    result = add_derived_features(df)

    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")

    print(f"Input rows: {len(df)}")
    print(f"Output rows: {len(result)}")
    print(f"Saved output to: {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
