from __future__ import annotations

from pathlib import Path

import pandas as pd

from pipeline_common import INPUT_DIR, OUTPUT_DIR, ensure_parent, read_table, read_excel_all_sheets


# ============================== CONFIG ==============================
DIAGNOSIS_SLUG = "target_diagnosis"

INPUT_RECORDS_AFTER_DIAGNOSIS = INPUT_DIR / "clinical_before_after" / DIAGNOSIS_SLUG / "records_after_diagnosis.xlsx"
INPUT_ECG_RECORDS = INPUT_DIR / "ecg" / "ecg_descriptions.csv"
INPUT_PATIENT_IDS_WITHOUT_TARGET = INPUT_DIR / "patient_selection" / DIAGNOSIS_SLUG / "patient_ids_without_target.csv"

OUTPUT_ECG_BEFORE_DIAGNOSIS = OUTPUT_DIR / "ecg_by_group" / DIAGNOSIS_SLUG / "ecg_before_diagnosis.csv"
OUTPUT_ECG_WITHOUT_TARGET = OUTPUT_DIR / "ecg_by_group" / DIAGNOSIS_SLUG / "ecg_without_target.csv"

ID_COL = "ID_PT"
CLINICAL_DATE_COL = "DATE_F"
ECG_DATE_COL = "DATE_F"
# ====================================================================


def load_clinical_after(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return read_excel_all_sheets(path)
    return read_table(path)


def first_diagnosis_dates(records_after: pd.DataFrame) -> pd.DataFrame:
    missing = {ID_COL, CLINICAL_DATE_COL} - set(records_after.columns)
    if missing:
        raise KeyError(f"Missing columns in clinical records: {sorted(missing)}")

    df = records_after[[ID_COL, CLINICAL_DATE_COL]].copy()
    df[ID_COL] = df[ID_COL].astype(str).str.strip()
    df[CLINICAL_DATE_COL] = pd.to_datetime(df[CLINICAL_DATE_COL], errors="coerce")
    df = df.dropna(subset=[ID_COL, CLINICAL_DATE_COL])

    return (
        df.groupby(ID_COL, as_index=False)[CLINICAL_DATE_COL]
        .min()
        .rename(columns={CLINICAL_DATE_COL: "first_diagnosis_date"})
    )


def select_ecg_before_diagnosis(ecg_records: pd.DataFrame, diagnosis_dates: pd.DataFrame) -> pd.DataFrame:
    missing = {ID_COL, ECG_DATE_COL} - set(ecg_records.columns)
    if missing:
        raise KeyError(f"Missing columns in ECG records: {sorted(missing)}")

    ecg = ecg_records.copy()
    ecg[ID_COL] = ecg[ID_COL].astype(str).str.strip()
    ecg[ECG_DATE_COL] = pd.to_datetime(ecg[ECG_DATE_COL], errors="coerce")

    merged = ecg.merge(diagnosis_dates, on=ID_COL, how="inner")
    before = merged[
        merged[ECG_DATE_COL].notna()
        & merged["first_diagnosis_date"].notna()
        & (merged[ECG_DATE_COL] < merged["first_diagnosis_date"])
    ].copy()

    return before.drop(columns=["first_diagnosis_date"])


def select_ecg_without_target(ecg_records: pd.DataFrame, patient_ids_without_target: pd.DataFrame) -> pd.DataFrame:
    if ID_COL not in ecg_records.columns:
        raise KeyError(f"Missing column in ECG records: {ID_COL}")
    if ID_COL not in patient_ids_without_target.columns:
        raise KeyError(f"Missing column in patient ID file: {ID_COL}")

    ids = set(patient_ids_without_target[ID_COL].astype(str).str.strip())
    ecg = ecg_records.copy()
    ecg[ID_COL] = ecg[ID_COL].astype(str).str.strip()

    return ecg[ecg[ID_COL].isin(ids)].copy()


def main() -> None:
    records_after = load_clinical_after(INPUT_RECORDS_AFTER_DIAGNOSIS)
    ecg_records = read_table(INPUT_ECG_RECORDS)
    ids_without = read_table(INPUT_PATIENT_IDS_WITHOUT_TARGET)

    diagnosis_dates = first_diagnosis_dates(records_after)
    ecg_before = select_ecg_before_diagnosis(ecg_records, diagnosis_dates)
    ecg_without = select_ecg_without_target(ecg_records, ids_without)

    ensure_parent(OUTPUT_ECG_BEFORE_DIAGNOSIS)
    ensure_parent(OUTPUT_ECG_WITHOUT_TARGET)

    ecg_before.to_csv(OUTPUT_ECG_BEFORE_DIAGNOSIS, index=False, encoding="utf-8-sig")
    ecg_without.to_csv(OUTPUT_ECG_WITHOUT_TARGET, index=False, encoding="utf-8-sig")

    print(f"ECG before diagnosis rows: {len(ecg_before)}")
    print(f"ECG without target rows: {len(ecg_without)}")
    print(f"Saved output to: {OUTPUT_ECG_BEFORE_DIAGNOSIS.parent}")


if __name__ == "__main__":
    main()
