from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from pandas.tseries.offsets import DateOffset

from pipeline_common import INPUT_DIR, OUTPUT_DIR, read_excel_all_sheets, write_excel_chunked


# ============================== CONFIG ==============================
DIAGNOSIS_SLUG = "target_diagnosis"
DIAGNOSIS_PATTERN = r"I2[1-4]"

INPUT_RECORDS_WITH_TARGET = INPUT_DIR / "patient_selection" / DIAGNOSIS_SLUG / "records_with_target.xlsx"
OUTPUT_SUBDIR = OUTPUT_DIR / "clinical_before_after" / DIAGNOSIS_SLUG

ID_COL = "ID_PT"
DATE_COL = "DATE_F"
DIAGNOSIS_COLUMNS = ["D_OSN", "D_SOP1", "D_SOP2"]

MONTHS_BEFORE_EXCLUSION = 0
MONTHS_AFTER_INCLUSION_SHIFT = 0
# ====================================================================


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    missing = {ID_COL, DATE_COL} - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[ID_COL, DATE_COL])
    df[ID_COL] = df[ID_COL].astype(str).str.strip()
    df = df.sort_values([ID_COL, DATE_COL], ascending=[True, True]).reset_index(drop=True)
    return df


def row_has_diagnosis(df: pd.DataFrame) -> pd.Series:
    available_cols = [col for col in DIAGNOSIS_COLUMNS if col in df.columns]
    if not available_cols:
        raise KeyError(f"None of the diagnosis columns were found: {DIAGNOSIS_COLUMNS}")

    mask = pd.Series(False, index=df.index)
    for col in available_cols:
        mask |= df[col].astype(str).str.contains(DIAGNOSIS_PATTERN, flags=re.I, na=False)
    return mask


def split_patient_group(group: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask = row_has_diagnosis(group)
    hits = group[mask]

    if hits.empty:
        return pd.DataFrame(), pd.DataFrame()

    first_diagnosis_date = hits[DATE_COL].min()

    before = group[
        group[DATE_COL] < (first_diagnosis_date - DateOffset(months=MONTHS_BEFORE_EXCLUSION))
    ].copy()

    after = group[
        group[DATE_COL] >= (first_diagnosis_date - DateOffset(months=MONTHS_AFTER_INCLUSION_SHIFT))
    ].copy()

    return before, after


def main() -> None:
    raw = read_excel_all_sheets(INPUT_RECORDS_WITH_TARGET)
    df = prepare_dataframe(raw)

    patients_with_target = df.loc[row_has_diagnosis(df), ID_COL].drop_duplicates()
    df = df[df[ID_COL].isin(patients_with_target)].reset_index(drop=True)

    before_parts = []
    after_parts = []

    for _, group in df.groupby(ID_COL, sort=False):
        before, after = split_patient_group(group.drop_duplicates())
        if not before.empty:
            before_parts.append(before)
        if not after.empty:
            after_parts.append(after)

    records_before = pd.concat(before_parts, ignore_index=True) if before_parts else pd.DataFrame(columns=df.columns)
    records_after = pd.concat(after_parts, ignore_index=True) if after_parts else pd.DataFrame(columns=df.columns)

    OUTPUT_SUBDIR.mkdir(parents=True, exist_ok=True)
    write_excel_chunked(records_before, OUTPUT_SUBDIR / "records_before_diagnosis.xlsx")
    write_excel_chunked(records_after, OUTPUT_SUBDIR / "records_after_diagnosis.xlsx")

    print(f"Patients with target diagnosis: {len(patients_with_target)}")
    print(f"Records before diagnosis: {len(records_before)}")
    print(f"Records after diagnosis: {len(records_after)}")
    print(f"Saved output to: {OUTPUT_SUBDIR}")


if __name__ == "__main__":
    main()
