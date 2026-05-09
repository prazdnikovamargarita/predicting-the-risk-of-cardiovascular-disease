from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from pipeline_common import INPUT_DIR, OUTPUT_DIR, ensure_parent, read_excel_all_sheets, write_excel_chunked


# ============================== CONFIG ==============================
DIAGNOSIS_SLUG = "target_diagnosis"
DIAGNOSIS_PATTERN = r"I2[1-4]"

INPUT_RECORDS_WITHOUT_TARGET = INPUT_DIR / "patient_selection" / DIAGNOSIS_SLUG / "records_without_target.xlsx"
INPUT_RECORDS_WITH_TARGET = INPUT_DIR / "patient_selection" / DIAGNOSIS_SLUG / "records_with_target.xlsx"

OUTPUT_SUBDIR = OUTPUT_DIR / "validation" / DIAGNOSIS_SLUG
DIAGNOSIS_COLUMNS = ["D_OSN", "D_SOP1", "D_SOP2"]
ID_COL = "ID_PT"
# ====================================================================


def load_records(path: Path) -> pd.DataFrame:
    return read_excel_all_sheets(path)


def patient_has_diagnosis(df: pd.DataFrame, pattern: str) -> pd.Series:
    available_cols = [col for col in DIAGNOSIS_COLUMNS if col in df.columns]
    if not available_cols:
        raise KeyError(f"None of the diagnosis columns were found: {DIAGNOSIS_COLUMNS}")

    row_has_code = pd.Series(False, index=df.index)
    for col in available_cols:
        row_has_code |= df[col].astype(str).str.contains(pattern, flags=re.I, na=False)

    return row_has_code.groupby(df[ID_COL]).transform("any")


def split_by_patient_diagnosis(df: pd.DataFrame, pattern: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if ID_COL not in df.columns:
        raise KeyError(f"Missing required column: {ID_COL}")

    has_target = patient_has_diagnosis(df, pattern)
    return df[has_target].copy(), df[~has_target].copy()


def export_validation_result(input_path: Path, dataset_label: str) -> None:
    df = load_records(input_path)
    records_with_code, records_without_code = split_by_patient_diagnosis(df, DIAGNOSIS_PATTERN)

    dataset_dir = OUTPUT_SUBDIR / dataset_label
    dataset_dir.mkdir(parents=True, exist_ok=True)

    records_with_code[[ID_COL]].drop_duplicates().to_csv(
        ensure_parent(dataset_dir / "patient_ids_with_detected_code.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    records_without_code[[ID_COL]].drop_duplicates().to_csv(
        ensure_parent(dataset_dir / "patient_ids_without_detected_code.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    write_excel_chunked(records_with_code, dataset_dir / "records_with_detected_code.xlsx")
    write_excel_chunked(records_without_code, dataset_dir / "records_without_detected_code.xlsx")

    print(f"[{dataset_label}] rows with code: {len(records_with_code)}")
    print(f"[{dataset_label}] rows without code: {len(records_without_code)}")


def main() -> None:
    export_validation_result(INPUT_RECORDS_WITHOUT_TARGET, "source_without_target")
    export_validation_result(INPUT_RECORDS_WITH_TARGET, "source_with_target")
    print(f"Saved output to: {OUTPUT_SUBDIR}")


if __name__ == "__main__":
    main()
