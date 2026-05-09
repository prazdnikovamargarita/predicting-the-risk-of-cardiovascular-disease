from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_common import INPUT_DIR, OUTPUT_DIR, read_excel_all_sheets, read_table


# ============================== CONFIG ==============================
DIAGNOSIS_SLUG = "target_diagnosis"

INPUT_RECORDS_WITH_TARGET = INPUT_DIR / "clinical_before_after" / DIAGNOSIS_SLUG / "records_before_diagnosis.xlsx"
INPUT_RECORDS_WITHOUT_TARGET = INPUT_DIR / "patient_selection" / DIAGNOSIS_SLUG / "records_without_target.xlsx"

OUTPUT_WITH_TARGET_CSV = OUTPUT_DIR / "blood_pressure" / DIAGNOSIS_SLUG / "blood_pressure_before_diagnosis.csv"
OUTPUT_WITHOUT_TARGET_CSV = OUTPUT_DIR / "blood_pressure" / DIAGNOSIS_SLUG / "blood_pressure_without_target.csv"

TEXT_COL = "REZ"
ID_COL = "ID_PT"
DATE_COL = "DATE_F"

OUTPUT_COLUMNS = [
    "ID_PT",
    "DATE_F",
    "D_OSN",
    "D_SOP1",
    "D_SOP2",
    "upper_at",
    "down_at",
    "match_text",
]
# ====================================================================


SYMBOL_BETWEEN_LETTERS = r"[\\/|.\-]"
AT_OR_AD = (
    r"(?:(?:[аa]\s*(?:"
    + SYMBOL_BETWEEN_LETTERS
    + r"\s*)?[тt])|(?:[аa]\s*(?:"
    + SYMBOL_BETWEEN_LETTERS
    + r"\s*)?[дd]))"
)
FULL_PHRASES = r"\b(?:артер[іи]альн\w*\s+тиск|артериал\w*\s+давлен\w*|арт\.?\s*тиск)\b"
PREFIX_TERMS = rf"(?:{FULL_PHRASES}|(?<!\w){AT_OR_AD}\.?(?!\w))"
SEPARATOR_NUMBERS = r"(?:\s*[/\\\-:|–—xх·∙•;_]\s*|\s+)"
BP_PAIR = rf"(?P<up>\d{{2,3}}){SEPARATOR_NUMBERS}(?P<down>\d{{2,3}})"
GAP_NO_DIGITS = r"(?:[^\d]{0,20})"
UNITS_M = r"\b[мm]{1,2}\s*\.?\s*(?:р\s*\.?\s*т\s*\.?\s*ст\s*\.?)?\b"

BLOOD_PRESSURE_PATTERN = re.compile(
    rf"{PREFIX_TERMS}{GAP_NO_DIGITS}{BP_PAIR}{GAP_NO_DIGITS}{UNITS_M}",
    flags=re.I | re.U,
)


def load_records(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return read_excel_all_sheets(path)
    return read_table(path)


def extract_all_blood_pressure_values(df: pd.DataFrame, text_col: str = TEXT_COL) -> pd.DataFrame:
    if text_col not in df.columns:
        raise KeyError(f'Missing text column: "{text_col}"')

    source_text = df[text_col].astype(str)
    matches = source_text.str.extractall(BLOOD_PRESSURE_PATTERN)

    if matches.empty:
        empty = df.loc[[]].copy()
        empty["orig_index"] = pd.Series(dtype="Int64")
        empty["match_no"] = pd.Series(dtype="Int64")
        empty["upper_at"] = pd.Series(dtype="Int64")
        empty["down_at"] = pd.Series(dtype="Int64")
        empty["match_text"] = pd.Series(dtype="string")
        return empty

    matches = matches.reset_index(names=["orig_index", "match_no"])
    upper = pd.to_numeric(matches["up"], errors="coerce")
    lower = pd.to_numeric(matches["down"], errors="coerce")

    valid = upper.gt(lower)
    valid_matches = matches.loc[valid].copy()
    valid_matches["upper_at"] = upper.loc[valid].astype("Int64")
    valid_matches["down_at"] = lower.loc[valid].astype("Int64")

    positions = []
    for index, text in source_text.to_dict().items():
        for match_no, match in enumerate(BLOOD_PRESSURE_PATTERN.finditer(text)):
            positions.append((index, match_no, match.group(0)))

    position_df = pd.DataFrame(positions, columns=["orig_index", "match_no", "match_text"])
    valid_matches = valid_matches.merge(position_df, on=["orig_index", "match_no"], how="left")

    result = valid_matches.merge(
        df.reset_index().rename(columns={"index": "orig_index"}),
        on="orig_index",
        how="left",
    )

    lead_cols = ["orig_index", "match_no", "upper_at", "down_at", "match_text"]
    return result[lead_cols + [col for col in result.columns if col not in lead_cols]]


def clean_output(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    available = [col for col in OUTPUT_COLUMNS if col in df.columns]
    df = df[available].copy()

    for col in [ID_COL, "D_OSN", "D_SOP1", "D_SOP2", "match_text"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().replace({"": np.nan, "nan": np.nan})

    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce").dt.strftime("%Y-%m-%d")

    for col in ["upper_at", "down_at"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    df = df[
        df["upper_at"].between(40, 300)
        & df["down_at"].between(20, 220)
    ]

    df = df.drop_duplicates()
    sort_cols = [col for col in [ID_COL, DATE_COL] if col in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, kind="stable")

    return df.reindex(columns=OUTPUT_COLUMNS)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
        sep=",",
        quoting=csv.QUOTE_MINIMAL,
        na_rep="",
        lineterminator="\n",
    )


def main() -> None:
    records_with_target = load_records(INPUT_RECORDS_WITH_TARGET)
    records_without_target = load_records(INPUT_RECORDS_WITHOUT_TARGET)

    bp_with_target = clean_output(extract_all_blood_pressure_values(records_with_target))
    bp_without_target = clean_output(extract_all_blood_pressure_values(records_without_target))

    save_csv(bp_with_target, OUTPUT_WITH_TARGET_CSV)
    save_csv(bp_without_target, OUTPUT_WITHOUT_TARGET_CSV)

    print(f"Blood pressure rows before diagnosis: {len(bp_with_target)}")
    print(f"Blood pressure rows without target: {len(bp_without_target)}")
    print(f"Saved output to: {OUTPUT_WITH_TARGET_CSV.parent}")


if __name__ == "__main__":
    main()
