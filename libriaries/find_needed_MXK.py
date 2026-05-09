from __future__ import annotations

import csv
import os
import random
import re
from pathlib import Path
from typing import Iterable

import pandas as pd
from pandas import DataFrame

try:
    from dbfread import DBF, FieldParser
except Exception:  # lets the rest of the project import even without dbfread installed
    DBF = None
    FieldParser = object

from pipeline_common import INPUT_DIR, OUTPUT_DIR, ensure_parent, write_excel_chunked


# ============================== INPUT ==============================
INPUT_DBF_DIR = INPUT_DIR / "dbf"
# Put source DBF files here: input/dbf/**/daily*.dbf
# ====================================================================

# ============================== OUTPUT =============================
OUTPUT_SAMPLE_XLSX = OUTPUT_DIR / "samples" / "patient_records_sample.xlsx"
# ====================================================================

COLUMN_NAMES = ["ID_PT", "DATE_F", "D_OSN", "D_SOP1", "D_SOP2", "REZ"]
DIAGNOSIS_COLUMNS = ["D_OSN", "D_SOP1", "D_SOP2"]


class TestFieldParser(FieldParser):
    """DBF parser that keeps raw W fields unchanged for Cyrillic-safe reading."""

    def parseW(self, field, data):
        return data


def _require_dbfread() -> None:
    if DBF is None:
        raise ImportError("Package 'dbfread' is required for DBF input files.")


def take_list_of_dbf_files(input_dbf_dir: str | Path = INPUT_DBF_DIR) -> list[Path]:
    """Return all DBF files under input/dbf that contain 'daily' in the filename."""
    input_dbf_dir = Path(input_dbf_dir)
    if not input_dbf_dir.exists():
        raise FileNotFoundError(f"Input DBF directory does not exist: {input_dbf_dir}")

    return sorted(
        path for path in input_dbf_dir.rglob("*.dbf")
        if "daily" in path.name.lower()
    )


def unique(input_list: Iterable) -> list:
    """Return unique values while preserving order."""
    seen = set()
    result = []
    for value in input_list:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


def is_non_empty(value) -> bool:
    return len(str(value).strip()) > 0


def _read_dbf_file(dbf_file_path: str | Path, encoding: str = "cp1251") -> pd.DataFrame:
    _require_dbfread()
    dbf_table = DBF(str(dbf_file_path), encoding=encoding, parserclass=TestFieldParser)
    return DataFrame(iter(dbf_table))


def load_dbf_records(input_dbf_dir: str | Path = INPUT_DBF_DIR, encoding: str = "cp1251") -> pd.DataFrame:
    """Load and merge all standardized DBF inputs from input/dbf."""
    frames = []
    for dbf_file_path in take_list_of_dbf_files(input_dbf_dir):
        print(f"Processing input DBF: {dbf_file_path}")
        frame = _read_dbf_file(dbf_file_path, encoding=encoding)
        available_columns = [col for col in COLUMN_NAMES if col in frame.columns]
        if available_columns:
            frame = frame[available_columns]
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=COLUMN_NAMES)

    combined = pd.concat(frames, ignore_index=True)
    if "ID_PT" in combined.columns and "DATE_F" in combined.columns:
        combined = combined.sort_values(["ID_PT", "DATE_F"], kind="stable")
    return combined


def find_needed_id(
    general_mkb_code: str = "I",
    target_mkb_code: str = r"^I6",
    only_target: bool = False,
    input_dbf_dir: str | Path = INPUT_DBF_DIR,
) -> list:
    """Find patient IDs with or without the target ICD/MКХ diagnosis.

    Input is always read from input/dbf unless another path is explicitly provided.
    """
    combined_dataframe = load_dbf_records(input_dbf_dir)
    if combined_dataframe.empty:
        return []

    if "ID_PT" not in combined_dataframe.columns:
        raise KeyError("Missing required column: ID_PT")

    target_ids = []
    general_ids = []

    for patient_id, patient_records in combined_dataframe.groupby("ID_PT"):
        target_mask = pd.Series(False, index=patient_records.index)
        general_mask = pd.Series(False, index=patient_records.index)

        for col in DIAGNOSIS_COLUMNS:
            if col in patient_records.columns:
                target_mask |= patient_records[col].astype(str).str.contains(target_mkb_code, flags=re.I, regex=True, na=False)
                general_mask |= patient_records[col].astype(str).str.contains(general_mkb_code, flags=re.I, regex=True, na=False)

        if target_mask.any():
            target_ids.append(patient_id)
        if general_mask.any():
            general_ids.append(patient_id)

    target_ids = set(target_ids)
    general_ids = set(general_ids)

    if only_target:
        return sorted(target_ids)
    return sorted(general_ids - target_ids)


def make_frame_by_id(
    needed_id_arr: Iterable,
    input_dbf_dir: str | Path = INPUT_DBF_DIR,
) -> pd.DataFrame:
    """Collect all records for a provided list of patient IDs."""
    all_records = load_dbf_records(input_dbf_dir)
    if all_records.empty:
        return all_records

    ids = {str(value).strip() for value in needed_id_arr}
    all_records["ID_PT"] = all_records["ID_PT"].astype(str).str.strip()
    result = all_records[all_records["ID_PT"].isin(ids)].copy()

    if {"ID_PT", "DATE_F"}.issubset(result.columns):
        result = result.sort_values(["ID_PT", "DATE_F"], kind="stable")
    return result.reset_index(drop=True)


def write_excel_file(dataframe: pd.DataFrame, filename: str | Path):
    """Compatibility wrapper: write output Excel under output/ paths."""
    return write_excel_chunked(dataframe, ensure_parent(filename))


if __name__ == "__main__":
    ids_without_target = find_needed_id()
    random.shuffle(ids_without_target)
    selected_ids = ids_without_target[:1500]
    result_df = make_frame_by_id(selected_ids)
    write_excel_file(result_df, OUTPUT_SAMPLE_XLSX)
    print(f"Saved output to: {OUTPUT_SAMPLE_XLSX}")
