from __future__ import annotations

from pathlib import Path

import pandas as pd

from pipeline_common import ensure_parent, read_excel_all_sheets, write_excel_chunked


# ============================== INPUT ==============================
# Use this function with a path under: input/<stage>/<file>.xlsx
# Example: INPUT_RECORDS_XLSX = INPUT_DIR / "patient_selection" / "target_diagnosis" / "records_with_target.xlsx"
# ====================================================================
def load_and_merge_sheets(input_excel_path: str | Path) -> pd.DataFrame:
    """Load all sheets from one Excel file and merge them into a single DataFrame."""
    return read_excel_all_sheets(input_excel_path)


# ============================== OUTPUT =============================
# Use this function with a path under: output/<stage>/<file>.xlsx
# Example: OUTPUT_RECORDS_XLSX = OUTPUT_DIR / "clinical_before_after" / "target_diagnosis" / "records_before_diagnosis.xlsx"
# ====================================================================
def write_excel_file(dataframe: pd.DataFrame, output_excel_path: str | Path) -> list[Path]:
    """Save a DataFrame to Excel; split into parts automatically if needed."""
    return write_excel_chunked(dataframe, ensure_parent(output_excel_path))
