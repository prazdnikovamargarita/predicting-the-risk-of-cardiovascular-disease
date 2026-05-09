from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent

# ============================== INPUT ROOT ==============================
# Put all manually prepared source files under this directory.
# Never put personal/local absolute paths into scripts.
INPUT_DIR = PROJECT_ROOT / "input"

# ============================== OUTPUT ROOT =============================
# All generated files are written here.
# Files in this directory can be deleted and regenerated from input/.
OUTPUT_DIR = PROJECT_ROOT / "output"


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_table(path: str | Path, **kwargs) -> pd.DataFrame:
    """Read CSV/XLSX/XLS table by extension."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path, **kwargs)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, **kwargs)

    raise ValueError(f"Unsupported input file extension: {path.suffix}. Use CSV or Excel.")


def read_excel_all_sheets(path: str | Path) -> pd.DataFrame:
    """Read all Excel sheets and merge them into one DataFrame."""
    sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    if not sheets:
        return pd.DataFrame()
    return pd.concat(sheets.values(), ignore_index=True)


def write_excel_chunked(
    df: pd.DataFrame,
    output_path: str | Path,
    max_rows_per_file: int = 1_000_000,
) -> list[Path]:
    """
    Save a DataFrame to Excel. If it is too large for one file, split it into numbered parts.
    """
    output_path = ensure_parent(output_path)
    output_path = Path(output_path)

    if len(df) <= max_rows_per_file:
        df.to_excel(output_path, index=False, engine="openpyxl")
        return [output_path]

    saved_paths = []
    stem = output_path.with_suffix("")
    suffix = output_path.suffix or ".xlsx"

    for part_no, start in enumerate(range(0, len(df), max_rows_per_file), start=1):
        part = df.iloc[start:start + max_rows_per_file]
        part_path = Path(f"{stem}_part_{part_no:02d}{suffix}")
        part.to_excel(part_path, index=False, engine="openpyxl")
        saved_paths.append(part_path)

    return saved_paths


def normalize_id_column(df: pd.DataFrame, id_col: str = "ID_PT") -> pd.DataFrame:
    df = df.copy()
    if id_col in df.columns:
        df[id_col] = df[id_col].astype(str).str.strip()
    return df


def columns_present(df: pd.DataFrame, cols: Iterable[str]) -> list[str]:
    return [col for col in cols if col in df.columns]
