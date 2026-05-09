from __future__ import annotations

from pathlib import Path

import pandas as pd

from pipeline_common import INPUT_DIR, OUTPUT_DIR, ensure_parent


# ============================== CONFIG ==============================
INPUT_CSV_DIR = INPUT_DIR / "ecg" / "main_info_parts"
OUTPUT_CSV_PATH = OUTPUT_DIR / "ecg" / "ecg_main_info.csv"
# ====================================================================


def main() -> None:
    if not INPUT_CSV_DIR.exists():
        raise FileNotFoundError(f"Input directory does not exist: {INPUT_CSV_DIR}")

    csv_paths = sorted(INPUT_CSV_DIR.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in: {INPUT_CSV_DIR}")

    parts = []
    for csv_path in csv_paths:
        part = pd.read_csv(csv_path)
        part["source_file"] = csv_path.name
        parts.append(part)

    merged = pd.concat(parts, ignore_index=True).drop_duplicates().reset_index(drop=True)

    ensure_parent(OUTPUT_CSV_PATH)
    merged.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")

    print(f"Input files: {len(csv_paths)}")
    print(f"Output rows: {len(merged)}")
    print(f"Saved output to: {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
