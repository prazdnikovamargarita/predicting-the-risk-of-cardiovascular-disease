from __future__ import annotations

from pathlib import Path

import pandas as pd

from pipeline_common import INPUT_DIR, OUTPUT_DIR, ensure_parent, read_table, write_excel_chunked


# ============================== CONFIG ==============================
BASE_RECORDS_PATH = INPUT_DIR / "group_correction" / "base_records.xlsx"
SOURCE_RECORDS_PATH = INPUT_DIR / "group_correction" / "source_records.xlsx"
IDS_TO_APPEND_PATH = INPUT_DIR / "group_correction" / "patient_ids_to_append.csv"

OUTPUT_PATH = OUTPUT_DIR / "group_correction" / "corrected_records.xlsx"
ID_COL = "ID_PT"
# ====================================================================


def main() -> None:
    base_records = read_table(BASE_RECORDS_PATH)
    source_records = read_table(SOURCE_RECORDS_PATH)
    ids_to_append = read_table(IDS_TO_APPEND_PATH)

    if ID_COL not in base_records.columns:
        raise KeyError(f"Missing column in base records: {ID_COL}")
    if ID_COL not in source_records.columns:
        raise KeyError(f"Missing column in source records: {ID_COL}")
    if ID_COL not in ids_to_append.columns:
        raise KeyError(f"Missing column in ID file: {ID_COL}")

    ids = set(ids_to_append[ID_COL].astype(str).str.strip())
    source_records[ID_COL] = source_records[ID_COL].astype(str).str.strip()
    base_records[ID_COL] = base_records[ID_COL].astype(str).str.strip()

    records_to_append = source_records[source_records[ID_COL].isin(ids)].copy()

    corrected = (
        pd.concat([base_records, records_to_append], ignore_index=True)
        .drop_duplicates()
        .reset_index(drop=True)
    )

    write_excel_chunked(corrected, OUTPUT_PATH)
    ensure_parent(OUTPUT_PATH.with_suffix(".summary.csv"))
    pd.DataFrame(
        {
            "metric": ["base_rows", "source_rows_selected", "output_rows"],
            "value": [len(base_records), len(records_to_append), len(corrected)],
        }
    ).to_csv(OUTPUT_PATH.with_suffix(".summary.csv"), index=False, encoding="utf-8-sig")

    print(f"Saved corrected records to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
