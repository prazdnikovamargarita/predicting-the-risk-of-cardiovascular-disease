from __future__ import annotations

import csv
from pathlib import Path

from libriaries.find_needed_MXK import find_needed_id, make_frame_by_id

from pipeline_common import INPUT_DIR, OUTPUT_DIR, ensure_dir, write_excel_chunked


# ============================== CONFIG ==============================
DIAGNOSIS_SLUG = "target_diagnosis"
GENERAL_DIAGNOSIS_CODE = "I"
TARGET_DIAGNOSIS_PATTERN = r"I2[1-4]"

# ============================== INPUT ==============================
INPUT_DBF_DIR = INPUT_DIR / "dbf"
# Expected files: input/dbf/**/daily*.dbf
# ================================================================

# ============================== OUTPUT =============================
OUTPUT_SUBDIR = OUTPUT_DIR / "patient_selection" / DIAGNOSIS_SLUG
OUTPUT_PATIENT_IDS_WITH_TARGET = OUTPUT_SUBDIR / "patient_ids_with_target.csv"
OUTPUT_PATIENT_IDS_WITHOUT_TARGET = OUTPUT_SUBDIR / "patient_ids_without_target.csv"
OUTPUT_RECORDS_WITH_TARGET = OUTPUT_SUBDIR / "records_with_target.xlsx"
OUTPUT_RECORDS_WITHOUT_TARGET = OUTPUT_SUBDIR / "records_without_target.xlsx"
# ====================================================================
# ====================================================================


def write_patient_ids(ids: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["ID_PT"])
        for patient_id in ids:
            writer.writerow([patient_id])


def main() -> None:
    ensure_dir(OUTPUT_SUBDIR)

    ids_with_target = find_needed_id(
        only_target=True,
        general_mkb_code=GENERAL_DIAGNOSIS_CODE,
        target_mkb_code=TARGET_DIAGNOSIS_PATTERN,
        input_dbf_dir=INPUT_DBF_DIR,
    )
    ids_without_target = find_needed_id(
        only_target=False,
        general_mkb_code=GENERAL_DIAGNOSIS_CODE,
        target_mkb_code=TARGET_DIAGNOSIS_PATTERN,
        input_dbf_dir=INPUT_DBF_DIR,
    )

    write_patient_ids(ids_with_target, OUTPUT_PATIENT_IDS_WITH_TARGET)
    write_patient_ids(ids_without_target, OUTPUT_PATIENT_IDS_WITHOUT_TARGET)

    records_with_target = make_frame_by_id(ids_with_target, input_dbf_dir=INPUT_DBF_DIR)
    records_without_target = make_frame_by_id(ids_without_target, input_dbf_dir=INPUT_DBF_DIR)

    write_excel_chunked(records_with_target, OUTPUT_RECORDS_WITH_TARGET)
    write_excel_chunked(records_without_target, OUTPUT_RECORDS_WITHOUT_TARGET)

    print(f"Saved output to: {OUTPUT_SUBDIR}")


if __name__ == "__main__":
    main()
