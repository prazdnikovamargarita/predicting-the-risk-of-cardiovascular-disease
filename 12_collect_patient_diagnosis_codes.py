from __future__ import annotations

from pathlib import Path

import pandas as pd

from pipeline_common import INPUT_DIR, OUTPUT_DIR, read_excel_all_sheets, read_table


# ============================== CONFIG ==============================
INPUT_RECORDS_PATH = INPUT_DIR / "diagnosis_codes" / "records.csv"
OUTPUT_CSV_PATH = OUTPUT_DIR / "diagnosis_codes" / "patient_diagnosis_codes.csv"

ID_COL = "ID_PT"
DIAGNOSIS_COLUMNS = ["D_OSN", "D_SOP1", "D_SOP2"]
OUTPUT_CODES_COL = "diagnosis_codes"
# ====================================================================


def load_records(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return read_excel_all_sheets(path)
    return read_table(path)


def collect_codes_for_patient(group: pd.DataFrame) -> str:
    codes: list[str] = []
    for col in DIAGNOSIS_COLUMNS:
        if col not in group.columns:
            continue
        values = group[col].dropna().astype(str).str.strip()
        values = values[values.str.len() > 0]
        values = values[values.str.lower() != "nan"]
        codes.extend(values.tolist())

    # зберігаємо порядок, але прибираємо дублікати
    seen = set()
    unique_codes = []
    for code in codes:
        if code not in seen:
            seen.add(code)
            unique_codes.append(code)

    return ", ".join(unique_codes)


def main() -> None:
    df = load_records(INPUT_RECORDS_PATH)

    if ID_COL not in df.columns:
        raise KeyError(f"Missing ID column: {ID_COL}")

    df[ID_COL] = df[ID_COL].astype(str).str.strip()

    rows = []
    for patient_id, group in df.groupby(ID_COL, sort=False):
        rows.append(
            {
                ID_COL: patient_id,
                OUTPUT_CODES_COL: collect_codes_for_patient(group),
            }
        )

    result = pd.DataFrame(rows)
    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")

    print(f"Patients: {len(result)}")
    print(f"Saved output to: {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
