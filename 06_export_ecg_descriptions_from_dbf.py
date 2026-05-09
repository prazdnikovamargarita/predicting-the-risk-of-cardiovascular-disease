from __future__ import annotations

from pathlib import Path

import pandas as pd
from dbfread import DBF, FieldParser

from pipeline_common import INPUT_DIR, OUTPUT_DIR, ensure_parent, write_excel_chunked


# ============================== CONFIG ==============================
INPUT_DBF_PATH = INPUT_DIR / "ecg" / "ecg_descriptions.dbf"
OUTPUT_CSV_PATH = OUTPUT_DIR / "ecg" / "ecg_descriptions.csv"
OUTPUT_XLSX_PATH = OUTPUT_DIR / "ecg" / "ecg_descriptions.xlsx"

DBF_ENCODING = "cp1251"
SOURCE_ID_COL = "IDBDISSL"
SOURCE_DATE_COL = "DATOTDP"
SOURCE_TEXT_COL = "PROTREZ"

ID_COL = "ID_PT"
DATE_COL = "DATE_F"
TEXT_COL = "REZ"
# ====================================================================


class CyrillicFieldParser(FieldParser):
    def parseW(self, field, data):
        return data


def main() -> None:
    if not INPUT_DBF_PATH.exists():
        raise FileNotFoundError(f"Input DBF file does not exist: {INPUT_DBF_PATH}")

    table = DBF(INPUT_DBF_PATH, encoding=DBF_ENCODING, parserclass=CyrillicFieldParser)
    df = pd.DataFrame(iter(table))

    required_cols = [SOURCE_ID_COL, SOURCE_DATE_COL, SOURCE_TEXT_COL]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing DBF columns: {missing}")

    result = df[required_cols].copy()
    result[ID_COL] = result[SOURCE_ID_COL].astype(str).str.slice(0, 6)
    result = result.rename(
        columns={
            SOURCE_DATE_COL: DATE_COL,
            SOURCE_TEXT_COL: TEXT_COL,
        }
    )
    result = result[[ID_COL, DATE_COL, TEXT_COL]]
    result = result.dropna(subset=[TEXT_COL])
    result = result[result[TEXT_COL].astype(str).str.len() > 0]
    result[DATE_COL] = pd.to_datetime(result[DATE_COL], errors="coerce")

    ensure_parent(OUTPUT_CSV_PATH)
    result.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")
    write_excel_chunked(result, OUTPUT_XLSX_PATH)

    print(f"Rows exported: {len(result)}")
    print(f"Saved CSV to: {OUTPUT_CSV_PATH}")
    print(f"Saved Excel to: {OUTPUT_XLSX_PATH}")


if __name__ == "__main__":
    main()
