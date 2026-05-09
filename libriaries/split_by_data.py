from __future__ import annotations

import re

import pandas as pd
from pandas.tseries.offsets import DateOffset


DATE_COL = "DATE_F"
ID_COL = "ID_PT"
TEXT_COL = "REZ"
DIAGNOSIS_COLUMNS = ["D_OSN", "D_SOP1", "D_SOP2"]


def clean_rez_column(df: pd.DataFrame, text_col: str = TEXT_COL) -> pd.DataFrame:
    """Remove empty text rows and normalize the text column."""
    df = df.copy()
    if text_col not in df.columns:
        return df
    df[text_col] = df[text_col].astype(str).str.strip()
    df = df[df[text_col].ne("") & df[text_col].str.lower().ne("nan")]
    return df


def row_matches_diagnosis(
    df: pd.DataFrame,
    diagnosis_pattern: str,
    text_pattern: str | None = None,
) -> pd.Series:
    """Return a row-level mask for diagnosis codes and optional REZ text pattern."""
    mask = pd.Series(False, index=df.index)

    for col in DIAGNOSIS_COLUMNS:
        if col in df.columns:
            mask |= df[col].astype(str).str.contains(diagnosis_pattern, flags=re.I, regex=True, na=False)

    if text_pattern and TEXT_COL in df.columns:
        mask |= df[TEXT_COL].astype(str).str.contains(text_pattern, flags=re.I, regex=True, na=False)

    return mask


def process_patient_group(
    df_group: pd.DataFrame,
    diagnosis_pattern: str,
    text_pattern: str | None = None,
    months_before: int = 0,
    months_after: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split one patient's records into records before and after the first target diagnosis date."""
    if DATE_COL not in df_group.columns:
        raise KeyError(f"Missing date column: {DATE_COL}")

    df_group = df_group.copy().drop_duplicates().reset_index(drop=True)
    df_group[DATE_COL] = pd.to_datetime(df_group[DATE_COL], errors="coerce")
    df_group = df_group.dropna(subset=[DATE_COL])

    hits = df_group.loc[row_matches_diagnosis(df_group, diagnosis_pattern, text_pattern)]
    if hits.empty:
        return pd.DataFrame(), pd.DataFrame()

    first_date = hits[DATE_COL].min()

    before_df = df_group.loc[
        df_group[DATE_COL] < (first_date - DateOffset(months=months_before))
    ].copy()
    after_df = df_group.loc[
        df_group[DATE_COL] >= (first_date + DateOffset(months=months_after))
    ].copy()

    return (
        clean_rez_column(before_df).drop_duplicates().reset_index(drop=True),
        clean_rez_column(after_df).drop_duplicates().reset_index(drop=True),
    )
