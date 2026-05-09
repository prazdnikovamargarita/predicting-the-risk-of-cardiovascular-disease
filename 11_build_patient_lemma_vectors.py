from __future__ import annotations

import os
import re
from itertools import tee
from pathlib import Path

import pandas as pd
import spacy

from pipeline_common import INPUT_DIR, OUTPUT_DIR, read_table


# ============================== CONFIG ==============================
INPUT_RECORDS_PATH = INPUT_DIR / "text_features" / "records.csv"
OUTPUT_DIR_PATH = OUTPUT_DIR / "text_features" / "lemma_vectors"

ID_COL = "ID_PT"
TEXT_COL = "REZ"

SPACY_MODEL_NAME = "uk_core_news_md"
MIN_TOKEN_LENGTH = 4
BIGRAMS_FROM = "lemmas"  # "lemmas", "surface", "both", or None
# ====================================================================


UA_ALPHA_PATTERN = re.compile(r"^[А-Яа-яЇїІіЄєҐґ]+$")


def load_spacy_model(model_name: str):
    try:
        return spacy.load(model_name)
    except OSError as error:
        raise RuntimeError(
            f"spaCy model '{model_name}' is not installed. "
            f"Install it before running this script."
        ) from error


def ensure_sentencizer(nlp) -> None:
    if not nlp.has_pipe("senter") and not nlp.has_pipe("parser"):
        try:
            nlp.add_pipe("sentencizer")
        except Exception:
            pass


def bigrams(seq: list[str]) -> list[str]:
    left, right = tee(seq)
    next(right, None)
    return [" ".join(pair) for pair in zip(left, right)]


def extract_lemmas_and_bigrams_from_texts(
    texts: pd.Series,
    nlp,
    unique: bool = False,
    bigrams_from: str | None = BIGRAMS_FROM,
    min_token_len: int = MIN_TOKEN_LENGTH,
) -> list[str]:
    ensure_sentencizer(nlp)
    bag: list[str] = []

    for raw in texts.dropna().astype(str):
        text = raw.replace("_", " ")
        doc = nlp(text)

        sentences = list(doc.sents)
        if not sentences:
            parts = re.split(r"[\.!?…;:\n]+", text)
            sentences = [nlp(part.strip()) for part in parts if part.strip()]

        for sentence in sentences:
            tokens = [
                token
                for token in sentence
                if token.is_alpha
                and UA_ALPHA_PATTERN.match(token.text)
                and len(token.text) >= min_token_len
            ]
            if not tokens:
                continue

            lemmas = [token.lemma_.lower() for token in tokens if len(token.lemma_) >= min_token_len]
            surfaces = [token.text.lower() for token in tokens]

            bag.extend(lemmas)

            if bigrams_from:
                if bigrams_from in {"lemmas", "both"} and len(lemmas) >= 2:
                    bag.extend(bigrams(lemmas))
                if bigrams_from in {"surface", "both"} and len(surfaces) >= 2:
                    bag.extend(bigrams(surfaces))

    return sorted(set(bag)) if unique else bag


def build_patient_lemma_vectors(df: pd.DataFrame, nlp) -> pd.DataFrame:
    if ID_COL not in df.columns:
        raise KeyError(f"Missing ID column: {ID_COL}")
    if TEXT_COL not in df.columns:
        raise KeyError(f"Missing text column: {TEXT_COL}")

    rows = []
    df = df.dropna(subset=[TEXT_COL]).copy()
    df = df[df[TEXT_COL].astype(str).str.len() > 0]
    df[ID_COL] = df[ID_COL].astype(str).str.strip()

    for patient_id, group in df.groupby(ID_COL, sort=False):
        lemmas = extract_lemmas_and_bigrams_from_texts(group[TEXT_COL], nlp, unique=True)
        rows.append({ID_COL: patient_id, "LEMMA_VECTOR": ", ".join(lemmas)})

    return pd.DataFrame(rows)


def main() -> None:
    nlp = load_spacy_model(SPACY_MODEL_NAME)
    df = read_table(INPUT_RECORDS_PATH)

    vectors = build_patient_lemma_vectors(df, nlp)
    all_lemmas = extract_lemmas_and_bigrams_from_texts(df[TEXT_COL], nlp, unique=False)
    unique_lemmas = sorted(set(all_lemmas))

    OUTPUT_DIR_PATH.mkdir(parents=True, exist_ok=True)
    vectors.to_csv(OUTPUT_DIR_PATH / "patient_lemma_vectors.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"LEMMA": all_lemmas}).to_csv(OUTPUT_DIR_PATH / "all_lemmas.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"LEMMA": unique_lemmas}).to_csv(OUTPUT_DIR_PATH / "unique_lemmas.csv", index=False, encoding="utf-8-sig")

    print(f"Patient vectors: {len(vectors)}")
    print(f"Unique lemmas: {len(unique_lemmas)}")
    print(f"Saved output to: {OUTPUT_DIR_PATH}")


if __name__ == "__main__":
    main()
