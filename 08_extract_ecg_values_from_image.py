from __future__ import annotations

import json
import os
import re
from pathlib import Path

from pipeline_common import INPUT_DIR, OUTPUT_DIR

import cv2
import numpy as np
import pandas as pd
import pytesseract


# ============================== CONFIG ==============================
# ============================== INPUT ==============================
INPUT_IMAGE_PATH = INPUT_DIR / "ecg_images" / "ecg_image.jpeg"
# ============================== OUTPUT =============================
OUTPUT_JSON_FLAT = OUTPUT_DIR / "ecg_ocr" / "ecg_values_flat.json"
OUTPUT_JSON_HIERARCHICAL = OUTPUT_DIR / "ecg_ocr" / "ecg_values_by_lead.json"
OUTPUT_ANNOTATED_IMAGE = OUTPUT_DIR / "ecg_ocr" / "ecg_grid_overlay.png"
OUTPUT_CSV = OUTPUT_DIR / "ecg_ocr" / "ecg_values_flat.csv"
# ====================================================================

# Optional. If Tesseract is not in PATH, set environment variable:
# Windows example: set TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
TESSERACT_CMD = os.getenv("TESSERACT_CMD")

VAL_X0_FRAC = 0.66
VAL_X1_FRAC = 0.97
Y_TOP_FRAC = 0.045
Y_BOT_FRAC = 0.945
PAD_V_FRAC = 0.004
PAD_H_FRAC = 0.006

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
ATTRIBUTES = ["PA", "QA", "RA", "STM", "TA"]
LEAD_SUFFIX = {
    "I": "1",
    "II": "2",
    "III": "3",
    "aVR": "aVR",
    "aVL": "aVL",
    "aVF": "aVF",
    "V1": "V1",
    "V2": "V2",
    "V3": "V3",
    "V4": "V4",
    "V5": "V5",
    "V6": "V6",
    "aVB": "aVR",
}
# ====================================================================


NUMERIC_MV_PATTERN = re.compile(r"[+\-]?\d+(?:\.\d+)?\s*mV", re.I)


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise IOError(f"Could not decode image: {path}")
    return image


def ocr_numeric_value(roi: np.ndarray) -> str:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    if np.mean(thresholded) < 127:
        thresholded = 255 - thresholded

    text = pytesseract.image_to_string(
        thresholded,
        config='--psm 7 -c tessedit_char_whitelist="0123456789+-().mV"',
    )
    match = NUMERIC_MV_PATTERN.search(text.replace("\n", " "))
    return match.group(0).replace(" ", "") if match else text.strip()


def build_grid_boxes(image: np.ndarray) -> list[tuple[int, int, int, int, str, str]]:
    height, width = image.shape[:2]

    x0 = int(VAL_X0_FRAC * width)
    x1 = int(VAL_X1_FRAC * width)
    y0 = int(Y_TOP_FRAC * height)
    y1 = int(Y_BOT_FRAC * height)

    row_height = (y1 - y0) / len(LEADS)
    cell_height = row_height / len(ATTRIBUTES)
    pad_vertical = int(PAD_V_FRAC * height)
    pad_horizontal = int(PAD_H_FRAC * width)

    boxes = []
    for row_no, lead in enumerate(LEADS):
        y_row = y0 + int(row_no * row_height)
        for col_no, attribute in enumerate(ATTRIBUTES):
            y_cell = y_row + int(col_no * cell_height)
            x = x0 + pad_horizontal
            y = y_cell + pad_vertical
            box_width = max(1, (x1 - x0) - 2 * pad_horizontal)
            box_height = max(1, int(cell_height) - 2 * pad_vertical)
            boxes.append((x, y, box_width, box_height, lead, attribute))
    return boxes


def flatten_results(results: dict[str, dict[str, str]]) -> dict[str, str]:
    flat = {}
    for lead in LEADS:
        if lead not in results:
            continue
        suffix = LEAD_SUFFIX.get(lead, lead)
        for attribute, value in results[lead].items():
            flat[f"{attribute}_{suffix}"] = value
    return flat


def annotate_image(image: np.ndarray, boxes: list[tuple[int, int, int, int, str, str]]) -> np.ndarray:
    overlay = image.copy()
    for index, (x, y, w, h, _, _) in enumerate(boxes, start=1):
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            overlay,
            str(index),
            (x, max(0, y - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return overlay


def main() -> None:
    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

    image = read_image(INPUT_IMAGE_PATH)
    boxes = build_grid_boxes(image)

    results = {lead: {} for lead in LEADS}
    for x, y, w, h, lead, attribute in boxes:
        roi = image[y:y + h, x:x + w]
        results[lead][attribute] = ocr_numeric_value(roi)

    flat = flatten_results(results)
    annotated = annotate_image(image, boxes)

    OUTPUT_JSON_FLAT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_JSON_HIERARCHICAL.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    with OUTPUT_JSON_FLAT.open("w", encoding="utf-8") as file:
        json.dump(flat, file, ensure_ascii=False, indent=2)

    pd.DataFrame([flat]).to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    cv2.imwrite(str(OUTPUT_ANNOTATED_IMAGE), annotated)

    print(f"Saved output to: {OUTPUT_JSON_FLAT.parent}")


if __name__ == "__main__":
    main()
