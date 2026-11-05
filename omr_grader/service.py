"""Glue between the engine and the web UI: decoding uploads, result tables, CSV / ZIP exports."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd

from .engine import GradeResult, OMRConfig, STATUS_CORRECT, encode_png, grade_image

LETTERS = "ABCDEFGHIJ"
SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"


def decode_image(data: bytes) -> Optional[np.ndarray]:
    """Bytes -> BGR image (EXIF orientation is applied by OpenCV), or None if unreadable."""
    arr = np.frombuffer(data, dtype=np.uint8)
    if arr.size == 0:
        return None
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def grade_bytes(data: bytes, cfg: OMRConfig) -> GradeResult:
    img = decode_image(data)
    if img is None:
        return GradeResult(ok=False, error="This file is not a readable image (use JPG or PNG).")
    return grade_image(img, cfg)


def _label(indices: list[int]) -> str:
    return ", ".join(LETTERS[i] for i in indices) if indices else "-"


def question_table(result: GradeResult) -> pd.DataFrame:
    rows = []
    for q in result.questions:
        rows.append({
            "Q": q.number,
            "Marked": _label(q.selected),
            "Correct answer": LETTERS[q.correct],
            "Result": q.status.capitalize(),
            "Marks": q.marks,
        })
    return pd.DataFrame(rows)


def summary_row(name: str, result: GradeResult) -> dict:
    if not result.ok:
        return {"File": name, "Status": "Could not read", "Marks": None, "Total": None, "Percentage": None,
                "Grade": "", "Correct": None, "Wrong": None, "Skipped": None, "Multiple marks": None,
                "Needs review": True, "Note": result.error}
    needs_review = result.multiple > 0 or result.skipped > 0
    note = []
    if result.multiple:
        note.append(f"{result.multiple} question(s) with more than one mark")
    if result.skipped:
        note.append(f"{result.skipped} unanswered")
    return {"File": name, "Status": "Graded", "Marks": result.marks, "Total": result.total_marks,
            "Percentage": result.percentage, "Grade": result.grade, "Correct": result.correct,
            "Wrong": result.wrong, "Skipped": result.skipped, "Multiple marks": result.multiple,
            "Needs review": needs_review, "Note": "; ".join(note)}


def batch_zip(results: list[tuple[str, GradeResult]]) -> bytes:
    """ZIP with the annotated image of every graded sheet plus a summary CSV."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        rows = []
        for name, res in results:
            rows.append(summary_row(name, res))
            if res.ok and res.annotated is not None:
                z.writestr(f"graded/{Path(name).stem}_graded.png", encode_png(res.annotated))
        z.writestr("results.csv", pd.DataFrame(rows).to_csv(index=False))
    return buf.getvalue()


def sample_files() -> dict[str, Path]:
    if not SAMPLES_DIR.exists():
        return {}
    return {p.stem: p for p in sorted(SAMPLES_DIR.glob("sheet*.jpg"))}
