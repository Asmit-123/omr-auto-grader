"""Generate printable blank OMR sheets that the engine is built to read.

Layout (same idea as the prototype sheet): a Name box on top, one bordered answer grid with
questions x choices bubbles, and a bordered GRADE box underneath. Question numbers and choice
letters sit *outside* the grid border so the grid is the largest clean rectangle on the page.
"""
from __future__ import annotations

import random
from typing import Optional, Sequence

import cv2
import numpy as np

LETTERS = "ABCDEFGHIJ"
GREY = (110, 110, 110)


def grid_box(questions: int, choices: int, width: int, height: int):
    """Grid rectangle: wide enough for the choices, as tall as the page allows."""
    col_w = min(190, (width - 360) / choices)
    gx0 = 230
    gx1 = int(gx0 + col_w * choices)
    gy0 = 380
    gy1 = height - 380
    return gx0, gx1, gy0, gy1


def bubble_radii(row_h: float, col_w: float):
    ry = int(min(row_h * 0.33, col_w * 0.30))
    rx = int(min(col_w * 0.38, ry * 1.5))
    return max(rx, 6), max(ry, 6)


def make_sheet(questions: int, choices: int, title: str = "MULTIPLE CHOICE",
               width: int = 1240, height: int = 1754) -> np.ndarray:
    """Blank A4 sheet at 150 dpi (1240 x 1754 px)."""
    img = np.full((height, width, 3), 255, np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, _), _ = cv2.getTextSize(title, font, 1.3, 3)
    cv2.putText(img, title, ((width - tw) // 2, 110), font, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    # name box
    cv2.putText(img, "Name", (110, 235), font, 0.9, GREY, 2, cv2.LINE_AA)
    cv2.rectangle(img, (230, 195), (width - 130, 250), GREY, 4)

    # answer grid
    gx0, gx1, gy0, gy1 = grid_box(questions, choices, width, height)
    row_h = (gy1 - gy0) / questions
    col_w = (gx1 - gx0) / choices
    cv2.rectangle(img, (gx0, gy0), (gx1, gy1), GREY, 5)

    for c in range(choices):
        cx = int(gx0 + col_w * (c + 0.5))
        cv2.putText(img, LETTERS[c], (cx - 12, gy0 - 22), font, 0.9, GREY, 2, cv2.LINE_AA)
    rx, ry = bubble_radii(row_h, col_w)
    scale = 0.9 if questions <= 40 else 0.55
    for r in range(questions):
        cy = int(gy0 + row_h * (r + 0.5))
        label = str(r + 1)
        (lw, lh), _ = cv2.getTextSize(label, font, scale, 2)
        cv2.putText(img, label, (gx0 - 25 - lw, cy + lh // 2), font, scale, GREY, 2, cv2.LINE_AA)
        for c in range(choices):
            cx = int(gx0 + col_w * (c + 0.5))
            cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, GREY, 3, cv2.LINE_AA)

    # grade box
    bx1, by1 = width - 130, height - 120
    cv2.rectangle(img, (bx1 - 420, by1 - 190), (bx1, by1), GREY, 5)
    cv2.putText(img, "GRADE", (bx1 - 300, by1 + 60), font, 1.0, GREY, 3, cv2.LINE_AA)
    return img


def fill_sheet(sheet: np.ndarray, questions: int, choices: int, answers: Sequence[Optional[Sequence[int]]],
               darkness: int = 25, seed: int = 0) -> np.ndarray:
    """Simulate a student filling the sheet. answers[i] = list of chosen indices (None/[] = skipped)."""
    rng = random.Random(seed)
    img = sheet.copy()
    height, width = img.shape[:2]
    gx0, gx1, gy0, gy1 = grid_box(questions, choices, width, height)
    row_h = (gy1 - gy0) / questions
    col_w = (gx1 - gx0) / choices
    rx, ry = bubble_radii(row_h, col_w)
    for r, picks in enumerate(answers):
        for c in picks or []:
            cx = int(gx0 + col_w * (c + 0.5)) + rng.randint(-3, 3)
            cy = int(gy0 + row_h * (r + 0.5)) + rng.randint(-3, 3)
            cv2.ellipse(img, (cx, cy), (max(rx - 3, 3), max(ry - 3, 3)), 0, 0, 360, (darkness,) * 3, cv2.FILLED, cv2.LINE_AA)
    return img


def sheet_to_pdf_bytes(img: np.ndarray) -> bytes:
    """Wrap the sheet image in a single-page A4 PDF (no extra dependencies, uses Pillow)."""
    from io import BytesIO
    from PIL import Image

    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    buf = BytesIO()
    pil.save(buf, format="PDF", resolution=150.0)
    return buf.getvalue()
