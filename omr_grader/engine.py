"""OMR engine: sheet detection, perspective correction, bubble reading, grading.

This is a generalised, hardened version of the original prototype (Main.py / utils.py):

* any number of questions x choices (the prototype was fixed at 5 x 5)
* configurable answer key, marks per correct answer and negative marking
* adaptive mark threshold instead of a fixed pixel count
* detects multiple marks and skipped questions explicitly
* validates the detected grid so a page border / table edge is not mistaken for it
* never raises on a bad photo: returns a result with ``error`` set instead
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

STATUS_CORRECT = "correct"
STATUS_WRONG = "wrong"
STATUS_SKIPPED = "skipped"
STATUS_MULTIPLE = "multiple"

# BGR colours for the overlay (same scheme as the prototype)
GREEN = (0, 200, 0)
RED = (0, 0, 230)
YELLOW = (0, 220, 255)
ORANGE = (0, 140, 255)

MAX_RELIABLE_QUESTIONS = 30  # tested limit for one answer grid on a typical phone photo

GRADE_BANDS = [  # from Table 6 of the project report
    (90, "A1", "Outstanding"),
    (80, "A2", "Excellent"),
    (70, "B1", "Very Good"),
    (60, "B2", "Good"),
    (50, "C1", "Average"),
    (40, "C2", "Below Average"),
    (0, "D", "Unsatisfactory"),
]


def grade_for(percentage: float) -> tuple[str, str]:
    for lower, grade, remark in GRADE_BANDS:
        if percentage >= lower:
            return grade, remark
    return "D", "Unsatisfactory"


@dataclass
class OMRConfig:
    questions: int = 5
    choices: int = 5
    answer_key: list[int] = field(default_factory=lambda: [1, 4, 0, 3, 2])  # 0-based
    marks_correct: float = 1.0
    marks_wrong: float = 0.0  # marks deducted per wrong answer (negative marking)
    ink_level: int = 105  # lighting-normalised grey level (0-255) below which a pixel counts as pen ink
    interior: float = 0.55  # fraction of the bubble radius that is measured (keeps the outline out)
    min_fill: float = 0.20  # a bubble must be at least this inked to count as marked
    relative_fill: float = 0.35  # ...and at least this fraction of a typical filled bubble
    cell_px: int = 80  # size of one bubble cell in the flattened sheet
    max_side: int = 2800  # input images are scaled so the longer side is at most this

    def validate(self) -> None:
        if not 1 <= self.questions <= 100:
            raise ValueError("questions must be between 1 and 100")
        if not 2 <= self.choices <= 10:
            raise ValueError("choices must be between 2 and 10")
        if len(self.answer_key) != self.questions:
            raise ValueError(
                f"answer key has {len(self.answer_key)} entries but there are {self.questions} questions"
            )
        for i, a in enumerate(self.answer_key):
            if not 0 <= a < self.choices:
                raise ValueError(f"answer key entry {i + 1} is outside the {self.choices} available choices")


@dataclass
class QuestionResult:
    number: int  # 1-based
    status: str
    selected: list[int]  # 0-based indices of marked bubbles
    correct: int  # 0-based index of the right answer
    fills: list[float]
    marks: float


@dataclass
class GradeResult:
    ok: bool
    error: Optional[str] = None
    questions: list[QuestionResult] = field(default_factory=list)
    correct: int = 0
    wrong: int = 0
    skipped: int = 0
    multiple: int = 0
    marks: float = 0.0
    total_marks: float = 0.0
    percentage: float = 0.0
    grade: str = ""
    remark: str = ""
    annotated: Optional[np.ndarray] = None  # original photo with feedback drawn on it
    warped: Optional[np.ndarray] = None  # flattened answer grid with feedback
    stages: dict[str, np.ndarray] = field(default_factory=dict)  # pipeline images for debugging
    grade_box_found: bool = False


# ----------------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------------
def reorder(points: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-left, bottom-right (same as the prototype)."""
    pts = points.reshape(4, 2).astype(np.float32)
    out = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    out[0] = pts[np.argmin(s)]
    out[3] = pts[np.argmax(s)]
    out[1] = pts[np.argmin(d)]
    out[2] = pts[np.argmax(d)]
    return out


def _quad_candidates(gray: np.ndarray, min_frac: float, max_frac: float):
    """All convex 4-corner contours with a plausible area, biggest first (de-duplicated)."""
    h, w = gray.shape[:2]
    img_area = float(h * w)
    blur = cv2.GaussianBlur(gray, (5, 5), 1)

    edge_maps = []
    edges = cv2.Canny(blur, 10, 50)
    edge_maps.append(cv2.dilate(edges, np.ones((3, 3), np.uint8)))
    thr = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10)
    edge_maps.append(thr)

    found: list[tuple[float, np.ndarray]] = []
    for em in edge_maps:
        contours, _ = cv2.findContours(em, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_frac * img_area or area > max_frac * img_area:
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                found.append((area, reorder(approx)))

    found.sort(key=lambda t: t[0], reverse=True)
    unique: list[tuple[float, np.ndarray]] = []
    for area, quad in found:
        centre = quad.mean(axis=0)
        dup = False
        for a2, q2 in unique:
            if np.linalg.norm(centre - q2.mean(axis=0)) < 0.03 * max(h, w) and abs(area - a2) / a2 < 0.25:
                dup = True
                break
        if not dup:
            unique.append((area, quad))
    return unique


def cell_size(quad: np.ndarray, cfg: OMRConfig) -> tuple[int, int]:
    """Cell width/height for the flattened grid, keeping the real aspect ratio of the printed cells."""
    tl, tr, bl, br = quad
    w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    aspect = float(np.clip((w / cfg.choices) / max(h / cfg.questions, 1e-6), 0.25, 4.0))
    cw = int(round(cfg.cell_px * np.sqrt(aspect)))
    ch = int(round(cfg.cell_px / np.sqrt(aspect)))
    return max(cw, 20), max(ch, 20)


def _warp(img: np.ndarray, quad: np.ndarray, width: int, height: int) -> np.ndarray:
    dst = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    m = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(img, m, (width, height))


def _bubble_blobs(warped_gray: np.ndarray, cfg: OMRConfig, cw: int, ch: int):
    """Bounding boxes (x, y, w, h) of bubble-like blobs inside a flattened grid."""
    thr = cv2.adaptiveThreshold(warped_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 8)
    # Remove the grid's own border (and any ruled lines): anything straight and longer than ~1.5 cells.
    horiz = cv2.morphologyEx(thr, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (int(cw * 1.5), 1)))
    vert = cv2.morphologyEx(thr, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(ch * 1.5))))
    lines = cv2.dilate(cv2.bitwise_or(horiz, vert), np.ones((3, 3), np.uint8))
    thr = cv2.bitwise_and(thr, cv2.bitwise_not(lines))
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cell_area = cw * ch
    boxes = []
    for c in contours:
        a = cv2.contourArea(c)
        if not 0.08 * cell_area < a < 0.9 * cell_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if 0.4 < w / max(h, 1) < 2.5 and a / max(w * h, 1) > 0.5:
            boxes.append((x, y, w, h))
    return boxes


def _grid_fit(warped_gray: np.ndarray, cfg: OMRConfig, cw: int, ch: int) -> int:
    """Number of cells that contain a bubble sitting near the cell centre (max questions x choices).
    Only the true answer grid scores high: the paper edge or a table edge puts bubbles off-centre."""
    occupied = set()
    for x, y, w, h in _bubble_blobs(warped_gray, cfg, cw, ch):
        cx, cy = x + w / 2, y + h / 2
        r, c = int(cy // ch), int(cx // cw)
        if 0 <= r < cfg.questions and 0 <= c < cfg.choices:
            if abs(cx - (c + 0.5) * cw) < 0.3 * cw and abs(cy - (r + 0.5) * ch) < 0.3 * ch:
                occupied.add((r, c))
    return len(occupied)


def _pick_grid(gray: np.ndarray, cands, cfg: OMRConfig):
    """Choose the candidate whose contents look most like Q x C bubbles (ties go to the smaller shape)."""
    expected = cfg.questions * cfg.choices
    best, best_score = None, -1.0
    for area, quad in cands[:8]:
        cw, ch = cell_size(quad, cfg)
        gw, gh = cfg.choices * cw, cfg.questions * ch
        wg = cv2.cvtColor(_warp(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), quad, gw, gh), cv2.COLOR_BGR2GRAY)
        score = _grid_fit(wg, cfg, cw, ch) / expected
        if score > best_score + 1e-9 or (abs(score - best_score) <= 0.05 and best is not None and area < best[0]):
            best, best_score = (area, quad), max(score, best_score)
    if best is None or best_score < 0.35:
        # nothing looked like a bubble grid; fall back to the largest quad (prototype behaviour)
        return (cands[0] if cands else None), best_score
    return best, best_score


def _pick_grade_box(cands, grid_quad: np.ndarray, grid_area: float, shape):
    """Optional box below the grid where the score is printed."""
    gy_bottom = grid_quad[:, 1].max()
    best = None
    for area, quad in cands:
        if area >= grid_area * 0.98 or area < grid_area * 0.02:
            continue
        cy = quad[:, 1].mean()
        if cy <= gy_bottom:
            continue
        if best is None or area > best[0]:
            best = (area, quad)
    return best[1] if best else None


# ----------------------------------------------------------------------------
# bubble reading
# ----------------------------------------------------------------------------
def normalise_lighting(gray: np.ndarray, cw: int, ch: int) -> np.ndarray:
    """Divide out the paper brightness so shadows / gradients vanish: paper ~255, ink well below."""
    small = cv2.resize(gray, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    k = int(max(cw, ch) * 0.25 * 1.8) | 1
    background = cv2.morphologyEx(small, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    background = cv2.GaussianBlur(background, (0, 0), max(k / 4, 1))
    background = cv2.resize(background, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    norm = gray.astype(np.float32) / np.maximum(background.astype(np.float32), 1.0) * 255.0
    return np.clip(norm, 0, 255).astype(np.uint8)


def read_fills(warped_gray: np.ndarray, cfg: OMRConfig, cw: int, ch: int, centres=None) -> np.ndarray:
    """Fraction of the bubble interior that is inked, shape (questions, choices).

    Only the inside of each bubble is measured, so the printed outline never counts as a mark,
    whatever the lighting. ``centres`` come from bubble_centres(); cell centres are the fallback.
    """
    g = cv2.GaussianBlur(normalise_lighting(warped_gray, cw, ch), (3, 3), 0)
    ink = (g < cfg.ink_level).astype(np.uint8)
    centres = centres or {}
    fills = np.zeros((cfg.questions, cfg.choices), dtype=np.float32)
    for r in range(cfg.questions):
        for c in range(cfg.choices):
            ccx, ccy = c * cw + cw / 2, r * ch + ch / 2
            half_w, half_h = cw * 0.2, ch * 0.2  # default: a window around the cell centre
            cx, cy = ccx, ccy
            if (r, c) in centres:
                bx, by, rad = centres[(r, c)]
                if abs(bx - ccx) < 0.3 * cw and abs(by - ccy) < 0.3 * ch:  # trust the blob only if sane
                    cx, cy = bx, by
                    half_w = float(np.clip(rad * cfg.interior, cw * 0.12, cw * 0.25))
                    half_h = float(np.clip(rad * cfg.interior * 0.8, ch * 0.12, ch * 0.25))
            x0, x1 = int(max(cx - half_w, 0)), int(min(cx + half_w, g.shape[1]))
            y0, y1 = int(max(cy - half_h, 0)), int(min(cy + half_h, g.shape[0]))
            region = ink[y0:y1, x0:x1]
            fills[r, c] = region.mean() if region.size else 0.0
    return fills


def bubble_centres(warped_gray: np.ndarray, cfg: OMRConfig, cw: int, ch: int) -> dict[tuple[int, int], tuple[int, int, int]]:
    """Real (cx, cy, radius) of each printed bubble, so feedback lands on hand-drawn / uneven sheets too."""
    out: dict[tuple[int, int], tuple[int, int, int]] = {}
    for x, y, w, h in _bubble_blobs(warped_gray, cfg, cw, ch):
        cx, cy = x + w // 2, y + h // 2
        r, col = cy // ch, cx // cw
        if 0 <= r < cfg.questions and 0 <= col < cfg.choices and (r, col) not in out:
            out[(r, col)] = (cx, cy, int(0.45 * min(w, h)))
    return out


def decide_marks(fills: np.ndarray, cfg: OMRConfig) -> list[list[int]]:
    """Which bubbles count as marked in each row? Threshold adapts to how dark this sheet's pen is."""
    row_max = fills.max(axis=1)
    strong = row_max[row_max >= cfg.min_fill]
    ref = float(np.percentile(strong, 75)) if strong.size else 0.0
    threshold = max(cfg.min_fill, cfg.relative_fill * ref)
    return [[int(i) for i in np.where(row >= threshold)[0]] for row in fills]


# ----------------------------------------------------------------------------
# drawing
# ----------------------------------------------------------------------------
def _draw_feedback(canvas: np.ndarray, questions: list[QuestionResult], cw: int, ch: int, centres=None) -> np.ndarray:
    """Draw marks on a flattened grid canvas (width = choices*cell, height = questions*cell)."""
    out = np.zeros_like(canvas)
    default_r = int(min(cw, ch) * 0.36)
    centres = centres or {}

    def centre(q: int, c: int):
        if (q, c) in centres:
            return centres[(q, c)][:2]
        return (c * cw + cw // 2, q * ch + ch // 2)

    def rad(q: int, c: int) -> int:
        return centres[(q, c)][2] if (q, c) in centres else default_r

    for q in questions:
        r = q.number - 1
        if q.status == STATUS_CORRECT:
            cv2.circle(out, centre(r, q.correct), rad(r, q.correct), GREEN, cv2.FILLED)
        elif q.status == STATUS_WRONG:
            cv2.circle(out, centre(r, q.correct), rad(r, q.correct), GREEN, cv2.FILLED)
            cv2.circle(out, centre(r, q.selected[0]), rad(r, q.selected[0]), RED, cv2.FILLED)
        elif q.status == STATUS_SKIPPED:
            cv2.circle(out, centre(r, q.correct), rad(r, q.correct), YELLOW, cv2.FILLED)
        else:  # multiple marks: flag each marked bubble in orange, show the right one in green
            cv2.circle(out, centre(r, q.correct), rad(r, q.correct), GREEN, cv2.FILLED)
            for s in q.selected:
                cv2.circle(out, centre(r, s), rad(r, s), ORANGE, cv2.FILLED)
    return out


def _blend(base: np.ndarray, overlay: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    mask = overlay.any(axis=2)
    result = base.copy()
    result[mask] = cv2.addWeighted(base, 1 - alpha, overlay, alpha, 0)[mask]
    return result


def _score_card(text_lines: list[str], size=(325, 150)) -> np.ndarray:
    w, h = size
    card = np.full((h, w, 3), 255, np.uint8)
    y = int(h * 0.42) if len(text_lines) > 1 else h // 2 + 14
    for i, line in enumerate(text_lines):
        scale = 1.3 if i == 0 else 0.75
        thick = 3 if i == 0 else 2
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        cv2.putText(card, line, ((w - tw) // 2, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (200, 0, 0), thick, cv2.LINE_AA)
        y += int(h * 0.32)
    return card


# ----------------------------------------------------------------------------
# main entry point
# ----------------------------------------------------------------------------
def _fail(msg: str, stages: dict) -> GradeResult:
    return GradeResult(ok=False, error=msg, stages=stages)


def grade_image(img_bgr: np.ndarray, cfg: OMRConfig) -> GradeResult:
    cfg.validate()
    if img_bgr is None or img_bgr.size == 0:
        return _fail("The image could not be read.", {})

    h0, w0 = img_bgr.shape[:2]
    scale = min(1.0, cfg.max_side / max(h0, w0))
    img = cv2.resize(img_bgr, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else img_bgr.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    stages: dict[str, np.ndarray] = {"1 Original": img}
    stages["2 Grayscale"] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    stages["3 Blurred"] = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)
    stages["4 Canny edges"] = cv2.cvtColor(cv2.Canny(blur, 10, 50), cv2.COLOR_GRAY2BGR)

    cands = _quad_candidates(gray, min_frac=0.02, max_frac=0.97)
    if not cands:
        return _fail("No answer grid was found. Make sure the whole sheet is in the photo, flat and well lit.", stages)
    picked, score = _pick_grid(gray, cands, cfg)
    if picked is None:
        return _fail("No answer grid was found.", stages)
    grid_area, grid_quad = picked

    contour_view = img.copy()
    cv2.polylines(contour_view, [grid_quad.astype(np.int32).reshape(-1, 1, 2)[[0, 1, 3, 2]]], True, (0, 255, 0), 4)
    grade_quad = _pick_grade_box(cands, grid_quad, grid_area, gray.shape)
    if grade_quad is not None:
        cv2.polylines(contour_view, [grade_quad.astype(np.int32).reshape(-1, 1, 2)[[0, 1, 3, 2]]], True, (255, 0, 0), 3)
    stages["5 Detected grid (green) / grade box (blue)"] = contour_view

    cw, ch = cell_size(grid_quad, cfg)
    gw, gh = cfg.choices * cw, cfg.questions * ch
    warped = _warp(img, grid_quad, gw, gh)
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    stages["6 Flattened grid"] = warped

    centres = bubble_centres(warped_gray, cfg, cw, ch)
    fills = read_fills(warped_gray, cfg, cw, ch, centres)
    marked = decide_marks(fills, cfg)

    questions: list[QuestionResult] = []
    correct = wrong = skipped = multiple = 0
    marks = 0.0
    for r in range(cfg.questions):
        sel = marked[r]
        key = cfg.answer_key[r]
        if not sel:
            status, m = STATUS_SKIPPED, 0.0
            skipped += 1
        elif len(sel) > 1:
            status, m = STATUS_MULTIPLE, -cfg.marks_wrong
            multiple += 1
        elif sel[0] == key:
            status, m = STATUS_CORRECT, cfg.marks_correct
            correct += 1
        else:
            status, m = STATUS_WRONG, -cfg.marks_wrong
            wrong += 1
        marks += m
        questions.append(QuestionResult(r + 1, status, sel, key, [round(float(f), 3) for f in fills[r]], m))

    total = cfg.questions * cfg.marks_correct
    marks = max(marks, 0.0)
    pct = round(100.0 * marks / total, 1) if total else 0.0
    grade, remark = grade_for(pct)

    # feedback on the flattened grid
    overlay = _draw_feedback(warped, questions, cw, ch, centres)
    warped_marked = _blend(warped, overlay)
    stages["7 Feedback on flattened grid"] = warped_marked

    # warp feedback (and the score card) back onto the original photo
    inv = cv2.getPerspectiveTransform(np.float32([[0, 0], [gw, 0], [0, gh], [gw, gh]]), grid_quad)
    back = cv2.warpPerspective(overlay, inv, (img.shape[1], img.shape[0]))
    annotated = _blend(img, back)

    label = [f"{marks:g}/{total:g}", f"{pct:g}%  Grade {grade}"]
    card = _score_card(label)
    if grade_quad is not None:
        m2 = cv2.getPerspectiveTransform(np.float32([[0, 0], [325, 0], [0, 150], [325, 150]]), grade_quad)
        card_back = cv2.warpPerspective(card, m2, (img.shape[1], img.shape[0]))
        inner = np.zeros(card.shape[:2], np.uint8)
        inner[12:-12, 12:-12] = 255  # keep the printed box border visible
        card_mask = cv2.warpPerspective(inner, m2, (img.shape[1], img.shape[0])) > 0
        annotated[card_mask] = card_back[card_mask]
    else:  # no grade box on the sheet: print the score in the top-left corner
        cv2.rectangle(annotated, (10, 10), (330, 90), (255, 255, 255), cv2.FILLED)
        cv2.putText(annotated, label[0] + "  " + label[1], (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 0, 0), 2, cv2.LINE_AA)
    stages["8 Final result"] = annotated

    return GradeResult(
        ok=True, questions=questions, correct=correct, wrong=wrong, skipped=skipped, multiple=multiple,
        marks=round(marks, 2), total_marks=total, percentage=pct, grade=grade, remark=remark,
        annotated=annotated, warped=warped_marked, stages=stages, grade_box_found=grade_quad is not None,
    )


def parse_answer_key(text: str, choices: int) -> list[int]:
    """'B, E, A, D, C' / 'BEADC' / '2 5 1 4 3' -> [1, 4, 0, 3, 2]."""
    tokens = [t for t in text.replace(",", " ").replace(";", " ").split() if t]
    if len(tokens) == 1 and tokens[0].isalpha() and len(tokens[0]) > 1:
        tokens = list(tokens[0])  # "BEADC"
    out = []
    for t in tokens:
        if t.isdigit():
            idx = int(t) - 1
        elif len(t) == 1 and t.isalpha():
            idx = ord(t.upper()) - ord("A")
        else:
            raise ValueError(f"Could not read '{t}' in the answer key")
        if not 0 <= idx < choices:
            raise ValueError(f"'{t}' is outside the {choices} available choices")
        out.append(idx)
    return out


def encode_png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()
