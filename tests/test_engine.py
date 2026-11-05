import random
from pathlib import Path

import cv2
import numpy as np
import pytest

from omr_grader.engine import OMRConfig, grade_image, grade_for, parse_answer_key
from omr_grader.sheet import make_sheet, fill_sheet

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


# ---------------------------------------------------------------- the original hand-drawn sheets
@pytest.mark.parametrize("name,expected", [
    ("sheet1", [0, 2, 1, 3, 4]),
    ("sheet2", [1, 4, 0, 2, 1]),
    ("sheet3", [4, 2, 1, 2, 0]),
])
def test_prototype_sheets(name, expected):
    r = grade_image(cv2.imread(str(SAMPLES / f"{name}.jpg")), OMRConfig())
    assert r.ok, r.error
    assert [q.selected[0] for q in r.questions] == expected
    assert r.grade_box_found


def test_prototype_scores():
    cfg = OMRConfig()  # key B,E,A,D,C from the prototype
    scores = {n: grade_image(cv2.imread(str(SAMPLES / f"{n}.jpg")), cfg).marks for n in ("sheet1", "sheet2", "sheet3")}
    assert scores == {"sheet1": 1.0, "sheet2": 3.0, "sheet3": 0.0}


def test_blank_sheet_is_all_skipped():
    r = grade_image(cv2.imread(str(SAMPLES / "blank_template.jpg")), OMRConfig())
    assert r.ok and r.skipped == 5 and r.marks == 0


# ---------------------------------------------------------------- synthetic phone photos
def phone_photo(sheet, seed=0, angle=6, persp=0.04, shadow=True, noise=6, background=True):
    rng = np.random.default_rng(seed)
    h, w = sheet.shape[:2]
    canvas_w, canvas_h = int(w * 1.25), int(h * 1.15)
    bg = np.full((canvas_h, canvas_w, 3), (60, 90, 120), np.uint8) if background else np.full((canvas_h, canvas_w, 3), 255, np.uint8)
    bg = (bg + rng.normal(0, 8, bg.shape)).clip(0, 255).astype(np.uint8)
    ox, oy = (canvas_w - w) // 2, (canvas_h - h) // 2
    src = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    j = lambda: rng.uniform(-persp, persp)
    dst = np.float32([[ox + j() * w, oy + j() * h], [ox + w + j() * w, oy + j() * h],
                      [ox + j() * w, oy + h + j() * h], [ox + w + j() * w, oy + h + j() * h]])
    m = cv2.getPerspectiveTransform(src, dst)
    rot = cv2.getRotationMatrix2D((canvas_w / 2, canvas_h / 2), angle, 1.0)
    m = np.vstack([rot, [0, 0, 1]]) @ m
    warped = cv2.warpPerspective(sheet, m, (canvas_w, canvas_h), borderMode=cv2.BORDER_TRANSPARENT, dst=bg.copy())
    img = warped.astype(np.float32)
    if shadow:
        grad = np.linspace(1.0, 0.7, canvas_w)[None, :, None]
        img *= grad
    img += rng.normal(0, noise, img.shape)
    img = cv2.GaussianBlur(img.clip(0, 255).astype(np.uint8), (3, 3), 0)
    return img


def random_answers(q, c, seed):
    rnd = random.Random(seed)
    return [rnd.randrange(c) for _ in range(q)]


@pytest.mark.parametrize("q,c,seed", [(20, 4, 1), (20, 4, 2), (10, 5, 3), (25, 4, 4), (30, 5, 5), (5, 5, 6), (30, 4, 7)])
def test_generated_sheet_through_phone_photo(q, c, seed):
    key = random_answers(q, c, seed)
    student = random_answers(q, c, seed + 100)
    cfg = OMRConfig(questions=q, choices=c, answer_key=key)
    sheet = fill_sheet(make_sheet(q, c), q, c, [[a] for a in student], seed=seed)
    photo = phone_photo(sheet, seed=seed)
    r = grade_image(photo, cfg)
    assert r.ok, r.error
    read = [x.selected[0] if len(x.selected) == 1 else None for x in r.questions]
    assert read == student
    assert r.correct == sum(a == b for a, b in zip(key, student))


def test_scan_without_background():
    q, c = 20, 4
    student = random_answers(q, c, 9)
    cfg = OMRConfig(questions=q, choices=c, answer_key=student)
    sheet = fill_sheet(make_sheet(q, c), q, c, [[a] for a in student])
    r = grade_image(sheet, cfg)
    assert r.ok and r.correct == q and r.percentage == 100 and r.grade == "A1"


def test_light_pen_still_detected():
    q, c = 10, 4
    student = random_answers(q, c, 11)
    cfg = OMRConfig(questions=q, choices=c, answer_key=student)
    sheet = fill_sheet(make_sheet(q, c), q, c, [[a] for a in student], darkness=70)
    r = grade_image(phone_photo(sheet, seed=11), cfg)
    assert r.ok and r.correct == q


def test_skipped_and_multiple_marks():
    q, c = 10, 4
    cfg = OMRConfig(questions=q, choices=c, answer_key=[0] * q, marks_wrong=0.25)
    answers = [[0], [0], [], [1], [0, 2], [0], [], [3], [0], [0]]
    sheet = fill_sheet(make_sheet(q, c), q, c, answers)
    r = grade_image(phone_photo(sheet, seed=3), cfg)
    assert r.ok
    assert r.skipped == 2 and r.multiple == 1 and r.wrong == 2 and r.correct == 5
    # 5 correct, 3 penalised (2 wrong + 1 multiple) at -0.25 each
    assert r.marks == pytest.approx(5 - 0.75)


def test_negative_marking_never_below_zero():
    q, c = 5, 4
    cfg = OMRConfig(questions=q, choices=c, answer_key=[0] * q, marks_wrong=1.0)
    sheet = fill_sheet(make_sheet(q, c), q, c, [[1]] * q)
    r = grade_image(sheet, cfg)
    assert r.ok and r.marks == 0 and r.percentage == 0


def test_page_edge_not_mistaken_for_grid():
    """Sheet photographed on a table: the paper edge is the biggest rectangle, but the grid must win."""
    q, c = 20, 4
    student = random_answers(q, c, 21)
    cfg = OMRConfig(questions=q, choices=c, answer_key=student)
    sheet = fill_sheet(make_sheet(q, c), q, c, [[a] for a in student])
    r = grade_image(phone_photo(sheet, seed=21, angle=3, persp=0.02), cfg)
    assert r.ok and r.correct == q


def test_garbage_image_fails_gracefully():
    r = grade_image(np.full((600, 800, 3), 128, np.uint8), OMRConfig())
    assert not r.ok and r.error


def test_config_validation():
    with pytest.raises(ValueError):
        OMRConfig(questions=5, choices=5, answer_key=[0, 1]).validate()
    with pytest.raises(ValueError):
        OMRConfig(questions=2, choices=4, answer_key=[0, 4]).validate()


# ---------------------------------------------------------------- helpers
def test_parse_answer_key():
    assert parse_answer_key("B, E, A, D, C", 5) == [1, 4, 0, 3, 2]
    assert parse_answer_key("BEADC", 5) == [1, 4, 0, 3, 2]
    assert parse_answer_key("2 5 1 4 3", 5) == [1, 4, 0, 3, 2]
    with pytest.raises(ValueError):
        parse_answer_key("A, F", 4)


@pytest.mark.parametrize("pct,grade", [(100, "A1"), (90, "A1"), (89.9, "A2"), (65, "B2"), (45, "C2"), (39.9, "D"), (0, "D")])
def test_grade_bands(pct, grade):
    assert grade_for(pct)[0] == grade
