# OMR Sheet Auto-Grader

Photograph or scan a filled multiple-choice OMR sheet and get the score, percentage, grade and colour-coded
feedback in seconds. A web app built with Python, OpenCV and Streamlit. No special scanner needed.

This is the finished product of the project in the reports: the prototype (`Main.py` / `utils.py`) was a
fixed 5 × 5 script with a hard-coded key; this version is a configurable, tested, deployable web app.

## What it does

| Feature | Status |
|---|---|
| Any number of questions x choices (tested 5 to 30 questions, 4 to 5 choices per grid) | done |
| Answer key typed in the UI (`B, E, A, D, C`, `BEADC` or `2 5 1 4 3`) | done |
| Marks per question and optional **negative marking** (score never drops below 0) | done |
| Correct / wrong / **skipped** / **multiple marks** detection | done |
| Grade bands A1 to D from the report (Table 6) | done |
| Colour feedback drawn onto your original photo, score written in the GRADE box | done |
| Upload, **live camera** or sample input | done |
| **Batch grading**, summary table, CSV and ZIP of graded images, "needs review" flags | done |
| Printable blank sheet generator (PDF/PNG) for any question count | done |
| Handles tilt, perspective, shadows, noise, sheet photographed on a table | done (tested) |
| Never crashes on a bad photo; shows what the grader saw instead | done |

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py               # opens http://localhost:8501
```

Try it immediately: choose **Grade one sheet -> Try a project sample**. With the default key `B, E, A, D, C`,
`sheet2` scores 3/5 (60%, grade B2).

## Deploy

The app is one container, so any host works.

**Streamlit Community Cloud (free, easiest)**
1. Push this folder to a GitHub repo.
2. Go to share.streamlit.io, click **New app**, pick the repo, main file `app.py`, and deploy.
   It installs `requirements.txt` automatically.

**Render (free tier, uses the included Dockerfile)**
1. Push to GitHub, then Render -> **New + -> Blueprint** -> pick the repo (`render.yaml` is included).

**Any Docker host (Railway, Fly.io, Cloud Run, your own server)**
```bash
docker build -t omr-grader .
docker run -p 8501:8501 omr-grader        # honours $PORT on hosts that set it
```

Health check endpoint: `/_stcore/health`. Upload limit is 25 MB per file (`.streamlit/config.toml`).

## Using it well

* Use **Print a sheet** to make sheets this grader is designed for. Fill bubbles fully.
* Keep the whole sheet in frame, flat, with even light. Heavy shadow across the grid is the main cause of misreads.
* Your own sheet design also works if it has one bordered answer grid with bubbles in a regular rows x columns
  layout (like the hand-drawn project sheets), with an optional bordered box below it for the grade.
* Questions with two marks or no mark are flagged so you can review them.
* **Known limit:** one grid of more than about 30 questions was not reliable on phone photos because the rows get
  too thin. Split longer exams over two sheets, or use a flatbed scan.

## Project layout

```
app.py                  Streamlit web app (UI only)
omr_grader/engine.py    detection, perspective warp, lighting normalisation, bubble reading, grading
omr_grader/service.py   decoding uploads, result tables, CSV / ZIP export
omr_grader/sheet.py     printable sheet generator
samples/                the project's hand-drawn sample sheets
tests/                  30 tests (real samples, synthetic phone photos, UI smoke tests)
Dockerfile, render.yaml, .streamlit/config.toml, requirements.txt
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The suite grades the original project sheets, plus randomly filled generated sheets that are tilted,
perspective-skewed, shadowed, noisy and placed on a table background, and drives the real UI.

## How it works

1. Grayscale, blur, Canny edges.
2. Candidate rectangles are tested; the one whose bubbles sit centred in a rows x columns lattice is the answer grid
   (so the paper edge is never picked by mistake).
3. Four-point perspective transform flattens the grid, keeping the real cell proportions.
4. Paper brightness is divided out, so shadows and gradients disappear.
5. The inside of each bubble is measured, so printed outlines never count as marks. The threshold adapts to how dark
   this sheet's pen is.
6. Grading, score card and colour feedback are warped back onto the original photo.
