"""OMR Sheet Auto-Grading Software - web app (Streamlit).

Run locally:   streamlit run app.py
"""
from __future__ import annotations

import cv2
import pandas as pd
import streamlit as st

from omr_grader import __version__
from omr_grader.engine import (GRADE_BANDS, MAX_RELIABLE_QUESTIONS, OMRConfig, encode_png, parse_answer_key)
from omr_grader.service import (LETTERS, batch_zip, grade_bytes, question_table, sample_files, summary_row)
from omr_grader.sheet import make_sheet, sheet_to_pdf_bytes

st.set_page_config(page_title="OMR Auto-Grader", page_icon="📝", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; max-width: 1200px;}
      div[data-testid="stMetric"] {background: rgba(128,128,128,0.08); border-radius: 10px; padding: 10px 14px;}
      .legend span {display:inline-block; width:12px; height:12px; border-radius:50%; margin:0 6px 0 14px; vertical-align:middle;}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- sidebar: exam setup
st.sidebar.title("📝 Exam setup")
preset = st.sidebar.radio(
    "Sheet type",
    ["Sample sheet (5 × 5, from the project)", "Custom sheet"],
    help="The sample preset matches the hand-drawn sheets from the project folder.",
)
if preset.startswith("Sample"):
    questions, choices, default_key = 5, 5, "B, E, A, D, C"
    st.sidebar.caption("5 questions · 5 choices (A–E)")
else:
    questions = st.sidebar.number_input("Number of questions", 1, 100, 20)
    choices = st.sidebar.number_input("Choices per question", 2, 10, 4)
    default_key = ", ".join(LETTERS[i % int(choices)] for i in range(int(questions)))

key_text = st.sidebar.text_area(
    "Answer key", value=default_key, height=90,
    help="Letters or numbers, e.g. B, E, A, D, C  or  BEADC  or  2 5 1 4 3",
    key=f"key_{preset}_{questions}_{choices}",
)
marks_correct = st.sidebar.number_input("Marks per correct answer", 0.25, 100.0, 1.0, 0.25)
negative = st.sidebar.checkbox("Negative marking")
marks_wrong = st.sidebar.number_input("Marks deducted per wrong answer", 0.0, 100.0, 0.25, 0.25) if negative else 0.0
with st.sidebar.expander("Advanced"):
    sensitivity = st.slider(
        "Mark sensitivity", 0.15, 0.60, 0.35, 0.05,
        help="Lower = counts lighter pencil/pen marks. Raise it if stray smudges are being read as answers.",
    )

cfg, cfg_error = None, None
try:
    key = parse_answer_key(key_text, int(choices))
    cfg = OMRConfig(questions=int(questions), choices=int(choices), answer_key=key,
                    marks_correct=float(marks_correct), marks_wrong=float(marks_wrong), relative_fill=float(sensitivity))
    cfg.validate()
except ValueError as e:
    cfg_error = str(e)

if cfg_error:
    st.sidebar.error(cfg_error)
elif int(questions) > MAX_RELIABLE_QUESTIONS:
    st.sidebar.warning(
        f"More than {MAX_RELIABLE_QUESTIONS} questions in one grid was not reliable in testing with phone photos. "
        "Use a flatbed scan, or split the exam over two sheets."
    )

st.title("OMR Sheet Auto-Grader")
st.caption("Photograph or scan a filled OMR sheet and get the score, grade and colour-coded feedback in seconds.")

tab_single, tab_batch, tab_print, tab_how = st.tabs(["Grade one sheet", "Batch grading", "Print a sheet", "How it works"])


def show_legend():
    st.markdown(
        '<div class="legend"><b>Feedback:</b>'
        '<span style="background:#00c800"></span>correct answer'
        '<span style="background:#e60000"></span>wrong answer given'
        '<span style="background:#ffdc00"></span>skipped (shows right answer)'
        '<span style="background:#ff8c00"></span>more than one mark</div>',
        unsafe_allow_html=True,
    )


def show_result(res, label: str):
    if not res.ok:
        st.error(res.error)
        if res.stages:
            with st.expander("What the grader saw"):
                st.image(cv2.cvtColor(res.stages["1 Original"], cv2.COLOR_BGR2RGB), width="stretch")
        return
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Marks", f"{res.marks:g} / {res.total_marks:g}")
    c2.metric("Percentage", f"{res.percentage:g}%")
    c3.metric("Grade", res.grade, res.remark, delta_color="off")
    c4.metric("Correct / Wrong", f"{res.correct} / {res.wrong}")
    c5.metric("Skipped / Multiple", f"{res.skipped} / {res.multiple}")
    if res.multiple or res.skipped:
        st.warning("Some questions were skipped or have more than one mark. Check them below before using this score.")

    left, right = st.columns([3, 2])
    with left:
        st.image(cv2.cvtColor(res.annotated, cv2.COLOR_BGR2RGB), caption="Graded sheet", width="stretch")
        show_legend()
    with right:
        df = question_table(res)
        st.dataframe(
            df.style.apply(lambda r: [""] * len(r) if r["Result"] == "Correct" else ["background-color: rgba(230,0,0,0.12)"] * len(r), axis=1),
            hide_index=True, width="stretch", height=min(60 + 35 * len(df), 520),
        )
        st.download_button("⬇ Graded image (PNG)", encode_png(res.annotated), f"{label}_graded.png", "image/png")
        st.download_button("⬇ Question table (CSV)", df.to_csv(index=False), f"{label}_questions.csv", "text/csv")
    with st.expander("Pipeline stages (what the grader did)"):
        names = list(res.stages)
        cols = st.columns(4)
        for i, n in enumerate(names):
            cols[i % 4].image(cv2.cvtColor(res.stages[n], cv2.COLOR_BGR2RGB), caption=n, width="stretch")


# --------------------------------------------------------------------------- single sheet
with tab_single:
    if cfg is None:
        st.info("Fix the setup in the sidebar to continue.")
    else:
        source = st.radio("Image source", ["Upload a file", "Use the camera", "Try a project sample"], horizontal=True)
        data, label = None, "sheet"
        if source == "Upload a file":
            up = st.file_uploader("Photo or scan of the OMR sheet", type=["jpg", "jpeg", "png", "bmp", "webp"])
            if up:
                data, label = up.getvalue(), up.name.rsplit(".", 1)[0]
        elif source == "Use the camera":
            cam = st.camera_input("Hold the whole sheet in view, flat and well lit")
            if cam:
                data, label = cam.getvalue(), "camera"
        else:
            samples = sample_files()
            if not preset.startswith("Sample"):
                st.info("The project samples are 5 × 5 sheets. Switch the sidebar to the sample preset to use them.")
            elif samples:
                pick = st.selectbox("Sample", list(samples))
                data, label = samples[pick].read_bytes(), pick
        if data:
            with st.spinner("Grading…"):
                res = grade_bytes(data, cfg)
            show_result(res, label)

# --------------------------------------------------------------------------- batch
with tab_batch:
    if cfg is None:
        st.info("Fix the setup in the sidebar to continue.")
    else:
        files = st.file_uploader("Upload all the sheets for this exam", type=["jpg", "jpeg", "png", "bmp", "webp"],
                                 accept_multiple_files=True, key="batch")
        if files:
            results, rows = [], []
            bar = st.progress(0.0, text="Grading…")
            for i, f in enumerate(files):
                res = grade_bytes(f.getvalue(), cfg)
                results.append((f.name, res))
                rows.append(summary_row(f.name, res))
                bar.progress((i + 1) / len(files), text=f"Graded {i + 1} of {len(files)}")
            bar.empty()
            df = pd.DataFrame(rows)
            graded = df[df["Status"] == "Graded"]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Sheets", len(df))
            m2.metric("Graded", len(graded))
            m3.metric("Average %", f"{graded['Percentage'].mean():.1f}" if len(graded) else "-")
            m4.metric("Need review", int(df["Needs review"].sum()))
            st.dataframe(df, hide_index=True, width="stretch")
            d1, d2 = st.columns(2)
            d1.download_button("⬇ Results (CSV)", df.to_csv(index=False), "omr_results.csv", "text/csv")
            d2.download_button("⬇ Results + graded images (ZIP)", batch_zip(results), "omr_graded.zip", "application/zip")
            if len(graded):
                st.bar_chart(graded["Grade"].value_counts().reindex([g for _, g, _ in GRADE_BANDS]).fillna(0), height=220)
            pick = st.selectbox("Inspect a sheet", [n for n, _ in results])
            show_result(dict(results)[pick], pick.rsplit(".", 1)[0])

# --------------------------------------------------------------------------- printable sheet
with tab_print:
    if cfg is None:
        st.info("Fix the setup in the sidebar to continue.")
    else:
        st.write(f"A blank sheet with **{cfg.questions} questions × {cfg.choices} choices**, laid out for this grader. "
                 "Print at 100% on A4, have students fill the bubbles fully, then photograph or scan the sheet.")
        title = st.text_input("Sheet title", "MULTIPLE CHOICE")
        sheet = make_sheet(cfg.questions, cfg.choices, title=title)
        st.image(cv2.cvtColor(sheet, cv2.COLOR_BGR2RGB), width=420)
        p1, p2 = st.columns(2)
        p1.download_button("⬇ Printable PDF", sheet_to_pdf_bytes(sheet), "omr_sheet.pdf", "application/pdf")
        p2.download_button("⬇ PNG image", encode_png(sheet), "omr_sheet.png", "image/png")

# --------------------------------------------------------------------------- how it works
with tab_how:
    st.markdown(
        """
**Pipeline** (the same idea as the project report, hardened for real photos)

1. **Preprocess**: grayscale, Gaussian blur, Canny edges.
2. **Find the answer grid**: candidate rectangles are tested and the one whose bubbles sit on a regular
   rows × columns lattice wins, so the paper edge or a table edge is never mistaken for the grid.
3. **Flatten**: a four-point perspective transform gives a top-down view, even for tilted photos.
4. **Even out the lighting**: the paper brightness is divided out so shadows and gradients disappear.
5. **Read the bubbles**: the inside of each bubble is measured, so the printed outline never counts as a mark.
6. **Grade**: correct / wrong / skipped / multiple marks, optional negative marking, percentage and grade band.
7. **Show it**: colour feedback is warped back onto your original photo, and the score is written in the GRADE box.
        """
    )
    st.table(pd.DataFrame([{"Score": f">= {lo}%" if lo else "< 40%", "Grade": g, "Remark": r} for lo, g, r in GRADE_BANDS]))
    st.markdown(
        """
**For best results**: keep the whole sheet in frame, flat, with even light and no heavy shadow across the grid.
Fill bubbles fully. A bubble with two marks is flagged and scored as wrong (minus the negative mark, if enabled).
        """
    )

st.divider()
st.caption(f"OMR Auto-Grader v{__version__}")
