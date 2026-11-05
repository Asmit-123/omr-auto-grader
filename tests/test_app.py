from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_app_boots():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, at.exception
    assert at.title[0].value == "OMR Sheet Auto-Grader"


def test_grades_project_sample_end_to_end():
    at = AppTest.from_file(APP, default_timeout=60).run()
    # radio[0] is the image source (main area), radio[1] the sheet type (sidebar)
    at.radio[0].set_value("Try a project sample").run()
    assert not at.exception, at.exception
    at.selectbox[0].set_value("sheet2").run()
    assert not at.exception, at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Marks"] == "3 / 5"
    assert metrics["Percentage"] == "60%"
    assert metrics["Grade"] == "B2"


def test_custom_sheet_and_bad_key():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.radio[0].set_value("Custom sheet").run()
    assert not at.exception
    at.sidebar.text_area[0].set_value("A, B, Z").run()
    assert at.sidebar.error  # invalid key is reported, app does not crash
