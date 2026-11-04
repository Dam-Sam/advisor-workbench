"""Runs every page headlessly, clicks the main buttons, and fails on any exception."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")
PAGES = ["views/home.py", "views/portfolio.py", "views/reconciliation.py", "views/ai_assistant.py"]


def run_page(page: str) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    if page != PAGES[0]:
        at.switch_page(page).run()
    return at


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_errors(page):
    at = run_page(page)
    assert not at.exception, [e.message for e in at.exception]


def test_explain_break_and_draft_summary():
    at = run_page("views/ai_assistant.py")
    [b for b in at.button if b.label == "Explain this break"][0].click().run()
    assert not at.exception
    assert any("Likely cause" in m.value for m in at.markdown)
    [b for b in at.button if b.label == "Draft update"][0].click().run()
    assert not at.exception
    assert any("Draft (edit before sending)" == t.label for t in at.text_area)
