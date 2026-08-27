"""The page shell, and the small print of the mobile layout.

The rail used to be display:none below 760px, which left a phone with no
navigation at all: you landed on Activity and could not reach the library or
the backends from there. These pin down the parts that failure depended on.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import ui  # noqa: E402


def page(active="jobs"):
    return ui.shell(title="t", active=active, heading="H", body="<p>b</p>")


def test_every_page_is_reachable_from_every_other():
    html = page()
    for href in ("/", "/library", "/backends"):
        assert f"href='{href}'" in html


def test_the_rail_is_never_hidden_outright():
    """Hiding it is what stranded the phone; it may be restyled, not removed."""
    mobile = ui.CSS.split("@media(max-width:760px)")[1]
    assert ".rail{display:none}" not in ui.CSS.replace(" ", "")
    assert ".rail{" in mobile          # restyled instead


def test_the_nav_lies_flat_on_a_phone():
    mobile = ui.CSS.split("@media(max-width:760px)")[1]
    assert "display:flex" in mobile
    assert "margin-left:0" in mobile   # content no longer indented for the rail


def test_the_body_is_tagged_with_its_page():
    """The column rules are scoped per page, so the tag has to be there."""
    for active in ("jobs", "library", "backends"):
        assert f"content page-{active}" in page(active)


def test_panels_scroll_rather_than_the_page():
    mobile = ui.CSS.split("@media(max-width:760px)")[1]
    assert re.search(r"\.panel\{[^}]*overflow-x:auto", mobile)


def test_the_selection_bar_stays_in_reach():
    mobile = ui.CSS.split("@media(max-width:760px)")[1]
    assert re.search(r"#selbar\{[^}]*position:fixed", mobile)


def test_the_active_page_is_marked():
    assert "class='on'" in page("library")
    assert page("library").count("class='on'") == 1
