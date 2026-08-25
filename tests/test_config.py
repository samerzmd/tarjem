"""Env parsing. The empty-string cases are the ones that matter in Docker:
Compose substitutes "" for any variable missing from .env, and a naive
os.getenv(name, default) would take that "" as a real value.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings, _bool, _int, _list, _str  # noqa: E402


@pytest.fixture
def env(monkeypatch):
    def _set(value):
        if value is None:
            monkeypatch.delenv("TARJEM_T", raising=False)
        else:
            monkeypatch.setenv("TARJEM_T", value)
    return _set


@pytest.mark.parametrize("value", [None, "", "   "])
def test_str_falls_back_when_unset_or_blank(env, value):
    env(value)
    assert _str("TARJEM_T", "claude-opus-5") == "claude-opus-5"


def test_str_takes_a_real_value_and_trims(env):
    env("  claude-sonnet-5 ")
    assert _str("TARJEM_T", "claude-opus-5") == "claude-sonnet-5"


@pytest.mark.parametrize("value", [None, "", "   "])
def test_bool_keeps_a_true_default_when_blank(env, value):
    env(value)
    assert _bool("TARJEM_T", True) is True
    assert _bool("TARJEM_T", False) is False


@pytest.mark.parametrize("value,expected", [
    ("true", True), ("True", True), ("1", True), ("yes", True), ("on", True),
    ("false", False), ("0", False), ("no", False), ("nonsense", False),
])
def test_bool_parses_real_values(env, value, expected):
    env(value)
    assert _bool("TARJEM_T", True) is expected


@pytest.mark.parametrize("value", [None, "", "not-a-number"])
def test_int_falls_back_on_blank_or_garbage(env, value):
    env(value)
    assert _int("TARJEM_T", 40) == 40


def test_int_takes_a_real_value(env):
    env("12")
    assert _int("TARJEM_T", 40) == 12


@pytest.mark.parametrize("value", [None, "", "  "])
def test_list_falls_back_when_blank(env, value):
    env(value)
    assert _list("TARJEM_T", "/media/movies,/media/tv") == ["/media/movies", "/media/tv"]


def test_list_splits_and_trims(env):
    env(" en , fr ,, es ")
    assert _list("TARJEM_T", "en") == ["en", "fr", "es"]


# -- LLM_EXTRA_BODY: raw JSON merged into the request, so bad input must not
#    take the service down at startup.

def test_extra_body_parses_json():
    assert Settings(llm_extra_body='{"think": false}').extra_body == {"think": False}


@pytest.mark.parametrize("raw", ["", "   ", "not json", "[1,2]", '"a string"', "null"])
def test_extra_body_ignores_anything_that_is_not_a_json_object(raw):
    assert Settings(llm_extra_body=raw).extra_body == {}
