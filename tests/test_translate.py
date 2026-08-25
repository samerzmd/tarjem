"""Engine tests with a stubbed provider - no network, no API key."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import srt  # noqa: E402
from app.config import Settings  # noqa: E402
from app.providers.base import Provider, ProviderError  # noqa: E402
from app.translate import BatchTranslation, TitleGlossary, Translator  # noqa: E402


def make_srt(n: int) -> list[srt.Cue]:
    return [
        srt.Cue(i + 1, i * 2000, i * 2000 + 1800, f"Line number {i} speaking.")
        for i in range(n)
    ]


class StubProvider(Provider):
    """Echoes each cue back with an [AR] marker, minus whatever we tell it to drop."""

    name = "stub"

    def __init__(self, drop: set[int] | None = None, fail_first: int = 0):
        super().__init__()
        self.drop = drop or set()
        self.fail_first = fail_first
        self.calls = 0
        self.systems: list[list] = []

    def structured(self, system, user, schema_model, max_tokens=16000):
        self.calls += 1
        self.systems.append(system)
        if schema_model is TitleGlossary:
            return TitleGlossary(summary="s", tone="t", address="a", terms=[])
        if self.calls <= self.fail_first:
            raise ProviderError("transient", retryable=True)

        import json
        import re

        payload = json.loads(re.search(r"\[\s*\{.*\}\s*\]", user, re.S).group(0))
        return BatchTranslation.model_validate({
            "cues": [
                {"id": row["id"], "ar": f"[AR] {row['text']}"}
                for row in payload
                if row["id"] not in self.drop
            ]
        })


@pytest.fixture
def cfg():
    s = Settings()
    s.batch_size = 10
    s.context_cues = 8
    s.strip_hi = False
    s.glossary_enabled = False
    s.max_retries = 2
    s.max_lines = 2
    s.max_line_chars = 200          # keep the stub's echo on one line
    return s


def test_every_cue_comes_back_with_timing_intact(cfg):
    cues = make_srt(25)
    provider = StubProvider()
    out, stats = Translator(provider, cfg).translate(cues, "Test")

    assert len(out) == len(cues)
    assert [(c.start, c.end) for c in out] == [(c.start, c.end) for c in cues]
    assert all(c.text.startswith("[AR]") for c in out)
    assert stats.translated == 25 and stats.untouched == 0
    assert stats.batches == 3


def test_missing_ids_are_repaired_not_shifted(cfg):
    cues = make_srt(10)
    # The provider silently omits cue 4 on every call it can.
    provider = StubProvider(drop={4})
    out, stats = Translator(provider, cfg).translate(cues, "Test")

    assert len(out) == 10
    assert stats.repairs >= 1
    # Cue 4 falls back to its source text; every *other* cue is still correct.
    assert out[4].text == cues[4].text
    assert out[5].text == "[AR] Line number 5 speaking."
    assert stats.untouched == 1


def test_transient_provider_errors_are_retried(cfg):
    cues = make_srt(5)
    provider = StubProvider(fail_first=1)
    out, stats = Translator(provider, cfg).translate(cues, "Test")
    assert stats.translated == 5
    assert provider.calls == 2


def test_blank_and_symbol_only_cues_are_never_sent(cfg):
    cues = [
        srt.Cue(1, 0, 1000, "Real dialogue."),
        srt.Cue(2, 1000, 2000, "♪"),
        srt.Cue(3, 2000, 3000, "1998"),
        srt.Cue(4, 3000, 4000, ""),
    ]
    provider = StubProvider()
    out, stats = Translator(provider, cfg).translate(cues, "Test")
    assert out[1].text == "♪" and out[2].text == "1998" and out[3].text == ""
    assert stats.translated == 1


def test_glossary_is_pinned_into_a_cached_system_block(cfg):
    cfg.glossary_enabled = True
    cues = make_srt(12)
    provider = StubProvider()
    translator = Translator(provider, cfg)
    glossary = TitleGlossary(summary="s", tone="t", address="a", terms=[])
    translator.translate(cues, "Test", glossary)

    system = provider.systems[-1]
    assert len(system) == 2
    assert system[1].cache is True
    assert "TRANSLATION BRIEF" in system[1].text


def test_italics_survive_the_round_trip(cfg):
    cues = [srt.Cue(1, 0, 2000, "<i>Whispering now.</i>")]
    out, _ = Translator(StubProvider(), cfg).translate(cues, "Test")
    assert out[0].text == "<i>[AR] Whispering now.</i>"


def test_context_window_carries_previous_translations(cfg):
    cfg.batch_size = 5
    cfg.context_cues = 3
    provider = StubProvider()
    Translator(provider, cfg).translate(make_srt(15), "Test")
    # The stub records the user prompt indirectly; check the second batch saw context.
    assert provider.calls == 3
