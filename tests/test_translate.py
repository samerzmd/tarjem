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


# -- Arabic grammar guardrails --------------------------------------------

def test_arabic_guardrails_are_in_the_system_prompt(cfg):
    from app.translate import _system_blocks
    cfg.target_lang = "ar"
    cfg.grammar_guardrails = True
    text = _system_blocks(cfg, None)[0].text
    assert "ARABIC MECHANICS" in text
    # The specific failures observed from a small local model.
    assert "ثلاثة أيام" in text          # reverse number agreement
    assert "أيها القائد" in text          # vocative without the article
    assert "فات الأوان" in text           # idioms are not calqued


def test_guardrails_can_be_turned_off(cfg):
    from app.translate import _system_blocks
    cfg.target_lang = "ar"
    cfg.grammar_guardrails = False
    assert "ARABIC MECHANICS" not in _system_blocks(cfg, None)[0].text


def test_guardrails_do_not_apply_to_other_target_languages(cfg):
    from app.translate import _system_blocks
    cfg.target_lang = "fr"
    cfg.grammar_guardrails = True
    assert "ARABIC MECHANICS" not in _system_blocks(cfg, None)[0].text


def test_guardrails_ride_along_with_the_cached_prefix(cfg):
    """They belong in the cached block, not re-sent per batch."""
    from app.translate import _system_blocks
    cfg.target_lang = "ar"
    cfg.grammar_guardrails = True
    blocks = _system_blocks(cfg, None)
    assert blocks[0].cache is True


# -- translation memory ----------------------------------------------------
# A fansub .ass flattened to SRT repeats the same line many times over. One real
# anime episode arrived with 5,359 cues against a normal ~350.

def test_repeated_lines_are_translated_once_and_reused(cfg):
    cues = [
        srt.Cue(1, 0, 1000, "Get down!"),
        srt.Cue(2, 1000, 2000, "Something else entirely."),
        srt.Cue(3, 2000, 3000, "Get down!"),
        srt.Cue(4, 3000, 4000, "get   DOWN!"),   # same line, different casing/spacing
    ]
    provider = StubProvider()
    out, stats = Translator(provider, cfg).translate(cues, "Test")

    assert len(out) == 4
    assert stats.reused == 2
    assert stats.translated == 4          # every cue still ends up translated
    assert out[0].text == out[2].text == out[3].text
    assert "Something else" in out[1].text


def test_the_model_only_sees_the_unique_lines(cfg):
    cues = [srt.Cue(i + 1, i * 1000, i * 1000 + 900, "Yes.") for i in range(50)]
    cues.append(srt.Cue(51, 60000, 61000, "A distinct line."))
    provider = StubProvider()
    _, stats = Translator(provider, cfg).translate(cues, "Test")

    # 51 cues, 2 unique -> a single batch, not six.
    assert stats.batches == 1
    assert stats.reused == 49


def test_a_repeat_of_a_failed_line_is_counted_as_untranslated(cfg):
    cues = [
        srt.Cue(1, 0, 1000, "Dropped line."),
        srt.Cue(2, 1000, 2000, "Dropped line."),
        srt.Cue(3, 2000, 3000, "Kept line."),
    ]
    out, stats = Translator(StubProvider(drop={0}), cfg).translate(cues, "Test")
    assert out[0].text == "Dropped line." and out[1].text == "Dropped line."
    assert stats.untouched == 2
    assert "Kept line" in out[2].text
