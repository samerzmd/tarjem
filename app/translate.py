"""The translation engine.

The design goal is a file that is *whole*: same number of cues, same timings,
consistent names and register from the first line to the last. Three things get
us there and none of them survive a copy-paste-into-a-chat-window workflow:

1. **A glossary pass.** Before translating, the model reads a stratified sample
   of the whole file and fixes the Arabic rendering of every recurring name,
   place and coined term. That glossary is pinned into the cached system prefix
   for every batch, and is reused across episodes of the same series.
2. **A rolling context window.** Each batch sees the last few source cues *and*
   their accepted translations, so pronouns, honorifics and running jokes carry
   across the batch boundary.
3. **Structured output with id round-tripping.** The model returns ids, not a
   blob. A mismatch is detected, repaired, and - if repair fails - narrowed to
   the individual cue, so one bad batch can never shift an entire file.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Callable, Iterable

from pydantic import BaseModel, Field

from . import srt
from .config import Settings
from .providers import Provider, ProviderError, SystemBlock

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Structured output schemas
# --------------------------------------------------------------------------

class TranslatedCue(BaseModel):
    id: int = Field(description="The id of the source cue this translates.")
    ar: str = Field(description="The Arabic subtitle text. May contain a newline.")


class BatchTranslation(BaseModel):
    cues: list[TranslatedCue]


class GlossaryTerm(BaseModel):
    source: str = Field(description="The term as it appears in the source subtitle.")
    arabic: str = Field(description="The Arabic rendering to use everywhere in this title.")
    kind: str = Field(description="One of: character, place, organisation, term, other.")
    note: str = Field(description="Short reason or usage note. May be empty.")


class TitleGlossary(BaseModel):
    summary: str = Field(description="Two sentences on the setting and genre.")
    tone: str = Field(description="How the dialogue should feel in Arabic.")
    address: str = Field(description="Formality between the main characters.")
    terms: list[GlossaryTerm]


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------

RULES = """\
You are a senior subtitle translator. You produce Arabic subtitles of the quality \
a streaming service would ship: fluent, idiomatic, and invisible to the viewer.

Target register: {register}

Rules, in priority order:

1. STRUCTURE IS SACRED. Exactly one output entry per input cue, with the same id. \
Never merge two cues, never split one across two, never drop or reorder. If a \
sentence spans several cues, translate it across those same cues so each one \
still reads naturally on its own.

2. TRANSLATE MEANING, NOT WORDS. Render idioms with the closest natural Arabic \
idiom. A word-for-word rendering that reads like machine output is a failure, \
even when every word is technically correct.

3. HOLD THE VOICE. A soldier barking an order, a child pleading, a lawyer \
hedging - each should sound like that person in Arabic. Keep contractions and \
interruptions colloquial; keep formal speech formal.

4. LENGTH DISCIPLINE. A subtitle is read in a couple of seconds. Prefer the \
shorter phrasing. Aim to stay at or under {max_line} characters per line and \
{max_lines} lines. Each cue carries its on-screen duration in milliseconds - \
tighten hard when it is short.

5. PRESERVE MARKUP EXACTLY. Keep <i>...</i> and any other inline tags around the \
same words they wrapped. Keep a leading "- " on a line when the source has one \
(it marks a second speaker). Keep music notes. Keep line breaks where they \
divide two speakers.

6. NAMES AND TERMS. Follow the glossary without exception. For anything not in \
it, use the established Arabic form of a well-known name or place; transliterate \
only when there is no established form, and then stay consistent.

7. NUMERALS. Use Western Arabic digits (0 1 2 3), which is what Arabic subtitles \
use. Convert units only when the source itself is casual about them.

8. REGISTER OF PROFANITY AND SLANG. Carry the force of the original at a level an \
Arabic broadcaster would ship: do not sanitise an insult into politeness, and do \
not escalate.

9. OUTPUT ONLY THE TRANSLATION. No notes, no romanisation, no quotation marks \
you did not find in the source, no explanations, no alternatives.

10. LEAVE ALONE what should not change: URLs, on-screen text already in Arabic, \
bare numbers, and codes. Return them unchanged rather than inventing a translation.
"""

# Smaller models get the structure right and then fumble the grammar. These are
# the mistakes they actually make, so they are worth naming with examples rather
# than trusting the model to remember its own rules. Prompt tokens are processed
# several times faster than they are generated, so this is cheap insurance.
ARABIC_MECHANICS = """\
ARABIC MECHANICS

Below is a table of mistakes that are made constantly in machine-translated
Arabic subtitles. The left column is always wrong. Never emit anything in it.

  WRONG                     CORRECT                  WHY
  ثلاث أيام                  ثلاثة أيام                numerals 3-10 reverse gender
  ثلاثة سنوات                ثلاث سنوات                same rule, other direction
  خمس رجال                   خمسة رجال                 same rule
  تلك الشتاء                 ذلك الشتاء                الشتاء is masculine
  هذه الصيف                  هذا الصيف                 الصيف is masculine
  يا القائد                  أيها القائد               يا never takes الـ
  يا الطبيبة                 أيتها الطبيبة             feminine form
  لم تعد تعود                لم تعد                    لم already carries the past
  لم يذهب ذهب                لم يذهب                   one verb, not two
  لم تعود                    لم تعد                    لم takes the jussive
  لم يذهبون                  لم يذهبوا                 same rule, plural
  السفينة غادرت              فات الأوان                "that ship has sailed"
  اذهب! (for "get down")     انبطح! / انخفض!           it means take cover
  راقبته (for "chasing")     طاردته / لاحقته           pursuit, not surveillance
  أمسك (for "hold on")       انتظر / اصمد              unless gripping something
  إنه على لي                على حسابي                 "it's on me"

Two habits behind most of these:

* An idiom is translated by MEANING, then rendered in the Arabic a speaker
  would actually use. A word-for-word idiom is a mistranslation even when
  every individual word is correct.
* A verb is chosen for what physically happens in the scene, not for the
  dictionary gloss of the English word.

يا is correct only before a name or an indefinite noun: يا أحمد, يا رجل.
"""

GLOSSARY_PROMPT = """\
Below is a representative sample of cues drawn from across one subtitle file\
{title_line}.

Read it as a whole and build the translation brief for this title:

- `summary`: two sentences on the setting and genre.
- `tone`: how the Arabic dialogue should feel.
- `address`: the formality between the main characters, and whether they would \
use the second person singular informally.
- `terms`: every recurring proper noun and coined term - characters, places, \
organisations, ranks, invented technology, running gags - with the exact Arabic \
rendering to use for the rest of the file. Include a term only if it recurs or \
would be easy to render inconsistently. Aim for the 10-40 that actually matter, \
not every noun.

Sample:
{sample}
"""

BATCH_PROMPT = """\
{title_block}{context_block}Translate every cue below into Arabic.

Return one entry per cue, carrying the same `id`. `ms` is how long the cue is on \
screen - use it to judge how much text will fit. Do not translate anything from \
the context section above.

{payload}
"""


def _system_blocks(cfg: Settings, glossary: TitleGlossary | None) -> list[SystemBlock]:
    rules = RULES.format(
        register=cfg.register_desc, max_line=cfg.max_line_chars, max_lines=cfg.max_lines
    )
    if cfg.target_lang == "ar" and cfg.grammar_guardrails:
        rules = f"{rules}\n{ARABIC_MECHANICS}"
    blocks = [SystemBlock(rules)]
    if glossary:
        blocks.append(SystemBlock(render_glossary(glossary), cache=True))
    else:
        blocks[0] = SystemBlock(rules, cache=True)
    return blocks


def render_glossary(g: TitleGlossary) -> str:
    lines = [
        "TRANSLATION BRIEF FOR THIS TITLE",
        "",
        f"Setting: {g.summary}",
        f"Tone: {g.tone}",
        f"Address between characters: {g.address}",
    ]
    if g.terms:
        lines += ["", "Fixed renderings (use these exactly, every time):"]
        for t in g.terms:
            note = f"  - {t.note}" if t.note else ""
            lines.append(f"  {t.source} -> {t.arabic}  [{t.kind}]{note}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------

@dataclass
class TranslationStats:
    total: int = 0
    translated: int = 0
    untouched: int = 0        # cues we could not translate and passed through
    batches: int = 0
    repairs: int = 0
    seconds: float = 0.0
    glossary_terms: int = 0

    def as_dict(self) -> dict:
        return {
            "total_cues": self.total,
            "translated": self.translated,
            "untranslated": self.untouched,
            "batches": self.batches,
            "repairs": self.repairs,
            "glossary_terms": self.glossary_terms,
            "seconds": round(self.seconds, 1),
        }


ProgressFn = Callable[[float, str], None]


class Translator:
    def __init__(self, provider: Provider, cfg: Settings):
        self.provider = provider
        self.cfg = cfg

    # -- glossary --------------------------------------------------------

    def build_glossary(self, cues: list[srt.Cue], title: str = "") -> TitleGlossary | None:
        if not self.cfg.glossary_enabled:
            return None
        sample = _stratified_sample(cues, self.cfg.glossary_sample)
        if len(sample) < 10:
            return None
        body = "\n".join(c.plain().replace("\n", " ") for c in sample)
        prompt = GLOSSARY_PROMPT.format(
            title_line=f" for: {title}" if title else "",
            sample=body,
        )
        try:
            glossary = self._call(
                [SystemBlock(
                    "You prepare translation briefs for subtitle localisation into Arabic. "
                    "You are precise about names and consistent about register.",
                    cache=False,
                )],
                prompt,
                TitleGlossary,
                max_tokens=8000,
            )
        except ProviderError as exc:
            log.warning("glossary pass failed, continuing without one: %s", exc)
            return None
        log.info("glossary: %d terms", len(glossary.terms))
        return glossary

    # -- main pass -------------------------------------------------------

    def translate(
        self,
        cues: list[srt.Cue],
        title: str = "",
        glossary: TitleGlossary | None = None,
        progress: ProgressFn | None = None,
    ) -> tuple[list[srt.Cue], TranslationStats]:
        started = time.monotonic()
        stats = TranslationStats(total=len(cues))
        if glossary:
            stats.glossary_terms = len(glossary.terms)

        translatable, passthrough = _partition(cues, self.cfg)
        system = _system_blocks(self.cfg, glossary)
        title_block = f"<title>{title}</title>\n\n" if title else ""

        results: dict[int, str] = {}
        history: list[tuple[str, str]] = []   # (source, arabic) accepted so far
        batches = list(_chunks(translatable, self.cfg.batch_size))

        for n, batch in enumerate(batches, start=1):
            context = _render_context(history, self.cfg.context_cues)
            got = self._translate_batch(system, title_block, context, batch, stats)
            for idx, cue in batch:
                text = got.get(idx)
                if text:
                    results[idx] = text
                    history.append((cue.plain().replace("\n", " "), text.replace("\n", " ")))
                else:
                    stats.untouched += 1
            history = history[-self.cfg.context_cues * 2:]
            stats.batches += 1
            if progress:
                progress(n / max(1, len(batches)), f"batch {n}/{len(batches)}")

        out: list[srt.Cue] = []
        for i, cue in enumerate(cues):
            if i in passthrough:
                out.append(cue)
                continue
            translated = results.get(i)
            if translated is None:
                out.append(cue)
                continue
            _, skin = srt.undress(cue.text)
            dressed = srt.dress(
                translated, skin, max_line=self.cfg.max_line_chars, max_lines=self.cfg.max_lines
            )
            out.append(srt.Cue(cue.index, cue.start, cue.end, dressed, cue.coords))
            stats.translated += 1

        stats.seconds = time.monotonic() - started
        return out, stats

    # -- one batch, with repair -------------------------------------------

    def _translate_batch(
        self,
        system: list[SystemBlock],
        title_block: str,
        context: str,
        batch: list[tuple[int, srt.Cue]],
        stats: TranslationStats,
        depth: int = 0,
    ) -> dict[int, str]:
        payload = _render_payload(batch, self.cfg)
        prompt = BATCH_PROMPT.format(
            title_block=title_block, context_block=context, payload=payload
        )
        wanted = {idx for idx, _ in batch}

        try:
            result = self._call(system, prompt, BatchTranslation, max_tokens=16000)
        except ProviderError as exc:
            if not exc.retryable or depth >= 2 or len(batch) == 1:
                log.error("batch failed permanently (%d cues): %s", len(batch), exc)
                return {}
            log.warning("batch failed (%s) - splitting %d cues", exc, len(batch))
            return self._split_retry(system, title_block, context, batch, stats, depth)

        got = {c.id: c.ar.strip() for c in result.cues if c.id in wanted and c.ar.strip()}
        missing = wanted - set(got)
        if not missing:
            return got

        stats.repairs += 1
        log.warning("batch returned %d/%d cues, repairing", len(got), len(wanted))
        if depth >= 2:
            return got
        leftover = [(i, c) for i, c in batch if i in missing]
        got.update(self._split_retry(system, title_block, context, leftover, stats, depth))
        return got

    def _split_retry(
        self,
        system: list[SystemBlock],
        title_block: str,
        context: str,
        batch: list[tuple[int, srt.Cue]],
        stats: TranslationStats,
        depth: int,
    ) -> dict[int, str]:
        if len(batch) <= 1:
            return self._translate_batch(system, title_block, context, batch, stats, depth + 1)
        mid = len(batch) // 2
        out = self._translate_batch(system, title_block, context, batch[:mid], stats, depth + 1)
        out.update(self._translate_batch(system, title_block, context, batch[mid:], stats, depth + 1))
        return out

    # -- transport with backoff -------------------------------------------

    def _call(self, system: list[SystemBlock], user: str, schema, max_tokens: int):
        last: ProviderError | None = None
        for attempt in range(self.cfg.max_retries):
            try:
                return self.provider.structured(system, user, schema, max_tokens=max_tokens)
            except ProviderError as exc:
                last = exc
                if not exc.retryable:
                    raise
                delay = exc.retry_after if exc.retry_after else min(60, 2 ** attempt * 5)
                log.warning("provider error (attempt %d): %s - retrying in %.0fs",
                            attempt + 1, exc, delay)
                time.sleep(delay)
        raise last or ProviderError("exhausted retries")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _partition(cues: list[srt.Cue], cfg: Settings) -> tuple[list[tuple[int, srt.Cue]], set[int]]:
    """Split cues into ones worth sending and ones to copy through verbatim."""
    send: list[tuple[int, srt.Cue]] = []
    skip: set[int] = set()
    for i, cue in enumerate(cues):
        body, _ = srt.undress(cue.text)
        if cfg.strip_hi:
            body = srt.strip_hi(body)
        if not body.strip() or not any(ch.isalpha() for ch in body):
            skip.add(i)
            continue
        send.append((i, srt.Cue(cue.index, cue.start, cue.end, body, cue.coords)))
    return send, skip


def _chunks(items: list, size: int) -> Iterable[list]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _stratified_sample(cues: list[srt.Cue], n: int) -> list[srt.Cue]:
    """Sample evenly across the file - names introduced late matter too."""
    meaty = [c for c in cues if len(c.plain().strip()) > 3]
    if len(meaty) <= n:
        return meaty
    step = len(meaty) / n
    return [meaty[int(i * step)] for i in range(n)]


def _render_payload(batch: list[tuple[int, srt.Cue]], cfg: Settings) -> str:
    rows = [
        {"id": idx, "ms": cue.duration, "text": cue.text}
        for idx, cue in batch
    ]
    return json.dumps(rows, ensure_ascii=False, indent=1)


def _render_context(history: list[tuple[str, str]], n: int) -> str:
    if not history or n <= 0:
        return ""
    tail = history[-n:]
    lines = "\n".join(f"  {src}\n  -> {ar}" for src, ar in tail)
    return (
        "<already_translated>\n"
        "The cues immediately before this batch, for continuity only.\n"
        f"{lines}\n"
        "</already_translated>\n\n"
    )
