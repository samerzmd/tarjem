"""SRT parsing and serialisation built to survive a round-trip through an LLM.

The goal is narrow: never let a translation pass corrupt timing. Text is the only
thing that changes; indices, timestamps and positioning metadata are carried
through untouched.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

BOM = "﻿"

_TS = r"(\d+):([0-5]\d):([0-5]\d)[,.](\d{1,3})"
TIMING_RE = re.compile(rf"^\s*{_TS}\s*-->\s*{_TS}\s*(.*?)\s*$")
BLOCK_SPLIT = re.compile(r"\n[ \t]*\n+")
INDEX_RE = re.compile(r"^\s*\d+\s*$")

# {\an8}, {\pos(..)} and friends: ASS override tags some releases smuggle into SRT.
# They can sit anywhere, including inside <font> markup that ffmpeg wrapped
# around them, but they only mean anything at the very start of a cue.
ASS_TAG_RE = re.compile(r"\{\\[^}]*\}")
# ffmpeg turns .ass styling into <font face=".." size=".." color="..">. The
# face is a font the player does not have and the size came from the script's
# own resolution, so it renders wrong even when it renders at all.
FONT_TAG_RE = re.compile(r"</?font[^>]*>", re.IGNORECASE)
# Anything inside angle brackets is one unit and must never be split across
# a line, however many spaces its attributes contain.
TOKEN_RE = re.compile(r"<[^>]*>|\{\\[^}]*\}|\S+")
ITALIC_WRAP_RE = re.compile(r"^<i>(?P<body>.*)</i>$", re.DOTALL | re.IGNORECASE)

# Hearing-impaired furniture: "[door creaks]", "(sighs)", "MARGE: ".
HI_BRACKET_RE = re.compile(r"[\[(][^\])]{0,80}[\])]")
HI_SPEAKER_RE = re.compile(r"^\s*(?:[-–—]\s*)?[A-Z][A-Z0-9 .'’\-#]{1,24}:\s+")

ARABIC_RE = re.compile(r"[؀-ۿݐ-ݿ]")

# Encodings worth trying, in order. cp1256 is the classic Windows-Arabic legacy
# encoding and shows up in subtitle packs from Arabic sites.
ENCODINGS = ("utf-8-sig", "utf-8", "utf-16", "cp1256", "cp1252", "iso-8859-1")


def decode(data: bytes) -> str:
    """Decode subtitle bytes, guessing the encoding conservatively."""
    for enc in ENCODINGS:
        try:
            text = data.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
        # utf-16 without a BOM decodes latin text into NUL-riddled garbage.
        if "\x00" in text:
            continue
        return text.lstrip(BOM)
    return data.decode("utf-8", errors="replace").lstrip(BOM)


def ts_to_ms(h: str, m: str, s: str, frac: str) -> int:
    return (int(h) * 3600 + int(m) * 60 + int(s)) * 1000 + int(frac.ljust(3, "0"))


def ms_to_ts(ms: int) -> str:
    ms = max(0, int(ms))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


@dataclass
class Cue:
    index: int
    start: int          # milliseconds
    end: int            # milliseconds
    text: str
    coords: str = ""    # trailing "X1:.. X2:.." some encoders append to the timing line

    @property
    def duration(self) -> int:
        return max(0, self.end - self.start)

    def plain(self) -> str:
        """Text with markup stripped - for glossary extraction and heuristics."""
        return re.sub(r"<[^>]+>|\{\\[^}]*\}", "", self.text)


def parse(text: str) -> list[Cue]:
    """Parse SRT text. Tolerates missing indices, CRLF, and stray blank lines."""
    text = text.lstrip(BOM).replace("\r\n", "\n").replace("\r", "\n")
    cues: list[Cue] = []
    for block in BLOCK_SPLIT.split(text):
        lines = block.split("\n")
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        if not lines:
            continue

        index: int | None = None
        if INDEX_RE.match(lines[0]) and len(lines) > 1 and TIMING_RE.match(lines[1]):
            index = int(lines[0].strip())
            lines = lines[1:]

        m = TIMING_RE.match(lines[0]) if lines else None
        if not m:
            continue  # not a cue - a stray header or a malformed block

        start = ts_to_ms(*m.group(1, 2, 3, 4))
        end = ts_to_ms(*m.group(5, 6, 7, 8))
        coords = m.group(9) or ""
        body = "\n".join(lines[1:]).strip("\n")
        cues.append(Cue(index if index is not None else len(cues) + 1, start, end, body, coords))

    return cues


def dumps(cues: list[Cue], renumber: bool = True) -> str:
    out: list[str] = []
    for i, cue in enumerate(cues, start=1):
        timing = f"{ms_to_ts(cue.start)} --> {ms_to_ts(cue.end)}"
        if cue.coords:
            timing += f" {cue.coords}"
        out.append(f"{i if renumber else cue.index}\n{timing}\n{cue.text}\n")
    return "\n".join(out)


# --------------------------------------------------------------------------
# Markup that reliably confuses translation models gets lifted off before the
# call and put back afterwards. Inline tags and dialogue dashes are deliberately
# left in place - the model handles those fine, and stripping them would lose
# the cue structure the translation needs to mirror.
# --------------------------------------------------------------------------

@dataclass
class Skin:
    ass_prefix: str = ""
    italic: bool = False
    source_lines: int = 1


def undress(text: str) -> tuple[str, Skin]:
    skin = Skin(source_lines=max(1, len([ln for ln in text.split("\n") if ln.strip()])))

    # Lift every ASS override tag out, wherever it sits. ffmpeg often leaves
    # them inside the <font> wrapper it generated, and a model asked to
    # "preserve markup" faithfully keeps {\an8} in the middle of the line -
    # where it positions nothing and simply shows up as literal text.
    found = ASS_TAG_RE.findall(text)
    if found:
        skin.ass_prefix = "".join(dict.fromkeys(found))
        text = ASS_TAG_RE.sub("", text)

    stripped = text.strip()
    m = ITALIC_WRAP_RE.match(stripped)
    if m and "<i>" not in m.group("body"):
        skin.italic = True
        text = m.group("body")
    return text.strip(), skin


def strip_style_tags(text: str) -> str:
    """Drop <font> wrappers, keeping <i>/<b>/<u> which mean the same everywhere."""
    return FONT_TAG_RE.sub("", text)


def dress(text: str, skin: Skin, max_line: int = 42, max_lines: int = 2) -> str:
    text = wrap(text, max_line=max_line, max_lines=max_lines)
    if skin.italic:
        text = f"<i>{text}</i>"
    return skin.ass_prefix + text


def wrap(text: str, max_line: int = 42, max_lines: int = 2) -> str:
    """Re-flow a translated cue so no line runs past the safe caption width.

    Breaks the model already made are respected when they fit; only over-long
    lines are split, and never past ``max_lines`` total.
    """
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if not lines:
        return text.strip()

    # Dialogue dashes mark speaker turns - those breaks are meaningful, keep them.
    if len(lines) > 1 and any(ln.startswith(("-", "–", "—")) for ln in lines):
        return "\n".join(lines)

    if all(len(ln) <= max_line for ln in lines):
        return "\n".join(lines[:max_lines]) if len(lines) > max_lines else "\n".join(lines)

    # Re-flow from scratch: the model's break points are not worth preserving
    # once a line has overrun anyway.
    joined = " ".join(lines)
    parts = max(1, min(max_lines, -(-len(joined) // max_line)))
    return "\n".join(_split_line(joined, max_line, parts))


def _split_line(line: str, max_line: int, max_parts: int) -> list[str]:
    if len(line) <= max_line or max_parts <= 1:
        return [line]
    # Split on token boundaries, not on every space: `<font size="20"
    # color="#ff0">` contains spaces but breaking it in half corrupts the tag.
    words = TOKEN_RE.findall(line)
    if len(words) < 2:
        return [line]
    # Balance the halves rather than greedily filling the first line: a 40/4
    # split reads far worse than 22/22 at the same total width.
    target = len(line) / max_parts
    best_at, best_cost = 1, None
    running = 0
    for i in range(1, len(words)):
        running += len(words[i - 1]) + 1
        cost = abs(running - target)
        if best_cost is None or cost < best_cost:
            best_at, best_cost = i, cost
    head = " ".join(words[:best_at])
    tail = " ".join(words[best_at:])
    return [head] + _split_line(tail, max_line, max_parts - 1)


def _norm(text: str) -> str:
    return " ".join(re.sub(r"<[^>]+>|\{\\[^}]*\}", "", text).split()).casefold()


def overlapping_duplicates(cues: list[Cue], window: int = 60) -> int:
    """Count cues that repeat a nearby cue's text *while it is still on screen*.

    Sequential repeats are ordinary - a character says "No!" twice. Repeats that
    overlap are a rendering artifact, usually an .ass with several styled layers
    flattened into SRT, and they stack on top of each other on playback.
    """
    hits = 0
    for i, cue in enumerate(cues):
        key = _norm(cue.text)
        if not key:
            continue
        for prev in cues[max(0, i - window):i]:
            if prev.end >= cue.start and cue.end >= prev.start and _norm(prev.text) == key:
                hits += 1
                break
    return hits


def collapse_duplicates(cues: list[Cue], window: int = 60) -> tuple[list[Cue], int]:
    """Merge identical cues that overlap into a single cue spanning both.

    Only overlapping repeats are merged; a line genuinely said twice, seconds
    apart, is left alone.
    """
    out: list[Cue] = []
    for cue in cues:
        key = _norm(cue.text)
        merged = False
        if key:
            for prev in out[-window:]:
                if prev.end >= cue.start and cue.end >= prev.start and _norm(prev.text) == key:
                    prev.end = max(prev.end, cue.end)
                    prev.start = min(prev.start, cue.start)
                    merged = True
                    break
        if not merged:
            out.append(Cue(cue.index, cue.start, cue.end, cue.text, cue.coords))
    return out, len(cues) - len(out)


def strip_hi(text: str) -> str:
    """Drop hearing-impaired annotations. Returns '' if nothing survives."""
    out = HI_BRACKET_RE.sub("", text)
    out = "\n".join(HI_SPEAKER_RE.sub("", ln) for ln in out.split("\n"))
    out = re.sub(r"[ \t]{2,}", " ", out)
    return "\n".join(ln.strip() for ln in out.split("\n") if ln.strip())


def looks_arabic(cues: list[Cue], threshold: float = 0.3) -> bool:
    sample = [c.plain() for c in cues[:200] if c.plain().strip()]
    if not sample:
        return False
    hits = sum(1 for s in sample if ARABIC_RE.search(s))
    return hits / len(sample) >= threshold
