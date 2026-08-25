"""Finding something to translate from.

Bazarr failing to find Arabic does not mean there is nothing to work with. In
order of preference:

1. A sidecar subtitle next to the video in one of the source languages.
2. A text subtitle track muxed into the container, extracted with ffmpeg.
3. Whisper transcription, if a subgen/whisper-asr endpoint is configured.
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import httpx

from . import srt
from .config import Settings

log = logging.getLogger(__name__)

SUB_EXTS = (".srt", ".ass", ".ssa", ".vtt", ".sub")
# Bitmap subtitle codecs need OCR, which is out of scope - skip them.
TEXT_CODECS = {"subrip", "srt", "ass", "ssa", "mov_text", "webvtt", "text", "eia_608", "subviewer"}

LANG_ALIASES = {
    "en": {"en", "eng", "english"},
    "fr": {"fr", "fre", "fra", "french"},
    "es": {"es", "spa", "esp", "spanish"},
    "de": {"de", "ger", "deu", "german"},
    "it": {"it", "ita", "italian"},
    "pt": {"pt", "por", "portuguese"},
    "nl": {"nl", "dut", "nld", "dutch"},
    "tr": {"tr", "tur", "turkish"},
    "ar": {"ar", "ara", "arabic"},
}


def aliases(code: str) -> set[str]:
    return LANG_ALIASES.get(code.lower(), {code.lower()})


@dataclass
class SubtitleSource:
    text: str
    origin: str        # "sidecar" | "embedded" | "whisper"
    detail: str        # path or stream description
    lang: str


def output_path(video: Path, cfg: Settings) -> Path:
    return video.with_suffix("").with_name(video.stem + cfg.output_suffix)


def has_target(video: Path, cfg: Settings) -> Path | None:
    """Return an existing Arabic sidecar for this video, if any."""
    stem = video.stem
    for candidate in video.parent.glob(f"{glob_escape(stem)}*"):
        if candidate.suffix.lower() not in SUB_EXTS:
            continue
        tail = candidate.name[len(stem):].lower()
        if any(f".{a}." in tail or tail.startswith(f".{a}.") for a in aliases(cfg.target_lang)):
            return candidate
    return None


def glob_escape(text: str) -> str:
    return re.sub(r"([\[\]*?])", r"[\1]", text)


def find_source(video: Path, cfg: Settings, subtitle_hint: str | None = None) -> SubtitleSource | None:
    if subtitle_hint:
        hinted = Path(cfg.to_local(subtitle_hint))
        if hinted.is_file():
            text = _load(hinted, cfg)
            if text:
                return SubtitleSource(text, "sidecar", str(hinted), _lang_of(hinted.name) or "??")

    found = _find_sidecar(video, cfg)
    if found:
        return found
    found = _extract_embedded(video, cfg)
    if found:
        return found
    return _whisper(video, cfg)


# -- sidecars ---------------------------------------------------------------

def _find_sidecar(video: Path, cfg: Settings) -> SubtitleSource | None:
    stem = video.stem
    candidates: list[tuple[int, Path, str]] = []
    for path in video.parent.glob(f"{glob_escape(stem)}*"):
        if path.suffix.lower() not in SUB_EXTS or not path.is_file():
            continue
        tail = path.name[len(stem):].lower()
        lang = _lang_of(path.name)
        if lang is None:
            # A bare "Movie.srt" - only usable if it is not already the target.
            if tail.strip(".") not in ("srt", "ass", "ssa", "vtt"):
                continue
            lang = ""
        if lang and lang in aliases(cfg.target_lang):
            continue
        rank = cfg.source_langs.index(lang) if lang in cfg.source_langs else len(cfg.source_langs)
        # A forced track is signage only - a few dozen cues. Last resort.
        if "forced" in tail:
            rank += 100
        candidates.append((rank, path, lang))

    for _, path, lang in sorted(candidates, key=lambda c: (c[0], len(c[1].name))):
        text = _load(path, cfg)
        if not text:
            continue
        cues = srt.parse(text)
        if len(cues) < 5:
            log.debug("skipping %s: only %d cues", path.name, len(cues))
            continue
        if srt.looks_arabic(cues):
            continue
        return SubtitleSource(text, "sidecar", str(path), lang or "??")
    return None


def _lang_of(name: str) -> str | None:
    parts = name.lower().split(".")
    for part in reversed(parts[:-1]):
        for code, names in LANG_ALIASES.items():
            if part in names:
                return code
    return None


def _load(path: Path, cfg: Settings) -> str | None:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        log.warning("cannot read %s: %s", path, exc)
        return None
    if path.suffix.lower() in (".ass", ".ssa", ".vtt", ".sub"):
        return _convert(path, cfg)
    return srt.decode(raw)


def _convert(path: Path, cfg: Settings) -> str | None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "converted.srt"
        code = _run([cfg.ffmpeg, "-y", "-loglevel", "error", "-i", str(path), str(out)])
        if code != 0 or not out.exists():
            return None
        return srt.decode(out.read_bytes())


# -- embedded tracks --------------------------------------------------------

def _extract_embedded(video: Path, cfg: Settings) -> SubtitleSource | None:
    streams = _probe(video, cfg)
    if not streams:
        return None

    ranked: list[tuple[int, int, dict]] = []
    for i, stream in enumerate(streams):
        if (stream.get("codec_name") or "").lower() not in TEXT_CODECS:
            continue
        tags = {k.lower(): str(v).lower() for k, v in (stream.get("tags") or {}).items()}
        lang = tags.get("language", "")
        code = next((c for c, names in LANG_ALIASES.items() if lang in names), "")
        if code and code in aliases(cfg.target_lang):
            continue
        rank = cfg.source_langs.index(code) if code in cfg.source_langs else len(cfg.source_langs)
        disposition = stream.get("disposition") or {}
        if disposition.get("forced") or "forced" in tags.get("title", ""):
            rank += 100
        ranked.append((rank, i, stream))

    for _, i, stream in sorted(ranked, key=lambda r: (r[0], r[1])):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "embedded.srt"
            code = _run([
                cfg.ffmpeg, "-y", "-loglevel", "error",
                "-i", str(video), "-map", f"0:s:{i}", "-c:s", "srt", str(out),
            ])
            if code != 0 or not out.exists() or out.stat().st_size < 200:
                continue
            text = srt.decode(out.read_bytes())
            cues = srt.parse(text)
            if len(cues) < 5 or srt.looks_arabic(cues):
                continue
            tags = stream.get("tags") or {}
            lang = str(tags.get("language", "??"))
            log.info("extracted embedded track %d (%s) from %s", i, lang, video.name)
            return SubtitleSource(text, "embedded", f"stream 0:s:{i} ({lang})", lang)
    return None


def _probe(video: Path, cfg: Settings) -> list[dict]:
    try:
        proc = subprocess.run(
            [cfg.ffprobe, "-v", "quiet", "-print_format", "json",
             "-show_streams", "-select_streams", "s", str(video)],
            capture_output=True, timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.warning("ffprobe failed on %s: %s", video, exc)
        return []
    if proc.returncode != 0:
        return []
    try:
        return json.loads(proc.stdout or b"{}").get("streams", [])
    except json.JSONDecodeError:
        return []


def _run(cmd: list[str], timeout: int = 900) -> int:
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.warning("command failed: %s (%s)", " ".join(cmd[:3]), exc)
        return 1
    if proc.returncode != 0:
        log.debug("ffmpeg stderr: %s", (proc.stderr or b"")[-400:].decode("utf-8", "replace"))
    return proc.returncode


# -- whisper fallback -------------------------------------------------------

def _whisper(video: Path, cfg: Settings) -> SubtitleSource | None:
    if not cfg.whisper_url:
        return None
    log.info("no text subtitle found for %s - trying whisper", video.name)
    try:
        with httpx.Client(timeout=httpx.Timeout(30.0, read=3600.0)) as client:
            with video.open("rb") as fh:
                response = client.post(
                    f"{cfg.whisper_url}/asr",
                    params={"task": "transcribe", "output": "srt"},
                    files={"audio_file": (video.name, fh, "application/octet-stream")},
                )
        if response.status_code >= 400:
            log.warning("whisper returned %s", response.status_code)
            return None
        text = response.text
    except httpx.RequestError as exc:
        log.warning("whisper unreachable: %s", exc)
        return None

    if len(srt.parse(text)) < 5:
        return None
    return SubtitleSource(text, "whisper", cfg.whisper_url, "??")
