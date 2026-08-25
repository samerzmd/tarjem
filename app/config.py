"""Environment-driven settings. Everything has a working default except the API key."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


# Docker Compose substitutes an *empty string* for a variable that isn't set in
# .env, so `os.getenv(name, default)` would hand back "" and silently override
# every default. Treat empty as absent throughout.

def _str(name: str, default: str = "") -> str:
    return (os.getenv(name) or "").strip() or default


def _bool(name: str, default: bool = False) -> bool:
    raw = _str(name)
    if not raw:
        return default
    return raw.lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    try:
        return int(_str(name) or default)
    except ValueError:
        return default


def _list(name: str, default: str) -> list[str]:
    return [x.strip() for x in _str(name, default).split(",") if x.strip()]


REGISTERS = {
    "msa": (
        "Modern Standard Arabic (فصحى) as used by professional streaming subtitles - "
        "clear, contemporary, and readable. Not classical or archaic."
    ),
    "gulf": "Gulf/Khaleeji colloquial Arabic, natural for a Saudi audience.",
    "egyptian": "Egyptian colloquial Arabic, the dialect of mainstream Arabic dubbing.",
    "levantine": "Levantine colloquial Arabic (Shami).",
    "msa-light": (
        "Modern Standard Arabic with a light, conversational touch - fluent MSA "
        "that does not sound like a news broadcast."
    ),
}


@dataclass
class Settings:
    # --- what we produce -------------------------------------------------
    target_lang: str = _str("TARGET_LANG", "ar")
    source_langs: list[str] = field(default_factory=lambda: _list("SOURCE_LANGS", "en,fr,es,de"))
    register: str = _str("ARABIC_REGISTER", "msa")
    output_suffix: str = _str("OUTPUT_SUFFIX", ".ar.srt")
    # Marks machine output so a later human/Bazarr pass can tell them apart.
    tag_output: bool = _bool("TAG_OUTPUT", True)
    strip_hi: bool = _bool("STRIP_HI", False)
    max_line_chars: int = _int("MAX_LINE_CHARS", 42)
    max_lines: int = _int("MAX_LINES", 2)

    # --- model -----------------------------------------------------------
    provider: str = _str("LLM_PROVIDER", "anthropic")
    model: str = _str("LLM_MODEL", "claude-opus-5")
    effort: str = _str("LLM_EFFORT", "low")
    batch_size: int = _int("BATCH_SIZE", 40)
    context_cues: int = _int("CONTEXT_CUES", 8)
    max_retries: int = _int("MAX_RETRIES", 3)
    glossary_enabled: bool = _bool("GLOSSARY_ENABLED", True)
    glossary_sample: int = _int("GLOSSARY_SAMPLE", 180)

    anthropic_api_key: str = _str("ANTHROPIC_API_KEY")
    openai_api_key: str = _str("OPENAI_API_KEY")
    openai_base_url: str = _str("OPENAI_BASE_URL", "https://api.openai.com/v1")
    openai_model: str = _str("OPENAI_MODEL", "gpt-4.1")

    # --- where the media lives ------------------------------------------
    media_roots: list[str] = field(default_factory=lambda: _list("MEDIA_ROOTS", "/media/movies,/media/tv"))
    video_exts: list[str] = field(default_factory=lambda: _list("VIDEO_EXTS", ".mkv,.mp4,.avi,.m4v,.mov,.ts,.webm"))
    # "bazarr_path:local_path" pairs, for when the two containers mount media differently.
    path_map: list[str] = field(default_factory=lambda: _list("PATH_MAP", ""))

    # --- integrations ----------------------------------------------------
    bazarr_url: str = _str("BAZARR_URL", "http://bazarr:6767").rstrip("/")
    bazarr_api_key: str = _str("BAZARR_API_KEY")
    notify_bazarr: bool = _bool("NOTIFY_BAZARR", True)
    whisper_url: str = _str("WHISPER_URL").rstrip("/")
    ffmpeg: str = _str("FFMPEG_BIN", "ffmpeg")
    ffprobe: str = _str("FFPROBE_BIN", "ffprobe")

    # --- runtime ---------------------------------------------------------
    data_dir: Path = Path(_str("DATA_DIR", "/data"))
    workers: int = _int("WORKERS", 1)
    sweep_enabled: bool = _bool("SWEEP_ENABLED", True)
    sweep_interval_min: int = _int("SWEEP_INTERVAL_MIN", 360)
    sweep_source: str = _str("SWEEP_SOURCE", "bazarr")  # bazarr | disk
    sweep_limit: int = _int("SWEEP_LIMIT", 50)
    dry_run: bool = _bool("DRY_RUN", False)
    api_token: str = _str("API_TOKEN")
    log_level: str = _str("LOG_LEVEL", "INFO").upper()

    @property
    def register_desc(self) -> str:
        return REGISTERS.get(self.register.lower(), REGISTERS["msa"])

    def to_local(self, path: str) -> str:
        """Map a path as Bazarr sees it onto this container's filesystem."""
        for entry in self.path_map:
            if ":" not in entry:
                continue
            remote, local = entry.split(":", 1)
            remote, local = remote.rstrip("/"), local.rstrip("/")
            if path.startswith(remote + "/") or path == remote:
                return local + path[len(remote):]
        return path


settings = Settings()
