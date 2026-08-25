"""Environment-driven settings. Everything has a working default except the API key."""
from __future__ import annotations

import json
import logging
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
    # Merge identical cues that overlap. An .ass with several styled layers
    # flattened into SRT repeats each line on top of itself; left alone they
    # render stacked. Only overlapping repeats are touched.
    collapse_duplicates: bool = _bool("COLLAPSE_DUPLICATES", True)

    # --- model -----------------------------------------------------------
    provider: str = _str("LLM_PROVIDER", "anthropic")
    model: str = _str("LLM_MODEL", "claude-opus-5")
    effort: str = _str("LLM_EFFORT", "low")
    batch_size: int = _int("BATCH_SIZE", 40)
    context_cues: int = _int("CONTEXT_CUES", 8)
    max_retries: int = _int("MAX_RETRIES", 3)
    glossary_enabled: bool = _bool("GLOSSARY_ENABLED", True)
    glossary_sample: int = _int("GLOSSARY_SAMPLE", 180)
    # Explicit Arabic grammar rules in the system prompt. Small local models
    # need them; a frontier model mostly knows this already, but they are
    # cheap - prompt tokens process far faster than they generate.
    grammar_guardrails: bool = _bool("GRAMMAR_GUARDRAILS", True)

    anthropic_api_key: str = _str("ANTHROPIC_API_KEY")
    openai_api_key: str = _str("OPENAI_API_KEY")
    openai_base_url: str = _str("OPENAI_BASE_URL", "https://api.openai.com/v1")
    openai_model: str = _str("OPENAI_MODEL", "gpt-4.1")
    ollama_url: str = _str("OLLAMA_URL", "http://localhost:11434")
    ollama_model: str = _str("OLLAMA_MODEL", "command-r7b-arabic")
    # Ollama's default context is small enough that a batch plus its glossary
    # can overflow it, and an overflow silently truncates the prompt.
    ollama_num_ctx: int = _int("OLLAMA_NUM_CTX", 8192)
    # Self-hosted models can take minutes per batch on CPU.
    llm_timeout: int = _int("LLM_TIMEOUT", 1800)
    # Raw JSON merged into the request body - for server-specific switches such
    # as Ollama's {"think": false} or a fixed {"temperature": 0.2}.
    llm_extra_body: str = _str("LLM_EXTRA_BODY")

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

    @property
    def active_model(self) -> str:
        """The model actually in use, for logs, /health and provenance.

        Each provider reads its own setting, so anything that reports "the
        model" has to resolve it the same way build_provider does - otherwise
        the dashboard confidently displays a model that is never called.
        """
        if self.provider in ("ollama", "local"):
            return self.ollama_model
        if self.provider in ("openai", "openai-compatible"):
            return self.openai_model
        return self.model

    @property
    def extra_body(self) -> dict:
        if not self.llm_extra_body:
            return {}
        try:
            parsed = json.loads(self.llm_extra_body)
        except json.JSONDecodeError:
            logging.getLogger(__name__).warning(
                "LLM_EXTRA_BODY is not valid JSON, ignoring: %s", self.llm_extra_body
            )
            return {}
        return parsed if isinstance(parsed, dict) else {}

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
