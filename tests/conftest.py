"""Keep the developer's shell environment out of the test run.

`Settings` reads os.environ, so a stray ARABIC_REGISTER or DRY_RUN exported in a
terminal would quietly change what the tests assert. Clear the lot before any
test module imports app.config.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

for name in list(os.environ):
    if name.startswith(("LLM_", "BAZARR_", "SWEEP_", "TARJEM_", "OPENAI_", "ANTHROPIC_")):
        del os.environ[name]

for name in ("TARGET_LANG", "SOURCE_LANGS", "ARABIC_REGISTER", "OUTPUT_SUFFIX", "TAG_OUTPUT",
             "STRIP_HI", "MAX_LINE_CHARS", "MAX_LINES", "BATCH_SIZE", "CONTEXT_CUES",
             "MAX_RETRIES", "GLOSSARY_ENABLED", "GLOSSARY_SAMPLE", "MEDIA_ROOTS", "PATH_MAP",
             "NOTIFY_BAZARR", "WHISPER_URL", "DATA_DIR", "WORKERS", "DRY_RUN", "API_TOKEN"):
    os.environ.pop(name, None)
