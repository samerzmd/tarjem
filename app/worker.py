"""Job pipeline and the background sweeper."""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from pathlib import Path

from . import sources, srt
from .bazarr import BazarrClient
from .config import Settings
from .providers import build_provider
from .store import DONE, FAILED, SKIPPED, Store
from .translate import TitleGlossary, Translator

log = logging.getLogger(__name__)

SEASON_DIR_RE = re.compile(r"^(season\s*\d+|s\d{1,3}|specials|series\s*\d+)$", re.IGNORECASE)
RELEASE_NOISE_RE = re.compile(
    r"\b(1080p|2160p|720p|480p|x264|x265|h\.?26[45]|hevc|bluray|blu-ray|web-?dl|webrip|"
    r"hdtv|dvdrip|remux|aac|ac3|dts|hdr|10bit|amzn|nf|dsnp|hmax|atvp|proper|repack)\b.*",
    re.IGNORECASE,
)


def series_key(video: Path) -> str:
    """A stable identity for the show or film, used to reuse a glossary."""
    parent = video.parent
    if SEASON_DIR_RE.match(parent.name.strip()):
        return parent.parent.name or parent.name
    return parent.name or video.stem


def display_title(video: Path) -> str:
    """A human title for the prompt, with release-group noise trimmed off."""
    key = series_key(video)
    name = RELEASE_NOISE_RE.sub("", video.stem).replace(".", " ").strip(" -_")
    if key and key.lower() not in name.lower():
        return f"{key} - {name}" if name else key
    return name or key


class Pipeline:
    """Turns one video into one Arabic sidecar."""

    def __init__(self, cfg: Settings, store: Store, bazarr: BazarrClient):
        self.cfg = cfg
        self.store = store
        self.bazarr = bazarr

    def run(self, job: dict) -> None:
        job_id = job["id"]
        video = Path(self.cfg.to_local(job["video"]))
        note = lambda pct, stage: self.store.update(job_id, progress=pct, stage=stage)

        if not video.is_file():
            self.store.finish(job_id, FAILED, error=f"video not found: {video}")
            return

        existing = sources.has_target(video, self.cfg)
        if existing:
            self.store.finish(job_id, SKIPPED, output=str(existing),
                              stage=f"{self.cfg.target_lang} subtitle already present")
            return

        note(0.02, "looking for a source subtitle")
        source = sources.find_source(video, self.cfg, job.get("detail") or None)
        if not source:
            self.store.finish(job_id, FAILED, error="no usable source subtitle found")
            return

        cues = srt.parse(source.text)
        if len(cues) < 5:
            self.store.finish(job_id, FAILED,
                              error=f"source has only {len(cues)} cues ({source.detail})")
            return

        self.store.update(job_id, origin=source.origin, detail=source.detail, progress=0.05)
        log.info("job %s: %d cues from %s (%s)", job_id, len(cues), source.origin, source.detail)

        provider = build_provider(self.cfg)
        translator = Translator(provider, self.cfg)
        title = job.get("title") or display_title(video)
        key = job.get("series_key") or series_key(video)

        try:
            note(0.08, "building the translation brief")
            glossary = self._glossary(translator, cues, title, key)
            note(0.12, "translating")

            def progress(fraction: float, stage: str) -> None:
                self.store.update(job_id, progress=0.12 + fraction * 0.85, stage=stage)

            translated, stats = translator.translate(cues, title, glossary, progress)

            if self.cfg.dry_run:
                self.store.finish(job_id, DONE, stage="dry run - nothing written",
                                  stats=stats.as_dict(), usage=provider.usage.as_dict())
                return

            note(0.98, "writing")
            out = self._write(video, translated, source)
            self.store.finish(job_id, DONE, output=str(out), progress=1.0, stage="done",
                              stats=stats.as_dict(), usage=provider.usage.as_dict())
            log.info("job %s: wrote %s (%s)", job_id, out.name, stats.as_dict())
            self._notify(job, video)
        except Exception as exc:  # noqa: BLE001 - a job must never kill the worker
            log.exception("job %s failed", job_id)
            self.store.finish(job_id, FAILED, error=f"{type(exc).__name__}: {exc}",
                              usage=provider.usage.as_dict())
        finally:
            provider.close()

    # -- steps ------------------------------------------------------------

    def _glossary(self, translator: Translator, cues, title: str, key: str) -> TitleGlossary | None:
        if not self.cfg.glossary_enabled:
            return None
        cached = self.store.get_glossary(key)
        if cached:
            log.info("reusing glossary for %s (%d terms)", key, len(cached.get("terms", [])))
            try:
                return TitleGlossary.model_validate(cached)
            except ValueError:
                self.store.drop_glossary(key)
        glossary = translator.build_glossary(cues, title)
        if glossary:
            self.store.put_glossary(key, glossary.model_dump())
        return glossary

    def _write(self, video: Path, cues: list[srt.Cue], source: sources.SubtitleSource) -> Path:
        out = sources.output_path(video, self.cfg)

        tmp = out.with_suffix(out.suffix + ".tmp")
        tmp.write_text(srt.dumps(cues), encoding="utf-8")
        try:
            stat = video.stat()
            os.chmod(tmp, 0o664)
            if hasattr(os, "chown") and os.getuid() == 0:  # pragma: no cover
                os.chown(tmp, stat.st_uid, stat.st_gid)
        except (OSError, AttributeError):
            pass
        tmp.replace(out)

        if self.cfg.tag_output:
            # Provenance lives beside the subtitle rather than inside it: an SRT
            # has no comment syntax, and a marker cue would render on screen.
            meta = out.with_suffix(out.suffix + ".tarjem.json")
            try:
                meta.write_text(json.dumps({
                    "translated_from": source.lang or "unknown",
                    "source": f"{source.origin}: {source.detail}",
                    "provider": self.cfg.provider,
                    "model": self.cfg.model if self.cfg.provider == "anthropic" else self.cfg.openai_model,
                    "register": self.cfg.register,
                    "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                }, indent=2), encoding="utf-8")
            except OSError as exc:
                log.debug("could not write provenance file: %s", exc)
        return out

    def _notify(self, job: dict, video: Path) -> None:
        """Nudge Bazarr to re-index the folder so the new sidecar appears in its UI.

        Without the ids from the webhook or a sweep there is nothing to target,
        and Bazarr will pick the file up on its own next scan instead.
        """
        kind = job.get("bz_kind") or (
            "episode" if SEASON_DIR_RE.match(video.parent.name.strip()) else "movie"
        )
        series_id = int(job.get("bz_series_id") or 0)
        item_id = int(job.get("bz_item_id") or 0)
        if not (series_id or item_id):
            return
        self.bazarr.rescan(kind, series_id, item_id)


class Worker(threading.Thread):
    """Drains the job queue one video at a time."""

    def __init__(self, cfg: Settings, store: Store, bazarr: BazarrClient, name: str = "worker"):
        super().__init__(name=name, daemon=True)
        self.cfg = cfg
        self.store = store
        self.pipeline = Pipeline(cfg, store, bazarr)
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        log.info("%s started", self.name)
        while not self._stop.is_set():
            job = self.store.claim_next()
            if not job:
                self._stop.wait(3)
                continue
            try:
                self.pipeline.run(dict(job))
            except Exception:  # noqa: BLE001
                log.exception("worker crashed on job %s", job["id"])
                self.store.finish(job["id"], FAILED, error="worker crashed")


class Sweeper(threading.Thread):
    """Periodically looks for videos still missing an Arabic subtitle."""

    def __init__(self, cfg: Settings, store: Store, bazarr: BazarrClient):
        super().__init__(name="sweeper", daemon=True)
        self.cfg = cfg
        self.store = store
        self.bazarr = bazarr
        self._stop = threading.Event()
        self._wake = threading.Event()
        self.last_run: float = 0.0
        self.last_result: dict = {}

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    def trigger(self) -> None:
        self._wake.set()

    def run(self) -> None:
        # Give Bazarr a moment to come up when the whole stack starts together.
        self._wake.wait(30)
        while not self._stop.is_set():
            self._wake.clear()
            try:
                self.last_result = self.sweep()
                self.last_run = time.time()
            except Exception:  # noqa: BLE001
                log.exception("sweep failed")
            self._wake.wait(max(60, self.cfg.sweep_interval_min * 60))

    def sweep(self, limit: int | None = None) -> dict:
        limit = limit or self.cfg.sweep_limit
        candidates = self._from_bazarr() if self.cfg.sweep_source == "bazarr" else []
        if not candidates:
            candidates = self._from_disk()

        queued = 0
        for cand in candidates:
            if queued >= limit:
                break
            video = cand["video"]
            path = str(video)
            if self.store.is_pending(path) or self.store.succeeded_for(path):
                continue
            if sources.has_target(video, self.cfg):
                continue
            enqueued = self.store.enqueue(
                path, cand["title"], cand["key"], "sweep",
                cand.get("kind", ""), cand.get("series_id", 0), cand.get("item_id", 0),
            )
            if enqueued is not None:
                queued += 1

        result = {"candidates": len(candidates), "queued": queued, "source": self.cfg.sweep_source}
        log.info("sweep: %s", result)
        return result

    def _from_bazarr(self) -> list[dict]:
        out: list[dict] = []
        for item in self.bazarr.wanted(self.cfg.target_lang):
            if not item.path:
                continue
            video = Path(item.path)
            if not video.is_file():
                log.debug("bazarr wants %s but the file is not visible here", item.path)
                continue
            out.append({
                "video": video,
                "title": item.title or display_title(video),
                "key": series_key(video),
                "kind": item.kind,
                "series_id": item.series_id,
                "item_id": item.item_id,
            })
        return out

    def _from_disk(self) -> list[dict]:
        exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in self.cfg.video_exts}
        out: list[dict] = []
        for root in self.cfg.media_roots:
            base = Path(root)
            if not base.is_dir():
                log.warning("media root missing: %s", base)
                continue
            for video in base.rglob("*"):
                if video.suffix.lower() in exts and video.is_file():
                    out.append({
                        "video": video,
                        "title": display_title(video),
                        "key": series_key(video),
                    })
        return out
