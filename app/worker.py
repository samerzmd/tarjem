"""Job pipeline and the background sweeper."""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import replace
from pathlib import Path

from . import sources, srt
from .bazarr import BazarrClient
from .config import Settings
from .endpoints import Endpoint, Pool
from .providers import ProviderError, build_provider
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


EPISODE_RE = re.compile(r"(?:^|[^a-z0-9])s(\d{1,3})[ ._-]?e(\d{1,4})", re.IGNORECASE)
EPISODE_ALT_RE = re.compile(r"(?:^|[^a-z0-9])(\d{1,3})x(\d{1,4})(?:[^a-z0-9]|$)", re.IGNORECASE)


MOVIE_ROOT_RE = re.compile(r"movie|film", re.IGNORECASE)
TV_ROOT_RE = re.compile("(^|[^a-z])(tv|show|serie|anime)", re.IGNORECASE)


def media_kind(video: Path, roots: list[str] | None = None) -> str:
    """Episode or film.

    Which library root the file sits under is the only reliable signal: plenty
    of anime is numbered straight through with no season folder and no SxxEyy,
    so filename patterns alone put whole series in the wrong tab. Those
    patterns are the fallback for a layout the root names do not describe.
    """
    for root in roots or []:
        try:
            video.relative_to(root)
        except ValueError:
            continue
        name = Path(root).name
        if MOVIE_ROOT_RE.search(name):
            return "movie"
        if TV_ROOT_RE.search(name):
            return "episode"

    if SEASON_DIR_RE.match(video.parent.name.strip()):
        return "episode"
    if EPISODE_RE.search(video.name) or EPISODE_ALT_RE.search(video.name):
        return "episode"
    return "movie"


def episode_number(video: Path) -> tuple[int, int]:
    """Season and episode, for sorting. (0, 0) when it cannot be read."""
    for pattern in (EPISODE_RE, EPISODE_ALT_RE):
        m = pattern.search(video.name)
        if m:
            return int(m.group(1)), int(m.group(2))
    return 0, 0


def display_title(video: Path) -> str:
    """A human title for the prompt, with release-group noise trimmed off."""
    key = series_key(video)
    name = RELEASE_NOISE_RE.sub("", video.stem).replace(".", " ").strip(" -_")
    if key and key.lower() not in name.lower():
        return f"{key} - {name}" if name else key
    return name or key


class Pipeline:
    """Turns one video into one Arabic sidecar."""

    def __init__(self, cfg: Settings, store: Store, bazarr: BazarrClient,
                 pool: Pool | None = None):
        self.cfg = cfg
        self.store = store
        self.bazarr = bazarr
        self.pool = pool

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

        # Do this before translating, not after: an .ass flattened into SRT can
        # be 90% cues that sit on top of each other, and carrying them through
        # the whole pipeline only to drop them at the end wastes the bookkeeping.
        if self.cfg.strip_style_tags:
            for cue in cues:
                cue.text = srt.strip_style_tags(cue.text)

        if self.cfg.collapse_duplicates:
            cues, merged = srt.collapse_duplicates(cues)
            if merged:
                log.info("job %s: merged %d stacked duplicate cues, %d remain",
                         job_id, merged, len(cues))

        self.store.update(job_id, origin=source.origin, detail=source.detail, progress=0.05)
        log.info("job %s: %d cues from %s (%s)", job_id, len(cues), source.origin, source.detail)
        if len(cues) > 2000:
            # A 45-minute episode is ~600 cues. Several thousand means a fansub
            # .ass with karaoke split per syllable, signs, and styled duplicates,
            # flattened by ffmpeg. Repeats collapse below; the rest is real work.
            log.warning("job %s: %d cues is very high for one file - expect this "
                        "to take proportionally longer", job_id, len(cues))

        # A job may name its own provider - "translate this one on Claude now"
        # while the library grinds through the local model.
        cfg = self.cfg
        if job.get("provider") and job["provider"] != self.cfg.provider:
            cfg = replace(self.cfg, provider=job["provider"])
            log.info("job %s: overriding provider to %s (%s)",
                     job_id, cfg.provider, cfg.active_model)

        # Take a backend of our own so two workers do not both drive the same
        # GPU while another sits idle. Naming a provider narrows which machines
        # qualify - it does not mean "skip the pool": a local rush should still
        # be spread across every ollama box, not pinned to the default URL.
        endpoint: Endpoint | None = None
        wanted = job.get("provider") or ""
        if self.pool and (not wanted or self.pool.has_kind(wanted)):
            endpoint = self.pool.acquire(wanted)
            if endpoint is None:
                self.store.update(job_id, status="queued",
                                  stage="waiting for a free backend")
                return
            self.store.update(job_id, stage=f"on {endpoint.name}",
                              backend=endpoint.name)
        elif not job.get("backend"):
            # Nothing in the pool serves this provider - Claude, usually, with
            # only local machines configured. Record it so the column is filled.
            self.store.update(job_id, backend=cfg.provider)

        try:
            provider = build_provider(cfg, endpoint)
        except ProviderError as exc:
            self.pool and self.pool.release(endpoint)
            self.store.finish(job_id, FAILED, error=f"cannot use {cfg.provider}: {exc}")
            return

        translator = Translator(provider, cfg)
        title = job.get("title") or display_title(video)
        key = job.get("series_key") or series_key(video)

        try:
            note(0.08, "building the translation brief")
            glossary = self._glossary(translator, cues, title, key)
            note(0.12, "translating")

            def progress(fraction: float, stage: str) -> None:
                self.store.update(job_id, progress=0.12 + fraction * 0.85, stage=stage)

            translated, stats = translator.translate(cues, title, glossary, progress)

            # A file where nothing translated is a failure, not a result. Writing
            # it would put the untranslated source under an .ar.srt name, which
            # then blocks every retry because the target "already exists".
            if stats.translated == 0:
                self.store.finish(
                    job_id, FAILED,
                    error=f"every batch failed - 0 of {stats.total} cues translated. "
                          f"Check the provider is reachable and the model name is right.",
                    stats=stats.as_dict(), usage=provider.usage.as_dict(),
                )
                return

            if stats.untouched:
                log.warning("job %s: %d of %d cues fell back to source text",
                            job_id, stats.untouched, stats.total)

            if cfg.dry_run:
                self.store.finish(job_id, DONE, stage="dry run - nothing written",
                                  stats=stats.as_dict(), usage=provider.usage.as_dict())
                return

            note(0.98, "writing")
            out = self._write(video, translated, source, cfg)
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
            if endpoint is not None:
                # A backend that could not be reached at all is parked for a
                # while - a desktop that went to sleep should not fail every
                # job handed to it before the pool notices.
                if getattr(provider, "unreachable", False):
                    endpoint.mark_down("connection failed")
                else:
                    endpoint.mark_up()
                self.pool.release(endpoint)

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

    def _write(self, video: Path, cues: list[srt.Cue], source: sources.SubtitleSource,
               cfg: Settings) -> Path:
        out = sources.output_path(video, cfg)

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

        if cfg.tag_output:
            # Provenance lives beside the subtitle rather than inside it: an SRT
            # has no comment syntax, and a marker cue would render on screen.
            meta = out.with_suffix(out.suffix + ".tarjem.json")
            try:
                meta.write_text(json.dumps({
                    "translated_from": source.lang or "unknown",
                    "source": f"{source.origin}: {source.detail}",
                    "provider": cfg.provider,
                    "model": cfg.active_model,
                    "register": cfg.register,
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
        kind = job.get("bz_kind") or ""
        series_id = int(job.get("bz_series_id") or 0)
        item_id = int(job.get("bz_item_id") or 0)

        if not (series_id or item_id):
            # Queued from the library page rather than by Bazarr, so it has no
            # ids. Ask Bazarr which item this file belongs to, otherwise the
            # subtitle sits on disk unnoticed until its next scan.
            found = self.bazarr.locate(str(video))
            if not found:
                log.debug("no bazarr item matches %s - it will be found on the "
                          "next disk scan", video.name)
                return
            kind, series_id, item_id = found

        if self.bazarr.rescan(kind, series_id, item_id):
            log.info("asked bazarr to rescan %s %s", kind, series_id or item_id)


class Worker(threading.Thread):
    """Drains the job queue one video at a time."""

    def __init__(self, cfg: Settings, store: Store, bazarr: BazarrClient,
                 name: str = "worker", priority_only: bool = False,
                 pool: Pool | None = None):
        super().__init__(name=name, daemon=True)
        self.cfg = cfg
        self.store = store
        self.pipeline = Pipeline(cfg, store, bazarr, pool)
        # The rush lane takes only priority jobs. They run on a different
        # provider, so they must not queue behind a local job with hours left.
        self.priority_only = priority_only
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        log.info("%s started%s", self.name, " (rush lane)" if self.priority_only else "")
        while not self._stop.is_set():
            # Claiming work we cannot start would only mark a job running and
            # immediately put it back - and with every backend disabled that
            # becomes a hot loop.
            if self.pipeline.pool and not self.pipeline.pool.available():
                # The rush lane must not stall on a full pool: its jobs may name
                # a provider no backend serves, and those need no lease.
                if not self.priority_only:
                    self._stop.wait(5)
                    continue
            job = self.store.claim_next(priority_only=self.priority_only)
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

    def candidates(self) -> tuple[list[dict], str]:
        """Everything still missing the target language, and where the list came from.

        Bazarr knows this already and respects the language profiles and
        exclusions, so it is preferred; walking the disk is the fallback for
        when there is no API key.
        """
        if self.cfg.sweep_source == "bazarr":
            found = self._from_bazarr()
            if found:
                return found, "bazarr"
        return self._from_disk(), "disk"

    def everything(self) -> tuple[list[dict], set[str]]:
        """Every video under MEDIA_ROOTS, plus the set Bazarr still wants.

        The sweep only cares about what is missing, but browsing wants the whole
        library - including titles already translated, so they can be redone.
        """
        wanted = {
            str(item["video"]) for item in self._from_bazarr()
        } if self.cfg.sweep_source == "bazarr" else set()
        return self._from_disk(), wanted

    def sweep(self, limit: int | None = None) -> dict:
        limit = limit or self.cfg.sweep_limit
        candidates, _ = self.candidates()

        queued = skipped = 0
        for cand in candidates:
            if queued >= limit:
                break
            video = cand["video"]
            path = str(video)
            if self.store.is_pending(path) or self.store.succeeded_for(path):
                continue
            if sources.has_target(video, self.cfg):
                continue
            if self.store.failed_recently(path, self.cfg.retry_failed_hours):
                skipped += 1
                continue
            enqueued = self.store.enqueue(
                path, cand["title"], cand["key"], "sweep",
                cand.get("kind", ""), cand.get("series_id", 0), cand.get("item_id", 0),
            )
            if enqueued is not None:
                queued += 1

        result = {"candidates": len(candidates), "queued": queued,
                  "skipped_recent_failures": skipped, "source": self.cfg.sweep_source}
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
