"""End-to-end: a video with an English sidecar becomes an Arabic sidecar on disk."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import sources, srt, worker  # noqa: E402
from app.config import Settings  # noqa: E402
from app.store import DONE, FAILED, SKIPPED, Store  # noqa: E402
from tests.test_translate import StubProvider  # noqa: E402

ENGLISH = "\n\n".join(
    f"{i}\n00:00:{i:02d},000 --> 00:00:{i + 1:02d},000\nLine {i} of dialogue."
    for i in range(1, 12)
)


class FakeBazarr:
    def __init__(self, knows=None):
        self.rescans = []
        self.lookups = []
        self.knows = knows          # what locate() should answer, if anything

    def rescan(self, kind, series_id, item_id):
        self.rescans.append((kind, series_id, item_id))
        return True

    def locate(self, video):
        self.lookups.append(video)
        return self.knows


@pytest.fixture
def rig(tmp_path, monkeypatch):
    # Explicit about every setting the assertions depend on: another test module
    # may have set env vars before app.config was first imported.
    cfg = Settings()
    cfg.target_lang = "ar"
    cfg.output_suffix = ".ar.srt"
    cfg.source_langs = ["en", "fr", "es", "de"]
    cfg.dry_run = False
    cfg.tag_output = True
    cfg.glossary_enabled = False
    cfg.batch_size = 5
    cfg.max_line_chars = 200
    cfg.data_dir = tmp_path / "data"

    provider = StubProvider()
    monkeypatch.setattr(worker, "build_provider", lambda _cfg, _ep=None: provider)

    store = Store(cfg.data_dir / "test.db")
    bazarr = FakeBazarr()
    return cfg, store, bazarr, provider, worker.Pipeline(cfg, store, bazarr)


def make_media(tmp_path: Path) -> Path:
    folder = tmp_path / "The Expanse" / "Season 2"
    folder.mkdir(parents=True)
    video = folder / "The.Expanse.S02E03.1080p.mkv"
    video.write_bytes(b"not a real video")
    (folder / "The.Expanse.S02E03.1080p.en.srt").write_text(ENGLISH, encoding="utf-8")
    return video


def test_writes_an_arabic_sidecar_with_identical_timings(rig, tmp_path):
    cfg, store, bazarr, _provider, pipeline = rig
    video = make_media(tmp_path)

    job_id = store.enqueue(str(video), "The Expanse S02E03", "The Expanse",
                           "bazarr", "episode", 7, 99)
    pipeline.run(dict(store.claim_next()))

    job = store.get(job_id)
    assert job["status"] == DONE, job.get("error")

    out = video.parent / "The.Expanse.S02E03.1080p.ar.srt"
    assert out.is_file()

    source_cues = srt.parse(ENGLISH)
    written = srt.parse(out.read_text(encoding="utf-8"))
    assert len(written) == len(source_cues)
    assert [(c.start, c.end) for c in written] == [(c.start, c.end) for c in source_cues]
    assert all(c.text.startswith("[AR]") for c in written)
    assert job["stats"]["translated"] == len(source_cues)


def test_records_provenance_beside_the_subtitle(rig, tmp_path):
    cfg, store, _bazarr, _provider, pipeline = rig
    video = make_media(tmp_path)
    store.enqueue(str(video), "", "", "manual")
    pipeline.run(dict(store.claim_next()))

    meta = video.parent / "The.Expanse.S02E03.1080p.ar.srt.tarjem.json"
    assert json.loads(meta.read_text())["translated_from"] == "en"


def test_asks_bazarr_to_rescan_when_it_has_an_id(rig, tmp_path):
    cfg, store, bazarr, _provider, pipeline = rig
    video = make_media(tmp_path)
    store.enqueue(str(video), "", "", "bazarr", "episode", 7, 99)
    pipeline.run(dict(store.claim_next()))
    assert bazarr.rescans == [("episode", 7, 99)]


def test_a_job_queued_here_is_looked_up_by_path(rig, tmp_path):
    """Nothing queued from the library page carries Bazarr's ids, so without
    the lookup the subtitle sat on disk and Bazarr kept showing it missing."""
    cfg, store, bazarr, _provider, pipeline = rig
    bazarr.knows = ("episode", 7, 0)
    video = make_media(tmp_path)
    store.enqueue(str(video), "", "", "manual")
    pipeline.run(dict(store.claim_next()))
    assert bazarr.lookups == [str(video)]
    assert bazarr.rescans == [("episode", 7, 0)]


def test_no_rescan_when_bazarr_does_not_know_the_file(rig, tmp_path):
    cfg, store, bazarr, _provider, pipeline = rig
    bazarr.knows = None
    video = make_media(tmp_path)
    store.enqueue(str(video), "", "", "manual")
    pipeline.run(dict(store.claim_next()))
    assert bazarr.rescans == []


def test_a_job_with_ids_does_not_need_the_lookup(rig, tmp_path):
    cfg, store, bazarr, _provider, pipeline = rig
    video = make_media(tmp_path)
    store.enqueue(str(video), "", "", "bazarr", "episode", 7, 99)
    pipeline.run(dict(store.claim_next()))
    assert bazarr.lookups == []


def test_skips_when_arabic_already_exists(rig, tmp_path):
    cfg, store, _bazarr, provider, pipeline = rig
    video = make_media(tmp_path)
    (video.parent / "The.Expanse.S02E03.1080p.ar.srt").write_text(ENGLISH, encoding="utf-8")

    job_id = store.enqueue(str(video), "", "", "sweep")
    pipeline.run(dict(store.claim_next()))
    assert store.get(job_id)["status"] == SKIPPED
    assert provider.calls == 0


def test_fails_cleanly_with_no_source(rig, tmp_path):
    cfg, store, _bazarr, _provider, pipeline = rig
    video = tmp_path / "Lonely.Movie.mkv"
    video.write_bytes(b"x")
    job_id = store.enqueue(str(video), "", "", "manual")
    pipeline.run(dict(store.claim_next()))

    job = store.get(job_id)
    assert job["status"] == FAILED
    assert "no usable source" in job["error"]


def test_dry_run_writes_nothing(rig, tmp_path):
    cfg, store, _bazarr, _provider, pipeline = rig
    cfg.dry_run = True
    video = make_media(tmp_path)
    job_id = store.enqueue(str(video), "", "", "manual")
    pipeline.run(dict(store.claim_next()))

    assert store.get(job_id)["status"] == DONE
    assert not (video.parent / "The.Expanse.S02E03.1080p.ar.srt").exists()


def test_glossary_is_reused_across_episodes(rig, tmp_path):
    cfg, store, _bazarr, provider, pipeline = rig
    cfg.glossary_enabled = True
    folder = tmp_path / "The Expanse" / "Season 2"
    folder.mkdir(parents=True)

    for episode in ("S02E01", "S02E02"):
        video = folder / f"The.Expanse.{episode}.mkv"
        video.write_bytes(b"x")
        (folder / f"The.Expanse.{episode}.en.srt").write_text(ENGLISH, encoding="utf-8")
        store.enqueue(str(video), "", "The Expanse", "sweep")
        pipeline.run(dict(store.claim_next()))

    # One glossary built, stored under the show, and reused for the second episode.
    assert [g["key"] for g in store.glossaries()] == ["The Expanse"]


class DeadProvider(StubProvider):
    """Every batch fails - e.g. the model name is wrong or the server is down."""

    def structured(self, system, user, schema_model, max_tokens=16000):
        self.calls += 1
        from app.providers.base import ProviderError
        raise ProviderError("model not found", retryable=False)


def test_a_file_where_nothing_translated_fails_and_writes_no_file(rig, tmp_path, monkeypatch):
    """The worst outcome is a .ar.srt full of untranslated English: it looks
    like success, and it blocks every retry because the target now exists."""
    cfg, store, _bazarr, _provider, pipeline = rig
    monkeypatch.setattr(worker, "build_provider", lambda _cfg, _ep=None: DeadProvider())
    video = make_media(tmp_path)

    job_id = store.enqueue(str(video), "", "", "sweep")
    pipeline.run(dict(store.claim_next()))

    job = store.get(job_id)
    assert job["status"] == FAILED
    assert "0 of" in job["error"]
    assert not (video.parent / "The.Expanse.S02E03.1080p.ar.srt").exists()
    # and nothing was left behind to block a retry
    assert sources.has_target(video, cfg) is None


def test_a_partial_failure_still_writes_what_it_got(rig, tmp_path, monkeypatch):
    cfg, store, _bazarr, _provider, pipeline = rig
    monkeypatch.setattr(worker, "build_provider", lambda _cfg, _ep=None: StubProvider(drop={0, 1}))
    video = make_media(tmp_path)

    job_id = store.enqueue(str(video), "", "", "sweep")
    pipeline.run(dict(store.claim_next()))

    job = store.get(job_id)
    assert job["status"] == DONE
    assert job["stats"]["untranslated"] == 2
    assert (video.parent / "The.Expanse.S02E03.1080p.ar.srt").is_file()


def _stacked(lines: int = 6, copies: int = 3) -> str:
    """`lines` distinct cues, each repeated `copies` times on top of itself."""
    blocks, n = [], 0
    for line in range(lines):
        start, end = line * 2, line * 2 + 2
        for copy in range(copies):
            n += 1
            blocks.append(
                f"{n}\n00:00:{start:02d},{copy * 100:03d} --> "
                f"00:00:{end:02d},000\nLine {line} of dialogue."
            )
    return "\n\n".join(blocks)


STACKED = _stacked()


def test_stacked_duplicates_are_merged_before_translating(rig, tmp_path):
    """An .ass flattened to SRT repeats each line on top of itself. One real
    episode: 5,390 cues, 346 unique lines, 958 after collapsing."""
    cfg, store, _bazarr, provider, pipeline = rig
    folder = tmp_path / "Show" / "Season 1"
    folder.mkdir(parents=True)
    video = folder / "Show.S01E01.mkv"
    video.write_bytes(b"x")
    (folder / "Show.S01E01.en.srt").write_text(STACKED, encoding="utf-8")

    job_id = store.enqueue(str(video), "", "", "sweep")
    pipeline.run(dict(store.claim_next()))

    job = store.get(job_id)
    assert job["status"] == DONE, job.get("error")
    # 18 source cues, 6 distinct lines stacked 3 deep -> 6 written.
    written = srt.parse((folder / "Show.S01E01.ar.srt").read_text(encoding="utf-8"))
    assert len(written) == 6
    assert job["stats"]["total_cues"] == 6


def test_collapsing_can_be_turned_off(rig, tmp_path):
    cfg, store, _bazarr, _provider, pipeline = rig
    cfg.collapse_duplicates = False
    folder = tmp_path / "Show" / "Season 1"
    folder.mkdir(parents=True)
    video = folder / "Show.S01E01.mkv"
    video.write_bytes(b"x")
    (folder / "Show.S01E01.en.srt").write_text(STACKED, encoding="utf-8")

    store.enqueue(str(video), "", "", "sweep")
    pipeline.run(dict(store.claim_next()))
    written = srt.parse((folder / "Show.S01E01.ar.srt").read_text(encoding="utf-8"))
    assert len(written) == 18


def test_a_sweep_walks_past_recent_failures(rig, tmp_path, monkeypatch):
    """Otherwise 217 candidates with no source mean the same five are retried
    forever while the rest are never looked at."""
    cfg, store, bazarr, _provider, _pipeline = rig
    cfg.sweep_limit = 2
    cfg.retry_failed_hours = 24

    made = []
    for n in range(5):
        v = tmp_path / f"show{n}" / "Season 1" / f"ep{n}.mkv"
        v.parent.mkdir(parents=True)
        v.write_bytes(b"x")
        made.append({"video": v, "title": f"show{n}", "key": f"show{n}"})

    sweeper = worker.Sweeper(cfg, store, bazarr)
    monkeypatch.setattr(sweeper, "candidates", lambda: (made, "disk"))

    first = sweeper.sweep()
    assert first["queued"] == 2
    # they all fail for want of a source
    while (job := store.claim_next()):
        store.finish(job["id"], FAILED, error="no usable source subtitle found")

    second = sweeper.sweep()
    assert second["queued"] == 2, "should move on to the next two"
    assert second["skipped_recent_failures"] == 2
    queued = {Path(j["video"]).name for j in store.recent(10, status="queued")}
    assert queued == {"ep2.mkv", "ep3.mkv"}
