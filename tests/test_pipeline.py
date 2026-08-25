"""End-to-end: a video with an English sidecar becomes an Arabic sidecar on disk."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import srt, worker  # noqa: E402
from app.config import Settings  # noqa: E402
from app.store import DONE, FAILED, SKIPPED, Store  # noqa: E402
from tests.test_translate import StubProvider  # noqa: E402

ENGLISH = "\n\n".join(
    f"{i}\n00:00:{i:02d},000 --> 00:00:{i + 1:02d},000\nLine {i} of dialogue."
    for i in range(1, 12)
)


class FakeBazarr:
    def __init__(self):
        self.rescans = []

    def rescan(self, kind, series_id, item_id):
        self.rescans.append((kind, series_id, item_id))
        return True


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
    monkeypatch.setattr(worker, "build_provider", lambda _cfg: provider)

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


def test_no_rescan_without_an_id(rig, tmp_path):
    cfg, store, bazarr, _provider, pipeline = rig
    video = make_media(tmp_path)
    store.enqueue(str(video), "", "", "manual")
    pipeline.run(dict(store.claim_next()))
    assert bazarr.rescans == []


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
