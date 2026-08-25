"""Source discovery and library-layout heuristics. No ffmpeg needed."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import sources  # noqa: E402
from app.config import Settings  # noqa: E402
from app.worker import display_title, series_key  # noqa: E402


@pytest.fixture
def cfg():
    s = Settings()
    s.target_lang = "ar"
    s.output_suffix = ".ar.srt"
    s.source_langs = ["en", "fr", "es", "de"]
    s.whisper_url = ""
    return s


SRT = "\n\n".join(
    f"{i}\n00:00:{i:02d},000 --> 00:00:{i + 1:02d},000\nLine {i}" for i in range(1, 10)
)


def test_output_path(cfg, tmp_path):
    video = tmp_path / "Movie.2019.1080p.mkv"
    assert sources.output_path(video, cfg).name == "Movie.2019.1080p.ar.srt"


def test_has_target_finds_existing_arabic(cfg, tmp_path):
    video = tmp_path / "Movie.mkv"
    video.write_bytes(b"x")
    assert sources.has_target(video, cfg) is None
    (tmp_path / "Movie.ar.srt").write_text(SRT)
    assert sources.has_target(video, cfg).name == "Movie.ar.srt"


def test_has_target_ignores_english(cfg, tmp_path):
    video = tmp_path / "Movie.mkv"
    video.write_bytes(b"x")
    (tmp_path / "Movie.en.srt").write_text(SRT)
    assert sources.has_target(video, cfg) is None


def test_sidecar_prefers_english_over_other_languages(cfg, tmp_path):
    video = tmp_path / "Movie.mkv"
    video.write_bytes(b"x")
    (tmp_path / "Movie.de.srt").write_text(SRT)
    (tmp_path / "Movie.en.srt").write_text(SRT)
    found = sources.find_source(video, cfg)
    assert found.lang == "en" and found.origin == "sidecar"


def test_forced_track_loses_to_a_full_one(cfg, tmp_path):
    video = tmp_path / "Movie.mkv"
    video.write_bytes(b"x")
    (tmp_path / "Movie.en.forced.srt").write_text(SRT)
    (tmp_path / "Movie.en.srt").write_text(SRT)
    assert sources.find_source(video, cfg).detail.endswith("Movie.en.srt")


def test_arabic_sidecar_is_never_used_as_a_source(cfg, tmp_path):
    video = tmp_path / "Movie.mkv"
    video.write_bytes(b"x")
    arabic = "\n\n".join(
        f"{i}\n00:00:{i:02d},000 --> 00:00:{i + 1:02d},000\nمرحبا بك" for i in range(1, 10)
    )
    (tmp_path / "Movie.fa.srt").write_text(arabic, encoding="utf-8")
    # Unlabelled-as-Arabic but actually Arabic text: caught by the script check.
    assert sources.find_source(video, cfg) is None


def test_bare_sidecar_is_accepted(cfg, tmp_path):
    video = tmp_path / "Movie.mkv"
    video.write_bytes(b"x")
    (tmp_path / "Movie.srt").write_text(SRT)
    found = sources.find_source(video, cfg)
    assert found and found.origin == "sidecar"


def test_tiny_sidecar_is_rejected(cfg, tmp_path):
    video = tmp_path / "Movie.mkv"
    video.write_bytes(b"x")
    (tmp_path / "Movie.en.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\nHi\n")
    assert sources.find_source(video, cfg) is None


def test_brackets_in_filenames_do_not_break_the_glob(cfg, tmp_path):
    video = tmp_path / "Movie [2019] [1080p].mkv"
    video.write_bytes(b"x")
    (tmp_path / "Movie [2019] [1080p].en.srt").write_text(SRT)
    assert sources.find_source(video, cfg).origin == "sidecar"


@pytest.mark.parametrize("path,expected", [
    ("/media/tv/The Expanse/Season 2/The.Expanse.S02E03.mkv", "The Expanse"),
    ("/media/tv/The Expanse/S03/The.Expanse.S03E01.mkv", "The Expanse"),
    ("/media/tv/Doctor Who/Specials/Special.mkv", "Doctor Who"),
    ("/media/movies/Dune (2021)/Dune.2021.2160p.mkv", "Dune (2021)"),
])
def test_series_key_groups_episodes_under_the_show(path, expected):
    assert series_key(Path(path)) == expected


def test_display_title_strips_release_noise():
    title = display_title(Path("/media/movies/Dune (2021)/Dune.2021.2160p.WEB-DL.x265-GRP.mkv"))
    assert "2160p" not in title and "x265" not in title
    assert "Dune" in title


def test_find_source_on_a_missing_video_returns_none(cfg, tmp_path):
    """It used to fall through to the whisper fallback and crash on open()."""
    cfg.whisper_url = "http://whisper:9000"
    assert sources.find_source(tmp_path / "nope.mkv", cfg) is None
