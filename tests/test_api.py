"""Boots the real app (workers, sqlite, routes) against a temp data dir."""
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    # Configure the live settings object rather than the environment: app.config
    # reads os.environ once at import, so env changes here would depend on this
    # module happening to import first.
    settings.data_dir = Path(tempfile.mkdtemp(prefix="tarjem-test-"))
    settings.sweep_enabled = False
    settings.bazarr_api_key = ""
    settings.api_token = "secret"
    settings.target_lang = "ar"
    settings.output_suffix = ".ar.srt"
    settings.dry_run = True
    with TestClient(app) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["target_lang"] == "ar"


def test_dashboard_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "tarjem" in response.text


def test_endpoints_require_the_token(client):
    assert client.get("/jobs").status_code == 401
    assert client.get("/jobs", headers={"x-api-token": "secret"}).status_code == 200


def test_hook_ignores_arabic_downloads(client):
    response = client.post(
        "/hook/bazarr",
        headers={"x-api-token": "secret"},
        json={"video": "/media/movies/X/X.mkv", "subtitle": "/media/movies/X/X.ar.srt", "lang": "ar"},
    )
    assert response.status_code == 200
    assert response.json()["queued"] is False


def test_hook_404s_on_a_missing_video(client):
    response = client.post(
        "/hook/bazarr",
        headers={"x-api-token": "secret"},
        json={"video": "/nope/never.mkv", "lang": "en"},
    )
    assert response.status_code == 404


def test_hook_accepts_form_encoded_posts(client, tmp_path):
    """The shape Bazarr's `curl --data-urlencode` command actually sends."""
    video = tmp_path / "Ocean's Eleven (2001)" / "Ocean's.Eleven.2001.mkv"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"x")

    response = client.post(
        "/hook/bazarr",
        headers={"x-api-token": "secret"},
        data={"video": str(video), "lang": "en", "movie_id": "42",
              "subtitle": str(video.with_suffix(".en.srt"))},
    )
    assert response.status_code == 200, response.text
    assert response.json()["queued"] is True

    job = client.get(f"/jobs/{response.json()['job']}", headers={"x-api-token": "secret"}).json()
    assert job["bz_kind"] == "movie" and job["bz_item_id"] == 42


def test_hook_queues_a_real_video(client, tmp_path):
    video = tmp_path / "Some Movie (2019)" / "Some.Movie.2019.1080p.mkv"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"not really a video")

    response = client.post(
        "/hook/bazarr",
        headers={"x-api-token": "secret"},
        json={"video": str(video), "subtitle": str(video.with_suffix(".en.srt")), "lang": "en"},
    )
    assert response.status_code == 200 and response.json()["queued"] is True

    # A second identical hook must not double-queue the same file.
    again = client.post(
        "/hook/bazarr",
        headers={"x-api-token": "secret"},
        json={"video": str(video), "lang": "en"},
    )
    assert again.json()["queued"] is False


def test_existing_arabic_sidecar_short_circuits(client, tmp_path):
    video = tmp_path / "Done.Movie.2020.mkv"
    video.write_bytes(b"x")
    (tmp_path / "Done.Movie.2020.ar.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\nx\n")

    response = client.post(
        "/translate", headers={"x-api-token": "secret"}, json={"video": str(video)}
    )
    assert response.json()["queued"] is False
    assert "already have" in response.json()["reason"]
