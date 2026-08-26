"""Telling Bazarr about a subtitle it did not ask for.

Jobs that arrive from Bazarr carry its ids, so the rescan is aimed straight at
the right show. Anything queued from tarjem's own library page has no ids at
all - the file was written and Bazarr went on showing the episode as missing
until its next scheduled disk scan, which can be hours. So we look the item up
by path instead.
"""
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.bazarr import BazarrClient  # noqa: E402
from app.config import Settings  # noqa: E402

SERIES = [
    {"sonarrSeriesId": 12, "path": "/tv/Detective Conan Remastered"},
    {"sonarrSeriesId": 13, "path": "/tv/Severance"},
]
MOVIES = [{"radarrId": 7, "path": "/movies/Dune (2021)/Dune.2021.mkv"}]


def client(series=SERIES, movies=MOVIES, calls=None, **over):
    cfg = Settings(bazarr_url="http://bazarr:6767", bazarr_api_key="k", **over)
    bz = BazarrClient(cfg)

    def handle(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(request.url.path)
        if request.url.path.endswith("/series"):
            return httpx.Response(200, json={"data": series})
        if request.url.path.endswith("/movies"):
            return httpx.Response(200, json={"data": movies})
        return httpx.Response(404)

    bz._client = httpx.Client(base_url="http://bazarr:6767/api",
                              transport=httpx.MockTransport(handle))
    return bz


def test_an_episode_is_matched_to_its_series_by_folder():
    found = client().locate("/tv/Severance/Season 2/Severance.S02E01.mkv")
    assert found == ("episode", 13, 0)


def test_anime_numbered_straight_through_still_matches():
    """No season folder and no SxxEyy - the folder is the only signal."""
    found = client().locate("/tv/Detective Conan Remastered/Conan - 031.mp4")
    assert found == ("episode", 12, 0)


def test_a_film_is_matched_on_its_exact_path():
    assert client().locate("/movies/Dune (2021)/Dune.2021.mkv") == ("movie", 0, 7)


def test_the_innermost_series_wins():
    """A nested library would otherwise resolve to whichever came back first."""
    nested = SERIES + [{"sonarrSeriesId": 99, "path": "/tv"}]
    found = client(series=nested).locate("/tv/Severance/S02E01.mkv")
    assert found == ("episode", 13, 0)


def test_a_sibling_folder_is_not_a_prefix_match():
    """'/tv/Severance' must not swallow '/tv/Severance Extras'."""
    found = client().locate("/tv/Severance Extras/blooper.mkv")
    assert found is None


def test_an_unknown_file_is_left_for_the_next_disk_scan():
    assert client().locate("/tv/Something Bazarr Never Heard Of/x.mkv") is None


def test_paths_are_mapped_into_this_container():
    bz = client(series=[{"sonarrSeriesId": 5, "path": "/tv/Severance"}],
                path_map=["/tv:/media/tv"])
    assert bz.locate("/media/tv/Severance/S01E01.mkv") == ("episode", 5, 0)


def test_the_index_is_fetched_once_not_per_episode():
    """Queueing a whole series would otherwise be two API calls per file."""
    calls = []
    bz = client(calls=calls)
    for n in range(1, 25):
        bz.locate(f"/tv/Severance/Season 1/S01E{n:02d}.mkv")
    assert calls == ["/api/series", "/api/movies"]


def test_a_stale_index_is_refetched():
    calls = []
    bz = client(calls=calls)
    bz.locate("/tv/Severance/a.mkv")
    bz.locate("/tv/Severance/b.mkv", ttl=0)
    assert calls.count("/api/series") == 2


def test_nothing_is_asked_when_notifying_is_switched_off():
    calls = []
    bz = client(calls=calls, notify_bazarr=False)
    assert bz.locate("/tv/Severance/a.mkv") is None
    assert calls == []


def test_a_bazarr_that_is_down_does_not_raise():
    cfg = Settings(bazarr_url="http://bazarr:6767", bazarr_api_key="k")
    bz = BazarrClient(cfg)

    def refuse(request):
        raise httpx.ConnectError("connection refused")

    bz._client = httpx.Client(base_url="http://bazarr:6767/api",
                              transport=httpx.MockTransport(refuse))
    assert bz.locate("/tv/Severance/a.mkv") is None


@pytest.mark.parametrize("row", [
    {"path": "/tv/X"},                       # no id
    {"sonarrSeriesId": 4},                   # no path
    {"sonarrSeriesId": 0, "path": "/tv/X"},  # id zero
])
def test_incomplete_rows_are_skipped(row):
    assert client(series=[row]).locate("/tv/X/ep.mkv") is None
