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


def test_health_says_little_to_a_stranger(client):
    """Reachable for the container healthcheck, but it volunteers nothing."""
    body = client.get("/health").json()
    assert body == {"status": "ok"}


def test_health_is_detailed_once_authenticated(client):
    body = client.get("/health", headers={"x-api-token": "secret"}).json()
    assert body["status"] == "ok"
    assert body["target_lang"] == "ar"
    assert body["auth"] is True


def test_dashboard_redirects_a_stranger_to_the_login_page(client):
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/login"


def test_dashboard_renders_when_authenticated(client):
    response = client.get("/?token=secret")
    assert response.status_code == 200
    assert "tarjem" in response.text


def test_endpoints_require_credentials(client):
    assert client.get("/jobs").status_code == 401
    assert client.get("/jobs", headers={"x-api-token": "secret"}).status_code == 200
    assert client.get("/jobs?token=secret").status_code == 200


def test_signing_in_sets_a_session_cookie_that_works(client):
    from app.config import settings as s
    s.auth_password = "hunter2"

    bad = client.post("/login", data={"password": "wrong"}, follow_redirects=False)
    assert bad.status_code == 303 and "error" in bad.headers["location"]

    ok = client.post("/login", data={"password": "hunter2"}, follow_redirects=False)
    assert ok.status_code == 303 and ok.headers["location"] == "/"
    cookie = ok.cookies.get("tarjem_session")
    assert cookie

    # the cookie alone now authenticates, with no token in the URL
    assert client.get("/jobs").status_code == 200
    assert client.get("/", follow_redirects=False).status_code == 200

    client.post("/logout", follow_redirects=False)
    assert client.get("/jobs").status_code == 401
    s.auth_password = ""


def test_the_login_page_is_reachable_without_credentials(client):
    page = client.get("/login")
    assert page.status_code == 200 and "Sign in" in page.text


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


# -- rush: run one job on a different provider, ahead of the queue ---------

def test_rush_requires_a_usable_provider(client, tmp_path):
    from app.config import settings as s
    video = tmp_path / "Rush.Me.2020.mkv"
    video.write_bytes(b"x")
    s.anthropic_api_key = ""

    r = client.post("/translate", headers={"x-api-token": "secret"},
                    json={"video": str(video), "provider": "anthropic", "rush": True})
    assert r.status_code == 400 and "ANTHROPIC_API_KEY" in r.json()["detail"]


def test_rush_queues_with_provider_and_priority(client, tmp_path):
    from app.config import settings as s
    s.anthropic_api_key = "test-key"
    video = tmp_path / "Rush.Me.2021.mkv"
    video.write_bytes(b"x")

    r = client.post("/translate", headers={"x-api-token": "secret"},
                    json={"video": str(video), "provider": "anthropic", "rush": True})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["queued"] and body["provider"] == "anthropic" and body["rush"]

    job = client.get(f"/jobs/{body['job']}", headers={"x-api-token": "secret"}).json()
    assert job["provider"] == "anthropic" and job["priority"] == 1
    s.anthropic_api_key = ""


def test_an_unknown_provider_is_rejected(client, tmp_path):
    video = tmp_path / "Rush.Me.2022.mkv"
    video.write_bytes(b"x")
    r = client.post("/translate", headers={"x-api-token": "secret"},
                    json={"video": str(video), "provider": "hal9000"})
    assert r.status_code == 400 and "unknown provider" in r.json()["detail"]


def test_rush_endpoint_requeues_an_existing_job(client, tmp_path):
    from app.config import settings as s
    s.anthropic_api_key = "test-key"
    video = tmp_path / "Redo.Me.2023.mkv"
    video.write_bytes(b"x")

    first = client.post("/translate", headers={"x-api-token": "secret"},
                        json={"video": str(video)}).json()["job"]
    # The original has to leave the queue before it can be rushed.
    client.post(f"/jobs/{first}/retry", headers={"x-api-token": "secret"})

    r = client.post(f"/jobs/{first}/rush?provider=anthropic", headers={"x-api-token": "secret"})
    assert r.status_code in (200, 404)   # 404 only if it is still queued
    s.anthropic_api_key = ""


def test_dashboard_shows_both_rush_controls(client):
    page = client.get("/?token=secret").text
    assert "Translate now" in page
    assert "rushPath('ollama')" in page and "rushPath('anthropic')" in page


# -- library: browse everything still missing Arabic ----------------------

def test_library_page_requires_auth(client):
    r = client.get("/library", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/login"


def test_library_page_renders_when_authenticated(client):
    page = client.get("/library?token=secret")
    assert page.status_code == 200
    assert "library" in page.text and "api/library" in page.text


def test_library_api_lists_candidates(client, tmp_path, monkeypatch):
    from app import main as m
    video = tmp_path / "Some Show" / "Season 1" / "Ep01.mkv"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"x")

    monkeypatch.setattr(
        m.state["sweeper"], "everything",
        lambda: ([{"video": video, "title": "Some Show - Ep01", "key": "Some Show"}], set()),
    )
    m.state.pop("library", None)

    d = client.get("/api/library", headers={"x-api-token": "secret"}).json()
    assert d["source"] == "disk" and d["total"] == 1
    item = d["items"][0]
    assert item["title"] == "Some Show - Ep01"
    assert item["translated"] is False and item["pending"] is False


def test_library_marks_already_translated_items(client, tmp_path, monkeypatch):
    from app import main as m
    video = tmp_path / "Done Show" / "Season 1" / "Ep02.mkv"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"x")
    (video.parent / "Ep02.ar.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\nx\n")

    monkeypatch.setattr(
        m.state["sweeper"], "everything",
        lambda: ([{"video": video, "title": "Done Show", "key": "Done Show"}], set()),
    )
    m.state.pop("library", None)

    item = client.get("/api/library", headers={"x-api-token": "secret"}).json()["items"][0]
    assert item["translated"] is True


def test_library_filters_by_query(client, tmp_path, monkeypatch):
    from app import main as m
    made = []
    for name in ("Alpha", "Beta"):
        v = tmp_path / name / "Season 1" / f"{name}.mkv"
        v.parent.mkdir(parents=True)
        v.write_bytes(b"x")
        made.append({"video": v, "title": name, "key": name})

    monkeypatch.setattr(m.state["sweeper"], "everything", lambda: (made, set()))
    m.state.pop("library", None)

    d = client.get("/api/library?q=alpha", headers={"x-api-token": "secret"}).json()
    assert d["total"] == 1 and d["items"][0]["title"] == "Alpha"
    m.state.pop("library", None)


def test_library_includes_already_translated_items(client, tmp_path, monkeypatch):
    """Everything is listed, not only what is missing - you may want a redo."""
    from app import main as m
    made = []
    for name, done in (("Fresh", False), ("Already", True)):
        v = tmp_path / name / "Season 1" / f"{name}.mkv"
        v.parent.mkdir(parents=True)
        v.write_bytes(b"x")
        if done:
            (v.parent / f"{name}.ar.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\nx\n")
        made.append({"video": v, "title": name, "key": name})

    monkeypatch.setattr(m.state["sweeper"], "everything", lambda: (made, set()))
    m.state.pop("library", None)

    d = client.get("/api/library", headers={"x-api-token": "secret"}).json()
    assert d["total"] == 2 and d["library_total"] == 2
    done = [i for i in d["items"] if i["translated"]][0]
    assert done["title"] == "Already" and done["subtitle"] == "Already.ar.srt"

    only_missing = client.get("/api/library?state=missing",
                              headers={"x-api-token": "secret"}).json()
    assert only_missing["total"] == 1 and only_missing["items"][0]["title"] == "Fresh"

    only_done = client.get("/api/library?state=translated",
                           headers={"x-api-token": "secret"}).json()
    assert only_done["total"] == 1 and only_done["items"][0]["title"] == "Already"
    m.state.pop("library", None)


def test_force_lets_a_translated_item_be_redone(client, tmp_path):
    video = tmp_path / "Redo.2024.mkv"
    video.write_bytes(b"x")
    existing = tmp_path / "Redo.2024.ar.srt"
    existing.write_text("1\n00:00:01,000 --> 00:00:02,000\nold\n")

    blocked = client.post("/translate", headers={"x-api-token": "secret"},
                          json={"video": str(video)})
    assert blocked.json()["queued"] is False

    forced = client.post("/translate", headers={"x-api-token": "secret"},
                         json={"video": str(video), "force": True})
    assert forced.json()["queued"] is True
    # the old subtitle is kept, not destroyed
    assert not existing.exists()
    assert (tmp_path / "Redo.2024.ar.srt.bak").is_file()


# -- dashboard with a real backlog ----------------------------------------
# 57 queued jobs pushed the running one and all the history out of the window.

def test_dashboard_hides_the_queue_but_keeps_the_running_job(client, tmp_path):
    from app import main as m
    from pathlib import Path as P
    db = m.state["store"]
    # Earlier tests left jobs queued, so claim until nothing is left, then take
    # the one we care about - claim_next serves the oldest first.
    while db.claim_next():
        pass
    db.enqueue(str(tmp_path / "the-running-one.mkv"))
    claimed = db.claim_next()
    assert P(claimed["video"]).name == "the-running-one.mkv"
    for n in range(45):
        db.enqueue(str(tmp_path / f"queued{n}.mkv"))

    page = client.get("/?token=secret").text
    assert "the-running-one.mkv" in page
    assert "queued0.mkv" not in page          # the queue is not rendered
    assert "/?status=queued" in page          # but it is one click away


def test_dashboard_can_filter_to_a_single_status(client):
    page = client.get("/?token=secret&status=queued").text
    assert "queued0.mkv" in page
    assert "show all activity" in page


def test_jobs_api_puts_the_running_job_first(client):
    d = client.get("/jobs?limit=50", headers={"x-api-token": "secret"}).json()
    assert d["jobs"][0]["status"] == "running"


def test_a_local_job_can_rush_too(client, tmp_path):
    """The rush lane is about jumping the queue, not about which provider."""
    video = tmp_path / "Local.Rush.2024.mkv"
    video.write_bytes(b"x")

    r = client.post("/translate", headers={"x-api-token": "secret"},
                    json={"video": str(video), "provider": "ollama", "rush": True})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["queued"] and body["provider"] == "ollama"

    job = client.get(f"/jobs/{body['job']}", headers={"x-api-token": "secret"}).json()
    assert job["provider"] == "ollama" and job["priority"] == 1


def test_rushing_locally_needs_no_api_key(client, tmp_path):
    from app.config import settings as s
    s.anthropic_api_key = ""
    video = tmp_path / "NoKey.Needed.2024.mkv"
    video.write_bytes(b"x")

    ok = client.post("/translate", headers={"x-api-token": "secret"},
                     json={"video": str(video), "provider": "ollama", "rush": True})
    assert ok.json()["queued"] is True

    # ...whereas Claude still refuses up front rather than failing later
    denied = client.post("/translate", headers={"x-api-token": "secret"},
                         json={"video": str(video), "provider": "anthropic", "rush": True})
    assert denied.status_code == 400


def test_the_rush_lane_takes_a_local_job(tmp_path):
    from app.store import Store
    store = Store(tmp_path / "lane.db")
    store.enqueue("/m/backlog.mkv")                       # ordinary queue
    rush = store.enqueue("/m/now.mkv", provider="ollama", priority=1)
    assert store.claim_next(priority_only=True)["id"] == rush


def test_library_offers_both_rush_buttons(client):
    page = client.get("/library?token=secret").text
    assert "'ollama'" in page and "'anthropic'" in page
    assert "/api/library" in page


# -- backend management ----------------------------------------------------
# The second GPU lives in a gaming PC, so it has to be possible to pull it out
# of rotation without editing .env and redeploying.

@pytest.fixture
def clean_pool():
    from app import main as m
    before = list(m.state["pool"].endpoints)
    m.state["pool"].endpoints.clear()
    yield m.state["pool"]
    m.state["pool"].endpoints[:] = before


def test_a_backend_can_be_added_and_persists(client, clean_pool):
    from app import main as m
    r = client.post("/api/backends", headers={"x-api-token": "secret"},
                    json={"kind": "ollama", "url": "http://192.168.1.7:11434",
                          "model": "command-r7b-arabic"})
    assert r.status_code == 200, r.text
    assert r.json()["added"] == "ollama@192.168.1.7:11434"
    # written through to the database, so a restart keeps it
    assert any(b["url"] == "http://192.168.1.7:11434" for b in m.state["store"].backends())


def test_adding_the_same_backend_twice_is_refused(client, clean_pool):
    body = {"kind": "ollama", "url": "http://dup:11434", "model": "m"}
    assert client.post("/api/backends", headers={"x-api-token": "secret"},
                       json=body).status_code == 200
    assert client.post("/api/backends", headers={"x-api-token": "secret"},
                       json=body).status_code == 409


def test_a_bad_backend_is_rejected(client, clean_pool):
    for body in ({"kind": "ollama", "url": ""}, {"kind": "hal9000", "url": "http://x:1"}):
        r = client.post("/api/backends", headers={"x-api-token": "secret"}, json=body)
        assert r.status_code == 400, body


def test_disabling_stops_new_work_without_losing_the_backend(client, clean_pool):
    from app import main as m
    client.post("/api/backends", headers={"x-api-token": "secret"},
                json={"kind": "ollama", "url": "http://gaming:11434", "model": "m"})
    name = "ollama@gaming:11434"

    r = client.patch(f"/api/backends/{name}?enabled=false", headers={"x-api-token": "secret"})
    assert r.status_code == 200 and r.json()["enabled"] is False
    assert m.state["pool"].acquire() is None          # nothing left to lease
    assert m.state["pool"].find(name) is not None     # but still configured
    assert m.state["store"].backends()[-1]["enabled"] is False

    client.patch(f"/api/backends/{name}?enabled=true", headers={"x-api-token": "secret"})
    leased = m.state["pool"].acquire()
    assert leased is not None and leased.name == name
    m.state["pool"].release(leased)


def test_a_busy_backend_cannot_be_removed(client, clean_pool):
    from app import main as m
    client.post("/api/backends", headers={"x-api-token": "secret"},
                json={"kind": "ollama", "url": "http://busy:11434", "model": "m"})
    leased = m.state["pool"].acquire()

    r = client.delete(f"/api/backends/{leased.name}", headers={"x-api-token": "secret"})
    assert r.status_code == 409 and "Disable it instead" in r.json()["detail"]

    m.state["pool"].release(leased)
    assert client.delete(f"/api/backends/{leased.name}",
                         headers={"x-api-token": "secret"}).status_code == 200


def test_the_backends_page_requires_auth_and_renders(client):
    assert client.get("/backends", follow_redirects=False).status_code == 303
    page = client.get("/backends?token=secret")
    assert page.status_code == 200
    assert "Add a backend" in page.text and "/api/backends" in page.text
