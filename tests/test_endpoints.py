"""The backend pool: two machines with a GPU each, sharing the work.

Routing alone buys nothing here - one worker sends one batch at a time, so
alternating which machine answers is no faster. The pool exists so several
workers can run at once, each on a backend of its own.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings  # noqa: E402
from app.endpoints import Endpoint, Pool, parse  # noqa: E402
from app.providers import _for_endpoint  # noqa: E402

SPEC = ("ollama@http://host.docker.internal:11434#command-r7b-arabic,"
        "openai@http://192.168.1.7:1234/v1#command-r7b-arabic")


# -- parsing ---------------------------------------------------------------

def test_parses_kind_url_and_model():
    a, b = parse(SPEC)
    assert (a.kind, a.url, a.model) == (
        "ollama", "http://host.docker.internal:11434", "command-r7b-arabic")
    assert (b.kind, b.url, b.model) == (
        "openai", "http://192.168.1.7:1234/v1", "command-r7b-arabic")
    assert "192.168.1.7:1234" in b.name


def test_a_missing_model_is_allowed():
    (e,) = parse("ollama@http://box:11434")
    assert e.model == ""


def test_malformed_entries_are_skipped_not_fatal():
    eps = parse("nonsense,ollama@,@http://x,ollama@http://good:11434")
    assert [e.url for e in eps] == ["http://good:11434"]


def test_an_empty_spec_is_an_empty_pool():
    assert parse("") == []
    assert not Pool([])


# -- each backend keeps its own provider -----------------------------------

def test_an_ollama_backend_keeps_the_native_provider():
    """The native API is what allows think:false and schema-constrained
    decoding; routing it through the OpenAI shim would lose both."""
    a, _ = parse(SPEC)
    cfg = _for_endpoint(Settings(), a)
    assert cfg.provider == "ollama"
    assert cfg.ollama_url == "http://host.docker.internal:11434"
    assert cfg.active_model == "command-r7b-arabic"


def test_an_lm_studio_backend_uses_the_openai_provider():
    _, b = parse(SPEC)
    cfg = _for_endpoint(Settings(), b)
    assert cfg.provider == "openai"
    assert cfg.openai_base_url == "http://192.168.1.7:1234/v1"
    assert cfg.openai_api_key                      # LM Studio wants some header


def test_lmstudio_is_accepted_as_a_kind():
    (e,) = parse("lmstudio@http://192.168.1.7:1234/v1#m")
    assert _for_endpoint(Settings(), e).provider == "openai"


# -- leasing ---------------------------------------------------------------

def test_two_workers_get_different_backends():
    pool = Pool(parse(SPEC))
    first, second = pool.acquire(), pool.acquire()
    assert first is not None and second is not None
    assert first is not second
    assert pool.acquire() is None            # both are busy
    pool.release(first)
    assert pool.acquire() is first           # and available again


def test_releasing_twice_is_harmless():
    pool = Pool(parse(SPEC))
    e = pool.acquire()
    pool.release(e)
    pool.release(e)
    assert pool.acquire() is not None


def test_a_sleeping_machine_is_taken_out_of_rotation():
    pool = Pool(parse(SPEC))
    a, b = pool.endpoints
    a.mark_down("connection refused")
    pool.release(pool.acquire())             # nothing leased
    assert not a.healthy() and b.healthy()

    leased = [pool.acquire() for _ in range(2)]
    assert leased[0] is b and leased[1] is None   # only the healthy one


def test_a_backend_comes_back_after_its_cooldown():
    e = Endpoint(kind="ollama", url="http://box:11434", name="box")
    e.mark_down("asleep")
    assert not e.healthy()
    e.down_until = time.monotonic() - 1       # pretend the cooldown elapsed
    assert e.healthy()

    e.mark_up()
    assert e.failures == 0 and e.down_until == 0.0


def test_status_reports_what_each_backend_is_doing():
    pool = Pool(parse(SPEC))
    leased = pool.acquire()
    rows = pool.status()
    assert len(rows) == 2
    busy = [r for r in rows if r["busy"]]
    assert len(busy) == 1 and busy[0]["name"] == leased.name
    assert all(r["healthy"] for r in rows)
