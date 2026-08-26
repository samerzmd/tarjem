"""A pool of LLM backends, so more than one machine can share the work.

The point is not routing for its own sake - sending the same sequential batches
to two GPUs in turn is no faster than sending them to one. The point is running
several jobs at once, each pinned to a backend of its own.

Configured as a comma-separated list, each entry ``kind@url#model``:

    LLM_ENDPOINTS=ollama@http://host.docker.internal:11434#command-r7b-arabic,
                  openai@http://192.168.1.7:1234/v1#command-r7b-arabic

``kind`` picks the provider, so an Ollama host keeps the native API (think off,
schema-constrained decoding) while an LM Studio box gets the OpenAI-compatible
one. A backend that stops answering - a desktop that went to sleep - is taken
out of rotation and retried later rather than failing the jobs sent to it.
"""
from __future__ import annotations

import itertools
import logging
import threading
import time
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

DOWN_SECONDS = 180.0

# A job names a provider; a backend declares a kind. They are the same idea
# under two spellings, so asking for "ollama" has to match a "local" backend.
KINDS = {
    "ollama": "ollama", "local": "ollama",
    "openai": "openai", "openai-compatible": "openai", "lmstudio": "openai",
    "anthropic": "anthropic",
}


def normalise(kind: str) -> str:
    return KINDS.get((kind or "").strip().lower(), (kind or "").strip().lower())


@dataclass
class Endpoint:
    kind: str                 # ollama | openai | anthropic
    url: str
    model: str = ""
    name: str = ""
    # Turned off from the UI rather than removed - a gaming PC gets its GPU
    # back without losing the backend's configuration.
    enabled: bool = True
    failures: int = 0
    down_until: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def healthy(self) -> bool:
        return time.monotonic() >= self.down_until

    def usable(self) -> bool:
        return self.enabled and self.healthy()

    def mark_down(self, why: str) -> None:
        self.failures += 1
        self.down_until = time.monotonic() + DOWN_SECONDS
        log.warning("endpoint %s taken out of rotation for %.0fs: %s",
                    self.name, DOWN_SECONDS, why)

    def mark_up(self) -> None:
        if self.failures or self.down_until:
            log.info("endpoint %s is answering again", self.name)
        self.failures = 0
        self.down_until = 0.0


def parse(spec: str) -> list[Endpoint]:
    """Parse LLM_ENDPOINTS. Malformed entries are skipped, not fatal."""
    out: list[Endpoint] = []
    for raw in (s.strip() for s in spec.split(",")):
        if not raw:
            continue
        kind, _, rest = raw.partition("@")
        url, _, model = rest.partition("#")
        kind, url, model = kind.strip().lower(), url.strip(), model.strip()
        if not kind or not url:
            log.warning("ignoring endpoint %r: expected kind@url#model", raw)
            continue
        out.append(Endpoint(kind=kind, url=url, model=model,
                            name=f"{kind}@{url.split('//')[-1].split('/')[0]}"))
    return out


class Pool:
    """Hands a backend to each worker, skipping ones that are not answering.

    Endpoints are leased rather than merely chosen: two workers should not both
    be pointed at the same GPU while another sits idle.
    """

    def __init__(self, endpoints: list[Endpoint]):
        self.endpoints = endpoints
        self._lock = threading.Lock()
        self._turn = itertools.cycle(range(len(endpoints))) if endpoints else None

    def __bool__(self) -> bool:
        return bool(self.endpoints)

    def has_kind(self, kind: str) -> bool:
        want = normalise(kind)
        return any(normalise(e.kind) == want for e in self.endpoints)

    def available(self, kind: str = "") -> bool:
        """Is any backend usable and idle? Checked before claiming a job, so a
        worker does not claim work it would only have to put back."""
        want = normalise(kind) if kind else ""
        with self._lock:
            return any(e.usable() and not e.lock.locked()
                       and (not want or normalise(e.kind) == want)
                       for e in self.endpoints)

    def acquire(self, kind: str = "") -> Endpoint | None:
        """Take an idle, usable backend, optionally of one kind.

        A job that asks for "ollama" should still be spread across every ollama
        machine - naming a provider is about which software answers, not which
        box does the work.
        """
        want = normalise(kind) if kind else ""
        with self._lock:
            usable = [e for e in self.endpoints
                      if e.usable() and (not want or normalise(e.kind) == want)]
            if not usable:
                return None
            for endpoint in usable:
                if endpoint.lock.acquire(blocking=False):
                    return endpoint
        return None

    def release(self, endpoint: Endpoint | None) -> None:
        if endpoint is None:
            return
        try:
            endpoint.lock.release()
        except RuntimeError:      # already released
            pass

    def find(self, name: str) -> Endpoint | None:
        return next((e for e in self.endpoints if e.name == name), None)

    def add(self, endpoint: Endpoint) -> None:
        with self._lock:
            if not self.find(endpoint.name):
                self.endpoints.append(endpoint)

    def remove(self, name: str) -> bool:
        with self._lock:
            endpoint = self.find(name)
            if endpoint is None:
                return False
            self.endpoints.remove(endpoint)
            return True

    def set_enabled(self, name: str, enabled: bool) -> bool:
        """Turning one off stops new work reaching it. A job already running on
        it finishes rather than being thrown away."""
        endpoint = self.find(name)
        if endpoint is None:
            return False
        endpoint.enabled = enabled
        if enabled:
            endpoint.mark_up()
        log.info("backend %s %s", name, "enabled" if enabled else "disabled")
        return True

    def status(self) -> list[dict]:
        return [
            {
                "name": e.name,
                "kind": e.kind,
                "url": e.url,
                "model": e.model,
                "enabled": e.enabled,
                "busy": e.lock.locked(),
                "healthy": e.healthy(),
                "failures": e.failures,
                "down_for_s": max(0, round(e.down_until - time.monotonic())),
            }
            for e in self.endpoints
        ]
