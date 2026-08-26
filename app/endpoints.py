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


@dataclass
class Endpoint:
    kind: str                 # ollama | openai | anthropic
    url: str
    model: str = ""
    name: str = ""
    failures: int = 0
    down_until: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def healthy(self) -> bool:
        return time.monotonic() >= self.down_until

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

    def acquire(self) -> Endpoint | None:
        """Take an idle, healthy backend. None if every one is busy or down."""
        with self._lock:
            healthy = [e for e in self.endpoints if e.healthy()]
            if not healthy:
                return None
            for endpoint in healthy:
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

    def status(self) -> list[dict]:
        return [
            {
                "name": e.name,
                "kind": e.kind,
                "model": e.model,
                "busy": e.lock.locked(),
                "healthy": e.healthy(),
                "failures": e.failures,
                "down_for_s": max(0, round(e.down_until - time.monotonic())),
            }
            for e in self.endpoints
        ]
