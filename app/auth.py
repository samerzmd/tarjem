"""Authentication.

Two audiences, two mechanisms:

* **People** get a password and a signed session cookie, so the dashboard is
  usable without pasting a token into every URL.
* **Machines** - Bazarr's post-processing hook, curl, scripts - send the token
  as a header or query parameter. They cannot fill in a login form.

The cookie is signed with a key derived from the password, so changing the
password invalidates every session, and sessions survive a restart. Nothing is
stored server-side; the cookie carries only its own expiry.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import time

from fastapi import HTTPException, Request, status

from .config import Settings

log = logging.getLogger(__name__)

COOKIE = "tarjem_session"
_VERSION = "v1"


def secret_key(cfg: Settings) -> bytes:
    """Derive the signing key. An explicit AUTH_SECRET wins; otherwise the
    password itself seeds it, which ties session validity to the password."""
    if cfg.auth_secret:
        return hashlib.sha256(cfg.auth_secret.encode()).digest()
    return hashlib.sha256(f"tarjem-session|{cfg.password}".encode()).digest()


def issue(cfg: Settings) -> str:
    expires = int(time.time() + cfg.session_hours * 3600)
    payload = f"{_VERSION}|{expires}"
    sig = hmac.new(secret_key(cfg), payload.encode(), hashlib.sha256).digest()
    return f"{expires}.{base64.urlsafe_b64encode(sig).decode().rstrip('=')}"


def valid_cookie(cfg: Settings, value: str | None) -> bool:
    if not value or "." not in value:
        return False
    raw_expires, _, sig = value.partition(".")
    try:
        expires = int(raw_expires)
    except ValueError:
        return False
    if expires < time.time():
        return False
    expected = issue_at(cfg, expires)
    return hmac.compare_digest(sig, expected)


def issue_at(cfg: Settings, expires: int) -> str:
    payload = f"{_VERSION}|{expires}"
    sig = hmac.new(secret_key(cfg), payload.encode(), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(sig).decode().rstrip("=")


def password_ok(cfg: Settings, candidate: str) -> bool:
    """Compare in constant time - a plain != leaks length and prefix by timing."""
    if not cfg.password:
        return False
    return hmac.compare_digest(candidate.encode(), cfg.password.encode())


def token_ok(cfg: Settings, candidate: str | None) -> bool:
    if not (cfg.api_token and candidate):
        return False
    return hmac.compare_digest(candidate.encode(), cfg.api_token.encode())


def is_https(request: Request) -> bool:
    """Did this request reach us over HTTPS?

    Behind a tunnel or reverse proxy the connection to the app itself is plain
    http on loopback, so the only honest signal is the header the proxy sets.
    Cloudflare and nginx both send X-Forwarded-Proto.
    """
    forwarded = request.headers.get("x-forwarded-proto", "")
    if forwarded:
        return forwarded.split(",")[0].strip().lower() == "https"
    return request.url.scheme == "https"


def authenticated(cfg: Settings, request: Request) -> bool:
    """True when the caller proved who they are, by any accepted route."""
    if not cfg.auth_enabled:
        return True
    if token_ok(cfg, request.headers.get("x-api-token")):
        return True
    if token_ok(cfg, request.query_params.get("token")):
        return True
    return valid_cookie(cfg, request.cookies.get(COOKIE))


def make_dependency(cfg: Settings):
    """FastAPI dependency that 401s anything unauthenticated."""

    def require(request: Request) -> None:
        if not authenticated(cfg, request):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="not authenticated - sign in, or send x-api-token",
            )

    return require


def warn_if_open(cfg: Settings) -> None:
    if cfg.auth_enabled:
        return
    log.warning(
        "AUTH IS OFF - no AUTH_PASSWORD and no API_TOKEN. Anyone who can reach "
        "this port can queue jobs, including paid ones. Set AUTH_PASSWORD."
    )
