"""Auth: a password for people, a token for machines, nothing for strangers."""
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import auth  # noqa: E402
from app.config import Settings  # noqa: E402


@pytest.fixture
def cfg():
    return Settings(auth_password="hunter2", api_token="machine-token", auth_secret="")


# -- cookies ---------------------------------------------------------------

def test_a_freshly_issued_cookie_is_accepted(cfg):
    assert auth.valid_cookie(cfg, auth.issue(cfg))


@pytest.mark.parametrize("bad", [
    "", None, "garbage", "notanumber.sig", "9999999999.wrongsignature",
])
def test_malformed_cookies_are_rejected(cfg, bad):
    assert not auth.valid_cookie(cfg, bad)


def test_an_expired_cookie_is_rejected(cfg):
    past = int(time.time()) - 10
    assert not auth.valid_cookie(cfg, f"{past}.{auth.issue_at(cfg, past)}")


def test_the_expiry_cannot_be_extended_without_the_key(cfg):
    """The signature covers the expiry, so editing it invalidates the cookie."""
    value = auth.issue(cfg)
    _, _, sig = value.partition(".")
    forged = f"{int(time.time()) + 10 ** 7}.{sig}"
    assert not auth.valid_cookie(cfg, forged)


def test_changing_the_password_signs_everyone_out(cfg):
    cookie = auth.issue(cfg)
    assert not auth.valid_cookie(Settings(auth_password="different"), cookie)


def test_an_explicit_secret_survives_a_password_change():
    a = Settings(auth_password="one", auth_secret="fixed-secret")
    b = Settings(auth_password="two", auth_secret="fixed-secret")
    assert auth.valid_cookie(b, auth.issue(a))


# -- credentials -----------------------------------------------------------

def test_password_check(cfg):
    assert auth.password_ok(cfg, "hunter2")
    assert not auth.password_ok(cfg, "hunter3")
    assert not auth.password_ok(cfg, "")


def test_no_password_configured_means_no_password_works():
    assert not auth.password_ok(Settings(), "anything")
    assert not auth.password_ok(Settings(), "")


def test_api_token_falls_back_to_being_the_password():
    """An existing install that only set API_TOKEN still gets a usable login."""
    cfg = Settings(api_token="only-a-token")
    assert cfg.password == "only-a-token"
    assert auth.password_ok(cfg, "only-a-token")


def test_token_check(cfg):
    assert auth.token_ok(cfg, "machine-token")
    assert not auth.token_ok(cfg, "guess")
    assert not auth.token_ok(cfg, None)
    assert not auth.token_ok(Settings(), "anything")


def test_auth_is_off_only_when_nothing_is_configured():
    assert not Settings().auth_enabled
    assert Settings(api_token="t").auth_enabled
    assert Settings(auth_password="p").auth_enabled


# -- https detection -------------------------------------------------------
# Behind a tunnel the app itself is plain http on loopback, so forcing Secure
# cookies on would break LAN access while looking like it had worked.

class _Req:
    def __init__(self, headers=None, scheme="http"):
        self.headers = headers or {}
        self.url = type("U", (), {"scheme": scheme})()


def test_forwarded_proto_is_believed():
    assert auth.is_https(_Req({"x-forwarded-proto": "https"}))
    assert not auth.is_https(_Req({"x-forwarded-proto": "http"}))


def test_a_forwarded_proto_chain_uses_the_first_hop():
    assert auth.is_https(_Req({"x-forwarded-proto": "https, http"}))


def test_falls_back_to_the_connection_scheme():
    assert auth.is_https(_Req(scheme="https"))
    assert not auth.is_https(_Req(scheme="http"))
