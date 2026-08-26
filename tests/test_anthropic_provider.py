"""Claude backend. The interesting behaviour is what it does when a model
rejects part of the request - the model line-up changes, a hardcoded list rots.
"""
import sys
from pathlib import Path

import anthropic
import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.providers.anthropic_provider import AnthropicProvider  # noqa: E402
from app.providers.base import ProviderError, SystemBlock  # noqa: E402
from app.translate import BatchTranslation  # noqa: E402


class FakeParse:
    """Stands in for client.messages.parse, recording each call."""

    def __init__(self, reject=(), parsed=None):
        self.reject = reject      # substrings that trigger a 400 if present
        self.calls = []
        self.parsed = parsed or BatchTranslation.model_validate(
            {"cues": [{"id": 0, "ar": "مرحبا"}]}
        )

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        for bad in self.reject:
            if bad in kwargs:
                raise anthropic.BadRequestError(
                    message=f"{bad} is not supported for this model",
                    response=httpx.Response(400, request=httpx.Request("POST", "http://x")),
                    body=None,
                )
        return type("R", (), {
            "parsed_output": self.parsed, "stop_reason": "end_turn",
            "stop_details": None,
            "usage": type("U", (), {
                "input_tokens": 100, "output_tokens": 50,
                "cache_read_input_tokens": 10, "cache_creation_input_tokens": 5,
            })(),
        })()


def provider(fake, **kw) -> AnthropicProvider:
    p = AnthropicProvider(api_key="test", **kw)
    p.client = type("C", (), {"messages": type("M", (), {"parse": fake})()})()
    return p


def call(p):
    return p.structured([SystemBlock("rules")], "translate", BatchTranslation, max_tokens=4000)


def test_effort_and_disabled_thinking_are_sent_when_accepted():
    fake = FakeParse()
    p = provider(fake, effort="low", thinking="disabled")
    call(p)
    assert fake.calls[0]["output_config"] == {"effort": "low"}
    assert fake.calls[0]["thinking"] == {"type": "disabled"}


def test_adaptive_thinking_sends_no_thinking_field():
    """Omitting it is the documented way to leave thinking at the default."""
    fake = FakeParse()
    call(provider(fake, thinking="adaptive"))
    assert "thinking" not in fake.calls[0]


def test_effort_is_dropped_for_a_model_that_rejects_it():
    """Haiku 4.5 and Sonnet 4.5 error on output_config.effort."""
    fake = FakeParse(reject=("output_config",))
    p = provider(fake, effort="low", thinking="adaptive")
    assert len(call(p).cues) == 1
    assert "output_config" in fake.calls[0]
    assert "output_config" not in fake.calls[1]
    assert p._send_effort is False

    call(p)                       # the downgrade sticks
    assert "output_config" not in fake.calls[-1]


def test_thinking_is_dropped_for_a_model_that_rejects_it():
    fake = FakeParse(reject=("thinking",))
    p = provider(fake, thinking="disabled")
    assert len(call(p).cues) == 1
    assert p._send_thinking is False


def test_a_model_rejecting_both_still_succeeds():
    fake = FakeParse(reject=("output_config", "thinking"))
    p = provider(fake, effort="low", thinking="disabled")
    assert len(call(p).cues) == 1
    assert "output_config" not in fake.calls[-1] and "thinking" not in fake.calls[-1]


def test_an_unrelated_bad_request_is_not_swallowed():
    fake = FakeParse(reject=("max_tokens",))
    with pytest.raises(ProviderError) as exc:
        call(provider(fake))
    assert "400" in str(exc.value) and not exc.value.retryable


def test_usage_is_recorded_including_cache():
    fake = FakeParse()
    p = provider(fake)
    call(p)
    assert p.usage.input_tokens == 100 and p.usage.output_tokens == 50
    assert p.usage.cache_read_tokens == 10 and p.usage.cache_write_tokens == 5
