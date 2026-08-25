"""Ollama's native endpoint. The details here are the ones that bite in practice:
thinking must be off, the schema must constrain decoding, and the context window
must be set explicitly or long prompts are silently truncated.
"""
import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.providers.base import ProviderError, SystemBlock  # noqa: E402
from app.providers.ollama_provider import OllamaProvider  # noqa: E402
from app.translate import BatchTranslation  # noqa: E402

CUES = {"cues": [{"id": 0, "ar": "مرحبا"}, {"id": 1, "ar": "إلى اللقاء"}]}


def reply(content: str, **extra) -> dict:
    return {"message": {"content": content}, "prompt_eval_count": 300,
            "eval_count": 120, "done_reason": "stop", **extra}


def provider(handler, **kw) -> OllamaProvider:
    p = OllamaProvider(base_url="http://ollama:11434", model="qwen3:14b", **kw)
    p.client = httpx.Client(base_url="http://ollama:11434",
                            transport=httpx.MockTransport(handler))
    return p


def call(p: OllamaProvider) -> BatchTranslation:
    return p.structured([SystemBlock("rules")], "translate", BatchTranslation, max_tokens=4000)


def test_thinking_is_disabled_and_the_schema_is_sent():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=reply(json.dumps(CUES)))

    assert len(call(provider(handler)).cues) == 2
    # Without this a reasoning model spends its output budget thinking; on the
    # /v1 endpoint the field is ignored entirely, which is why this provider exists.
    assert seen["think"] is False
    assert seen["stream"] is False
    assert seen["format"]["properties"]["cues"]["type"] == "array"


def test_context_window_is_always_explicit():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=reply(json.dumps(CUES)))

    call(provider(handler, num_ctx=16384))
    assert seen["options"]["num_ctx"] == 16384
    assert seen["options"]["num_predict"] == 4000


def test_extra_options_merge_without_dropping_the_defaults():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=reply(json.dumps(CUES)))

    call(provider(handler, options={"temperature": 0.1, "top_p": 0.9}))
    assert seen["options"]["temperature"] == 0.1
    assert seen["options"]["top_p"] == 0.9
    assert "num_ctx" in seen["options"]


def test_a_v1_suffix_in_the_url_is_stripped():
    p = OllamaProvider(base_url="http://ollama:11434/v1", model="m")
    assert str(p.client.base_url).rstrip("/") == "http://ollama:11434"
    p.close()


def test_truncated_generation_is_reported_not_silently_accepted():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=reply(json.dumps(CUES), done_reason="length"))

    with pytest.raises(ProviderError) as exc:
        call(provider(handler))
    assert "num_predict" in str(exc.value) and exc.value.retryable


def test_missing_model_is_not_retried():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text='{"error":"model \'qwen9\' not found"}')

    with pytest.raises(ProviderError) as exc:
        call(provider(handler))
    assert not exc.value.retryable


def test_server_error_is_retryable():
    with pytest.raises(ProviderError) as exc:
        call(provider(lambda r: httpx.Response(500, text="out of memory")))
    assert exc.value.retryable


def test_usage_comes_from_ollamas_own_counters():
    p = provider(lambda r: httpx.Response(200, json=reply(json.dumps(CUES))))
    call(p)
    assert p.usage.input_tokens == 300 and p.usage.output_tokens == 120


def test_empty_and_unparseable_responses_are_retryable():
    with pytest.raises(ProviderError) as exc:
        call(provider(lambda r: httpx.Response(200, json=reply("   "))))
    assert exc.value.retryable

    with pytest.raises(ProviderError) as exc:
        call(provider(lambda r: httpx.Response(200, json=reply("not json at all"))))
    assert exc.value.retryable
