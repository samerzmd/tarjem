"""The OpenAI-compatible provider against servers that aren't OpenAI.

Local runtimes disagree about response_format, token parameter names, and
whether a reasoning model's scratchpad ends up in the content. All of that is
handled without failing a batch.
"""
import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.providers.base import ProviderError  # noqa: E402
from app.providers.openai_provider import OpenAIProvider, _json_payload  # noqa: E402
from app.translate import BatchTranslation  # noqa: E402

CUES = {"cues": [{"id": 0, "ar": "مرحبا"}, {"id": 1, "ar": "إلى اللقاء"}]}


def reply(content: str) -> dict:
    return {
        "choices": [{"finish_reason": "stop", "message": {"content": content}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50},
    }


def provider(handler) -> OpenAIProvider:
    p = OpenAIProvider(api_key="", base_url="http://ollama:11434/v1", model="qwen3:14b")
    p.client = httpx.Client(
        base_url="http://ollama:11434/v1", transport=httpx.MockTransport(handler)
    )
    return p


def call(p: OpenAIProvider) -> BatchTranslation:
    return p.structured([], "translate", BatchTranslation, max_tokens=1000)


# -- content extraction ----------------------------------------------------

def test_strips_a_reasoning_block():
    raw = '<think>Let me consider the register...</think>\n{"cues": []}'
    assert json.loads(_json_payload(raw)) == {"cues": []}


def test_strips_code_fences():
    assert json.loads(_json_payload('```json\n{"cues": []}\n```')) == {"cues": []}


def test_strips_conversational_preamble():
    raw = 'Sure! Here is the translation:\n{"cues": []}\nHope that helps.'
    assert json.loads(_json_payload(raw)) == {"cues": []}


def test_leaves_clean_json_alone():
    assert _json_payload('{"cues": []}') == '{"cues": []}'


# -- capability negotiation ------------------------------------------------

def test_falls_back_from_json_schema_to_json_object():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append((body.get("response_format") or {}).get("type"))
        if (body.get("response_format") or {}).get("type") == "json_schema":
            return httpx.Response(400, json={"error": "response_format.json_schema unsupported"})
        return httpx.Response(200, json=reply(json.dumps(CUES)))

    p = provider(handler)
    assert len(call(p).cues) == 2
    assert seen == ["json_schema", "json_object"]
    # The downgrade sticks, so the next batch doesn't pay for it again.
    call(p)
    assert seen[-1] == "json_object"


def test_falls_back_all_the_way_to_no_response_format():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if "response_format" in body:
            return httpx.Response(400, json={"error": "response_format is not supported"})
        return httpx.Response(200, json=reply(json.dumps(CUES)))

    p = provider(handler)
    assert len(call(p).cues) == 2
    assert p._json_mode == "none"


def test_switches_to_max_completion_tokens_when_told():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append("max_tokens" in body)
        if "max_tokens" in body:
            return httpx.Response(400, json={
                "error": "Unsupported parameter 'max_tokens'; use 'max_completion_tokens'"})
        return httpx.Response(200, json=reply(json.dumps(CUES)))

    p = provider(handler)
    assert len(call(p).cues) == 2
    assert p._token_param == "max_completion_tokens"
    assert seen == [True, False]


def test_a_thinking_model_that_wraps_its_answer_still_parses():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=reply(
            "<think>The speaker is angry, so...</think>\n```json\n" + json.dumps(CUES) + "\n```"))

    assert len(call(provider(handler)).cues) == 2


# -- errors ----------------------------------------------------------------

def test_truncated_response_is_retryable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "choices": [{"finish_reason": "length", "message": {"content": "{"}}]})

    with pytest.raises(ProviderError) as exc:
        call(provider(handler))
    assert exc.value.retryable


def test_server_error_is_retryable_but_bad_request_is_not():
    with pytest.raises(ProviderError) as exc:
        call(provider(lambda r: httpx.Response(503, text="overloaded")))
    assert exc.value.retryable

    with pytest.raises(ProviderError) as exc:
        call(provider(lambda r: httpx.Response(404, text="model 'qwen9' not found")))
    assert not exc.value.retryable


def test_extra_body_is_merged_into_the_request():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=reply(json.dumps(CUES)))

    p = provider(handler)
    p.extra_body = {"think": False, "temperature": 0.2}
    call(p)
    assert seen["think"] is False and seen["temperature"] == 0.2


def test_usage_is_accumulated():
    p = provider(lambda r: httpx.Response(200, json=reply(json.dumps(CUES))))
    call(p)
    call(p)
    assert p.usage.calls == 2
    assert p.usage.input_tokens == 200 and p.usage.output_tokens == 100


# -- shape coercion --------------------------------------------------------
# Only needed where the server cannot enforce a schema. LM Studio, for one,
# accepts json_schema or text and rejects json_object outright - so a model
# that fails constrained decoding ends up answering free-form.

def test_a_bare_list_is_accepted():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=reply(json.dumps(
            [{"id": 0, "ar": "مرحبا"}, {"id": 1, "ar": "وداعا"}])))

    assert len(call(provider(handler)).cues) == 2


def test_the_input_field_names_echoed_back_are_remapped():
    """A small model often replies with the shape it was given."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=reply(json.dumps(
            {"cues": [{"id": 0, "ms": 2100, "text": "مرحبا"}]})))

    out = call(provider(handler))
    assert out.cues[0].ar == "مرحبا" and out.cues[0].id == 0


def test_an_alternative_container_key_is_found():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=reply(json.dumps(
            {"translations": [{"id": 3, "arabic": "نعم"}]})))

    out = call(provider(handler))
    assert out.cues[0].id == 3 and out.cues[0].ar == "نعم"


def test_json_schema_falls_back_on_a_400_whatever_the_wording():
    """LM Studio says 'does not match the expected peg-native format' - no
    mention of schema or response_format, so keyword matching missed it."""
    modes = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        modes.append((body.get("response_format") or {}).get("type", "none"))
        if modes[-1] == "json_schema":
            return httpx.Response(400, json={
                "error": "Engine protocol predict stream returned an error: "
                         "the model produced output that does not match the "
                         "expected peg-native format"})
        return httpx.Response(200, json=reply(json.dumps(CUES)))

    assert len(call(provider(handler)).cues) == 2
    assert modes[0] == "json_schema" and modes[1] == "json_object"
