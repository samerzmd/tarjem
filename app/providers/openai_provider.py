"""OpenAI-compatible backend.

Kept deliberately generic so the same code reaches OpenAI, Ollama, vLLM, LM
Studio or llama.cpp via ``OPENAI_BASE_URL``. "OpenAI-compatible" is a spectrum
rather than a spec, so the provider negotiates down on first contact instead of
assuming a dialect:

* ``response_format: json_schema`` -> ``json_object`` -> nothing, depending on
  what the server accepts.
* ``max_tokens`` <-> ``max_completion_tokens``.

Both results are remembered on the instance, so the negotiation costs at most
one extra request per run. There is no prompt caching to manage here - the
system prefix is simply resent each call.
"""
from __future__ import annotations

import json
import logging
import re

import httpx

from .base import Provider, ProviderError, SystemBlock, Usage, T

log = logging.getLogger(__name__)

THINK_RE = re.compile(r"<(think|thinking|reasoning)>.*?</\1>", re.DOTALL | re.IGNORECASE)
FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class OpenAIProvider(Provider):
    name = "openai"

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 1800.0,
        extra_body: dict | None = None,
    ):
        super().__init__()
        self.model = model
        self.extra_body = extra_body or {}
        # Local servers ignore the key but some reject a missing header outright.
        self.client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key or 'no-key'}"},
            timeout=timeout,
        )
        self._json_mode = "json_schema"      # -> "json_object" -> "none"
        self._token_param = "max_tokens"     # -> "max_completion_tokens"

    def structured(
        self,
        system: list[SystemBlock],
        user: str,
        schema_model: type[T],
        max_tokens: int = 16000,
    ) -> T:
        schema = _strict_schema(schema_model)
        messages = [
            {"role": "system", "content": "\n\n".join(b.text for b in system)},
            {"role": "user", "content": user},
        ]

        # Up to three attempts, each one dropping a capability the server rejected.
        for _ in range(3):
            payload = {"model": self.model, "messages": messages, **self.extra_body}
            payload[self._token_param] = max_tokens
            if self._json_mode == "json_schema":
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_model.__name__, "strict": True, "schema": schema,
                    },
                }
            elif self._json_mode == "json_object":
                payload["response_format"] = {"type": "json_object"}

            try:
                response = self.client.post("/chat/completions", json=payload)
            except httpx.RequestError as exc:
                self.unreachable = True
                raise ProviderError(f"connection error: {exc}", retryable=True) from exc

            if response.status_code < 400:
                return self._parse(response, schema_model)

            detail = response.text[:400]
            if response.status_code in (400, 404, 422) and self._degrade(detail):
                continue

            raise ProviderError(
                f"api error {response.status_code}: {detail}",
                retryable=response.status_code == 429 or response.status_code >= 500,
                retry_after=_retry_after(response),
            )

        raise ProviderError("server rejected every request shape we know", retryable=False)

    def _degrade(self, detail: str) -> bool:
        """Drop one unsupported feature. Returns False when there's nothing left.

        Deliberately not keyed on the wording of the error. Servers describe the
        same failure in wildly different terms - LM Studio answers a schema its
        grammar engine cannot satisfy with "the model produced output that does
        not match the expected peg-native format", which mentions neither the
        schema nor response_format. Matching on keywords meant a 400 that was
        obviously about structured output sailed straight past the fallback.
        """
        low = detail.lower()
        if "max_completion_tokens" in low and self._token_param == "max_tokens":
            log.info("server wants max_completion_tokens; switching")
            self._token_param = "max_completion_tokens"
            return True
        if self._json_mode == "json_schema":
            log.warning("json_schema rejected (%s); falling back to json_object",
                        detail[:120].replace("\n", " "))
            self._json_mode = "json_object"
            return True
        if self._json_mode == "json_object":
            log.warning("json_object rejected too; relying on prompt instructions alone")
            self._json_mode = "none"
            return True
        return False

    def _parse(self, response: httpx.Response, schema_model: type[T]) -> T:
        body = response.json()
        usage = body.get("usage") or {}
        self.usage.add(
            Usage(
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
                calls=1,
            )
        )

        choice = (body.get("choices") or [{}])[0]
        if choice.get("finish_reason") == "length":
            raise ProviderError("hit token limit - batch too large", retryable=True)

        content = (choice.get("message") or {}).get("content") or ""
        text = _json_payload(content)
        if not text:
            raise ProviderError("empty response", retryable=True)

        try:
            return schema_model.model_validate(_coerce(_loads(text)))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ProviderError(
                f"unparseable structured output: {exc} | {text[:200]}", retryable=True
            ) from exc

    def close(self) -> None:
        self.client.close()


ALIASES = ("ar", "arabic", "translation", "translated", "text", "value")


def _loads(text: str):
    """Parse the first JSON value and ignore whatever follows it.

    An unconstrained model often adds a closing remark, or a second copy of the
    answer, after the JSON. json.loads rejects the lot with "Extra data"; the
    translation sitting in front of it is perfectly good.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        value, _ = json.JSONDecoder().raw_decode(text)
        return value


def _coerce(data):
    """Nudge a near-miss into the shape the schema wants.

    Only needed where the server could not enforce a schema. Left to itself a
    small model will return a bare list, or echo the input's field names back -
    the translation is right there and correct, and rejecting the batch over the
    key it arrived under would be a waste.
    """
    if isinstance(data, list):
        data = {"cues": data}
    if not isinstance(data, dict):
        return data

    cues = data.get("cues")
    if cues is None:
        for key in ("items", "results", "translations", "subtitles", "lines"):
            if isinstance(data.get(key), list):
                cues, data = data[key], {**data, "cues": data[key]}
                break
    if not isinstance(cues, list):
        return data

    fixed = []
    for entry in cues:
        if not isinstance(entry, dict):
            continue
        if "ar" in entry and isinstance(entry["ar"], str):
            fixed.append({"id": entry.get("id"), "ar": entry["ar"]})
            continue
        for key in ALIASES:
            if isinstance(entry.get(key), str):
                fixed.append({"id": entry.get("id"), "ar": entry[key]})
                break
    return {**data, "cues": fixed}


def _json_payload(content: str) -> str:
    """Dig the JSON object out of whatever a local model wrapped it in.

    Reasoning models leak <think> blocks, chat-tuned models add ```json fences
    and a sentence of preamble. None of that survives json.loads, and none of it
    is worth failing a batch over.
    """
    text = THINK_RE.sub("", content).strip()
    fenced = FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    if text[:1] in ("{", "["):
        return text
    # A model may answer with either an object or a bare array, so take
    # whichever opens first rather than assuming an object and slicing an
    # array down to its first element.
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if not starts:
        return text
    start = min(starts)
    end = text.rfind("}" if text[start] == "{" else "]")
    return text[start:end + 1] if end > start else text


def _strict_schema(model_cls) -> dict:
    """Pydantic emits $defs/$ref; strict json_schema mode wants them inlined."""
    schema = model_cls.model_json_schema()
    defs = schema.pop("$defs", {})

    def inline(node):
        if isinstance(node, dict):
            if "$ref" in node:
                name = node["$ref"].rsplit("/", 1)[-1]
                return inline(dict(defs.get(name, {})))
            node = {k: inline(v) for k, v in node.items()}
            if node.get("type") == "object":
                node.setdefault("additionalProperties", False)
                node.setdefault("required", list((node.get("properties") or {}).keys()))
            return node
        if isinstance(node, list):
            return [inline(v) for v in node]
        return node

    return inline(schema)


def _retry_after(response: httpx.Response) -> float | None:
    try:
        return float(response.headers.get("retry-after", ""))
    except (TypeError, ValueError):
        return None
