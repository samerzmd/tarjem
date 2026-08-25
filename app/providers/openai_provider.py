"""OpenAI-compatible backend.

Kept deliberately generic so the same code reaches OpenAI, a local Ollama or an
LM Studio server via ``OPENAI_BASE_URL``. There is no prompt caching to manage
here - the system prefix is simply resent each call.
"""
from __future__ import annotations

import json
import logging

import httpx

from .base import Provider, ProviderError, SystemBlock, Usage, T

log = logging.getLogger(__name__)


class OpenAIProvider(Provider):
    name = "openai"

    def __init__(self, api_key: str, base_url: str, model: str, timeout: float = 300.0):
        super().__init__()
        if not api_key:
            raise ProviderError("OPENAI_API_KEY is not set", retryable=False)
        self.model = model
        self.client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    def structured(
        self,
        system: list[SystemBlock],
        user: str,
        schema_model: type[T],
        max_tokens: int = 16000,
    ) -> T:
        schema = _strict_schema(schema_model)
        payload = {
            "model": self.model,
            "max_completion_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": "\n\n".join(b.text for b in system)},
                {"role": "user", "content": user},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema_model.__name__, "strict": True, "schema": schema},
            },
        }

        try:
            response = self.client.post("/chat/completions", json=payload)
        except httpx.RequestError as exc:
            raise ProviderError(f"connection error: {exc}", retryable=True) from exc

        if response.status_code >= 400:
            retryable = response.status_code == 429 or response.status_code >= 500
            retry_after = _retry_after(response)
            raise ProviderError(
                f"api error {response.status_code}: {response.text[:300]}",
                retryable=retryable,
                retry_after=retry_after,
            )

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
        content = (choice.get("message") or {}).get("content")
        if not content:
            raise ProviderError("empty response", retryable=True)

        try:
            return schema_model.model_validate(json.loads(content))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ProviderError(f"unparseable structured output: {exc}", retryable=True) from exc

    def close(self) -> None:
        self.client.close()


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
