"""Ollama's native /api/chat.

Ollama does expose an OpenAI-compatible /v1 surface, and the generic provider
reaches it, but two things only work here:

* ``think: false`` genuinely disables a reasoning model's scratchpad. On /v1 the
  field is ignored, and a model like qwen3 spends most of its output budget
  thinking before it starts translating - measured at roughly ten times the work
  for the same result.
* ``format`` takes a JSON schema and constrains decoding, which is stricter than
  asking for JSON in the prompt and never needs a repair pass.

``num_ctx`` matters more than it looks. Ollama's default context is small enough
that a full batch plus its glossary can overflow it, and an overflow is silent -
the prompt is truncated and cues come back missing. It is set explicitly here.
"""
from __future__ import annotations

import json
import logging

import httpx

from .base import Provider, ProviderError, SystemBlock, Usage, T

log = logging.getLogger(__name__)


class OllamaProvider(Provider):
    name = "ollama"

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 1800.0,
        num_ctx: int = 8192,
        options: dict | None = None,
    ):
        super().__init__()
        self.model = model
        # Translation wants fidelity, not invention. Low temperature measurably
        # reduces the paraphrasing that small models drift into.
        self.options = {"temperature": 0.2, "num_ctx": num_ctx, **(options or {})}
        # /v1 belongs to the OpenAI-compatible provider; strip it if it was passed.
        self.client = httpx.Client(
            base_url=base_url.rstrip("/").removesuffix("/v1"), timeout=timeout
        )
        # Only reasoning models accept `think`; Ollama rejects it on the rest.
        # Assume it is wanted and drop it the first time a server objects.
        self._send_think = True

    def structured(
        self,
        system: list[SystemBlock],
        user: str,
        schema_model: type[T],
        max_tokens: int = 16000,
    ) -> T:
        payload = {
            "model": self.model,
            "stream": False,
            "format": schema_model.model_json_schema(),
            "options": {**self.options, "num_predict": max_tokens},
            "messages": [
                {"role": "system", "content": "\n\n".join(b.text for b in system)},
                {"role": "user", "content": user},
            ],
        }

        for _ in range(2):
            if self._send_think:
                payload["think"] = False
            else:
                payload.pop("think", None)

            try:
                response = self.client.post("/api/chat", json=payload)
            except httpx.RequestError as exc:
                raise ProviderError(f"connection error: {exc}", retryable=True) from exc

            if response.status_code < 400:
                break

            detail = response.text[:400]
            if self._send_think and "think" in detail.lower():
                log.info("%s does not take `think`; dropping it", self.model)
                self._send_think = False
                continue

            raise ProviderError(
                f"ollama error {response.status_code}: {detail}",
                # A missing model or a bad schema will never succeed on retry.
                retryable=response.status_code >= 500,
            )

        body = response.json()
        self.usage.add(
            Usage(
                input_tokens=body.get("prompt_eval_count", 0) or 0,
                output_tokens=body.get("eval_count", 0) or 0,
                calls=1,
            )
        )

        content = (body.get("message") or {}).get("content") or ""
        if not content.strip():
            raise ProviderError("empty response", retryable=True)

        # A truncated generation yields valid-looking but incomplete JSON.
        if body.get("done_reason") == "length":
            raise ProviderError("hit num_predict - batch too large", retryable=True)

        try:
            return schema_model.model_validate(json.loads(content))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ProviderError(
                f"unparseable output: {exc} | {content[:200]}", retryable=True
            ) from exc

    def close(self) -> None:
        self.client.close()
