"""Claude backend.

Two things matter for this workload and both are handled here:

* **Structured output** - ``messages.parse`` with a Pydantic ``output_format``
  guarantees the reply is a valid object, so a batch can never come back as
  prose that silently corrupts a subtitle file.
* **Prompt caching** - the translation rules and the per-title glossary are
  identical for every batch in a file. Marking them as a cache breakpoint turns
  the bulk of each request into a cache read.
"""
from __future__ import annotations

import logging

import anthropic
from .base import Provider, ProviderError, SystemBlock, Usage, T

log = logging.getLogger(__name__)


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, api_key: str = "", model: str = "claude-opus-5", effort: str = "low"):
        super().__init__()
        # An empty key is not the same as no credentials: the SDK also resolves
        # ANTHROPIC_AUTH_TOKEN and `ant auth login` profiles.
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self.model = model
        self.effort = effort

    def structured(
        self,
        system: list[SystemBlock],
        user: str,
        schema_model: type[T],
        max_tokens: int = 16000,
    ) -> T:
        blocks: list[dict] = []
        for block in system:
            entry: dict = {"type": "text", "text": block.text}
            if block.cache:
                entry["cache_control"] = {"type": "ephemeral"}
            blocks.append(entry)

        try:
            response = self.client.messages.parse(
                model=self.model,
                max_tokens=max_tokens,
                system=blocks,
                output_config={"effort": self.effort},
                messages=[{"role": "user", "content": user}],
                output_format=schema_model,
            )
        except anthropic.RateLimitError as exc:
            retry_after = _retry_after(exc)
            raise ProviderError(f"rate limited: {exc}", retryable=True, retry_after=retry_after) from exc
        except anthropic.APIConnectionError as exc:
            raise ProviderError(f"connection error: {exc}", retryable=True) from exc
        except anthropic.APIStatusError as exc:
            raise ProviderError(
                f"api error {exc.status_code}: {exc.message}",
                retryable=exc.status_code >= 500 or exc.status_code == 429,
            ) from exc

        self._record(response)

        if response.stop_reason == "refusal":
            detail = getattr(response.stop_details, "category", None)
            raise ProviderError(f"model declined this batch ({detail})", retryable=False)
        if response.stop_reason == "max_tokens":
            raise ProviderError("hit max_tokens - batch too large", retryable=True)

        parsed = response.parsed_output
        if parsed is None:
            raise ProviderError("no structured output in response", retryable=True)
        return parsed

    def _record(self, response) -> None:
        u = getattr(response, "usage", None)
        if not u:
            return
        self.usage.add(
            Usage(
                input_tokens=getattr(u, "input_tokens", 0) or 0,
                output_tokens=getattr(u, "output_tokens", 0) or 0,
                cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
                cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
                calls=1,
            )
        )


def _retry_after(exc: anthropic.RateLimitError) -> float | None:
    try:
        return float(exc.response.headers.get("retry-after", ""))
    except (AttributeError, TypeError, ValueError):
        return None
