from __future__ import annotations

from ..config import Settings
from .base import Provider, ProviderError, SystemBlock, Usage


def build_provider(cfg: Settings) -> Provider:
    if cfg.provider == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider(api_key=cfg.anthropic_api_key, model=cfg.model, effort=cfg.effort)
    if cfg.provider in ("openai", "openai-compatible"):
        from .openai_provider import OpenAIProvider

        return OpenAIProvider(
            api_key=cfg.openai_api_key, base_url=cfg.openai_base_url, model=cfg.openai_model
        )
    raise ProviderError(f"unknown LLM_PROVIDER: {cfg.provider}", retryable=False)


__all__ = ["Provider", "ProviderError", "SystemBlock", "Usage", "build_provider"]
