from __future__ import annotations

from ..config import Settings
from ..endpoints import Endpoint
from .base import Provider, ProviderError, SystemBlock, Usage


def build_provider(cfg: Settings, endpoint: Endpoint | None = None) -> Provider:
    """Build a provider, optionally pinned to one backend from the pool."""
    if endpoint is not None:
        cfg = _for_endpoint(cfg, endpoint)

    if cfg.provider == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider(
            api_key=cfg.anthropic_api_key, model=cfg.model,
            effort=cfg.effort, thinking=cfg.thinking,
        )
    if cfg.provider in ("ollama", "local"):
        from .ollama_provider import OllamaProvider

        return OllamaProvider(
            base_url=cfg.ollama_url,
            model=cfg.ollama_model,
            timeout=float(cfg.llm_timeout),
            num_ctx=cfg.ollama_num_ctx,
            options=cfg.extra_body,
        )
    if cfg.provider in ("openai", "openai-compatible"):
        from .openai_provider import OpenAIProvider

        return OpenAIProvider(
            api_key=cfg.openai_api_key,
            base_url=cfg.openai_base_url,
            model=cfg.openai_model,
            timeout=float(cfg.llm_timeout),
            extra_body=cfg.extra_body,
        )
    raise ProviderError(f"unknown LLM_PROVIDER: {cfg.provider}", retryable=False)


__all__ = ["Provider", "ProviderError", "SystemBlock", "Usage", "build_provider"]


def _for_endpoint(cfg: Settings, endpoint: Endpoint) -> Settings:
    """A copy of the settings pointed at one specific backend."""
    from dataclasses import replace

    if endpoint.kind in ("ollama", "local"):
        return replace(cfg, provider="ollama", ollama_url=endpoint.url,
                       ollama_model=endpoint.model or cfg.ollama_model)
    if endpoint.kind in ("openai", "openai-compatible", "lmstudio"):
        return replace(cfg, provider="openai", openai_base_url=endpoint.url,
                       openai_model=endpoint.model or cfg.openai_model,
                       openai_api_key=cfg.openai_api_key or "local")
    if endpoint.kind == "anthropic":
        return replace(cfg, provider="anthropic",
                       model=endpoint.model or cfg.model)
    raise ProviderError(f"unknown endpoint kind: {endpoint.kind}", retryable=False)
