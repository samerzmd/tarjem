"""Provider interface: a structured-JSON completion with a cacheable system prefix."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@dataclass
class SystemBlock:
    """A slice of the system prompt. ``cache=True`` marks a caching breakpoint."""
    text: str
    cache: bool = False


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    calls: int = 0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.cache_write_tokens += other.cache_write_tokens
        self.calls += other.calls

    def as_dict(self) -> dict:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
        }


class ProviderError(RuntimeError):
    """Raised for a failure the caller may retry."""

    def __init__(self, message: str, retryable: bool = True, retry_after: float | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.retry_after = retry_after


class Provider(ABC):
    name: str = "base"

    def __init__(self) -> None:
        self.usage = Usage()
        # Set when the backend could not be reached at all, as opposed to
        # answering with an error. The endpoint pool uses it to park a machine
        # that has gone to sleep instead of failing every job sent to it.
        self.unreachable = False

    @abstractmethod
    def structured(
        self,
        system: list[SystemBlock],
        user: str,
        schema_model: type[T],
        max_tokens: int = 16000,
    ) -> T:
        """Return an instance of ``schema_model`` parsed from the model's reply."""

    def close(self) -> None:  # pragma: no cover - most providers need nothing
        pass
