"""Data models, enums, and type definitions for NexusAI-Client.

Leverages modern Python 3.14 type annotations and dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal


class ProviderType(StrEnum):
    """Supported AI Providers in NexusAI-Client."""

    DEEPSEEK = "deepseek"
    GEMINI_PRO = "gemini_pro"
    GEMINI_FREE = "gemini_free"
    MISTRAL = "mistral"
    NVIDIA = "nvidia"
    NVIDIA_FREE = "nvidia_free"
    OPENROUTER = "openrouter"
    OPENROUTER_FREE = "openrouter_free"


type MessageRole = Literal["system", "user", "assistant"]


@dataclass(slots=True, kw_only=True, frozen=True)
class ChatMessage:
    """Represents a single message in a conversation thread."""

    role: MessageRole
    content: str

    def to_dict(self) -> dict[str, str]:
        """Convert to standard OpenAI-style message dictionary."""
        return {"role": self.role, "content": self.content}


@dataclass(slots=True, kw_only=True, frozen=True)
class UsageInfo:
    """Token usage metrics returned by the AI provider."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(slots=True, kw_only=True)
class ModelPricing:
    """Pricing estimation per million tokens in USD."""

    prompt_per_million: float = 0.0
    completion_per_million: float = 0.0
    cache_read_per_million: float | None = None

    @property
    def is_free(self) -> bool:
        """Return True if both prompt and completion costs are zero."""
        return self.prompt_per_million == 0.0 and self.completion_per_million == 0.0

    def format_pricing(self) -> str:
        """Human-readable price representation."""
        if self.is_free:
            return "Gratuit ($0 / 1M)"
        cache_info = f" (Cache: ${self.cache_read_per_million:.2f}/1M)" if self.cache_read_per_million is not None else ""
        return f"Input: ${self.prompt_per_million:.2f}/1M | Output: ${self.completion_per_million:.2f}/1M{cache_info}"


@dataclass(slots=True, kw_only=True)
class ModelInfo:
    """Detailed model descriptor including availability, context window, and pricing."""

    id: str
    name: str
    provider: str
    is_free: bool = False
    context_length: int | None = None
    pricing: ModelPricing | None = None
    description: str | None = None
    raw_data: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        cost_str = self.pricing.format_pricing() if self.pricing else ("Gratuit" if self.is_free else "Non spécifié")
        ctx_str = f"{self.context_length // 1000}k" if self.context_length else "N/A"
        return f"[{self.provider.upper()}] {self.id} (Contexte: {ctx_str} tokens | {cost_str})"


@dataclass(slots=True, kw_only=True)
class AccountInfo:
    """Account status, budget balance, usage metrics, and rate limits for a provider."""

    provider: str
    total_balance: float | None = None
    granted_balance: float | None = None
    topped_up_balance: float | None = None
    total_usage: float | None = None
    currency: str = "USD"
    is_free_tier: bool = False
    rate_limit_info: str | None = None
    extra_details: dict[str, Any] = field(default_factory=dict)

    def format_summary(self) -> str:
        """Format a clear human-readable summary of the account budget and quota."""
        parts: list[str] = []
        if self.total_balance is not None:
            curr_symbol = "$" if self.currency == "USD" else (self.currency + " ")
            parts.append(f"Solde restant: {curr_symbol}{self.total_balance:.2f}")
            if self.granted_balance is not None and self.granted_balance > 0:
                parts.append(f"(Offert: {curr_symbol}{self.granted_balance:.2f})")
        elif self.is_free_tier:
            parts.append("Tier Gratuit Actif ($0.00 facturé)")

        if self.total_usage is not None and self.total_usage > 0:
            parts.append(f"Consommé: ${self.total_usage:.4f}")

        if self.rate_limit_info:
            parts.append(f"Quotas: {self.rate_limit_info}")

        return " | ".join(parts) if parts else "Informations de compte non disponibles"


@dataclass(slots=True, kw_only=True)
class AIResponse:
    """Standardized response object returned by all NexusAI providers."""

    text: str
    provider: str
    model: str
    usage: UsageInfo | None = None
    finish_reason: str | None = None
    raw_response: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.text
