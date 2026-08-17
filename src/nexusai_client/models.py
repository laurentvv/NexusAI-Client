"""Data models, enums, and type definitions for NexusAI-Client.

Leverages modern Python 3.14 type annotations and dataclasses.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal


class ProviderType(StrEnum):
    """Supported AI Providers in NexusAI-Client."""

    CEREBRAS = "cerebras"
    CEREBRAS_FREE = "cerebras_free"
    COHERE = "cohere"
    COHERE_FREE = "cohere_free"
    DEEPSEEK = "deepseek"
    GEMINI_PRO = "gemini_pro"
    GEMINI_FREE = "gemini_free"
    GROQ = "groq"
    GROQ_FREE = "groq_free"
    MISTRAL = "mistral"
    NVIDIA = "nvidia"
    NVIDIA_FREE = "nvidia_free"
    OPENROUTER = "openrouter"
    OPENROUTER_FREE = "openrouter_free"
    ORCAROUTER = "orcarouter"
    ORCAROUTER_FREE = "orcarouter_free"


type MessageRole = Literal["system", "user", "assistant", "tool"]


@dataclass(slots=True, kw_only=True, frozen=True)
class ToolCall:
    """Represents a function or tool invocation requested by an AI model."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw_arguments: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to standard OpenAI-style tool_call dictionary."""
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.raw_arguments or json.dumps(self.arguments),
            },
        }


@dataclass(slots=True, kw_only=True)
class FunctionDefinition:
    """Definition of a callable function tool."""

    name: str
    description: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to standard OpenAI function dictionary."""
        d: dict[str, Any] = {"name": self.name}
        if self.description:
            d["description"] = self.description
        if self.parameters:
            d["parameters"] = self.parameters
        return d


@dataclass(slots=True, kw_only=True)
class ToolDefinition:
    """Top-level tool specification following the OpenAI JSON Schema standard."""

    type: Literal["function"] = "function"
    function: FunctionDefinition | dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to standard OpenAI tool object."""
        if isinstance(self.function, FunctionDefinition):
            fn_dict = self.function.to_dict()
        else:
            fn_dict = self.function
        return {"type": self.type, "function": fn_dict}


@dataclass(slots=True, kw_only=True, frozen=True)
class ChatMessage:
    """Represents a single message in a conversation thread."""

    role: MessageRole
    content: str = ""
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to standard OpenAI-style message dictionary."""
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name is not None:
            d["name"] = self.name
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            d["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        return d


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
        cache_info = (
            f" (Cache: ${self.cache_read_per_million:.2f}/1M)"
            if self.cache_read_per_million is not None
            else ""
        )
        return f"Input: ${self.prompt_per_million:.2f}/1M | Output: ${self.completion_per_million:.2f}/1M{cache_info}"

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate total USD cost for given token usage."""
        input_cost = (prompt_tokens / 1_000_000.0) * self.prompt_per_million
        output_cost = (completion_tokens / 1_000_000.0) * self.completion_per_million
        return input_cost + output_cost


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
        cost_str = (
            self.pricing.format_pricing()
            if self.pricing
            else ("Gratuit" if self.is_free else "Non spécifié")
        )
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
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_response: dict[str, Any] = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        """Return True if the model requested one or more tool/function calls."""
        return len(self.tool_calls) > 0

    def __str__(self) -> str:
        return self.text


@dataclass(slots=True, kw_only=True)
class StreamChunk:
    """A streaming text delta emitted by an active generation stream."""

    text: str
    provider: str
    model: str
    is_finished: bool = False
    finish_reason: str | None = None
