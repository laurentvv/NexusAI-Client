"""NexusAI-Client Providers Package."""

from __future__ import annotations

from nexusai_client.providers.base import BaseAIProvider
from nexusai_client.providers.deepseek import DeepSeekProvider
from nexusai_client.providers.gemini import (
    GeminiBaseProvider,
    GeminiFreeProvider,
    GeminiProProvider,
)
from nexusai_client.providers.mistral import MistralProvider
from nexusai_client.providers.nvidia import NvidiaProvider
from nexusai_client.providers.openai_compat import OpenAICompatibleProvider
from nexusai_client.providers.openrouter import OpenRouterProvider

__all__ = [
    "BaseAIProvider",
    "OpenAICompatibleProvider",
    "DeepSeekProvider",
    "GeminiBaseProvider",
    "GeminiFreeProvider",
    "GeminiProProvider",
    "MistralProvider",
    "NvidiaProvider",
    "OpenRouterProvider",
]
