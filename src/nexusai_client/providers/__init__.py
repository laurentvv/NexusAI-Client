"""NexusAI-Client Providers Package."""

from __future__ import annotations

from nexusai_client.providers.base import BaseAIProvider
from nexusai_client.providers.cerebras import CerebrasProvider
from nexusai_client.providers.cohere import CohereProvider
from nexusai_client.providers.deepseek import DeepSeekProvider
from nexusai_client.providers.gemini import (
    GeminiBaseProvider,
    GeminiFreeProvider,
    GeminiProProvider,
)
from nexusai_client.providers.groq import GroqProvider
from nexusai_client.providers.mistral import MistralProvider
from nexusai_client.providers.nvidia import NvidiaProvider
from nexusai_client.providers.openai_compat import OpenAICompatibleProvider
from nexusai_client.providers.openrouter import OpenRouterProvider
from nexusai_client.providers.orcarouter import OrcaRouterProvider

__all__ = [
    "BaseAIProvider",
    "OpenAICompatibleProvider",
    "CerebrasProvider",
    "CohereProvider",
    "DeepSeekProvider",
    "GeminiBaseProvider",
    "GeminiFreeProvider",
    "GeminiProProvider",
    "GroqProvider",
    "MistralProvider",
    "NvidiaProvider",
    "OpenRouterProvider",
    "OrcaRouterProvider",
]
