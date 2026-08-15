"""NexusAI-Client: A lightweight, unified asynchronous Python client for multi-provider AI APIs.

Supported Providers:
1. DeepSeek (Paid API)
2. Gemini Pro (Paid API)
3. Gemini Free (Via Google AI Studio)
4. Mistral (Free Tier / La Plateforme)
5. Nvidia NIM (Free Tier API)
6. OpenRouter (Free Tier)
"""

from __future__ import annotations

from nexusai_client.config import Config, ProviderConfig, ProviderDefaults
from nexusai_client.exceptions import (
    APIConnectionError,
    APIResponseError,
    APITimeoutError,
    AuthenticationError,
    ConfigurationError,
    InvalidModelError,
    MissingAPIKeyError,
    NetworkError,
    NexusAIError,
    ProviderError,
    ProviderNotFoundError,
    ProviderServerError,
    RateLimitError,
)
from nexusai_client.gateway import AIGateway, FallbackGateway
from nexusai_client.models import (
    AccountInfo,
    AIResponse,
    ChatMessage,
    MessageRole,
    ModelInfo,
    ModelPricing,
    ProviderType,
    StreamChunk,
    UsageInfo,
)
from nexusai_client.providers import (
    BaseAIProvider,
    CerebrasProvider,
    CohereProvider,
    DeepSeekProvider,
    GeminiBaseProvider,
    GeminiFreeProvider,
    GeminiProProvider,
    GroqProvider,
    MistralProvider,
    NvidiaProvider,
    OpenAICompatibleProvider,
    OpenRouterProvider,
)
from nexusai_client.utils import (
    load_image_as_base64_and_mime,
    load_image_as_data_uri,
)

__version__ = "0.1.0"

__all__ = [
    # Gateway & Factory
    "AIGateway",
    "FallbackGateway",
    # Enums & Models
    "ProviderType",
    "ChatMessage",
    "AIResponse",
    "StreamChunk",
    "UsageInfo",
    "MessageRole",
    "ModelInfo",
    "ModelPricing",
    "AccountInfo",
    # Providers
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
    # Configuration
    "Config",
    "ProviderConfig",
    "ProviderDefaults",
    # Image & Multimodal Utilities
    "load_image_as_base64_and_mime",
    "load_image_as_data_uri",
    # Exceptions
    "NexusAIError",
    "ConfigurationError",
    "MissingAPIKeyError",
    "ProviderError",
    "ProviderNotFoundError",
    "InvalidModelError",
    "NetworkError",
    "APIConnectionError",
    "APITimeoutError",
    "APIResponseError",
    "AuthenticationError",
    "RateLimitError",
    "ProviderServerError",
]
