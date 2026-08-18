"""NexusAI-Client: A lightweight, unified asynchronous Python client for multi-provider AI APIs.

Supported Providers:
1. Cerebras (Free Tier / CS-3 Wafer-Scale)
2. Cohere (Free Trial / V2 REST)
3. DeepSeek (Paid API)
4. Gemini Pro (Paid API)
5. Gemini Free (Via Google AI Studio)
6. Groq (Free Tier / LPU)
7. Mistral (Free Tier / La Plateforme)
8. Nvidia NIM (Free Tier API)
9. OpenRouter (Free & Paid Hub)
10. OrcaRouter (Free Tier & Zero-Margin Multi-provider Gateway)
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
    FunctionDefinition,
    MessageRole,
    ModelInfo,
    ModelPricing,
    ProviderType,
    StreamChunk,
    ToolCall,
    ToolDefinition,
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
    OrcaRouterProvider,
)
from nexusai_client.utils import (
    load_image_as_base64_and_mime,
    load_image_as_data_uri,
)

__version__ = "0.4.3"

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
    "ToolCall",
    "ToolDefinition",
    "FunctionDefinition",
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
    "OrcaRouterProvider",
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
