"""Configuration loader and environment manager for NexusAI-Client.

Handles loading credentials and settings from .env using python-dotenv,
with fallback mechanisms, type validation, and sensible defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import ClassVar, Final

from dotenv import find_dotenv, load_dotenv

from nexusai_client.exceptions import MissingAPIKeyError
from nexusai_client.models import ProviderType

# Auto-load environment variables from the nearest .env file
load_dotenv(find_dotenv(usecwd=True))


class ProviderDefaults:
    """Default endpoints and models for each provider."""

    # DeepSeek
    DEEPSEEK_BASE_URL: Final[str] = "https://api.deepseek.com"
    DEEPSEEK_MODEL: Final[str] = "deepseek-chat"
    DEEPSEEK_ENV_KEY: Final[str] = "DEEPSEEK_API_KEY"

    # Gemini Pro (Paid API / Vertex / Google AI Studio)
    GEMINI_PRO_BASE_URL: Final[str] = "https://generativelanguage.googleapis.com/v1beta"
    GEMINI_PRO_MODEL: Final[str] = "gemini-2.5-pro"
    GEMINI_PRO_ENV_KEY: Final[str] = "GEMINI_PRO_API_KEY"

    # Gemini Free (Google AI Studio Free Tier)
    GEMINI_FREE_BASE_URL: Final[str] = "https://generativelanguage.googleapis.com/v1beta"
    GEMINI_FREE_MODEL: Final[str] = "gemini-2.5-flash"
    GEMINI_FREE_ENV_KEY: Final[str] = "GEMINI_FREE_API_KEY"
    GEMINI_FALLBACK_ENV_KEY: Final[str] = "GEMINI_API_KEY"

    # Mistral (Free tier / La Plateforme)
    MISTRAL_BASE_URL: Final[str] = "https://api.mistral.ai/v1"
    MISTRAL_MODEL: Final[str] = "mistral-small-latest"
    MISTRAL_ENV_KEY: Final[str] = "MISTRAL_API_KEY"

    # Nvidia (Free NIM API)
    NVIDIA_BASE_URL: Final[str] = "https://integrate.api.nvidia.com/v1"
    NVIDIA_MODEL: Final[str] = "meta/llama-3.1-8b-instruct"
    NVIDIA_ENV_KEY: Final[str] = "NVIDIA_API_KEY"

    # OpenRouter (Free tier models)
    OPENROUTER_BASE_URL: Final[str] = "https://openrouter.ai/api/v1"
    OPENROUTER_MODEL: Final[str] = "openrouter/free"
    OPENROUTER_ENV_KEY: Final[str] = "OPENROUTER_API_KEY"

    # Global
    DEFAULT_TIMEOUT: Final[float] = 60.0


@dataclass(slots=True, kw_only=True)
class ProviderConfig:
    """Resolved configuration for an AI Provider instance."""

    api_key: str
    base_url: str
    default_model: str
    timeout: float = 60.0
    extra_headers: dict[str, str] | None = None


class Config:
    """Centralized configuration manager for NexusAI-Client."""

    @staticmethod
    def reload_env() -> None:
        """Force reloads environment variables from the nearest .env file."""
        load_dotenv(find_dotenv(usecwd=True), override=True)

    @classmethod
    def get_api_key(
        cls,
        env_var: str,
        fallback_var: str | None = None,
        provider_name: str = "Unknown",
        required: bool = True,
    ) -> str:
        """Retrieve an API key from environment variables.

        Args:
            env_var: Primary environment variable name.
            fallback_var: Optional secondary environment variable name.
            provider_name: Human-friendly provider name for error messages.
            required: If True, raises MissingAPIKeyError if neither var is set.

        Returns:
            The resolved API key string.

        Raises:
            MissingAPIKeyError: If key is not found and required is True.
        """
        key = os.getenv(env_var) or (os.getenv(fallback_var) if fallback_var else None)
        if not key or not key.strip():
            if required:
                raise MissingAPIKeyError(provider=provider_name, env_var=env_var)
            return ""
        return key.strip()

    @classmethod
    def get_provider_config(
        cls,
        provider: ProviderType | str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        require_api_key: bool = True,
    ) -> ProviderConfig:
        """Resolve full configuration for a specific provider.

        User-provided arguments always take precedence over environment variables
        and default constants.
        """
        norm = str(provider).lower().replace("-", "_")
        env_timeout = os.getenv("NEXUS_DEFAULT_TIMEOUT")
        resolved_timeout = timeout if timeout is not None else (
            float(env_timeout) if env_timeout else ProviderDefaults.DEFAULT_TIMEOUT
        )

        match norm:
            case ProviderType.DEEPSEEK | "deepseek":
                resolved_key = api_key or cls.get_api_key(
                    ProviderDefaults.DEEPSEEK_ENV_KEY,
                    provider_name="DeepSeek",
                    required=require_api_key,
                )
                return ProviderConfig(
                    api_key=resolved_key,
                    base_url=base_url or os.getenv("DEEPSEEK_BASE_URL", ProviderDefaults.DEEPSEEK_BASE_URL),
                    default_model=model or os.getenv("DEEPSEEK_DEFAULT_MODEL", ProviderDefaults.DEEPSEEK_MODEL),
                    timeout=resolved_timeout,
                )

            case ProviderType.GEMINI_PRO | "gemini_pro" | "gemini-pro":
                resolved_key = api_key or cls.get_api_key(
                    ProviderDefaults.GEMINI_PRO_ENV_KEY,
                    fallback_var=ProviderDefaults.GEMINI_FALLBACK_ENV_KEY,
                    provider_name="Gemini Pro",
                    required=require_api_key,
                )
                return ProviderConfig(
                    api_key=resolved_key,
                    base_url=base_url or os.getenv("GEMINI_PRO_BASE_URL", ProviderDefaults.GEMINI_PRO_BASE_URL),
                    default_model=model or os.getenv("GEMINI_PRO_DEFAULT_MODEL", ProviderDefaults.GEMINI_PRO_MODEL),
                    timeout=resolved_timeout,
                )

            case ProviderType.GEMINI_FREE | "gemini_free" | "gemini-free" | "gemini":
                resolved_key = api_key or cls.get_api_key(
                    ProviderDefaults.GEMINI_FREE_ENV_KEY,
                    fallback_var=ProviderDefaults.GEMINI_FALLBACK_ENV_KEY,
                    provider_name="Gemini Free",
                    required=require_api_key,
                )
                return ProviderConfig(
                    api_key=resolved_key,
                    base_url=base_url or os.getenv("GEMINI_FREE_BASE_URL", ProviderDefaults.GEMINI_FREE_BASE_URL),
                    default_model=model or os.getenv("GEMINI_FREE_DEFAULT_MODEL", ProviderDefaults.GEMINI_FREE_MODEL),
                    timeout=resolved_timeout,
                )

            case ProviderType.MISTRAL | "mistral" | "mistral_free" | "mistral-free":
                resolved_key = api_key or cls.get_api_key(
                    ProviderDefaults.MISTRAL_ENV_KEY,
                    provider_name="Mistral",
                    required=require_api_key,
                )
                return ProviderConfig(
                    api_key=resolved_key,
                    base_url=base_url or os.getenv("MISTRAL_BASE_URL", ProviderDefaults.MISTRAL_BASE_URL),
                    default_model=model or os.getenv("MISTRAL_DEFAULT_MODEL", ProviderDefaults.MISTRAL_MODEL),
                    timeout=resolved_timeout,
                )

            case ProviderType.NVIDIA | ProviderType.NVIDIA_FREE | "nvidia" | "nvidia_free" | "nvidia-free" | "nvidia_nim" | "nvidia-nim":
                resolved_key = api_key or cls.get_api_key(
                    ProviderDefaults.NVIDIA_ENV_KEY,
                    provider_name="Nvidia",
                    required=require_api_key,
                )
                return ProviderConfig(
                    api_key=resolved_key,
                    base_url=base_url or os.getenv("NVIDIA_BASE_URL", ProviderDefaults.NVIDIA_BASE_URL),
                    default_model=model or os.getenv("NVIDIA_DEFAULT_MODEL", ProviderDefaults.NVIDIA_MODEL),
                    timeout=resolved_timeout,
                )

            case ProviderType.OPENROUTER | ProviderType.OPENROUTER_FREE | "openrouter" | "openrouter_free" | "openrouter-free":
                resolved_key = api_key or cls.get_api_key(
                    ProviderDefaults.OPENROUTER_ENV_KEY,
                    provider_name="OpenRouter",
                    required=require_api_key,
                )
                return ProviderConfig(
                    api_key=resolved_key,
                    base_url=base_url or os.getenv("OPENROUTER_BASE_URL", ProviderDefaults.OPENROUTER_BASE_URL),
                    default_model=model or os.getenv("OPENROUTER_DEFAULT_MODEL", ProviderDefaults.OPENROUTER_MODEL),
                    timeout=resolved_timeout,
                    extra_headers={
                        "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "https://github.com/NexusAI-Client"),
                        "X-Title": os.getenv("OPENROUTER_APP_NAME", "NexusAI-Client"),
                    },
                )

            case _:
                from nexusai_client.exceptions import ProviderNotFoundError
                raise ProviderNotFoundError(
                    provider=str(provider),
                    available_providers=[p.value for p in ProviderType],
                )
