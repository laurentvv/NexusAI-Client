"""Configuration loader and environment manager for NexusAI-Client.

Handles loading credentials and settings from .env using python-dotenv,
with fallback mechanisms, type validation, and sensible defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Final

from dotenv import find_dotenv, load_dotenv

from nexusai_client.exceptions import MissingAPIKeyError, ProviderNotFoundError
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


@dataclass(slots=True, kw_only=True, frozen=True)
class _ProviderMeta:
    name: str
    env_key: str
    default_base_url: str
    default_model: str
    fallback_env_key: str | None = None
    env_base_url_var: str | None = None
    env_model_var: str | None = None


_PROVIDER_META_REGISTRY: dict[str, _ProviderMeta] = {
    # DeepSeek
    "deepseek": _ProviderMeta(
        name="DeepSeek",
        env_key=ProviderDefaults.DEEPSEEK_ENV_KEY,
        default_base_url=ProviderDefaults.DEEPSEEK_BASE_URL,
        default_model=ProviderDefaults.DEEPSEEK_MODEL,
        env_base_url_var="DEEPSEEK_BASE_URL",
        env_model_var="DEEPSEEK_DEFAULT_MODEL",
    ),
    # Gemini Pro
    "gemini_pro": _ProviderMeta(
        name="Gemini Pro",
        env_key=ProviderDefaults.GEMINI_PRO_ENV_KEY,
        default_base_url=ProviderDefaults.GEMINI_PRO_BASE_URL,
        default_model=ProviderDefaults.GEMINI_PRO_MODEL,
        fallback_env_key=ProviderDefaults.GEMINI_FALLBACK_ENV_KEY,
        env_base_url_var="GEMINI_PRO_BASE_URL",
        env_model_var="GEMINI_PRO_DEFAULT_MODEL",
    ),
    # Gemini Free
    "gemini_free": _ProviderMeta(
        name="Gemini Free",
        env_key=ProviderDefaults.GEMINI_FREE_ENV_KEY,
        default_base_url=ProviderDefaults.GEMINI_FREE_BASE_URL,
        default_model=ProviderDefaults.GEMINI_FREE_MODEL,
        fallback_env_key=ProviderDefaults.GEMINI_FALLBACK_ENV_KEY,
        env_base_url_var="GEMINI_FREE_BASE_URL",
        env_model_var="GEMINI_FREE_DEFAULT_MODEL",
    ),
    "gemini": _ProviderMeta(
        name="Gemini Free",
        env_key=ProviderDefaults.GEMINI_FREE_ENV_KEY,
        default_base_url=ProviderDefaults.GEMINI_FREE_BASE_URL,
        default_model=ProviderDefaults.GEMINI_FREE_MODEL,
        fallback_env_key=ProviderDefaults.GEMINI_FALLBACK_ENV_KEY,
        env_base_url_var="GEMINI_FREE_BASE_URL",
        env_model_var="GEMINI_FREE_DEFAULT_MODEL",
    ),
    # Mistral
    "mistral": _ProviderMeta(
        name="Mistral",
        env_key=ProviderDefaults.MISTRAL_ENV_KEY,
        default_base_url=ProviderDefaults.MISTRAL_BASE_URL,
        default_model=ProviderDefaults.MISTRAL_MODEL,
        env_base_url_var="MISTRAL_BASE_URL",
        env_model_var="MISTRAL_DEFAULT_MODEL",
    ),
    "mistral_free": _ProviderMeta(
        name="Mistral",
        env_key=ProviderDefaults.MISTRAL_ENV_KEY,
        default_base_url=ProviderDefaults.MISTRAL_BASE_URL,
        default_model=ProviderDefaults.MISTRAL_MODEL,
        env_base_url_var="MISTRAL_BASE_URL",
        env_model_var="MISTRAL_DEFAULT_MODEL",
    ),
    # Nvidia NIM
    "nvidia": _ProviderMeta(
        name="Nvidia",
        env_key=ProviderDefaults.NVIDIA_ENV_KEY,
        default_base_url=ProviderDefaults.NVIDIA_BASE_URL,
        default_model=ProviderDefaults.NVIDIA_MODEL,
        env_base_url_var="NVIDIA_BASE_URL",
        env_model_var="NVIDIA_DEFAULT_MODEL",
    ),
    "nvidia_free": _ProviderMeta(
        name="Nvidia",
        env_key=ProviderDefaults.NVIDIA_ENV_KEY,
        default_base_url=ProviderDefaults.NVIDIA_BASE_URL,
        default_model=ProviderDefaults.NVIDIA_MODEL,
        env_base_url_var="NVIDIA_BASE_URL",
        env_model_var="NVIDIA_DEFAULT_MODEL",
    ),
    "nvidia_nim": _ProviderMeta(
        name="Nvidia",
        env_key=ProviderDefaults.NVIDIA_ENV_KEY,
        default_base_url=ProviderDefaults.NVIDIA_BASE_URL,
        default_model=ProviderDefaults.NVIDIA_MODEL,
        env_base_url_var="NVIDIA_BASE_URL",
        env_model_var="NVIDIA_DEFAULT_MODEL",
    ),
    # OpenRouter
    "openrouter": _ProviderMeta(
        name="OpenRouter",
        env_key=ProviderDefaults.OPENROUTER_ENV_KEY,
        default_base_url=ProviderDefaults.OPENROUTER_BASE_URL,
        default_model=ProviderDefaults.OPENROUTER_MODEL,
        env_base_url_var="OPENROUTER_BASE_URL",
        env_model_var="OPENROUTER_DEFAULT_MODEL",
    ),
    "openrouter_free": _ProviderMeta(
        name="OpenRouter",
        env_key=ProviderDefaults.OPENROUTER_ENV_KEY,
        default_base_url=ProviderDefaults.OPENROUTER_BASE_URL,
        default_model=ProviderDefaults.OPENROUTER_MODEL,
        env_base_url_var="OPENROUTER_BASE_URL",
        env_model_var="OPENROUTER_DEFAULT_MODEL",
    ),
}


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
        """Retrieve an API key from environment variables."""
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
        """Resolve full configuration for a specific provider via table lookup."""
        norm = str(provider).lower().replace("-", "_").strip()
        meta = _PROVIDER_META_REGISTRY.get(norm)

        if meta is None:
            raise ProviderNotFoundError(
                provider=str(provider),
                available_providers=[p.value for p in ProviderType],
            )

        env_timeout = os.getenv("NEXUS_DEFAULT_TIMEOUT")
        resolved_timeout = timeout if timeout is not None else (
            float(env_timeout) if env_timeout else ProviderDefaults.DEFAULT_TIMEOUT
        )

        resolved_key = api_key or cls.get_api_key(
            meta.env_key,
            fallback_var=meta.fallback_env_key,
            provider_name=meta.name,
            required=require_api_key,
        )

        env_base = os.getenv(meta.env_base_url_var) if meta.env_base_url_var else None
        env_model = os.getenv(meta.env_model_var) if meta.env_model_var else None

        extra_headers: dict[str, str] | None = None
        if "openrouter" in norm:
            extra_headers = {
                "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "https://github.com/NexusAI-Client"),
                "X-Title": os.getenv("OPENROUTER_APP_NAME", "NexusAI-Client"),
            }

        return ProviderConfig(
            api_key=resolved_key,
            base_url=base_url or env_base or meta.default_base_url,
            default_model=model or env_model or meta.default_model,
            timeout=resolved_timeout,
            extra_headers=extra_headers,
        )
