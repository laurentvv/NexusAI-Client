"""Unified Gateway and Factory for NexusAI-Client.

Provides a single entry point to instantiate, query, and interact with all supported AI providers.
"""

from __future__ import annotations

import asyncio
from typing import Any, Self

from nexusai_client.exceptions import MissingAPIKeyError, ProviderNotFoundError
from nexusai_client.models import (
    AccountInfo,
    AIResponse,
    ChatMessage,
    ModelInfo,
    ProviderType,
)
from nexusai_client.providers.base import BaseAIProvider
from nexusai_client.providers.deepseek import DeepSeekProvider
from nexusai_client.providers.gemini import GeminiFreeProvider, GeminiProProvider
from nexusai_client.providers.mistral import MistralProvider
from nexusai_client.providers.nvidia import NvidiaProvider
from nexusai_client.providers.openrouter import OpenRouterProvider

type ProviderClassMap = dict[str, type[BaseAIProvider]]

_PROVIDER_REGISTRY: ProviderClassMap = {
    # DeepSeek
    ProviderType.DEEPSEEK: DeepSeekProvider,
    "deepseek": DeepSeekProvider,
    # Gemini Pro
    ProviderType.GEMINI_PRO: GeminiProProvider,
    "gemini_pro": GeminiProProvider,
    "gemini-pro": GeminiProProvider,
    # Gemini Free
    ProviderType.GEMINI_FREE: GeminiFreeProvider,
    "gemini_free": GeminiFreeProvider,
    "gemini-free": GeminiFreeProvider,
    "gemini": GeminiFreeProvider,
    # Mistral
    ProviderType.MISTRAL: MistralProvider,
    "mistral": MistralProvider,
    "mistral_free": MistralProvider,
    "mistral-free": MistralProvider,
    # Nvidia NIM
    ProviderType.NVIDIA: NvidiaProvider,
    ProviderType.NVIDIA_FREE: NvidiaProvider,
    "nvidia": NvidiaProvider,
    "nvidia_free": NvidiaProvider,
    "nvidia-free": NvidiaProvider,
    "nvidia_nim": NvidiaProvider,
    "nvidia-nim": NvidiaProvider,
    # OpenRouter
    ProviderType.OPENROUTER: OpenRouterProvider,
    ProviderType.OPENROUTER_FREE: OpenRouterProvider,
    "openrouter": OpenRouterProvider,
    "openrouter_free": OpenRouterProvider,
    "openrouter-free": OpenRouterProvider,
}


class AIGateway:
    """Unified Gateway & Client for interacting with any configured AI provider.

    Usage Examples:
    --------------
    ```python
    # 1. Direct generation with automatic .env configuration:
    client = AIGateway(provider="openrouter")
    response = await client.generate_text("Explique la théorie de la relativité.")
    await client.close()

    # 2. Check remaining budget & credits:
    async with AIGateway("deepseek") as client:
        info = await client.get_account_info()
        print(f"Solde restant: ${info.total_balance}")

    # 3. List available models & pricing:
    async with AIGateway("gemini_free") as client:
        models = await client.list_models()
        for m in models:
            print(m)
    ```
    """

    def __init__(
        self,
        provider: ProviderType | str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Instantiate an AIGateway client for a specific provider."""
        self._provider_name_raw = str(provider)
        self.provider: BaseAIProvider = self.create(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout=timeout,
            **kwargs,
        )

    @classmethod
    def create(
        cls,
        provider: ProviderType | str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> BaseAIProvider:
        """Factory method returning an initialized concrete BaseAIProvider instance."""
        norm_key = str(provider).lower().strip()
        provider_cls = _PROVIDER_REGISTRY.get(norm_key)

        if provider_cls is None:
            raise ProviderNotFoundError(
                provider=norm_key,
                available_providers=cls.available_providers(),
            )

        return provider_cls(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout=timeout,
            **kwargs,
        )

    @classmethod
    def available_providers(cls) -> list[str]:
        """Return a sorted list of unique supported provider names."""
        unique_names = {
            "deepseek",
            "gemini_pro",
            "gemini_free",
            "mistral",
            "nvidia (or nvidia_free)",
            "openrouter (or openrouter_free)",
        }
        return sorted(unique_names)

    @classmethod
    async def list_all_available_models(
        cls,
        providers: list[str] | None = None,
        *,
        free_only: bool = False,
    ) -> dict[str, list[ModelInfo]]:
        """Query multiple or all providers in parallel to list available models."""
        target_providers = providers or ["deepseek", "gemini_free", "gemini_pro", "mistral", "nvidia", "openrouter"]
        results: dict[str, list[ModelInfo]] = {}

        async def _fetch(p_name: str) -> tuple[str, list[ModelInfo]]:
            try:
                prov = cls.create(p_name)
                models = await prov.list_models(free_only=free_only)
                await prov.close()
                return p_name, models
            except (MissingAPIKeyError, Exception):
                return p_name, []

        tasks = [_fetch(p) for p in target_providers]
        fetched = await asyncio.gather(*tasks)

        for p_name, models in fetched:
            if models:
                results[p_name] = models

        return results

    @classmethod
    async def get_all_account_infos(
        cls,
        providers: list[str] | None = None,
    ) -> dict[str, AccountInfo]:
        """Query budget, credits, and quotas across multiple or all providers in parallel."""
        target_providers = providers or ["deepseek", "gemini_free", "gemini_pro", "mistral", "nvidia", "openrouter"]
        results: dict[str, AccountInfo] = {}

        async def _fetch_info(p_name: str) -> tuple[str, AccountInfo | None]:
            try:
                prov = cls.create(p_name)
                info = await prov.get_account_info()
                await prov.close()
                return p_name, info
            except (MissingAPIKeyError, Exception):
                return p_name, None

        tasks = [_fetch_info(p) for p in target_providers]
        fetched = await asyncio.gather(*tasks)

        for p_name, info in fetched:
            if info is not None:
                results[p_name] = info

        return results

    # ---------------------------------------------------------
    # Delegated Methods for Convenience
    # ---------------------------------------------------------

    async def generate_text(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AIResponse:
        """Generate text using the underlying configured provider."""
        return await self.provider.generate_text(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    async def chat(
        self,
        messages: list[ChatMessage | dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AIResponse:
        """Send chat messages using the underlying configured provider."""
        return await self.provider.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    async def list_models(self, *, free_only: bool = False) -> list[ModelInfo]:
        """Fetch available models for this specific provider."""
        return await self.provider.list_models(free_only=free_only)

    async def get_account_info(self) -> AccountInfo:
        """Fetch remaining budget, credits, and quota status for this provider."""
        return await self.provider.get_account_info()

    async def close(self) -> None:
        """Close the underlying provider client session."""
        await self.provider.close()

    async def __aenter__(self) -> Self:
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Async context manager exit."""
        await self.close()
