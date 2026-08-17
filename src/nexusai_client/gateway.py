"""Unified Gateway, Factory, and Fallback Manager for NexusAI-Client.

Provides a single entry point to instantiate, query, stream, and automatically failover
across all supported AI providers.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any, Self

from nexusai_client.exceptions import (
    MissingAPIKeyError,
    NexusAIError,
    ProviderNotFoundError,
)
from nexusai_client.models import (
    AccountInfo,
    AIResponse,
    ChatMessage,
    ModelInfo,
    ProviderType,
    ToolDefinition,
)
from nexusai_client.providers.base import BaseAIProvider
from nexusai_client.providers.cerebras import CerebrasProvider
from nexusai_client.providers.cohere import CohereProvider
from nexusai_client.providers.deepseek import DeepSeekProvider
from nexusai_client.providers.gemini import GeminiFreeProvider, GeminiProProvider
from nexusai_client.providers.groq import GroqProvider
from nexusai_client.providers.mistral import MistralProvider
from nexusai_client.providers.nvidia import NvidiaProvider
from nexusai_client.providers.openrouter import OpenRouterProvider
from nexusai_client.providers.orcarouter import OrcaRouterProvider

logger = logging.getLogger("nexusai_client")

type ProviderClassMap = dict[str, type[BaseAIProvider]]

_PROVIDER_REGISTRY: ProviderClassMap = {
    # Cerebras (Wafer-Scale Free Tier)
    ProviderType.CEREBRAS: CerebrasProvider,
    ProviderType.CEREBRAS_FREE: CerebrasProvider,
    "cerebras": CerebrasProvider,
    "cerebras_free": CerebrasProvider,
    "cerebras-free": CerebrasProvider,
    # Cohere (Enterprise & Free Trial Tier)
    ProviderType.COHERE: CohereProvider,
    ProviderType.COHERE_FREE: CohereProvider,
    "cohere": CohereProvider,
    "cohere_free": CohereProvider,
    "cohere-free": CohereProvider,
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
    # Groq (Free LPU Tier)
    ProviderType.GROQ: GroqProvider,
    ProviderType.GROQ_FREE: GroqProvider,
    "groq": GroqProvider,
    "groq_free": GroqProvider,
    "groq-free": GroqProvider,
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
    # OrcaRouter (Zero-Margin Free & Commercial Gateway)
    ProviderType.ORCAROUTER: OrcaRouterProvider,
    ProviderType.ORCAROUTER_FREE: OrcaRouterProvider,
    "orcarouter": OrcaRouterProvider,
    "orcarouter_free": OrcaRouterProvider,
    "orcarouter-free": OrcaRouterProvider,
}


class FallbackGateway(BaseAIProvider):
    """Automatic multi-provider fallback wrapper.

    Sequentially attempts calls across a list of configured providers until one succeeds.
    """

    def __init__(
        self,
        providers: list[ProviderType | str | BaseAIProvider],
        *,
        timeout: float = 60.0,
        **kwargs: Any,
    ) -> None:
        if not providers:
            raise ValueError(
                "FallbackGateway requires at least one provider in the chain."
            )

        self._provider_chain: list[BaseAIProvider] = []
        for p in providers:
            if isinstance(p, BaseAIProvider):
                self._provider_chain.append(p)
            else:
                self._provider_chain.append(
                    AIGateway.create(p, timeout=timeout, **kwargs)
                )

        primary = self._provider_chain[0]
        super().__init__(
            api_key=primary.api_key,
            base_url=primary.base_url,
            default_model=primary.default_model,
            timeout=timeout,
        )

    @property
    def provider_name(self) -> str:
        chain_names = " -> ".join(p.provider_name for p in self._provider_chain)
        return f"fallback({chain_names})"

    async def generate_text(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        json_mode: bool = False,
        tools: list[ToolDefinition | dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AIResponse:
        """Attempt text generation across providers in chain order."""
        last_error: Exception | None = None
        for prov in self._provider_chain:
            try:
                return await prov.generate_text(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=json_mode,
                    tools=tools,
                    tool_choice=tool_choice,
                    **kwargs,
                )
            except (NexusAIError, Exception) as err:
                logger.warning(
                    f"FallbackGateway: provider '{prov.provider_name}' failed ({err}). Trying next..."
                )
                last_error = err

        raise RuntimeError(
            f"All providers in FallbackGateway chain failed. Last error: {last_error}"
        )

    async def analyze_image(
        self,
        prompt: str,
        image: Any,
        *,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        json_mode: bool = False,
        **kwargs: Any,
    ) -> AIResponse:
        """Attempt multimodal vision analysis across providers in chain order."""
        last_error: Exception | None = None
        for prov in self._provider_chain:
            try:
                return await prov.analyze_image(
                    prompt=prompt,
                    image=image,
                    system_prompt=system_prompt,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=json_mode,
                    **kwargs,
                )
            except (NexusAIError, Exception) as err:
                logger.warning(
                    f"FallbackGateway Vision: provider '{prov.provider_name}' failed ({err}). Trying next..."
                )
                last_error = err

        raise RuntimeError(
            f"All vision providers in FallbackGateway chain failed. Last error: {last_error}"
        )

    async def chat(
        self,
        messages: list[ChatMessage | dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        json_mode: bool = False,
        tools: list[ToolDefinition | dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AIResponse:
        """Attempt chat completion across providers in chain order."""
        last_error: Exception | None = None
        for prov in self._provider_chain:
            try:
                return await prov.chat(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=json_mode,
                    tools=tools,
                    tool_choice=tool_choice,
                    **kwargs,
                )
            except (NexusAIError, Exception) as err:
                logger.warning(
                    f"FallbackGateway: provider '{prov.provider_name}' failed ({err}). Trying next..."
                )
                last_error = err

        raise RuntimeError(
            f"All providers in FallbackGateway chain failed. Last error: {last_error}"
        )

    async def stream_text(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream text from the first operational provider in the chain."""
        last_error: Exception | None = None
        for prov in self._provider_chain:
            try:
                async for chunk in prov.stream_text(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                ):
                    yield chunk
                return
            except (NexusAIError, Exception) as err:
                logger.warning(
                    f"FallbackGateway: stream '{prov.provider_name}' failed ({err}). Trying next..."
                )
                last_error = err

        raise RuntimeError(
            f"All providers in FallbackGateway stream failed. Last error: {last_error}"
        )

    async def stream_chat(
        self,
        messages: list[ChatMessage | dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream chat from the first operational provider in the chain."""
        last_error: Exception | None = None
        for prov in self._provider_chain:
            try:
                async for chunk in prov.stream_chat(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                ):
                    yield chunk
                return
            except (NexusAIError, Exception) as err:
                logger.warning(
                    f"FallbackGateway: stream_chat '{prov.provider_name}' failed ({err}). Trying next..."
                )
                last_error = err

        raise RuntimeError(
            f"All providers in FallbackGateway stream_chat failed. Last error: {last_error}"
        )

    async def list_models(self, *, free_only: bool = False) -> list[ModelInfo]:
        """Aggregate models from all chain providers."""
        all_models: list[ModelInfo] = []
        for prov in self._provider_chain:
            try:
                all_models.extend(await prov.list_models(free_only=free_only))
            except Exception:
                pass
        return all_models

    async def get_account_info(self) -> AccountInfo:
        """Return account info from primary provider."""
        return await self._provider_chain[0].get_account_info()

    async def close(self) -> None:
        """Close all underlying provider sessions in the chain."""
        for prov in self._provider_chain:
            await prov.close()


class AIGateway:
    """Unified Gateway & Client for interacting with any configured AI provider.

    Usage Examples:
    --------------
    ```python
    # 1. Direct generation:
    async with AIGateway("gemini_free") as client:
        res = await client.generate_text("Bonjour !")

    # 2. Real-time streaming:
    async with AIGateway("openrouter") as client:
        async for chunk in client.stream_text("Raconte une histoire courte."):
            print(chunk, end="", flush=True)

    # 3. Resilient Fallback Chain:
    async with AIGateway.with_fallback(["gemini_free", "nvidia_free", "openrouter", "deepseek"]) as client:
        res = await client.generate_text("Calculer pi avec 5 décimales.")
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
    def with_fallback(
        cls,
        providers: list[ProviderType | str | BaseAIProvider],
        *,
        timeout: float = 60.0,
        **kwargs: Any,
    ) -> FallbackGateway:
        """Create a resilient FallbackGateway that automatically fails over across providers."""
        return FallbackGateway(providers=providers, timeout=timeout, **kwargs)

    @classmethod
    def get_configured_providers(cls, *, prioritize_free: bool = True) -> list[str]:
        """Dynamically inspect environment and return all providers with valid API keys.

        Args:
            prioritize_free: If True (default), places zero-cost / free-tier providers
                             at the front of the chain, followed by paid fallback providers.

        Returns:
            List of configured provider strings (e.g. ['gemini_free', 'nvidia_free', 'openrouter', 'deepseek']).
        """
        free_candidates = [
            "gemini_free",
            "groq_free",
            "cerebras_free",
            "nvidia_free",
            "openrouter",
            "orcarouter_free",
            "cohere_free",
            "mistral",
        ]
        paid_candidates = ["deepseek", "gemini_pro"]

        configured_free: list[str] = []
        configured_paid: list[str] = []

        for p_name in free_candidates:
            try:
                prov = cls.create(p_name)
                if prov.api_key:
                    configured_free.append(p_name)
            except (MissingAPIKeyError, Exception):
                pass

        for p_name in paid_candidates:
            try:
                prov = cls.create(p_name)
                if prov.api_key:
                    configured_paid.append(p_name)
            except (MissingAPIKeyError, Exception):
                pass

        if prioritize_free:
            return configured_free + configured_paid
        return configured_paid + configured_free

    @classmethod
    def get_configured_vision_providers(
        cls, *, prioritize_free: bool = True
    ) -> list[str]:
        """Inspect environment and return active providers that support multimodal vision analysis."""
        free_vision_candidates = [
            "gemini_free",
            "nvidia_free",
            "cohere_free",
            "mistral",
            "openrouter",
        ]
        paid_vision_candidates = ["gemini_pro"]

        configured_free: list[str] = []
        configured_paid: list[str] = []

        for p_name in free_vision_candidates:
            try:
                prov = cls.create(p_name)
                if prov.api_key:
                    configured_free.append(p_name)
            except (MissingAPIKeyError, Exception):
                pass

        for p_name in paid_vision_candidates:
            try:
                prov = cls.create(p_name)
                if prov.api_key:
                    configured_paid.append(p_name)
            except (MissingAPIKeyError, Exception):
                pass

        if prioritize_free:
            return configured_free + configured_paid
        return configured_paid + configured_free

    @classmethod
    def auto_fallback_vision(
        cls,
        *,
        prioritize_free: bool = True,
        timeout: float = 60.0,
        **kwargs: Any,
    ) -> FallbackGateway:
        """Automatically create a FallbackGateway composed of active multimodal Vision providers.

        Example:
        -------
        ```python
        async with AIGateway.auto_fallback_vision() as client:
            res = await client.analyze_image("Extract table data to JSON", "invoice.png")
        ```
        """
        chain = cls.get_configured_vision_providers(prioritize_free=prioritize_free)
        if not chain:
            raise MissingAPIKeyError(
                provider="Any Vision Provider",
                env_var="GEMINI_FREE_API_KEY, NVIDIA_API_KEY, COHERE_API_KEY, MISTRAL_API_KEY, or OPENROUTER_API_KEY",
                message="No configured multimodal Vision providers found in .env.",
            )
        return cls.with_fallback(chain, timeout=timeout, **kwargs)

    @classmethod
    def auto_fallback(
        cls,
        *,
        prioritize_free: bool = True,
        timeout: float = 60.0,
        **kwargs: Any,
    ) -> FallbackGateway:
        """Automatically create a FallbackGateway composed of all active providers in .env.

        Example:
        -------
        ```python
        async with AIGateway.auto_fallback() as client:
            res = await client.generate_text("Hello!")
        ```
        """
        chain = cls.get_configured_providers(prioritize_free=prioritize_free)
        if not chain:
            raise MissingAPIKeyError(
                provider="Any",
                env_var="At least one valid API key in .env",
                message="No configured AI providers found in .env to build a fallback chain.",
            )
        return cls.with_fallback(chain, timeout=timeout, **kwargs)

    @classmethod
    def available_providers(cls) -> list[str]:
        """Return a sorted list of unique supported provider names."""
        unique_names = {
            "cerebras (or cerebras_free)",
            "cohere (or cohere_free)",
            "deepseek",
            "gemini_pro",
            "gemini_free",
            "groq (or groq_free)",
            "mistral",
            "nvidia (or nvidia_free)",
            "openrouter (or openrouter_free)",
            "orcarouter (or orcarouter_free)",
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
        target_providers = providers or [
            "cerebras",
            "cohere",
            "deepseek",
            "gemini_free",
            "gemini_pro",
            "groq",
            "mistral",
            "nvidia",
            "openrouter",
            "orcarouter",
        ]
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
        target_providers = providers or [
            "cerebras",
            "cohere",
            "deepseek",
            "gemini_free",
            "gemini_pro",
            "groq",
            "mistral",
            "nvidia",
            "openrouter",
            "orcarouter",
        ]
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
        json_mode: bool = False,
        tools: list[ToolDefinition | dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AIResponse:
        """Generate text using the underlying configured provider."""
        return await self.provider.generate_text(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        )

    async def analyze_image(
        self,
        prompt: str,
        image: Any,
        *,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        json_mode: bool = False,
        **kwargs: Any,
    ) -> AIResponse:
        """Analyze an image, chart, PDF, or screenshot using multimodal vision models."""
        return await self.provider.analyze_image(
            prompt=prompt,
            image=image,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
            **kwargs,
        )

    async def chat(
        self,
        messages: list[ChatMessage | dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        json_mode: bool = False,
        tools: list[ToolDefinition | dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AIResponse:
        """Send chat messages using the underlying configured provider."""
        return await self.provider.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        )

    def stream_text(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream generated text chunks in real time."""
        return self.provider.stream_text(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    def stream_chat(
        self,
        messages: list[ChatMessage | dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream generated chat chunks in real time."""
        return self.provider.stream_chat(
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
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: Any,
    ) -> None:
        """Async context manager exit."""
        await self.close()
