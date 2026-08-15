"""Groq Provider implementation for NexusAI-Client.

Provides access to Groq's ultra-fast LPU inference engine via its OpenAI-compatible API.
Supports popular models like Llama 3.3 70B, Llama 3.1 8B, DeepSeek R1 Distill, and Mixtral.
"""

from __future__ import annotations

from typing import Any, override

from nexusai_client.config import Config
from nexusai_client.models import AccountInfo, ModelInfo, ModelPricing
from nexusai_client.providers.openai_compat import OpenAICompatibleProvider


class GroqProvider(OpenAICompatibleProvider):
    """Groq AI Provider (Ultra-Fast LPU Free & Developer Tier)."""

    provider_name = "groq"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        default_model: str | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> None:
        config = Config.get_provider_config(
            "groq",
            api_key=api_key,
            base_url=base_url,
            model=model or default_model,
            timeout=timeout,
        )
        super().__init__(
            api_key=config.api_key,
            base_url=config.base_url,
            default_model=config.default_model,
            default_vision_model=config.default_vision_model,
            timeout=config.timeout,
            extra_headers=config.extra_headers,
            **kwargs,
        )

    @override
    async def list_models(self, *, free_only: bool = False) -> list[ModelInfo]:
        """Fetch available models from Groq /models endpoint."""
        models = await super().list_models(free_only=False)
        free_pricing = ModelPricing(prompt_per_million=0.0, completion_per_million=0.0)

        results: list[ModelInfo] = []
        for m in models:
            # Determine approximate context length based on model ID
            if "128k" in m.id or "llama-3.3" in m.id or "llama-3.1" in m.id or "deepseek" in m.id:
                ctx = 128_000
            elif "32768" in m.id or "mixtral" in m.id:
                ctx = 32_768
            elif "8192" in m.id or "gemma" in m.id:
                ctx = 8_192
            else:
                ctx = 8_192

            results.append(
                ModelInfo(
                    id=m.id,
                    name=m.id.replace("-", " ").title(),
                    provider=self.provider_name,
                    is_free=True,
                    context_length=ctx,
                    pricing=free_pricing,
                    description="High-speed LPU inference hosted on Groq Cloud.",
                    raw_data=m.raw_data,
                )
            )
        return results

    @override
    async def get_account_info(self) -> AccountInfo:
        """Return rate limits and status for Groq Free / Developer tier."""
        return AccountInfo(
            provider=self.provider_name,
            total_balance=0.0,
            currency="USD",
            is_free_tier=True,
            rate_limit_info="30 RPM | 14,400 RPD | 30,000 TPM (Free LPU Tier)",
            extra_details={
                "RPM": 30,
                "RPD": 14_400,
                "TPM": 30_000,
                "tier": "Free LPU Developer Tier",
            },
        )
