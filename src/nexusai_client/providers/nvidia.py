"""Nvidia NIM Provider (Free Tier / NVIDIA NGC).

Documentation: https://build.nvidia.com/
"""

from __future__ import annotations

from typing import Any, override

from nexusai_client.config import Config, ProviderDefaults
from nexusai_client.models import (
    AccountInfo,
    ModelInfo,
    ModelPricing,
    ProviderType,
)
from nexusai_client.providers.openai_compat import OpenAICompatibleProvider


class NvidiaProvider(OpenAICompatibleProvider):
    """Client for Nvidia NIM (Inference Microservices) Free API."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> None:
        config = Config.get_provider_config(
            ProviderType.NVIDIA,
            api_key=api_key,
            base_url=base_url,
            model=model or ProviderDefaults.NVIDIA_MODEL,
            timeout=timeout,
        )
        super().__init__(
            api_key=config.api_key,
            base_url=config.base_url,
            default_model=config.default_model,
            timeout=config.timeout,
            **kwargs,
        )

    @property
    @override
    def provider_name(self) -> str:
        return "nvidia"

    @override
    async def list_models(self, *, free_only: bool = False) -> list[ModelInfo]:
        """Fetch all models hosted on Nvidia NIM platform."""
        models = await super().list_models(free_only=False)
        free_pricing = ModelPricing(prompt_per_million=0.0, completion_per_million=0.0)

        results: list[ModelInfo] = []
        for m in models:
            results.append(
                ModelInfo(
                    id=m.id,
                    name=m.id.split("/")[-1].replace("-", " ").title(),
                    provider=self.provider_name,
                    is_free=True,
                    context_length=128_000 if "llama-3" in m.id or "deepseek" in m.id else 32_000,
                    pricing=free_pricing,
                    description="Modèle hébergé sur Nvidia NIM (Accès gratuit avec crédits NGC).",
                    raw_data=m.raw_data,
                )
            )
        return results

    @override
    async def get_account_info(self) -> AccountInfo:
        """Account information for Nvidia NIM / NGC Free Tier."""
        return AccountInfo(
            provider=self.provider_name,
            is_free_tier=True,
            rate_limit_info="1,000 crédits d'inférence gratuits offerts (NVIDIA NGC)",
            total_balance=1000.0,
            currency="Credits",
            extra_details={"platform": "NVIDIA NIM / NGC"},
        )
