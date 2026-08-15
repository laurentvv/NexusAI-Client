"""Cerebras Provider implementation for NexusAI-Client.

Provides access to Cerebras's Wafer-Scale Engine (CS-3) ultra-fast inference API.
World-record inference speeds (2,000+ tokens/second) via OpenAI-compatible endpoints.
"""

from __future__ import annotations

from typing import Any, override

from nexusai_client.config import Config
from nexusai_client.models import AccountInfo, ModelInfo, ModelPricing
from nexusai_client.providers.openai_compat import OpenAICompatibleProvider


class CerebrasProvider(OpenAICompatibleProvider):
    """Cerebras AI Provider (Wafer-Scale Ultra-Fast Inference)."""

    provider_name = "cerebras"

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
            "cerebras",
            api_key=api_key,
            base_url=base_url,
            model=model or default_model,
            timeout=timeout,
        )
        super().__init__(
            api_key=config.api_key,
            base_url=config.base_url,
            default_model=config.default_model,
            timeout=config.timeout,
            extra_headers=config.extra_headers,
            **kwargs,
        )

    @override
    async def list_models(self, *, free_only: bool = False) -> list[ModelInfo]:
        """Fetch available models from Cerebras /models endpoint."""
        models = await super().list_models(free_only=False)
        free_pricing = ModelPricing(prompt_per_million=0.0, completion_per_million=0.0)

        results: list[ModelInfo] = []
        for m in models:
            ctx = 128_000 if "llama" in m.id else 8_192
            results.append(
                ModelInfo(
                    id=m.id,
                    name=m.id.replace("-", " ").title(),
                    provider=self.provider_name,
                    is_free=True,
                    context_length=ctx,
                    pricing=free_pricing,
                    description="Wafer-Scale high-speed inference on Cerebras CS-3.",
                    raw_data=m.raw_data,
                )
            )
        return results

    @override
    async def get_account_info(self) -> AccountInfo:
        """Return rate limits and status for Cerebras Free tier."""
        return AccountInfo(
            provider=self.provider_name,
            total_balance=0.0,
            currency="USD",
            is_free_tier=True,
            rate_limit_info="30 RPM | 60,000 TPM | 1M tokens/day (Free Tier)",
            extra_details={
                "RPM": 30,
                "TPM": 60_000,
                "daily_limit": "1,000,000 tokens",
                "tier": "Cerebras Free Developer Tier",
            },
        )
