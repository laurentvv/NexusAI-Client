"""DeepSeek AI Provider (Paid API).

Documentation: https://api-docs.deepseek.com/
"""

from __future__ import annotations

from typing import Any, override

import httpx

from nexusai_client.config import Config, ProviderDefaults
from nexusai_client.exceptions import (
    APIConnectionError,
    APIResponseError,
    APITimeoutError,
)
from nexusai_client.models import (
    AccountInfo,
    ModelInfo,
    ModelPricing,
    ProviderType,
)
from nexusai_client.providers.openai_compat import OpenAICompatibleProvider

# Official DeepSeek Pricing table ($ per million tokens)
DEEPSEEK_PRICING_MAP: dict[str, dict[str, Any]] = {
    "deepseek-chat": {
        "name": "DeepSeek-V3 (Chat)",
        "prompt": 0.27,
        "completion": 1.10,
        "cache_read": 0.07,
        "context_length": 64_000,
        "description": "Modèle généraliste haute performance pour le dialogue, la rédaction et le code.",
    },
    "deepseek-reasoner": {
        "name": "DeepSeek-R1 (Reasoner)",
        "prompt": 0.55,
        "completion": 2.19,
        "cache_read": 0.14,
        "context_length": 64_000,
        "description": "Modèle de raisonnement avec chaîne de pensée (Thinking Mode) pour les tâches complexes.",
    },
}


class DeepSeekProvider(OpenAICompatibleProvider):
    """Client for the DeepSeek API platform."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        fallback_models: list[str] | tuple[str, ...] | None = None,
        auto_rotate_models: bool = True,
        **kwargs: Any,
    ) -> None:
        config = Config.get_provider_config(
            ProviderType.DEEPSEEK,
            api_key=api_key,
            base_url=base_url,
            model=model or ProviderDefaults.DEEPSEEK_MODEL,
            timeout=timeout,
        )
        resolved_fallbacks = (
            fallback_models
            if fallback_models is not None
            else list(ProviderDefaults.DEEPSEEK_FALLBACK_MODELS)
        )
        super().__init__(
            api_key=config.api_key,
            base_url=config.base_url,
            default_model=config.default_model,
            timeout=config.timeout,
            fallback_models=resolved_fallbacks,
            auto_rotate_models=auto_rotate_models,
            **kwargs,
        )

    @property
    @override
    def provider_name(self) -> str:
        return "deepseek"

    @property
    def balance_endpoint(self) -> str:
        """Endpoint for fetching DeepSeek account balance."""
        return f"{self.base_url}/user/balance"

    @override
    async def list_models(self, *, free_only: bool = False) -> list[ModelInfo]:
        """Fetch DeepSeek models and enrich them with pricing and context lengths."""
        if free_only:
            # DeepSeek is a strictly paid API
            return []

        models = await super().list_models(free_only=False)
        enriched: list[ModelInfo] = []

        for m in models:
            meta = DEEPSEEK_PRICING_MAP.get(m.id, {})
            pricing = ModelPricing(
                prompt_per_million=meta.get("prompt", 0.27),
                completion_per_million=meta.get("completion", 1.10),
                cache_read_per_million=meta.get("cache_read", 0.07),
            )
            enriched.append(
                ModelInfo(
                    id=m.id,
                    name=meta.get("name", m.name),
                    provider=self.provider_name,
                    is_free=False,
                    context_length=meta.get("context_length", 64_000),
                    pricing=pricing,
                    description=meta.get("description", m.description),
                    raw_data=m.raw_data,
                )
            )

        return enriched or [
            ModelInfo(
                id="deepseek-chat",
                name="DeepSeek-V3 (Chat)",
                provider=self.provider_name,
                is_free=False,
                context_length=64_000,
                pricing=ModelPricing(
                    prompt_per_million=0.27,
                    completion_per_million=1.10,
                    cache_read_per_million=0.07,
                ),
                description="Modèle généraliste DeepSeek V3.",
            ),
            ModelInfo(
                id="deepseek-reasoner",
                name="DeepSeek-R1 (Reasoner)",
                provider=self.provider_name,
                is_free=False,
                context_length=64_000,
                pricing=ModelPricing(
                    prompt_per_million=0.55,
                    completion_per_million=2.19,
                    cache_read_per_million=0.14,
                ),
                description="Modèle de raisonnement avancé DeepSeek R1.",
            ),
        ]

    @override
    async def get_account_info(self) -> AccountInfo:
        """Fetch real-time account budget balance from DeepSeek API."""
        headers = self._build_headers()
        client = await self.get_client()

        try:
            response = await client.get(self.balance_endpoint, headers=headers)
        except httpx.TimeoutException as exc:
            raise APITimeoutError(
                provider=self.provider_name, timeout_seconds=self.timeout
            ) from exc
        except (httpx.NetworkError, httpx.ConnectError) as exc:
            raise APIConnectionError(
                provider=self.provider_name, original_error=exc
            ) from exc

        self._handle_http_error(response)

        try:
            data = response.json()
            infos = data.get("balance_infos", [])
            # Search for USD balance info first, or take primary entry
            target_info = next(
                (i for i in infos if i.get("currency") == "USD"),
                (infos[0] if infos else {}),
            )

            currency = target_info.get("currency", "USD")
            total_bal = float(target_info.get("total_balance", 0.0))
            granted_bal = float(target_info.get("granted_balance", 0.0))
            topped_up_bal = float(target_info.get("topped_up_balance", 0.0))

            return AccountInfo(
                provider=self.provider_name,
                total_balance=total_bal,
                granted_balance=granted_bal,
                topped_up_balance=topped_up_bal,
                currency=currency,
                is_free_tier=False,
                rate_limit_info="DeepSeek Standard Tier",
                extra_details=data,
            )

        except Exception as exc:
            raise APIResponseError(
                provider=self.provider_name,
                status_code=response.status_code,
                response_body=response.text,
                message=f"Failed to parse DeepSeek balance: {exc}",
            ) from exc
