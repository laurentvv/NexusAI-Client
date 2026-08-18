"""OrcaRouter Provider (Multi-provider Gateway with Free Tier Models).

Documentation: https://docs.orcarouter.ai/
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


class OrcaRouterProvider(OpenAICompatibleProvider):
    """Client for OrcaRouter API (Access to free and commercial zero-margin models)."""

    provider_name = "orcarouter"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        default_model: str | None = None,
        vision_model: str | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> None:
        config = Config.get_provider_config(
            ProviderType.ORCAROUTER,
            api_key=api_key,
            base_url=base_url,
            model=model or default_model or ProviderDefaults.ORCAROUTER_MODEL,
            vision_model=vision_model or ProviderDefaults.ORCAROUTER_VISION_MODEL,
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
        """Fetch all models from OrcaRouter /v1/models with free tier detection."""
        headers = self._build_headers()
        client = await self.get_client()

        try:
            response = await client.get(self.models_endpoint, headers=headers)
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
            raw_models = data.get("data", []) if isinstance(data, dict) else data

            models: list[ModelInfo] = []
            for item in raw_models:
                m_id = item.get("id", "") if isinstance(item, dict) else str(item)
                if not m_id:
                    continue

                pricing_data = item.get("pricing", {}) if isinstance(item, dict) else {}
                prompt_price: float = 0.0
                completion_price: float = 0.0

                if pricing_data:
                    try:
                        prompt_price = float(pricing_data.get("prompt", 0)) * 1_000_000
                        completion_price = (
                            float(pricing_data.get("completion", 0)) * 1_000_000
                        )
                    except (ValueError, TypeError):
                        prompt_price = 0.0
                        completion_price = 0.0

                is_free = (
                    m_id.endswith("-free")
                    or m_id.endswith(":free")
                    or m_id == "orcarouter/free"
                    or (
                        prompt_price == 0.0
                        and completion_price == 0.0
                        and bool(pricing_data)
                    )
                )

                if free_only and not is_free:
                    continue

                pricing = (
                    ModelPricing(
                        prompt_per_million=prompt_price,
                        completion_per_million=completion_price,
                    )
                    if (prompt_price > 0.0 or completion_price > 0.0 or is_free)
                    else None
                )

                name = item.get("name", m_id) if isinstance(item, dict) else m_id
                ctx_len = item.get("context_length") if isinstance(item, dict) else None
                desc = item.get("description") if isinstance(item, dict) else None

                models.append(
                    ModelInfo(
                        id=m_id,
                        name=name,
                        provider=self.provider_name,
                        is_free=is_free,
                        context_length=ctx_len,
                        pricing=pricing,
                        description=desc,
                        raw_data=item if isinstance(item, dict) else {},
                    )
                )

            return models

        except Exception as exc:
            raise APIResponseError(
                provider=self.provider_name,
                status_code=response.status_code,
                response_body=response.text,
                message=f"Failed to parse OrcaRouter models: {exc}",
            ) from exc

    @override
    async def get_account_info(self) -> AccountInfo:
        """Return rate limits and status for OrcaRouter tier."""
        return AccountInfo(
            provider=self.provider_name,
            total_balance=0.0,
            currency="USD",
            is_free_tier=True,
            rate_limit_info="Fixed minute/daily windows | Prompt token cap | Free Tier (-free models)",
            extra_details={
                "tier": "OrcaRouter Free Tier",
                "free_models_suffix": "-free",
                "default_model": ProviderDefaults.ORCAROUTER_MODEL,
                "docs": "https://docs.orcarouter.ai/fr/routing/free-models",
            },
        )
