"""OpenRouter Provider (Free Tier & Router API).

Documentation: https://openrouter.ai/docs
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


class OpenRouterProvider(OpenAICompatibleProvider):
    """Client for OpenRouter API (Access to free and commercial models)."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        site_url: str | None = None,
        site_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        config = Config.get_provider_config(
            ProviderType.OPENROUTER,
            api_key=api_key,
            base_url=base_url,
            model=model or ProviderDefaults.OPENROUTER_MODEL,
            timeout=timeout,
        )

        headers = dict(config.extra_headers or {})
        if site_url:
            headers["HTTP-Referer"] = site_url
        if site_name:
            headers["X-Title"] = site_name

        super().__init__(
            api_key=config.api_key,
            base_url=config.base_url,
            default_model=config.default_model,
            default_vision_model=config.default_vision_model,
            timeout=config.timeout,
            extra_headers=headers,
            **kwargs,
        )

    @property
    @override
    def provider_name(self) -> str:
        return "openrouter"

    @property
    def auth_key_endpoint(self) -> str:
        """Endpoint for checking key balance, usage, and rate limits."""
        return f"{self.base_url}/auth/key"

    @property
    def credits_endpoint(self) -> str:
        """Endpoint for checking account credit balance."""
        return f"{self.base_url}/credits"

    @override
    async def list_models(self, *, free_only: bool = False) -> list[ModelInfo]:
        """Fetch all models from OpenRouter with real-time pricing data."""
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
            raw_models = data.get("data", [])

            models: list[ModelInfo] = []
            for item in raw_models:
                m_id = item.get("id", "")
                pricing_data = item.get("pricing", {})

                try:
                    prompt_price = float(pricing_data.get("prompt", 0)) * 1_000_000
                    completion_price = (
                        float(pricing_data.get("completion", 0)) * 1_000_000
                    )
                except (ValueError, TypeError):
                    prompt_price = 0.0
                    completion_price = 0.0

                is_free = (
                    prompt_price == 0.0 and completion_price == 0.0
                ) or m_id.endswith(":free")

                if free_only and not is_free:
                    continue

                pricing = ModelPricing(
                    prompt_per_million=prompt_price,
                    completion_per_million=completion_price,
                )

                models.append(
                    ModelInfo(
                        id=m_id,
                        name=item.get("name", m_id),
                        provider=self.provider_name,
                        is_free=is_free,
                        context_length=item.get("context_length"),
                        pricing=pricing,
                        description=item.get("description"),
                        raw_data=item,
                    )
                )

            return models

        except Exception as exc:
            raise APIResponseError(
                provider=self.provider_name,
                status_code=response.status_code,
                response_body=response.text,
                message=f"Failed to parse OpenRouter models: {exc}",
            ) from exc

    @override
    async def get_account_info(self) -> AccountInfo:
        """Fetch account usage, remaining budget/credits, and rate limits from OpenRouter."""
        headers = self._build_headers()
        client = await self.get_client()

        try:
            response = await client.get(self.auth_key_endpoint, headers=headers)
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
            data = response.json().get("data", {})
            usage = float(data.get("usage", 0.0))
            limit = data.get("limit")
            limit_val = float(limit) if limit is not None else None
            is_free = bool(data.get("is_free_tier", False))

            rate_limit_meta = data.get("rate_limit", {})
            rate_str = (
                f"{rate_limit_meta.get('requests', 20)} req/{rate_limit_meta.get('interval', '10s')}"
                if rate_limit_meta
                else "20 req/10s"
            )

            # Try to fetch exact remaining credits if available
            credits_balance: float | None = None
            try:
                c_resp = await client.get(self.credits_endpoint, headers=headers)
                if c_resp.is_success:
                    c_data = c_resp.json().get("data", {})
                    total_credits = c_data.get("total_credits")
                    if total_credits is not None:
                        credits_balance = float(total_credits) - float(
                            c_data.get("total_usage", 0.0)
                        )
            except Exception:
                pass

            remaining = (
                credits_balance
                if credits_balance is not None
                else ((limit_val - usage) if limit_val is not None else None)
            )

            return AccountInfo(
                provider=self.provider_name,
                total_balance=remaining,
                total_usage=usage,
                currency="USD",
                is_free_tier=is_free or (remaining is None and usage == 0.0),
                rate_limit_info=rate_str,
                extra_details=data,
            )

        except Exception as exc:
            raise APIResponseError(
                provider=self.provider_name,
                status_code=response.status_code,
                response_body=response.text,
                message=f"Failed to parse OpenRouter key information: {exc}",
            ) from exc
