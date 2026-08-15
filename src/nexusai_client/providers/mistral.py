"""Mistral AI Provider (Free tier / La Plateforme).

Documentation: https://docs.mistral.ai/
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

MISTRAL_PRICING_MAP: dict[str, dict[str, Any]] = {
    "mistral-small-latest": {
        "name": "Mistral Small (Latest)",
        "prompt": 0.10,
        "completion": 0.30,
        "context_length": 32_000,
        "is_free": True,
        "description": "Modèle rapide et léger, accessible sur le tier gratuit La Plateforme.",
    },
    "mistral-large-latest": {
        "name": "Mistral Large 2",
        "prompt": 2.00,
        "completion": 6.00,
        "context_length": 128_000,
        "is_free": False,
        "description": "Modèle de pointe pour le raisonnement complexe et le multilingue.",
    },
    "codestral-latest": {
        "name": "Codestral (Latest)",
        "prompt": 0.30,
        "completion": 0.90,
        "context_length": 256_000,
        "is_free": True,
        "description": "Modèle spécialisé pour le code, la complétion FIM et le débogage.",
    },
    "pixtral-12b-2409": {
        "name": "Pixtral 12B",
        "prompt": 0.15,
        "completion": 0.15,
        "context_length": 128_000,
        "is_free": True,
        "description": "Modèle multimodal pour l'analyse d'images et de documents.",
    },
    "mistral-embed": {
        "name": "Mistral Embed",
        "prompt": 0.10,
        "completion": 0.0,
        "context_length": 8_000,
        "is_free": False,
        "description": "Modèle d'embedding sémantique pour la recherche vectorielle et le RAG.",
    },
}


class MistralProvider(OpenAICompatibleProvider):
    """Client for the Mistral AI platform (Free & Paid tiers)."""

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
            ProviderType.MISTRAL,
            api_key=api_key,
            base_url=base_url,
            model=model or ProviderDefaults.MISTRAL_MODEL,
            timeout=timeout,
        )
        super().__init__(
            api_key=config.api_key,
            base_url=config.base_url,
            default_model=config.default_model,
            default_vision_model=config.default_vision_model,
            timeout=config.timeout,
            **kwargs,
        )

    @property
    @override
    def provider_name(self) -> str:
        return "mistral"

    @override
    async def list_models(self, *, free_only: bool = False) -> list[ModelInfo]:
        """Fetch Mistral models and enrich them with pricing and context lengths."""
        models = await super().list_models(free_only=False)
        enriched: list[ModelInfo] = []

        for m in models:
            meta = MISTRAL_PRICING_MAP.get(m.id, {})
            is_free = meta.get("is_free", True if "small" in m.id or "open" in m.id else False)
            if free_only and not is_free:
                continue

            pricing = ModelPricing(
                prompt_per_million=meta.get("prompt", 0.20),
                completion_per_million=meta.get("completion", 0.60),
            ) if not is_free else ModelPricing(prompt_per_million=0.0, completion_per_million=0.0)

            enriched.append(
                ModelInfo(
                    id=m.id,
                    name=meta.get("name", m.name),
                    provider=self.provider_name,
                    is_free=is_free,
                    context_length=meta.get("context_length", 32_000),
                    pricing=pricing,
                    description=meta.get("description", m.description),
                    raw_data=m.raw_data,
                )
            )

        return enriched

    @override
    async def get_account_info(self) -> AccountInfo:
        """Account information for Mistral AI platform."""
        return AccountInfo(
            provider=self.provider_name,
            is_free_tier=True,
            rate_limit_info="1 req/sec (Free Tier Experimentation) ou Standard Pay-as-you-go",
            extra_details={"platform": "Mistral La Plateforme"},
        )
