"""Google Gemini Providers (Free & Pro tiers via Google AI Studio / Gemini REST API).

Documentation: https://ai.google.dev/api/generate-content
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
    AIResponse,
    ChatMessage,
    ModelInfo,
    ModelPricing,
    ProviderType,
    UsageInfo,
)
from nexusai_client.providers.base import BaseAIProvider


class GeminiBaseProvider(BaseAIProvider):
    """Base provider implementing the Google Generative Language REST API."""

    def _build_url(self, model: str) -> str:
        """Construct the endpoint URL for generateContent."""
        clean_model = model.removeprefix("models/")
        return f"{self.base_url}/models/{clean_model}:generateContent"

    def _build_headers(self) -> dict[str, str]:
        """Construct headers for Gemini API."""
        return {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
            **self.extra_headers,
        }

    def _convert_messages_to_gemini(
        self,
        messages: list[ChatMessage | dict[str, str]],
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        """Convert standard messages into Gemini format."""
        system_instruction: dict[str, Any] | None = None
        gemini_contents: list[dict[str, Any]] = []

        for msg in messages:
            if isinstance(msg, ChatMessage):
                role = msg.role
                content = msg.content
            else:
                role = str(msg["role"])
                content = str(msg["content"])

            if role == "system":
                system_instruction = {"parts": [{"text": content}]}
            else:
                gemini_role = "model" if role == "assistant" else "user"
                gemini_contents.append({
                    "role": gemini_role,
                    "parts": [{"text": content}],
                })

        return system_instruction, gemini_contents

    @override
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
        """Generate response for a single text prompt."""
        messages: list[ChatMessage] = []
        if system_prompt:
            messages.append(ChatMessage(role="system", content=system_prompt))
        messages.append(ChatMessage(role="user", content=prompt))

        return await self.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    @override
    async def chat(
        self,
        messages: list[ChatMessage | dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AIResponse:
        """Send chat messages to Gemini REST API."""
        target_model = model or self.default_model
        system_instruction, contents = self._convert_messages_to_gemini(messages)

        generation_config: dict[str, Any] = {
            "temperature": temperature,
        }
        if max_tokens is not None:
            generation_config["maxOutputTokens"] = max_tokens

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": generation_config,
        }
        if system_instruction:
            payload["systemInstruction"] = system_instruction

        payload.update(self.extra_params)
        payload.update(kwargs)

        url = self._build_url(target_model)
        headers = self._build_headers()
        client = await self.get_client()

        try:
            response = await client.post(url, headers=headers, json=payload)
        except httpx.TimeoutException as exc:
            raise APITimeoutError(provider=self.provider_name, timeout_seconds=self.timeout) from exc
        except (httpx.NetworkError, httpx.ConnectError) as exc:
            raise APIConnectionError(provider=self.provider_name, original_error=exc) from exc

        self._handle_http_error(response)

        try:
            data = response.json()
            candidates = data.get("candidates", [])
            if not candidates:
                prompt_feedback = data.get("promptFeedback", {})
                block_reason = prompt_feedback.get("blockReason", "No response candidate generated")
                raise APIResponseError(
                    provider=self.provider_name,
                    status_code=response.status_code,
                    response_body=response.text,
                    message=f"Gemini returned no candidates: {block_reason}",
                )

            candidate = candidates[0]
            parts = candidate.get("content", {}).get("parts", [])
            text_output = "".join(part.get("text", "") for part in parts if "text" in part)
            finish_reason = candidate.get("finishReason")

            usage_metadata = data.get("usageMetadata", {})
            usage_info: UsageInfo | None = None
            if usage_metadata:
                usage_info = UsageInfo(
                    prompt_tokens=usage_metadata.get("promptTokenCount", 0),
                    completion_tokens=usage_metadata.get("candidatesTokenCount", 0),
                    total_tokens=usage_metadata.get("totalTokenCount", 0),
                )

            return AIResponse(
                text=text_output,
                provider=self.provider_name,
                model=target_model,
                usage=usage_info,
                finish_reason=finish_reason,
                raw_response=data,
            )

        except (KeyError, IndexError, ValueError) as exc:
            raise APIResponseError(
                provider=self.provider_name,
                status_code=response.status_code,
                response_body=response.text,
                message=f"Failed to parse Gemini response: {exc}",
            ) from exc

    @override
    async def list_models(self, *, free_only: bool = False) -> list[ModelInfo]:
        """Query Google AI Studio models endpoint."""
        url = f"{self.base_url}/models"
        headers = self._build_headers()
        client = await self.get_client()

        try:
            response = await client.get(url, headers=headers)
        except httpx.TimeoutException as exc:
            raise APITimeoutError(provider=self.provider_name, timeout_seconds=self.timeout) from exc
        except (httpx.NetworkError, httpx.ConnectError) as exc:
            raise APIConnectionError(provider=self.provider_name, original_error=exc) from exc

        self._handle_http_error(response)

        try:
            data = response.json()
            raw_models = data.get("models", [])

            models: list[ModelInfo] = []
            for item in raw_models:
                methods = item.get("supportedGenerationMethods", [])
                if "generateContent" not in methods:
                    continue

                raw_name = item.get("name", "")
                clean_id = raw_name.removeprefix("models/")
                display_name = item.get("displayName", clean_id)
                input_limit = item.get("inputTokenLimit", 1_000_000)

                models.append(
                    ModelInfo(
                        id=clean_id,
                        name=display_name,
                        provider=self.provider_name,
                        is_free=self._is_free_tier(),
                        context_length=input_limit,
                        pricing=self._get_model_pricing(clean_id),
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
                message=f"Failed to parse Gemini models: {exc}",
            ) from exc

    def _is_free_tier(self) -> bool:
        return False

    def _get_model_pricing(self, model_id: str) -> ModelPricing:
        return ModelPricing(prompt_per_million=0.0, completion_per_million=0.0)


class GeminiProProvider(GeminiBaseProvider):
    """Gemini Pro Provider (Paid tier / High throughput endpoint)."""

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
            ProviderType.GEMINI_PRO,
            api_key=api_key,
            base_url=base_url,
            model=model or ProviderDefaults.GEMINI_PRO_MODEL,
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
        return "gemini_pro"

    @override
    def _is_free_tier(self) -> bool:
        return False

    @override
    def _get_model_pricing(self, model_id: str) -> ModelPricing:
        if "flash" in model_id:
            return ModelPricing(prompt_per_million=0.075, completion_per_million=0.30)
        return ModelPricing(prompt_per_million=1.25, completion_per_million=5.00)

    @override
    async def get_account_info(self) -> AccountInfo:
        """Account information for Gemini Pro (GCP Billing)."""
        return AccountInfo(
            provider=self.provider_name,
            is_free_tier=False,
            rate_limit_info="Facturation Pay-as-you-go Google Cloud / Quotas Développeur",
            extra_details={"platform": "Google Cloud / AI Studio Pay-as-you-go"},
        )


class GeminiFreeProvider(GeminiBaseProvider):
    """Gemini Free Provider (Google AI Studio Free Tier)."""

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
            ProviderType.GEMINI_FREE,
            api_key=api_key,
            base_url=base_url,
            model=model or ProviderDefaults.GEMINI_FREE_MODEL,
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
        return "gemini_free"

    @override
    def _is_free_tier(self) -> bool:
        return True

    @override
    def _get_model_pricing(self, model_id: str) -> ModelPricing:
        return ModelPricing(prompt_per_million=0.0, completion_per_million=0.0)

    @override
    async def get_account_info(self) -> AccountInfo:
        """Account information for Gemini Free tier."""
        return AccountInfo(
            provider=self.provider_name,
            is_free_tier=True,
            rate_limit_info="15 RPM (Req/min) | 1,000,000 TPM | 1,500 RPD (Req/jour)",
            total_balance=0.0,
            extra_details={"platform": "Google AI Studio Free Tier"},
        )
