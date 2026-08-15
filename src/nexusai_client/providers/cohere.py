"""Cohere Provider implementation for NexusAI-Client.

Provides asynchronous integration with Cohere's official V2 Chat API.
Supports Command R+, Command R, Command Light, Aya, structured outputs, and token streaming.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any, override

import httpx

from nexusai_client.config import Config
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
    UsageInfo,
)
from nexusai_client.providers.base import BaseAIProvider


class CohereProvider(BaseAIProvider):
    """Cohere AI Provider (Enterprise LLMs & Free Trial Tier)."""

    provider_name = "cohere"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> None:
        config = Config.get_provider_config(
            "cohere",
            api_key=api_key,
            base_url=base_url,
            model=default_model,
            timeout=timeout,
        )
        super().__init__(
            api_key=config.api_key,
            base_url=config.base_url,
            default_model=config.default_model,
            timeout=config.timeout,
            **kwargs,
        )

    def _get_headers(self) -> dict[str, str]:
        """Construct standard HTTP headers for Cohere API."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "NexusAI-Client/0.1.0",
        }

    @override
    async def generate_text(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        json_mode: bool = False,
        **kwargs: Any,
    ) -> AIResponse:
        """Generate single-turn text completion via Cohere V2 Chat API."""
        messages: list[ChatMessage] = []
        if system_prompt:
            messages.append(ChatMessage(role="system", content=system_prompt))
        messages.append(ChatMessage(role="user", content=prompt))

        return await self.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
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
        json_mode: bool = False,
        **kwargs: Any,
    ) -> AIResponse:
        """Send a multi-turn conversation to Cohere V2 Chat API."""
        target_model = model or self.default_model
        formatted_messages = [
            {"role": m.role if isinstance(m, ChatMessage) else m["role"], "content": m.content if isinstance(m, ChatMessage) else m["content"]}
            for m in messages
        ]

        payload: dict[str, Any] = {
            "model": target_model,
            "messages": formatted_messages,
            "temperature": temperature,
            **kwargs,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        client = await self.get_client()
        url = f"{self.base_url}/chat"

        try:
            response = await client.post(url, json=payload, headers=self._get_headers())
        except httpx.TimeoutException as exc:
            raise APITimeoutError(provider=self.provider_name, timeout_seconds=self.timeout) from exc
        except (httpx.NetworkError, httpx.ConnectError) as exc:
            raise APIConnectionError(provider=self.provider_name, original_error=exc) from exc

        self._handle_http_error(response)

        try:
            data = response.json()
            # Extract content from Cohere v2 format: data["message"]["content"][0]["text"]
            content_list = data.get("message", {}).get("content", [])
            text_parts = [c.get("text", "") for c in content_list if c.get("type") == "text" or "text" in c]
            reply_text = "".join(text_parts) if text_parts else str(data.get("message", {}).get("content", ""))

            usage_data = data.get("usage", {}).get("tokens", {})
            prompt_tokens = usage_data.get("input_tokens")
            completion_tokens = usage_data.get("output_tokens")
            total_tokens = (prompt_tokens or 0) + (completion_tokens or 0) if prompt_tokens is not None else None

            usage = UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            ) if total_tokens is not None else None

            return AIResponse(
                text=reply_text,
                provider=self.provider_name,
                model=target_model,
                usage=usage,
                raw_response=data,
            )
        except Exception as exc:
            raise APIResponseError(
                provider=self.provider_name,
                status_code=response.status_code,
                response_body=response.text,
                message=f"Failed to parse Cohere V2 response: {exc}",
            ) from exc

    @override
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
        """Stream real-time text chunks from a prompt via Cohere V2 SSE."""
        messages: list[ChatMessage] = []
        if system_prompt:
            messages.append(ChatMessage(role="system", content=system_prompt))
        messages.append(ChatMessage(role="user", content=prompt))

        async for chunk in self.stream_chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        ):
            yield chunk

    @override
    async def stream_chat(
        self,
        messages: list[ChatMessage | dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream generated tokens in real time from Cohere V2 Chat API via SSE."""
        target_model = model or self.default_model
        formatted_messages = [
            {"role": m.role if isinstance(m, ChatMessage) else m["role"], "content": m.content if isinstance(m, ChatMessage) else m["content"]}
            for m in messages
        ]

        payload: dict[str, Any] = {
            "model": target_model,
            "messages": formatted_messages,
            "temperature": temperature,
            "stream": True,
            **kwargs,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        client = await self.get_client()
        url = f"{self.base_url}/chat"

        try:
            async with client.stream(
                "POST",
                url,
                json=payload,
                headers=self._get_headers(),
                timeout=self.timeout,
            ) as response:
                self._handle_http_error(response)

                async for line in response.aiter_lines():
                    clean_line = line.strip()
                    if not clean_line or not clean_line.startswith("data:"):
                        continue

                    data_str = clean_line.removeprefix("data:").strip()
                    if data_str == "[DONE]":
                        break

                    try:
                        chunk_json = json.loads(data_str)
                        # Cohere v2 streaming event: delta.message.content.text
                        delta = chunk_json.get("delta", {}).get("message", {}).get("content", {})
                        if isinstance(delta, dict) and "text" in delta:
                            token = delta["text"]
                            if token:
                                yield token
                        elif isinstance(delta, str) and delta:
                            yield delta
                    except json.JSONDecodeError:
                        continue

        except httpx.TimeoutException as exc:
            raise APITimeoutError(provider=self.provider_name, timeout_seconds=self.timeout) from exc
        except (httpx.NetworkError, httpx.ConnectError) as exc:
            raise APIConnectionError(provider=self.provider_name, original_error=exc) from exc

    @override
    async def list_models(self, *, free_only: bool = False) -> list[ModelInfo]:
        """Fetch available models from Cohere API."""
        client = await self.get_client()
        url = "https://api.cohere.com/v1/models"

        try:
            response = await client.get(url, headers=self._get_headers())
        except (httpx.TimeoutException, httpx.NetworkError, Exception):
            # Fallback to standard known Cohere models
            return [
                ModelInfo(
                    id="command-r-plus-08-2024",
                    name="Command R+ (08-2024)",
                    provider=self.provider_name,
                    is_free=True,
                    context_length=128_000,
                    pricing=ModelPricing(prompt_per_million=2.5, completion_per_million=10.0),
                    description="Cohere flagship reasoning and tool-use model.",
                ),
                ModelInfo(
                    id="command-r-08-2024",
                    name="Command R (08-2024)",
                    provider=self.provider_name,
                    is_free=True,
                    context_length=128_000,
                    pricing=ModelPricing(prompt_per_million=0.15, completion_per_million=0.60),
                    description="Cohere fast scalable enterprise model.",
                ),
            ]

        if response.status_code != 200:
            return []

        try:
            data = response.json()
            models_raw = data.get("models", [])
            results: list[ModelInfo] = []

            for m in models_raw:
                endpoints = m.get("endpoints", [])
                if "chat" not in endpoints:
                    continue

                m_id = m.get("name", "")
                ctx = m.get("context_length", 128_000)
                results.append(
                    ModelInfo(
                        id=m_id,
                        name=m_id.replace("-", " ").title(),
                        provider=self.provider_name,
                        is_free=True,  # Trial Key free tier
                        context_length=ctx,
                        pricing=ModelPricing(prompt_per_million=0.5, completion_per_million=1.5),
                        description="Cohere Command foundation model for enterprise reasoning.",
                        raw_data=m,
                    )
                )
            return results
        except Exception:
            return []

    @override
    async def get_account_info(self) -> AccountInfo:
        """Return rate limits and status for Cohere Free Trial Key."""
        return AccountInfo(
            provider=self.provider_name,
            total_balance=0.0,
            currency="USD",
            is_free_tier=True,
            rate_limit_info="20 RPM | 1,000 calls/month (Free Trial Tier)",
            extra_details={
                "RPM": 20,
                "monthly_limit": "1,000 API requests",
                "tier": "Free Developer Trial",
            },
        )
