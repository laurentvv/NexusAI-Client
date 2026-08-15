"""Generic OpenAI-compatible provider implementation.

Handles providers adhering to the OpenAI Chat Completions standard
(DeepSeek, Mistral, Nvidia NIM, OpenRouter).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any, override

import httpx

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
    UsageInfo,
)
from nexusai_client.providers.base import BaseAIProvider


class OpenAICompatibleProvider(BaseAIProvider):
    """Base provider for all APIs implementing the OpenAI chat/completions schema."""

    @property
    def chat_endpoint(self) -> str:
        """Endpoint for chat completions."""
        return f"{self.base_url}/chat/completions"

    @property
    def models_endpoint(self) -> str:
        """Endpoint for listing models."""
        return f"{self.base_url}/models"

    def _build_headers(self) -> dict[str, str]:
        """Build HTTP headers including authorization and extra headers."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        headers.update(self.extra_headers)
        return headers

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
        """Generate a response for a single text prompt."""
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
        """Send a multi-turn chat conversation to the OpenAI-compatible endpoint."""
        target_model = model or self.default_model
        normalized_messages = self._normalize_messages(messages)

        payload: dict[str, Any] = {
            "model": target_model,
            "messages": normalized_messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        payload.update(self.extra_params)
        payload.update(kwargs)

        headers = self._build_headers()
        client = await self.get_client()

        try:
            response = await client.post(
                self.chat_endpoint,
                headers=headers,
                json=payload,
            )
        except httpx.TimeoutException as exc:
            raise APITimeoutError(provider=self.provider_name, timeout_seconds=self.timeout) from exc
        except (httpx.NetworkError, httpx.ConnectError) as exc:
            raise APIConnectionError(provider=self.provider_name, original_error=exc) from exc

        self._handle_http_error(response)

        try:
            data = response.json()
            choice = data["choices"][0]
            message_content = choice.get("message", {}).get("content", "")
            finish_reason = choice.get("finish_reason")

            usage_data = data.get("usage")
            usage_info: UsageInfo | None = None
            if isinstance(usage_data, dict):
                usage_info = UsageInfo(
                    prompt_tokens=usage_data.get("prompt_tokens", 0),
                    completion_tokens=usage_data.get("completion_tokens", 0),
                    total_tokens=usage_data.get("total_tokens", 0),
                )

            return AIResponse(
                text=message_content or "",
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
                message=f"Failed to parse provider response: {exc}",
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
        """Stream real-time text chunks from a prompt."""
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
        """Stream real-time tokens from a conversation thread via Server-Sent Events (SSE)."""
        target_model = model or self.default_model
        normalized_messages = self._normalize_messages(messages)

        payload: dict[str, Any] = {
            "model": target_model,
            "messages": normalized_messages,
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        payload.update(self.extra_params)
        payload.update(kwargs)

        headers = self._build_headers()
        client = await self.get_client()

        try:
            async with client.stream("POST", self.chat_endpoint, headers=headers, json=payload) as response:
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
                        choices = chunk_json.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content")
                            if content:
                                yield content
                    except json.JSONDecodeError:
                        continue

        except httpx.TimeoutException as exc:
            raise APITimeoutError(provider=self.provider_name, timeout_seconds=self.timeout) from exc
        except (httpx.NetworkError, httpx.ConnectError) as exc:
            raise APIConnectionError(provider=self.provider_name, original_error=exc) from exc

    @override
    async def list_models(self, *, free_only: bool = False) -> list[ModelInfo]:
        """Query standard /models endpoint and return normalized ModelInfo objects."""
        headers = self._build_headers()
        client = await self.get_client()

        try:
            response = await client.get(self.models_endpoint, headers=headers)
        except httpx.TimeoutException as exc:
            raise APITimeoutError(provider=self.provider_name, timeout_seconds=self.timeout) from exc
        except (httpx.NetworkError, httpx.ConnectError) as exc:
            raise APIConnectionError(provider=self.provider_name, original_error=exc) from exc

        self._handle_http_error(response)

        try:
            data = response.json()
            raw_models = data.get("data", []) if isinstance(data, dict) else data

            models: list[ModelInfo] = []
            for item in raw_models:
                m_id = item.get("id") if isinstance(item, dict) else str(item)
                if not m_id:
                    continue
                models.append(
                    ModelInfo(
                        id=m_id,
                        name=item.get("name", m_id) if isinstance(item, dict) else m_id,
                        provider=self.provider_name,
                        is_free=False,
                        description=item.get("description") if isinstance(item, dict) else None,
                        raw_data=item if isinstance(item, dict) else {},
                    )
                )
            return models

        except Exception as exc:
            raise APIResponseError(
                provider=self.provider_name,
                status_code=response.status_code,
                response_body=response.text,
                message=f"Failed to parse models list: {exc}",
            ) from exc

    @override
    async def get_account_info(self) -> AccountInfo:
        """Default account info implementation for generic OpenAI endpoints."""
        return AccountInfo(
            provider=self.provider_name,
            is_free_tier=False,
            rate_limit_info="Standard Rate Limit",
        )
