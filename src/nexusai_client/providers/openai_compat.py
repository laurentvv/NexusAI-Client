"""Generic OpenAI-compatible provider implementation with dynamic model rotation.

Handles providers adhering to the OpenAI Chat Completions standard
(DeepSeek, Mistral, Nvidia NIM, OpenRouter, Groq, Cerebras, OrcaRouter).
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any, override

import httpx

from nexusai_client.exceptions import (
    APIConnectionError,
    APIResponseError,
    APITimeoutError,
    ProviderServerError,
    RateLimitError,
)
from nexusai_client.models import (
    AccountInfo,
    AIResponse,
    ChatMessage,
    ModelInfo,
    ToolCall,
    ToolDefinition,
    UsageInfo,
)
from nexusai_client.providers.base import BaseAIProvider

logger = logging.getLogger("nexusai_client")

_NON_CHAT_KEYWORDS: tuple[str, ...] = (
    "whisper",
    "guard",
    "embed",
    "tts",
    "stt",
    "rerank",
    "moderation",
    "safeguard",
    "bge",
    "clip",
    "audio",
    "transcription",
    "vector",
)


def _is_chat_model(model_id: str) -> bool:
    """Return True if the model is suitable for general text chat/completions."""
    m_lower = model_id.lower()
    return not any(kw in m_lower for kw in _NON_CHAT_KEYWORDS)


def _is_model_unsupported_error(exc: APIResponseError) -> bool:
    """Check if an API error indicates the requested model does not exist or is deprecated."""
    if exc.status_code in (400, 404):
        msg = str(exc).lower()
        unsupported_terms = (
            "does not exist",
            "not found",
            "model_not_found",
            "invalid_model",
            "deprecated",
            "not have access",
            "is not supported",
            "unknown model",
            "model not supported",
            "invalid model",
        )
        return any(term in msg for term in unsupported_terms)
    return False


class OpenAICompatibleProvider(BaseAIProvider):
    """Base provider for all APIs implementing the OpenAI chat/completions schema with dynamic model rotation."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        default_model: str,
        default_vision_model: str | None = None,
        timeout: float = 60.0,
        extra_headers: dict[str, str] | None = None,
        fallback_models: list[str] | tuple[str, ...] | None = None,
        fallback_vision_models: list[str] | tuple[str, ...] | None = None,
        auto_rotate_models: bool = True,
        cooldown_seconds: float = 60.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            default_model=default_model,
            default_vision_model=default_vision_model,
            timeout=timeout,
            extra_headers=extra_headers,
            **kwargs,
        )
        self.fallback_models: list[str] = list(fallback_models or [])
        self.fallback_vision_models: list[str] = list(fallback_vision_models or [])
        self.auto_rotate_models: bool = auto_rotate_models
        self.cooldown_seconds: float = cooldown_seconds
        self._model_cooldowns: dict[str, float] = {}
        self._discovered_chat_models: list[str] | None = None
        self._discovered_at: float = 0.0

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

    def _mark_model_cooldown(
        self, model: str, duration: float | None = None
    ) -> None:
        """Mark a model as temporarily unavailable or in cooldown."""
        cd = duration if duration is not None else self.cooldown_seconds
        self._model_cooldowns[model] = time.monotonic() + cd

    async def _discover_active_chat_models(self) -> list[str]:
        """Dynamically query /models and discover active text/chat models."""
        now = time.monotonic()
        if self._discovered_chat_models is not None and (now - self._discovered_at) < 600.0:
            return self._discovered_chat_models

        try:
            models = await self.list_models()
            valid = [
                m.id for m in models
                if _is_chat_model(m.id) and self._model_cooldowns.get(m.id, 0.0) <= now
            ]
            if valid:
                self._discovered_chat_models = valid
                self._discovered_at = now
                return valid
        except Exception as exc:
            logger.debug(
                f"{self.provider_name}: Dynamic model discovery failed: {exc}"
            )
        return []

    async def _get_candidate_models(
        self, requested_model: str | None, *, is_vision: bool = False
    ) -> list[str]:
        """Order candidate models, placing available models before models in cooldown."""
        primary = requested_model or (
            self.default_vision_model if is_vision else self.default_model
        )
        fallbacks = (
            self.fallback_vision_models if is_vision else self.fallback_models
        )

        ordered: list[str] = [primary] if primary else []
        if self.auto_rotate_models:
            for m in fallbacks:
                if m not in ordered:
                    ordered.append(m)

        now = time.monotonic()
        available: list[str] = []
        in_cooldown: list[str] = []

        for m in ordered:
            cooldown_until = self._model_cooldowns.get(m, 0.0)
            if cooldown_until > now:
                in_cooldown.append(m)
            else:
                available.append(m)

        candidates = available + in_cooldown
        if not candidates and not is_vision:
            discovered = await self._discover_active_chat_models()
            candidates = [
                m for m in discovered if self._model_cooldowns.get(m, 0.0) <= now
            ] or discovered

        return candidates or ([primary] if primary else [])

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
        tools: list[ToolDefinition | dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
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
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        )

    async def _single_analyze_image(
        self,
        *,
        model: str,
        prompt: str,
        image: Any,
        system_prompt: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        json_mode: bool = False,
        **kwargs: Any,
    ) -> AIResponse:
        """Execute single Vision inference call with a specific model."""
        from nexusai_client.utils import load_image_as_data_uri

        data_uri = load_image_as_data_uri(image)

        user_content: list[dict[str, Any]] = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_uri}},
        ]

        messages_payload: list[dict[str, Any]] = []
        if system_prompt:
            messages_payload.append({"role": "system", "content": system_prompt})
        messages_payload.append({"role": "user", "content": user_content})

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages_payload,
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
            choice = data["choices"][0]
            reply_text = choice["message"].get("content") or ""

            usage_data = data.get("usage", {})
            prompt_tokens = usage_data.get("prompt_tokens")
            completion_tokens = usage_data.get("completion_tokens")
            total_tokens = usage_data.get("total_tokens")

            usage = (
                UsageInfo(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )
                if total_tokens is not None
                else None
            )

            return AIResponse(
                text=reply_text,
                provider=self.provider_name,
                model=model,
                usage=usage,
                raw_response=data,
            )
        except Exception as exc:
            raise APIResponseError(
                provider=self.provider_name,
                status_code=response.status_code,
                response_body=response.text,
                message=f"Failed to parse Vision response: {exc}",
            ) from exc

    @override
    async def analyze_image(
        self,
        prompt: str,
        image: Any,
        *,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        json_mode: bool = False,
        **kwargs: Any,
    ) -> AIResponse:
        """Analyze an image, chart, PDF, or screenshot with dynamic vision model rotation."""
        candidates = await self._get_candidate_models(model, is_vision=True)
        last_error: Exception | None = None

        for candidate in candidates:
            try:
                return await self._single_analyze_image(
                    model=candidate,
                    prompt=prompt,
                    image=image,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=json_mode,
                    **kwargs,
                )
            except RateLimitError as exc:
                self._mark_model_cooldown(candidate)
                logger.warning(
                    f"{self.provider_name}: Vision model '{candidate}' rate limited (429). Rotating to next model..."
                )
                last_error = exc
                continue
            except APIResponseError as exc:
                if _is_model_unsupported_error(exc):
                    self._mark_model_cooldown(candidate, duration=3600.0)
                    logger.warning(
                        f"{self.provider_name}: Vision model '{candidate}' unavailable ({exc}). Rotating..."
                    )
                    last_error = exc
                    continue
                raise
            except (APITimeoutError, ProviderServerError) as exc:
                if self.auto_rotate_models:
                    self._mark_model_cooldown(candidate, duration=60.0)
                    logger.warning(
                        f"{self.provider_name}: Vision model '{candidate}' failed ({exc}). Trying next candidate..."
                    )
                    last_error = exc
                    continue
                raise

        if last_error:
            raise last_error
        raise RuntimeError(f"No candidate vision models available in provider '{self.provider_name}'.")

    async def _single_chat(
        self,
        *,
        model: str,
        messages: list[ChatMessage | dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        json_mode: bool = False,
        tools: list[ToolDefinition | dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AIResponse:
        """Send a single multi-turn chat completion request to the OpenAI-compatible endpoint."""
        normalized_messages = self._normalize_messages(messages)

        payload: dict[str, Any] = {
            "model": model,
            "messages": normalized_messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        normalized_tools = self._normalize_tools(tools)
        if normalized_tools:
            payload["tools"] = normalized_tools
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice

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
            choice = data["choices"][0]
            choice_msg = choice.get("message", {})
            message_content = choice_msg.get("content") or ""
            finish_reason = choice.get("finish_reason")

            # Parse tool calls
            raw_tool_calls = choice_msg.get("tool_calls", [])
            parsed_tool_calls: list[ToolCall] = []
            if raw_tool_calls and isinstance(raw_tool_calls, list):
                for tc in raw_tool_calls:
                    tc_id = tc.get("id", "")
                    fn_data = tc.get("function", {})
                    fn_name = fn_data.get("name", "")
                    raw_args = fn_data.get("arguments", "")
                    if isinstance(raw_args, dict):
                        args_dict = raw_args
                        raw_args_str = json.dumps(raw_args)
                    elif isinstance(raw_args, str):
                        raw_args_str = raw_args
                        try:
                            args_dict = json.loads(raw_args) if raw_args.strip() else {}
                        except Exception:
                            args_dict = {}
                    else:
                        raw_args_str = str(raw_args)
                        args_dict = {}

                    parsed_tool_calls.append(
                        ToolCall(
                            id=tc_id,
                            name=fn_name,
                            arguments=args_dict,
                            raw_arguments=raw_args_str,
                        )
                    )

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
                model=model,
                usage=usage_info,
                finish_reason=finish_reason,
                tool_calls=parsed_tool_calls,
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
    async def chat(
        self,
        messages: list[ChatMessage | dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        json_mode: bool = False,
        tools: list[ToolDefinition | dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AIResponse:
        """Send a multi-turn chat conversation to the OpenAI-compatible endpoint with automatic dynamic model rotation."""
        candidates = await self._get_candidate_models(model, is_vision=False)
        last_error: Exception | None = None
        tried_discovery = False

        i = 0
        while i < len(candidates):
            candidate = candidates[i]
            i += 1
            try:
                return await self._single_chat(
                    model=candidate,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=json_mode,
                    tools=tools,
                    tool_choice=tool_choice,
                    **kwargs,
                )
            except RateLimitError as exc:
                self._mark_model_cooldown(candidate)
                logger.warning(
                    f"{self.provider_name}: Model '{candidate}' rate limited (429). Rotating to next dynamic model..."
                )
                last_error = exc
                continue
            except APIResponseError as exc:
                if _is_model_unsupported_error(exc):
                    self._mark_model_cooldown(candidate, duration=3600.0)
                    logger.warning(
                        f"{self.provider_name}: Model '{candidate}' is unavailable/deprecated ({exc}). Auto-rotating to next dynamic model..."
                    )
                    last_error = exc
                    if i >= len(candidates) and not tried_discovery:
                        tried_discovery = True
                        discovered = await self._discover_active_chat_models()
                        for d in discovered:
                            if (
                                d not in candidates
                                and self._model_cooldowns.get(d, 0.0) <= time.monotonic()
                            ):
                                candidates.append(d)
                    continue
                raise
            except (APITimeoutError, ProviderServerError) as exc:
                if self.auto_rotate_models and i < len(candidates):
                    self._mark_model_cooldown(candidate, duration=60.0)
                    logger.warning(
                        f"{self.provider_name}: Model '{candidate}' failed ({exc}). Trying next candidate..."
                    )
                    last_error = exc
                    continue
                raise

        if last_error:
            raise last_error
        raise RuntimeError(f"No candidate models available in provider '{self.provider_name}'.")

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
        """Stream real-time text chunks from a prompt via SSE."""
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
        messages: list[ChatMessage | dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream real-time tokens from a conversation thread via Server-Sent Events (SSE)."""
        candidates = await self._get_candidate_models(model, is_vision=False)
        target_model = candidates[0] if candidates else (model or self.default_model)
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
            async with client.stream(
                "POST", self.chat_endpoint, headers=headers, json=payload
            ) as response:
                if not response.is_success:
                    await response.aread()
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
            raise APITimeoutError(
                provider=self.provider_name, timeout_seconds=self.timeout
            ) from exc
        except (httpx.NetworkError, httpx.ConnectError) as exc:
            raise APIConnectionError(
                provider=self.provider_name, original_error=exc
            ) from exc

    @override
    async def list_models(self, *, free_only: bool = False) -> list[ModelInfo]:
        """Query standard /models endpoint and return normalized ModelInfo objects."""
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
                m_id = item.get("id") if isinstance(item, dict) else str(item)
                if not m_id:
                    continue
                models.append(
                    ModelInfo(
                        id=m_id,
                        name=item.get("name", m_id) if isinstance(item, dict) else m_id,
                        provider=self.provider_name,
                        is_free=False,
                        description=item.get("description")
                        if isinstance(item, dict)
                        else None,
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
