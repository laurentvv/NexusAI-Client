"""Google Gemini Providers (Free & Pro tiers via Google AI Studio / Gemini REST API).

Documentation: https://ai.google.dev/api/generate-content
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any, override

import httpx

from nexusai_client.config import Config, ProviderDefaults
from nexusai_client.exceptions import (
    APIConnectionError,
    APIResponseError,
    APITimeoutError,
    RateLimitError,
)
from nexusai_client.models import (
    AccountInfo,
    AIResponse,
    ChatMessage,
    ModelInfo,
    ModelPricing,
    ProviderType,
    ToolCall,
    ToolDefinition,
    UsageInfo,
)
from nexusai_client.providers.base import BaseAIProvider

logger = logging.getLogger("nexusai_client")


class GeminiBaseProvider(BaseAIProvider):
    """Base provider implementing the Google Generative Language REST API."""

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
        auto_rotate_models: bool = False,
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

    def _get_candidate_models(
        self, requested_model: str | None, *, is_vision: bool = False
    ) -> list[str]:
        """Order candidate models, placing available models before models in cooldown."""
        primary = requested_model or (
            self.default_vision_model if is_vision else self.default_model
        )
        fallbacks = (
            self.fallback_vision_models if is_vision else self.fallback_models
        )

        ordered: list[str] = [primary]
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

        return available + in_cooldown

    def _mark_model_cooldown(
        self, model: str, duration: float | None = None
    ) -> None:
        """Mark a model as temporarily rate-limited or unavailable."""
        cd = duration if duration is not None else self.cooldown_seconds
        self._model_cooldowns[model] = time.monotonic() + cd

    def _normalize_gemini_model(self, model: str) -> str:
        """Map common shorthand aliases to exact Google AI Studio model names."""
        clean = model.removeprefix("models/").strip()
        aliases = {
            "gemini-3.1-pro": "gemini-3.1-pro-preview",
            "gemini-3-pro": "gemini-3.1-pro-preview",
            "gemini-3-flash": "gemini-3-flash-preview",
        }
        return aliases.get(clean, clean)

    def _build_url(self, model: str, stream: bool = False) -> str:
        """Construct the endpoint URL for generateContent or streamGenerateContent."""
        clean_model = self._normalize_gemini_model(model)
        if stream:
            return f"{self.base_url}/models/{clean_model}:streamGenerateContent?alt=sse"
        return f"{self.base_url}/models/{clean_model}:generateContent"

    def _build_headers(self) -> dict[str, str]:
        """Construct headers for Gemini API."""
        return {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
            **self.extra_headers,
        }

    def _convert_tools_to_gemini(
        self,
        tools: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]] | None:
        """Convert OpenAI-style tool definitions into Gemini functionDeclarations format."""
        if not tools:
            return None

        declarations: list[dict[str, Any]] = []
        for tool in tools:
            if "functionDeclarations" in tool:
                return tools

            fn = (
                tool.get("function", tool)
                if tool.get("type") == "function" or "function" in tool
                else tool
            )
            decl: dict[str, Any] = {"name": fn.get("name", "")}
            if fn.get("description"):
                decl["description"] = fn["description"]
            if fn.get("parameters"):
                decl["parameters"] = fn["parameters"]
            declarations.append(decl)

        if declarations:
            return [{"functionDeclarations": declarations}]
        return None

    def _convert_messages_to_gemini(
        self,
        messages: list[ChatMessage | dict[str, Any]],
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        """Convert standard messages into Gemini format."""
        system_instruction: dict[str, Any] | None = None
        gemini_contents: list[dict[str, Any]] = []

        for msg in messages:
            if isinstance(msg, ChatMessage):
                role = msg.role
                content = msg.content
                tool_calls = msg.tool_calls
                name = msg.name
            else:
                role = str(msg.get("role", "user"))
                content = str(msg.get("content", "") or "")
                raw_tc = msg.get("tool_calls", [])
                tool_calls = (
                    [
                        ToolCall(
                            id=tc.get("id", ""),
                            name=tc.get("function", {}).get("name", ""),
                            arguments=json.loads(
                                tc.get("function", {}).get("arguments", "{}")
                            )
                            if isinstance(tc.get("function", {}).get("arguments"), str)
                            else tc.get("function", {}).get("arguments", {}),
                            raw_arguments=str(
                                tc.get("function", {}).get("arguments", "")
                            ),
                        )
                        for tc in raw_tc
                    ]
                    if raw_tc
                    else []
                )
                name = msg.get("name")

            if role == "system":
                system_instruction = {"parts": [{"text": content}]}
            elif role == "assistant":
                parts: list[dict[str, Any]] = []
                if content:
                    parts.append({"text": content})
                if tool_calls:
                    for tc in tool_calls:
                        parts.append(
                            {"functionCall": {"name": tc.name, "args": tc.arguments}}
                        )
                if not parts:
                    parts.append({"text": ""})
                gemini_contents.append(
                    {
                        "role": "model",
                        "parts": parts,
                    }
                )
            elif role == "tool":
                try:
                    parsed_content = (
                        json.loads(content)
                        if isinstance(content, str) and content.strip().startswith("{")
                        else content
                    )
                except Exception:
                    parsed_content = content

                response_obj = (
                    parsed_content
                    if isinstance(parsed_content, dict)
                    else {"output": parsed_content}
                )
                gemini_contents.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "functionResponse": {
                                    "name": name or "tool_function",
                                    "response": response_obj,
                                }
                            }
                        ],
                    }
                )
            else:
                gemini_contents.append(
                    {
                        "role": "user",
                        "parts": [{"text": content}],
                    }
                )

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
        json_mode: bool = False,
        tools: list[ToolDefinition | dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
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
        """Execute single Gemini Vision inference call."""
        from nexusai_client.utils import load_image_as_base64_and_mime

        b64_str, mime_type = load_image_as_base64_and_mime(image)

        parts: list[dict[str, Any]] = [
            {"text": prompt},
            {
                "inline_data": {
                    "mime_type": mime_type,
                    "data": b64_str,
                }
            },
        ]

        contents = [{"role": "user", "parts": parts}]
        payload: dict[str, Any] = {"contents": contents}

        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        gen_config: dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None:
            gen_config["maxOutputTokens"] = max_tokens
        if json_mode:
            gen_config["responseMimeType"] = "application/json"

        payload["generationConfig"] = gen_config
        payload.update(self.extra_params)
        payload.update(kwargs)

        url = f"{self.base_url}/models/{model}:generateContent?key={self.api_key}"
        client = await self.get_client()

        try:
            response = await client.post(
                url, json=payload, headers={"Content-Type": "application/json"}
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
            candidates = data.get("candidates", [])
            reply_text = ""
            if candidates:
                cand_parts = candidates[0].get("content", {}).get("parts", [])
                reply_text = "".join(p.get("text", "") for p in cand_parts)

            usage_meta = data.get("usageMetadata", {})
            prompt_tokens = usage_meta.get("promptTokenCount")
            completion_tokens = usage_meta.get("candidatesTokenCount")
            total_tokens = usage_meta.get("totalTokenCount")

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
                message=f"Failed to parse Gemini Vision response: {exc}",
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
        """Analyze an image, chart, PDF, or screenshot with automatic model rotation on 429."""
        candidates = self._get_candidate_models(model, is_vision=True)
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
                    f"Gemini Vision model '{candidate}' rate limited (429). Rotating to next model..."
                )
                last_error = exc
                continue
            except APIResponseError as exc:
                if exc.status_code in (400, 404) and "not found" in str(exc).lower():
                    self._mark_model_cooldown(candidate, duration=3600.0)
                    logger.warning(
                        f"Gemini Vision model '{candidate}' not available. Rotating..."
                    )
                    last_error = exc
                    continue
                raise

        if last_error:
            raise last_error
        raise RuntimeError("No candidate vision models available in Gemini provider.")

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
        """Send chat messages to a single Gemini model."""
        system_instruction, contents = self._convert_messages_to_gemini(messages)

        generation_config: dict[str, Any] = {
            "temperature": temperature,
        }
        if max_tokens is not None:
            generation_config["maxOutputTokens"] = max_tokens
        if json_mode:
            generation_config["responseMimeType"] = "application/json"

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": generation_config,
        }
        if system_instruction:
            payload["systemInstruction"] = system_instruction

        gemini_tools = self._convert_tools_to_gemini(self._normalize_tools(tools))
        if gemini_tools:
            payload["tools"] = gemini_tools
            if tool_choice is not None:
                if isinstance(tool_choice, str):
                    mode_map = {"auto": "AUTO", "none": "NONE", "required": "ANY"}
                    gemini_mode = mode_map.get(tool_choice.lower(), tool_choice.upper())
                    payload["toolConfig"] = {
                        "functionCallingConfig": {"mode": gemini_mode}
                    }
                elif isinstance(tool_choice, dict):
                    payload["toolConfig"] = tool_choice

        payload.update(self.extra_params)
        payload.update(kwargs)

        url = self._build_url(model, stream=False)
        headers = self._build_headers()
        client = await self.get_client()

        try:
            response = await client.post(url, headers=headers, json=payload)
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
            candidates = data.get("candidates", [])
            if not candidates:
                prompt_feedback = data.get("promptFeedback", {})
                block_reason = prompt_feedback.get(
                    "blockReason", "No response candidate generated"
                )
                raise APIResponseError(
                    provider=self.provider_name,
                    status_code=response.status_code,
                    response_body=response.text,
                    message=f"Gemini returned no candidates: {block_reason}",
                )

            candidate = candidates[0]
            parts = candidate.get("content", {}).get("parts", [])
            text_parts: list[str] = []
            parsed_tool_calls: list[ToolCall] = []

            for i, part in enumerate(parts):
                if "text" in part:
                    text_parts.append(part["text"])
                if "functionCall" in part:
                    fc = part["functionCall"]
                    fc_name = fc.get("name", "")
                    fc_args = fc.get("args", {})
                    parsed_tool_calls.append(
                        ToolCall(
                            id=f"call_{fc_name}_{i + 1}",
                            name=fc_name,
                            arguments=fc_args if isinstance(fc_args, dict) else {},
                            raw_arguments=json.dumps(fc_args)
                            if isinstance(fc_args, dict)
                            else str(fc_args),
                        )
                    )

            text_output = "".join(text_parts)
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
                message=f"Failed to parse Gemini response: {exc}",
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
        """Send chat messages to Gemini REST API with automatic model rotation on rate limits."""
        candidates = self._get_candidate_models(model, is_vision=False)
        last_error: Exception | None = None

        for candidate in candidates:
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
                    f"Gemini model '{candidate}' rate limited (429). Rotating to next model..."
                )
                last_error = exc
                continue
            except APIResponseError as exc:
                if exc.status_code in (400, 404) and "not found" in str(exc).lower():
                    self._mark_model_cooldown(candidate, duration=3600.0)
                    logger.warning(
                        f"Gemini model '{candidate}' not available. Rotating..."
                    )
                    last_error = exc
                    continue
                raise

        if last_error:
            raise last_error
        raise RuntimeError("No candidate models available in Gemini provider.")

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
        """Stream generated text from a prompt using Gemini SSE."""
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

    async def _single_stream_chat(
        self,
        *,
        model: str,
        messages: list[ChatMessage | dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream tokens from a single Gemini model."""
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

        url = self._build_url(model, stream=True)
        headers = self._build_headers()
        client = await self.get_client()

        try:
            async with client.stream(
                "POST", url, headers=headers, json=payload
            ) as response:
                self._handle_http_error(response)

                async for line in response.aiter_lines():
                    clean_line = line.strip()
                    if not clean_line or not clean_line.startswith("data:"):
                        continue

                    data_str = clean_line.removeprefix("data:").strip()
                    try:
                        chunk_json = json.loads(data_str)
                        candidates = chunk_json.get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            for part in parts:
                                text_val = part.get("text")
                                if text_val:
                                    yield text_val
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
    async def stream_chat(
        self,
        messages: list[ChatMessage | dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream generated tokens in real time with automatic model rotation on rate limits."""
        candidates = self._get_candidate_models(model, is_vision=False)
        last_error: Exception | None = None

        for candidate in candidates:
            streamed_any = False
            try:
                async for chunk in self._single_stream_chat(
                    model=candidate,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                ):
                    streamed_any = True
                    yield chunk
                return
            except RateLimitError as exc:
                if streamed_any:
                    raise
                self._mark_model_cooldown(candidate)
                logger.warning(
                    f"Gemini stream model '{candidate}' rate limited (429). Rotating to next model..."
                )
                last_error = exc
                continue
            except APIResponseError as exc:
                if streamed_any:
                    raise
                if exc.status_code in (400, 404) and "not found" in str(exc).lower():
                    self._mark_model_cooldown(candidate, duration=3600.0)
                    logger.warning(
                        f"Gemini stream model '{candidate}' not available. Rotating..."
                    )
                    last_error = exc
                    continue
                raise

        if last_error:
            raise last_error
        raise RuntimeError("No candidate models available for Gemini streaming.")

    @override
    async def list_models(self, *, free_only: bool = False) -> list[ModelInfo]:
        """Query Google AI Studio models endpoint."""
        url = f"{self.base_url}/models"
        headers = self._build_headers()
        client = await self.get_client()

        try:
            response = await client.get(url, headers=headers)
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
            default_vision_model=config.default_vision_model,
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
    """Gemini Free Provider (Google AI Studio Free Tier) with intelligent Model Rotation."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        fallback_models: list[str] | tuple[str, ...] | None = None,
        fallback_vision_models: list[str] | tuple[str, ...] | None = None,
        auto_rotate_models: bool = True,
        cooldown_seconds: float = 60.0,
        **kwargs: Any,
    ) -> None:
        config = Config.get_provider_config(
            ProviderType.GEMINI_FREE,
            api_key=api_key,
            base_url=base_url,
            model=model or ProviderDefaults.GEMINI_FREE_MODEL,
            timeout=timeout,
        )
        resolved_fallbacks = (
            fallback_models
            if fallback_models is not None
            else list(ProviderDefaults.GEMINI_FREE_FALLBACK_MODELS)
        )
        resolved_vision_fallbacks = (
            fallback_vision_models
            if fallback_vision_models is not None
            else list(ProviderDefaults.GEMINI_FREE_VISION_FALLBACK_MODELS)
        )
        super().__init__(
            api_key=config.api_key,
            base_url=config.base_url,
            default_model=config.default_model,
            default_vision_model=config.default_vision_model,
            timeout=config.timeout,
            fallback_models=resolved_fallbacks,
            fallback_vision_models=resolved_vision_fallbacks,
            auto_rotate_models=auto_rotate_models,
            cooldown_seconds=cooldown_seconds,
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
        """Account information for Gemini Free tier with active model rotation status."""
        active_candidates = self._get_candidate_models(None)
        current_active = (
            active_candidates[0] if active_candidates else self.default_model
        )
        now = time.monotonic()
        cooldowns = [
            f"{m} ({int(exp - now)}s restants)"
            for m, exp in self._model_cooldowns.items()
            if exp > now
        ]
        return AccountInfo(
            provider=self.provider_name,
            is_free_tier=True,
            rate_limit_info="15 RPM | 250K TPM | 500 RPD (Flash Lite) / 20 RPD (Flash) / 14.4k RPD (Gemma)",
            total_balance=0.0,
            extra_details={
                "platform": "Google AI Studio Free Tier",
                "default_model": self.default_model,
                "current_active_model": current_active,
                "rotation_enabled": self.auto_rotate_models,
                "available_models_rotation": self.fallback_models,
                "models_in_cooldown": cooldowns
                if cooldowns
                else "None (Tous les modèles opérationnels)",
            },
        )
