"""Base abstract class for all AI providers in NexusAI-Client.

Defines the common asynchronous interface, HTTP connection management,
model discovery, streaming, and account budget/quota inspection.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any, Self

import httpx

from nexusai_client.exceptions import (
    APIResponseError,
    AuthenticationError,
    ProviderServerError,
    RateLimitError,
)
from nexusai_client.models import (
    AccountInfo,
    AIResponse,
    ChatMessage,
    ModelInfo,
)


class BaseAIProvider(ABC):
    """Abstract Base Class for all AI model providers.

    All concrete providers implement asynchronous generation, streaming, chat,
    and model discovery.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        default_model: str,
        default_vision_model: str | None = None,
        timeout: float = 60.0,
        extra_headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the AI provider."""
        self.api_key: str = api_key
        self.base_url: str = base_url.rstrip("/")
        self.default_model: str = default_model
        self.default_vision_model: str | None = default_vision_model
        self.timeout: float = timeout
        self.extra_headers: dict[str, str] = extra_headers or {}
        self.extra_params: dict[str, Any] = {
            k: v for k, v in kwargs.items() if v is not None and k not in ("model", "default_model", "vision_model")
        }
        self._client: httpx.AsyncClient | None = None

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the unique provider identifier."""
        ...

    @abstractmethod
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
        """Generate a single text response from a prompt."""
        ...

    @abstractmethod
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
        """Analyze an image, chart, PDF or screenshot using multimodal models."""
        ...

    @abstractmethod
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
        """Send a full conversation thread of messages to the model."""
        ...

    @abstractmethod
    def stream_text(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream generated text tokens in real time from a prompt."""
        ...

    @abstractmethod
    def stream_chat(
        self,
        messages: list[ChatMessage | dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream generated text tokens in real time from a multi-turn conversation."""
        ...

    @abstractmethod
    async def list_models(self, *, free_only: bool = False) -> list[ModelInfo]:
        """Fetch available models from the provider."""
        ...

    @abstractmethod
    async def get_account_info(self) -> AccountInfo:
        """Fetch remaining account balance, credit limits, usage, or quota information."""
        ...

    # ---------------------------------------------------------
    # HTTP Client Lifecycle & Connection Management
    # ---------------------------------------------------------

    async def get_client(self) -> httpx.AsyncClient:
        """Retrieve or initialize the underlying httpx AsyncClient."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                follow_redirects=True,
            )
        return self._client

    async def close(self) -> None:
        """Close the underlying HTTP client session."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> Self:
        """Async context manager entry point."""
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: Any,
    ) -> None:
        """Async context manager exit point."""
        await self.close()

    # ---------------------------------------------------------
    # Error Handling & Utilities
    # ---------------------------------------------------------

    def _handle_http_error(self, response: httpx.Response) -> None:
        """Map HTTP error status codes to custom NexusAI exceptions."""
        if response.is_success:
            return

        status = response.status_code
        text = response.text

        try:
            err_json = response.json()
            msg = (
                err_json.get("error", {}).get("message")
                if isinstance(err_json.get("error"), dict)
                else err_json.get("message") or err_json.get("error") or text
            )
        except Exception:
            msg = text

        match status:
            case 401 | 403:
                raise AuthenticationError(
                    provider=self.provider_name,
                    status_code=status,
                    response_body=text,
                    message=f"Authentication failed: {msg}",
                )
            case 429:
                raise RateLimitError(
                    provider=self.provider_name,
                    status_code=status,
                    response_body=text,
                    message=f"Rate limit or quota exceeded: {msg}",
                )
            case code if code >= 500:
                raise ProviderServerError(
                    provider=self.provider_name,
                    status_code=status,
                    response_body=text,
                    message=f"Provider server error ({status}): {msg}",
                )
            case _:
                raise APIResponseError(
                    provider=self.provider_name,
                    status_code=status,
                    response_body=text,
                    message=msg,
                )

    def _normalize_messages(
        self,
        messages: list[ChatMessage | dict[str, str]],
    ) -> list[dict[str, str]]:
        """Normalize ChatMessage objects or dicts into standard dictionary format."""
        normalized: list[dict[str, str]] = []
        for msg in messages:
            if isinstance(msg, ChatMessage):
                normalized.append(msg.to_dict())
            elif isinstance(msg, dict):
                normalized.append({"role": str(msg["role"]), "content": str(msg["content"])})
            else:
                raise TypeError(f"Invalid message type: {type(msg)}. Expected ChatMessage or dict.")
        return normalized
