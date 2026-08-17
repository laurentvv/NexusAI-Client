"""Custom exception hierarchy for NexusAI-Client.

Provides fine-grained and clean exception handling for configuration,
network, provider, and API response errors.
"""

from __future__ import annotations

from typing import Any


class NexusAIError(Exception):
    """Base exception for all NexusAI-Client errors."""

    def __init__(self, message: str, *args: Any) -> None:
        super().__init__(message, *args)
        self.message = message


# ---------------------------------------------------------
# Configuration Errors
# ---------------------------------------------------------


class ConfigurationError(NexusAIError):
    """Raised when there is an issue with client or provider configuration."""


class MissingAPIKeyError(ConfigurationError):
    """Raised when a required API key is missing from environment or parameters."""

    def __init__(self, provider: str, env_var: str | None = None) -> None:
        message = f"API key for provider '{provider}' is missing."
        if env_var:
            message += f" Please set the '{env_var}' environment variable in your .env file or pass 'api_key' directly."
        super().__init__(message)
        self.provider = provider
        self.env_var = env_var


# ---------------------------------------------------------
# Provider Lookup Errors
# ---------------------------------------------------------


class ProviderError(NexusAIError):
    """Base class for provider-related lookup or capability errors."""


class ProviderNotFoundError(ProviderError):
    """Raised when an unknown or unsupported provider is requested in AIGateway."""

    def __init__(
        self, provider: str, available_providers: list[str] | None = None
    ) -> None:
        message = f"Provider '{provider}' is not supported."
        if available_providers:
            message += f" Available providers are: {', '.join(available_providers)}"
        super().__init__(message)
        self.provider = provider
        self.available_providers = available_providers or []


class InvalidModelError(ProviderError):
    """Raised when an invalid or unsupported model name is supplied."""


# ---------------------------------------------------------
# Network & Connection Errors
# ---------------------------------------------------------


class NetworkError(NexusAIError):
    """Raised when an HTTP or socket-level network issue occurs."""


class APIConnectionError(NetworkError):
    """Raised when the client fails to establish a connection to the provider API."""

    def __init__(self, provider: str, original_error: Exception | str) -> None:
        message = f"Failed to connect to provider '{provider}': {original_error}"
        super().__init__(message)
        self.provider = provider
        self.original_error = original_error


class APITimeoutError(NetworkError):
    """Raised when a request to the provider API times out."""

    def __init__(self, provider: str, timeout_seconds: float) -> None:
        message = (
            f"Request to provider '{provider}' timed out after {timeout_seconds:.1f}s."
        )
        super().__init__(message)
        self.provider = provider
        self.timeout_seconds = timeout_seconds


# ---------------------------------------------------------
# API Response & HTTP Errors
# ---------------------------------------------------------


class APIResponseError(NexusAIError):
    """Raised when the provider API returns a non-2xx HTTP status code."""

    def __init__(
        self,
        provider: str,
        status_code: int,
        response_body: str,
        message: str | None = None,
    ) -> None:
        detail = message or response_body
        full_message = f"Provider '{provider}' returned HTTP {status_code}: {detail}"
        super().__init__(full_message)
        self.provider = provider
        self.status_code = status_code
        self.response_body = response_body


class AuthenticationError(APIResponseError):
    """Raised when the provider returns HTTP 401 or 403 (invalid/expired credentials)."""


class RateLimitError(APIResponseError):
    """Raised when the provider returns HTTP 429 (quota exceeded or rate limit hit)."""


class ProviderServerError(APIResponseError):
    """Raised when the provider returns an HTTP 5xx server error."""
