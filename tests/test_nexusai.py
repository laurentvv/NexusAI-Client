"""Comprehensive unit test suite for NexusAI-Client."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from nexusai_client import (
    AccountInfo,
    AIGateway,
    AIResponse,
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BaseAIProvider,
    CerebrasProvider,
    ChatMessage,
    CohereProvider,
    Config,
    DeepSeekProvider,
    FallbackGateway,
    GeminiFreeProvider,
    GeminiProProvider,
    GroqProvider,
    MissingAPIKeyError,
    MistralProvider,
    ModelInfo,
    NvidiaProvider,
    OpenRouterProvider,
    ProviderNotFoundError,
    ProviderType,
    RateLimitError,
)


def test_missing_api_key_raises() -> None:
    """Test that creating a provider without API key raises MissingAPIKeyError."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(MissingAPIKeyError) as exc_info:
            AIGateway(provider="deepseek")
        assert exc_info.value.provider == "DeepSeek"
        assert exc_info.value.env_var == "DEEPSEEK_API_KEY"


def test_invalid_provider_raises() -> None:
    """Test that requesting an unknown provider raises ProviderNotFoundError."""
    with pytest.raises(ProviderNotFoundError) as exc_info:
        AIGateway(provider="unknown_provider_xyz")
    assert "unknown_provider_xyz" in str(exc_info.value)
    assert len(exc_info.value.available_providers) > 0


def test_instantiate_with_direct_key() -> None:
    """Test instantiating all 6 providers with explicit API keys."""
    # 1. DeepSeek
    client_ds = AIGateway(provider="deepseek", api_key="sk-test-deepseek")
    assert isinstance(client_ds.provider, DeepSeekProvider)
    assert client_ds.provider.api_key == "sk-test-deepseek"

    # 2. Gemini Pro
    client_gp = AIGateway(provider="gemini_pro", api_key="test-gemini-pro")
    assert isinstance(client_gp.provider, GeminiProProvider)
    assert client_gp.provider.api_key == "test-gemini-pro"

    # 3. Gemini Free
    client_gf = AIGateway(provider="gemini_free", api_key="test-gemini-free")
    assert isinstance(client_gf.provider, GeminiFreeProvider)
    assert client_gf.provider.api_key == "test-gemini-free"

    # 4. Mistral
    client_m = AIGateway(provider="mistral", api_key="test-mistral")
    assert isinstance(client_m.provider, MistralProvider)
    assert client_m.provider.api_key == "test-mistral"

    # 5. Nvidia
    client_nv = AIGateway(provider="nvidia_free", api_key="test-nvidia")
    assert isinstance(client_nv.provider, NvidiaProvider)
    assert client_nv.provider.api_key == "test-nvidia"

    # 6. Groq
    client_groq = AIGateway(provider="groq", api_key="test-groq")
    assert isinstance(client_groq.provider, GroqProvider)
    assert client_groq.provider.api_key == "test-groq"

    # 7. Cerebras
    client_cb = AIGateway(provider="cerebras", api_key="test-cerebras")
    assert isinstance(client_cb.provider, CerebrasProvider)
    assert client_cb.provider.api_key == "test-cerebras"

    # 8. Cohere
    client_co = AIGateway(provider="cohere", api_key="test-cohere")
    assert isinstance(client_co.provider, CohereProvider)
    assert client_co.provider.api_key == "test-cohere"

    # 9. OpenRouter
    client_or = AIGateway(provider="openrouter", api_key="test-openrouter")
    assert isinstance(client_or.provider, OpenRouterProvider)
    assert client_or.provider.api_key == "test-openrouter"


def test_factory_create_method() -> None:
    """Test AIGateway.create() returns direct BaseAIProvider instances."""
    p_deepseek = AIGateway.create("deepseek", api_key="test-key")
    assert isinstance(p_deepseek, DeepSeekProvider)

    p_gemini = AIGateway.create("gemini_free", api_key="test-key")
    assert isinstance(p_gemini, GeminiFreeProvider)

    p_groq = AIGateway.create("groq", api_key="test-key")
    assert isinstance(p_groq, GroqProvider)

    p_cerebras = AIGateway.create("cerebras", api_key="test-key")
    assert isinstance(p_cerebras, CerebrasProvider)

    p_cohere = AIGateway.create("cohere", api_key="test-key")
    assert isinstance(p_cohere, CohereProvider)

    p_mistral = AIGateway.create("mistral", api_key="test-key")
    assert isinstance(p_mistral, MistralProvider)

    p_nvidia = AIGateway.create("nvidia", api_key="test-key")
    assert isinstance(p_nvidia, NvidiaProvider)

    p_openrouter = AIGateway.create("openrouter", api_key="test-key")
    assert isinstance(p_openrouter, OpenRouterProvider)


def test_available_providers() -> None:
    """Test available_providers list."""
    providers = AIGateway.available_providers()
    assert "cerebras (or cerebras_free)" in providers
    assert "cohere (or cohere_free)" in providers
    assert "deepseek" in providers
    assert "gemini_free" in providers
    assert "gemini_pro" in providers
    assert "groq (or groq_free)" in providers
    assert "mistral" in providers


@pytest.mark.asyncio
async def test_fallback_gateway_success_first_attempt() -> None:
    """Test FallbackGateway succeeds on primary provider."""
    p1 = AIGateway.create("gemini_free", api_key="test-key")
    p2 = AIGateway.create("openrouter", api_key="test-key")

    mock_resp = AIResponse(text="Success from P1", provider="gemini_free", model="gemini-2.5-flash")
    with patch.object(p1, "generate_text", new_callable=AsyncMock, return_value=mock_resp):
        with patch.object(p2, "generate_text", new_callable=AsyncMock) as mock_p2:
            gateway = FallbackGateway([p1, p2])
            res = await gateway.generate_text("Test prompt")
            assert res.text == "Success from P1"
            mock_p2.assert_not_called()


@pytest.mark.asyncio
async def test_fallback_gateway_fails_over_to_second() -> None:
    """Test FallbackGateway fails over to secondary when primary fails."""
    p1 = AIGateway.create("gemini_free", api_key="test-key")
    p2 = AIGateway.create("openrouter", api_key="test-key")

    mock_resp_p2 = AIResponse(text="Success from P2", provider="openrouter", model="openrouter/free")
    with patch.object(
        p1,
        "generate_text",
        side_effect=RateLimitError(
            provider="gemini_free",
            status_code=429,
            response_body="Quota exceeded",
            message="Quota",
        ),
    ):
        with patch.object(p2, "generate_text", new_callable=AsyncMock, return_value=mock_resp_p2):
            gateway = FallbackGateway([p1, p2])
            res = await gateway.generate_text("Test prompt")
            assert res.text == "Success from P2"
            assert res.provider == "openrouter"


@pytest.mark.asyncio
async def test_get_configured_providers_and_auto_fallback() -> None:
    """Test dynamic chain generation and auto_fallback."""
    with patch.dict(
        os.environ,
        {
            "GEMINI_FREE_API_KEY": "test-gemini-key",
            "DEEPSEEK_API_KEY": "test-deepseek-key",
        },
        clear=True,
    ):
        chain = AIGateway.get_configured_providers(prioritize_free=True)
        assert "gemini_free" in chain
        assert "deepseek" in chain
        assert chain.index("gemini_free") < chain.index("deepseek")

        auto_gateway = AIGateway.auto_fallback()
        assert isinstance(auto_gateway, FallbackGateway)
        assert len(auto_gateway._provider_chain) >= 2


@pytest.mark.asyncio
async def test_deepseek_account_balance_mocked() -> None:
    """Test DeepSeek get_account_info method."""
    mock_payload = {
        "is_available": True,
        "balance_infos": [
            {
                "currency": "USD",
                "total_balance": "15.50",
                "granted_balance": "5.00",
                "topped_up_balance": "10.50",
            }
        ],
    }

    mock_response = httpx.Response(
        status_code=200,
        json=mock_payload,
        request=httpx.Request("GET", "https://api.deepseek.com/user/balance"),
    )

    async with AIGateway(provider="deepseek", api_key="test-key") as client:
        with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            info = await client.get_account_info()
            assert isinstance(info, AccountInfo)
            assert info.total_balance == 15.50
            assert info.granted_balance == 5.00
            assert info.currency == "USD"
            assert "Solde restant: $15.50" in info.format_summary()


@pytest.mark.asyncio
async def test_openrouter_account_info_mocked() -> None:
    """Test OpenRouter get_account_info method."""
    mock_auth_payload = {
        "data": {
            "label": "My Dev Key",
            "usage": 1.25,
            "limit": 10.0,
            "is_free_tier": False,
            "rate_limit": {"requests": 20, "interval": "10s"},
        }
    }

    mock_response = httpx.Response(
        status_code=200,
        json=mock_auth_payload,
        request=httpx.Request("GET", "https://openrouter.ai/api/v1/auth/key"),
    )

    async with AIGateway(provider="openrouter", api_key="test-key") as client:
        with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            info = await client.get_account_info()
            assert isinstance(info, AccountInfo)
            assert info.total_usage == 1.25
            assert info.total_balance == 8.75
            assert "Solde restant: $8.75" in info.format_summary()


@pytest.mark.asyncio
async def test_groq_account_info_and_models() -> None:
    """Test Groq account info and models discovery."""
    mock_models_payload = {
        "data": [
            {"id": "llama-3.3-70b-versatile", "object": "model", "owned_by": "meta"},
            {"id": "mixtral-8x7b-32768", "object": "model", "owned_by": "mistralai"},
        ]
    }
    mock_response = httpx.Response(
        status_code=200,
        json=mock_models_payload,
        request=httpx.Request("GET", "https://api.groq.com/openai/v1/models"),
    )
    async with AIGateway(provider="groq", api_key="gsk_test") as client:
        info = await client.get_account_info()
        assert info.provider == "groq"
        assert "30 RPM" in (info.rate_limit_info or "")

        with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            models = await client.list_models()
            assert len(models) == 2
            assert models[0].id == "llama-3.3-70b-versatile"
            assert models[0].is_free is True
            assert models[0].context_length == 128_000


@pytest.mark.asyncio
async def test_cerebras_account_info_and_models() -> None:
    """Test Cerebras account info and models discovery."""
    mock_models_payload = {
        "data": [
            {"id": "llama-3.3-70b", "object": "model", "owned_by": "meta"},
        ]
    }
    mock_response = httpx.Response(
        status_code=200,
        json=mock_models_payload,
        request=httpx.Request("GET", "https://api.cerebras.ai/v1/models"),
    )
    async with AIGateway(provider="cerebras", api_key="csk_test") as client:
        info = await client.get_account_info()
        assert info.provider == "cerebras"
        assert "30 RPM" in (info.rate_limit_info or "")

        with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            models = await client.list_models()
            assert len(models) == 1
            assert models[0].id == "llama-3.3-70b"
            assert models[0].is_free is True


@pytest.mark.asyncio
async def test_cohere_account_info_and_generation() -> None:
    """Test Cohere account info, models discovery, and V2 chat generation."""
    mock_chat_payload = {
        "id": "cohere-msg-123",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello from Cohere Command R+"}],
        },
        "usage": {
            "tokens": {"input_tokens": 10, "output_tokens": 15},
        },
    }
    mock_response = httpx.Response(
        status_code=200,
        json=mock_chat_payload,
        request=httpx.Request("POST", "https://api.cohere.com/v2/chat"),
    )
    async with AIGateway(provider="cohere", api_key="cohere_test") as client:
        info = await client.get_account_info()
        assert info.provider == "cohere"
        assert "20 RPM" in (info.rate_limit_info or "")

        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            res = await client.generate_text("Hi Cohere")
            assert res.text == "Hello from Cohere Command R+"
            assert res.provider == "cohere"
            assert res.usage is not None
            assert res.usage.total_tokens == 25


@pytest.mark.asyncio
async def test_list_models_openrouter_mocked() -> None:
    """Test OpenRouter list_models parsing pricing and free filter."""
    mock_payload = {
        "data": [
            {
                "id": "google/gemini-2.0-flash-exp:free",
                "name": "Google: Gemini 2.0 Flash Exp (free)",
                "context_length": 1048576,
                "pricing": {"prompt": "0", "completion": "0"},
            },
            {
                "id": "openai/gpt-4o",
                "name": "OpenAI: GPT-4o",
                "context_length": 128000,
                "pricing": {"prompt": "0.0000025", "completion": "0.000010"},
            },
        ]
    }

    mock_response = httpx.Response(
        status_code=200,
        json=mock_payload,
        request=httpx.Request("GET", "https://openrouter.ai/api/v1/models"),
    )

    async with AIGateway(provider="openrouter", api_key="test-key") as client:
        with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            all_models = await client.list_models()
            assert len(all_models) == 2
            assert all_models[0].is_free is True
            assert all_models[1].is_free is False
            assert all_models[1].pricing is not None
            assert all_models[1].pricing.prompt_per_million == 2.5

            free_models = await client.list_models(free_only=True)
            assert len(free_models) == 1
            assert free_models[0].id == "google/gemini-2.0-flash-exp:free"


@pytest.mark.asyncio
async def test_openai_compatible_generation_mocked() -> None:
    """Test OpenAI-compatible generation with mocked HTTP response."""
    mock_payload = {
        "id": "chatcmpl-123",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello from OpenRouter!",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }

    mock_response = httpx.Response(
        status_code=200,
        json=mock_payload,
        request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
    )

    async with AIGateway(provider="openrouter", api_key="test-key") as client:
        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            resp = await client.generate_text(
                "Bonjour !",
                system_prompt="Tu es un assistant.",
            )

            assert isinstance(resp, AIResponse)
            assert resp.text == "Hello from OpenRouter!"
            assert resp.provider == "openrouter"
            assert resp.usage is not None
            assert resp.usage.total_tokens == 15
            assert resp.finish_reason == "stop"


@pytest.mark.asyncio
async def test_gemini_generation_mocked() -> None:
    """Test Google Gemini generation with mocked HTTP response."""
    mock_payload = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "Hello from Gemini!"}],
                    "role": "model",
                },
                "finishReason": "STOP",
                "index": 0,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 12,
            "candidatesTokenCount": 8,
            "totalTokenCount": 20,
        },
    }

    mock_response = httpx.Response(
        status_code=200,
        json=mock_payload,
        request=httpx.Request("POST", "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"),
    )

    async with AIGateway(provider="gemini_free", api_key="test-gemini-key") as client:
        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            resp = await client.generate_text(
                "Hello Gemini!",
                system_prompt="You are an expert.",
            )

            assert isinstance(resp, AIResponse)
            assert resp.text == "Hello from Gemini!"
            assert resp.provider == "gemini_free"
            assert resp.usage is not None
            assert resp.usage.total_tokens == 20
            assert resp.finish_reason == "STOP"


@pytest.mark.asyncio
async def test_chat_multi_turn() -> None:
    """Test chat multi-turn functionality."""
    mock_payload = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "Je vais très bien, merci !"},
                "finish_reason": "stop",
            }
        ]
    }
    mock_response = httpx.Response(
        status_code=200,
        json=mock_payload,
        request=httpx.Request("POST", "https://integrate.api.nvidia.com/v1/chat/completions"),
    )

    client = AIGateway("nvidia_free", api_key="test-nv-key")
    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        messages = [
            ChatMessage(role="user", content="Bonjour, comment vas-tu ?"),
        ]
        resp = await client.chat(messages=messages)
        assert resp.text == "Je vais très bien, merci !"
        assert resp.provider == "nvidia"
    await client.close()


@pytest.mark.asyncio
async def test_auth_error_handling() -> None:
    """Test that HTTP 401 raises AuthenticationError."""
    mock_response = httpx.Response(
        status_code=401,
        json={"error": {"message": "Invalid API key"}},
        request=httpx.Request("POST", "https://api.deepseek.com/chat/completions"),
    )

    async with AIGateway(provider="deepseek", api_key="invalid-key") as client:
        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            with pytest.raises(AuthenticationError) as exc_info:
                await client.generate_text("Test prompt")

            assert exc_info.value.status_code == 401
            assert exc_info.value.provider == "deepseek"


@pytest.mark.asyncio
async def test_rate_limit_error_handling() -> None:
    """Test that HTTP 429 raises RateLimitError."""
    mock_response = httpx.Response(
        status_code=429,
        json={"error": "Too Many Requests"},
        request=httpx.Request("POST", "https://api.mistral.ai/v1/chat/completions"),
    )

    async with AIGateway(provider="mistral", api_key="test-key") as client:
        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            with pytest.raises(RateLimitError) as exc_info:
                await client.generate_text("Test prompt")

            assert exc_info.value.status_code == 429
            assert exc_info.value.provider == "mistral"


@pytest.mark.asyncio
async def test_connection_error_handling() -> None:
    """Test that network failures raise APIConnectionError."""
    async with AIGateway(provider="mistral", api_key="test-key") as client:
        with patch.object(httpx.AsyncClient, "post", side_effect=httpx.ConnectError("Connection refused")):
            with pytest.raises(APIConnectionError) as exc_info:
                await client.generate_text("Test prompt")
            assert "mistral" in exc_info.value.provider


@pytest.mark.asyncio
async def test_timeout_error_handling() -> None:
    """Test that timeouts raise APITimeoutError."""
    async with AIGateway(provider="deepseek", api_key="test-key", timeout=5.0) as client:
        with patch.object(httpx.AsyncClient, "post", side_effect=httpx.TimeoutException("Read timed out")):
            with pytest.raises(APITimeoutError) as exc_info:
                await client.generate_text("Test prompt")
            assert exc_info.value.timeout_seconds == 5.0


def test_utils_image_encoding() -> None:
    """Test image loading from raw bytes and data URI formatting."""
    from nexusai_client.utils import load_image_as_base64_and_mime, load_image_as_data_uri

    # Synthetic 1x1 PNG bytes
    png_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4"
    b64, mime = load_image_as_base64_and_mime(png_bytes)
    assert mime == "image/png"
    assert len(b64) > 0

    data_uri = load_image_as_data_uri(png_bytes)
    assert data_uri.startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_analyze_image_gemini_mocked() -> None:
    """Test Gemini Vision analyze_image request and response parsing."""
    mock_payload = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "A futuristic glowing banner."}],
                    "role": "model",
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 50,
            "candidatesTokenCount": 8,
            "totalTokenCount": 58,
        },
    }
    mock_response = httpx.Response(
        status_code=200,
        json=mock_payload,
        request=httpx.Request("POST", "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"),
    )

    async with AIGateway("gemini_free", api_key="test-key") as client:
        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            png_bytes = b"\x89PNG\r\n\x1a\n"
            res = await client.analyze_image("Describe this image", png_bytes)
            assert res.text == "A futuristic glowing banner."
            assert res.provider == "gemini_free"
            assert res.model == "gemini-2.5-flash"
            assert res.usage is not None
            assert res.usage.total_tokens == 58


@pytest.mark.asyncio
async def test_analyze_image_openai_compat_mocked() -> None:
    """Test OpenAI-compatible / Nvidia / Mistral Vision analyze_image."""
    mock_payload = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "Visual description of the chart."},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 60, "completion_tokens": 10, "total_tokens": 70},
    }
    mock_response = httpx.Response(
        status_code=200,
        json=mock_payload,
        request=httpx.Request("POST", "https://integrate.api.nvidia.com/v1/chat/completions"),
    )

    async with AIGateway("nvidia_free", api_key="test-key") as client:
        with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            png_bytes = b"\x89PNG\r\n\x1a\n"
            res = await client.analyze_image("Explain the chart", png_bytes)
            assert res.text == "Visual description of the chart."
            assert res.provider == "nvidia"
            assert res.model == "meta/llama-3.2-11b-vision-instruct"

