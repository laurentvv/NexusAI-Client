"""Access validation and diagnostic benchmark for NexusAI-Client.

Verifies .env configuration, queries live account balances/quotas,
retrieves the model catalog (free vs paid with per-1M token costs),
and performs a live inference test to benchmark latency.
"""

from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import dataclass
from typing import Any

# Ensure UTF-8 output encoding on Windows platforms
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from nexusai_client import (
    AIGateway,
    AuthenticationError,
    MissingAPIKeyError,
    NexusAIError,
    RateLimitError,
)

PROVIDERS_TO_TEST = [
    {
        "id": "deepseek",
        "name": "1. DeepSeek (Paid API)",
        "env_key": "DEEPSEEK_API_KEY",
        "is_paid": True,
    },
    {
        "id": "gemini_pro",
        "name": "2. Google Gemini Pro (Paid Tier)",
        "env_key": "GEMINI_PRO_API_KEY",
        "is_paid": True,
    },
    {
        "id": "gemini_free",
        "name": "3. Google Gemini Free (Google AI Studio)",
        "env_key": "GEMINI_FREE_API_KEY",
        "is_paid": False,
    },
    {
        "id": "groq",
        "name": "4. Groq (Free Tier / Ultra-Fast LPU)",
        "env_key": "GROQ_API_KEY",
        "is_paid": False,
    },
    {
        "id": "cerebras",
        "name": "5. Cerebras (Free Tier / Wafer-Scale CS-3)",
        "env_key": "CEREBRAS_API_KEY",
        "is_paid": False,
    },
    {
        "id": "cohere",
        "name": "6. Cohere (Free Trial Tier / Command R+)",
        "env_key": "COHERE_API_KEY",
        "is_paid": False,
    },
    {
        "id": "mistral",
        "name": "7. Mistral AI (Free Tier / Platform)",
        "env_key": "MISTRAL_API_KEY",
        "is_paid": False,
    },
    {
        "id": "nvidia_free",
        "name": "8. Nvidia NIM (Free Tier / NGC)",
        "env_key": "NVIDIA_API_KEY",
        "is_paid": False,
    },
    {
        "id": "openrouter",
        "name": "9. OpenRouter (Free Tier / :free Models)",
        "env_key": "OPENROUTER_API_KEY",
        "is_paid": False,
    },
]


@dataclass
class ProviderTestResult:
    provider_id: str
    name: str
    status: str
    budget_info: str = ""
    latency_ms: float = 0.0
    models_count: int = 0
    test_response: str = ""
    error_message: str = ""


async def test_single_provider(item: dict[str, Any]) -> ProviderTestResult:
    """Execute complete validation suite for a given provider."""
    p_id = item["id"]
    p_name = item["name"]

    print("\n" + "=" * 75)
    print(f"🔍 Connection Test : {p_name}")
    print("=" * 75)

    try:
        async with AIGateway(provider=p_id) as client:
            print(f"🔑 API Key detected for '{p_id}'.")

            # 1. Budget & Quota Inspection
            print("💰 Fetching account balance and rate limits...")
            budget_str = "N/A"
            try:
                acc_info = await client.get_account_info()
                budget_str = acc_info.format_summary()
                print(f"  💵 Account Status : {budget_str}")
            except Exception as e:
                print(f"  ⚠️ Could not fetch balance/quota: {e}")

            # 2. Model Catalog Discovery
            print("\n📋 Fetching model catalog...")
            try:
                models = await client.list_models()
                free_models = [m for m in models if m.is_free]
                paid_models = [m for m in models if not m.is_free]

                print(f"  📊 Available Models : {len(models)} total ({len(free_models)} free, {len(paid_models)} paid)")

                if free_models:
                    print("  🟢 Sample Free Models :")
                    for m in free_models[:4]:
                        ctx = f"{m.context_length // 1000}k" if m.context_length else "N/A"
                        print(f"     • {m.id:<35} | Context: {ctx:<6} | {m.name}")

                if paid_models:
                    print("  💳 Sample Paid Models & Pricing :")
                    for m in paid_models[:4]:
                        cost = m.pricing.format_pricing() if m.pricing else "Standard rate"
                        ctx = f"{m.context_length // 1000}k" if m.context_length else "N/A"
                        print(f"     • {m.id:<30} | Context: {ctx:<6} | {cost}")

            except Exception as e:
                print(f"  ⚠️ Could not list models: {e}")
                models = []

            # 3. Live Inference Benchmark
            print(f"\n🚀 Live inference benchmark ({client.provider.default_model})...")
            start_t = time.perf_counter()
            response = await client.generate_text(
                prompt="Reply with exactly 2 words: 'Test Successful'.",
                temperature=0.1,
                max_tokens=20,
            )
            elapsed_ms = (time.perf_counter() - start_t) * 1000

            print(f"  ✅ Success! Latency : {elapsed_ms:.1f}ms")
            print(f"  💬 Output : \"{response.text.strip()}\"")
            if response.usage:
                print(f"  🔢 Tokens : {response.usage.total_tokens} (prompt: {response.usage.prompt_tokens}, completion: {response.usage.completion_tokens})")

            return ProviderTestResult(
                provider_id=p_id,
                name=p_name,
                status="✅ OPERATIONAL",
                budget_info=budget_str,
                latency_ms=elapsed_ms,
                models_count=len(models),
                test_response=response.text.strip()[:30],
            )

    except MissingAPIKeyError as e:
        print(f"⚠️  Missing Key : {e.message}")
        return ProviderTestResult(
            provider_id=p_id,
            name=p_name,
            status="⚠️ MISSING KEY",
            error_message=f"Set {item['env_key']} in .env",
        )
    except AuthenticationError as e:
        print(f"❌ Authentication Failed (HTTP {e.status_code}) : {e.message}")
        return ProviderTestResult(
            provider_id=p_id,
            name=p_name,
            status="❌ INVALID AUTH",
            error_message="Invalid or expired API key",
        )
    except RateLimitError as e:
        print(f"⏳ Rate Limit Exceeded (HTTP 429) : {e.message}")
        return ProviderTestResult(
            provider_id=p_id,
            name=p_name,
            status="⏳ RATE LIMITED",
            error_message="Quota or rate limit reached",
        )
    except NexusAIError as e:
        print(f"❌ API Error : {e.message}")
        return ProviderTestResult(
            provider_id=p_id,
            name=p_name,
            status="❌ API ERROR",
            error_message=str(e)[:40],
        )
    except Exception as e:
        print(f"❌ Unexpected Error : {e}")
        return ProviderTestResult(
            provider_id=p_id,
            name=p_name,
            status="❌ ERROR",
            error_message=str(e)[:40],
        )


async def main() -> None:
    """Run full diagnostic validation across all 6 providers."""
    print("=" * 75)
    print("🤖 NEXUSAI-CLIENT - ACCESS, BUDGET & MODEL VALIDATION SUITE")
    print("=" * 75)
    print("Testing environment configuration across all 6 supported AI providers.")

    results: list[ProviderTestResult] = []

    for item in PROVIDERS_TO_TEST:
        res = await test_single_provider(item)
        results.append(res)

    # Final Summary Table
    print("\n\n" + "=" * 90)
    print("📊 PROVIDER ACCESS & BUDGET SUMMARY")
    print("=" * 90)
    print(f"{'Provider':<28} | {'Status':<16} | {'Budget / Quota':<30} | {'Latency'}")
    print("-" * 90)

    for r in results:
        lat_str = f"{r.latency_ms:.0f}ms" if r.latency_ms > 0 else "-"
        budget_disp = (r.budget_info or r.error_message)[:28]
        print(f"{r.name:<28} | {r.status:<16} | {budget_disp:<30} | {lat_str}")

    print("=" * 90)
    print("💡 Tip: To enable a provider, add its corresponding API key to your .env file.")


if __name__ == "__main__":
    asyncio.run(main())
