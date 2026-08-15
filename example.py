"""NexusAI-Client Usage Demonstration.

Demonstrates:
1. Checking account balance / remaining credits / quotas per provider
2. Discovering available models (free vs paid and pricing)
3. Performing asynchronous inference and streaming with OpenRouter and Gemini Free.
"""

from __future__ import annotations

import asyncio
import sys

# Ensure UTF-8 output encoding on Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from nexusai_client import (
    AIGateway,
    ChatMessage,
    MissingAPIKeyError,
    NexusAIError,
)


async def demonstrate_account_info_and_models() -> None:
    """Demonstrate querying balances and model catalogs."""
    print("\n" + "=" * 65)
    print("💳 1. CHECKING BUDGET & MODEL CATALOGS (OpenRouter & Gemini)")
    print("=" * 65)

    try:
        async with AIGateway(provider="openrouter") as client:
            info = await client.get_account_info()
            print(f"📊 OpenRouter Status: {info.format_summary()}")

            free_models = await client.list_models(free_only=True)
            print(f"🟢 Free models found ({len(free_models)}):")
            for m in free_models[:3]:
                print(f"   • {m.id} (Context: {m.context_length} tokens)")

    except MissingAPIKeyError as e:
        print(f"ℹ️ OpenRouter not configured: {e.message}")
    except NexusAIError as e:
        print(f"⚠️ OpenRouter Error: {e}")


async def demonstrate_openrouter_call() -> None:
    """Demonstrate async inference call with OpenRouter."""
    print("\n" + "=" * 65)
    print("🤖 2. OPENROUTER AI INFERENCE (Free Tier)")
    print("=" * 65)

    prompt = "Explain quantum mechanics in 2 simple sentences."

    try:
        async with AIGateway(provider="openrouter") as client:
            print(f"👉 Sending request to OpenRouter ({client.provider.default_model})...")
            response = await client.generate_text(
                prompt=prompt,
                system_prompt="You are an expert science educator.",
                temperature=0.5,
            )

            print("\n✅ Response Received:")
            print(f"Model used     : {response.model}")
            print(f"Provider       : {response.provider}")
            if response.usage:
                print(
                    f"Tokens consumed: {response.usage.total_tokens} "
                    f"(prompt: {response.usage.prompt_tokens}, completion: {response.usage.completion_tokens})"
                )
            print("-" * 65)
            print(response.text)
            print("-" * 65)

    except MissingAPIKeyError as e:
        print(f"⚠️  Missing Key: {e.message}")
    except NexusAIError as e:
        print(f"❌ NexusAI Error: {e}")


async def demonstrate_gemini_streaming() -> None:
    """Demonstrate real-time token streaming with Google Gemini Free."""
    print("\n" + "=" * 65)
    print("✨ 3. GEMINI FREE REAL-TIME TOKEN STREAMING (Google AI Studio)")
    print("=" * 65)

    try:
        async with AIGateway(provider="gemini_free") as client:
            info = await client.get_account_info()
            print(f"📊 Gemini Quotas: {info.format_summary()}")

            print(f"👉 Streaming response from Gemini ({client.provider.default_model}):\n")
            async for chunk in client.stream_text("Name 3 key architectural advantages of asyncio in Python."):
                print(chunk, end="", flush=True)
            print("\n" + "-" * 65)

    except MissingAPIKeyError as e:
        print(f"⚠️  Missing Key: {e.message}")
    except NexusAIError as e:
        print(f"❌ NexusAI Error: {e}")


async def main() -> None:
    """Main demonstration entry point."""
    print("🚀 Starting NexusAI-Client Demonstration")
    print(f"Python Version: {sys.version.split()[0]}")

    await demonstrate_account_info_and_models()
    await demonstrate_openrouter_call()
    await demonstrate_gemini_streaming()

    print("\n🎉 Demonstration complete!")


if __name__ == "__main__":
    asyncio.run(main())
