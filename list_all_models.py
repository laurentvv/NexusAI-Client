"""CLI utility to explore, filter, and export the live model catalog across providers.

Usage examples:
    uv run python list_all_models.py                  # Display full summary
    uv run python list_all_models.py --free           # Filter free-tier models only
    uv run python list_all_models.py --provider nvidia # Filter by provider
    uv run python list_all_models.py --search llama   # Search by keyword
    uv run python list_all_models.py --export models.json # Export complete catalog to JSON
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from typing import Any

# Ensure UTF-8 output on Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from nexusai_client import AIGateway, ModelInfo

PROVIDERS = [
    ("deepseek", "1. DeepSeek (Paid API)"),
    ("gemini_free", "2. Google Gemini (Google AI Studio Free Tier)"),
    ("gemini_pro", "3. Google Gemini (Paid Pro Tier)"),
    ("mistral", "4. Mistral AI (Platform)"),
    ("nvidia_free", "5. Nvidia NIM (Free Tier NGC)"),
    ("openrouter", "6. OpenRouter (Multi-Provider Hub)"),
]


async def fetch_provider_models(provider_id: str, free_only: bool) -> tuple[str, list[ModelInfo]]:
    """Fetch model list for a given provider."""
    try:
        async with AIGateway(provider=provider_id) as client:
            models = await client.list_models(free_only=free_only)
            return provider_id, models
    except Exception as e:
        print(f"⚠️ Error fetching models for '{provider_id}': {e}", file=sys.stderr)
        return provider_id, []


def _filter_models(models: list[ModelInfo], search_term: str | None) -> list[ModelInfo]:
    """Filter models by search keyword."""
    if not search_term:
        return models
    s = search_term.lower()
    return [
        m for m in models
        if s in m.id.lower() or s in m.name.lower() or (m.description and s in m.description.lower())
    ]


def _display_models_table(label: str, models: list[ModelInfo], limit: int) -> None:
    """Display tabular model breakdown for a provider."""
    free_count = sum(1 for m in models if m.is_free)
    paid_count = len(models) - free_count

    print(f"\n📁 {label}")
    print(f"   Available Models: {len(models)} ({free_count} free, {paid_count} paid)")
    print("-" * 80)

    if not models:
        print("   (No matching models found)")
        return

    print(f"   {'Model Identifier (ID)':<40} | {'Context':<10} | {'Pricing / Status'}")
    print("   " + "-" * 75)

    for m in models[:limit]:
        ctx = f"{m.context_length // 1000}k" if m.context_length else "N/A"
        cost = m.pricing.format_pricing() if m.pricing else ("🟢 Free" if m.is_free else "💳 Standard Paid")
        badge = "🟢 " if m.is_free else "💳 "
        print(f"   {badge}{m.id:<38} | {ctx:<10} | {cost}")

    if len(models) > limit:
        print(f"   ... and {len(models) - limit} more models (use --limit 50 or --search to refine)")


def _export_to_json(filepath: str, data: dict[str, Any]) -> None:
    """Export data dictionary to a JSON file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"💾 Catalog successfully exported to '{filepath}'!")


def _build_parser() -> argparse.ArgumentParser:
    """Configure CLI argument parser."""
    parser = argparse.ArgumentParser(description="NexusAI-Client - Model Catalog Explorer")
    parser.add_argument("--free", action="store_true", help="Display free-tier models only")
    parser.add_argument("--provider", type=str, default=None, help="Filter by provider")
    parser.add_argument("--search", type=str, default=None, help="Search by keyword")
    parser.add_argument("--export", type=str, default=None, help="Export catalog to JSON file")
    parser.add_argument("--limit", type=int, default=15, help="Max models displayed per provider (default: 15)")
    return parser


async def main() -> None:
    """Main CLI entry point for model explorer."""
    parser = _build_parser()
    args = parser.parse_args()

    target_providers = [p for p in PROVIDERS if not args.provider or p[0] == args.provider.lower()]
    if not target_providers:
        print(f"❌ Unknown provider: '{args.provider}'. Available: {[p[0] for p in PROVIDERS]}")
        return

    print("=" * 80)
    print("🤖 NEXUSAI-CLIENT - LIVE MODEL CATALOG EXPLORER")
    if args.free:
        print("🎯 Filter: Free-Tier Models Only")
    if args.search:
        print(f"🔍 Search: '{args.search}'")
    print("=" * 80)

    tasks = [fetch_provider_models(p_id, args.free) for p_id, _ in target_providers]
    results = await asyncio.gather(*tasks)

    total_models_found = 0
    export_data: dict[str, Any] = {}

    for (p_id, label), (_, raw_models) in zip(target_providers, results, strict=False):
        models = _filter_models(raw_models, args.search)
        total_models_found += len(models)
        export_data[p_id] = [
            {
                "id": m.id,
                "name": m.name,
                "provider": m.provider,
                "is_free": m.is_free,
                "context_length": m.context_length,
                "pricing": asdict(m.pricing) if m.pricing else None,
                "description": m.description,
            }
            for m in models
        ]
        _display_models_table(label, models, args.limit)

    print("\n" + "=" * 80)
    print(f"✅ Total: {total_models_found} models indexed across active providers.")
    print("=" * 80)

    if args.export:
        _export_to_json(args.export, export_data)


if __name__ == "__main__":
    asyncio.run(main())
