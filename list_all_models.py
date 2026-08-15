"""Script utilitaire pour lister, filtrer et exporter le catalogue de modèles disponibles.

Permet de visualiser tous les modèles supportés par fournisseur avec leurs
caractéristiques (gratuit / payant, fenêtre de contexte, tarifs, description).

Exemples d'utilisation :
    uv run python list_all_models.py                  # Affiche le récapitulatif complet
    uv run python list_all_models.py --free           # Affiche uniquement les modèles gratuits
    uv run python list_all_models.py --provider nvidia # Filtre par fournisseur
    uv run python list_all_models.py --search llama   # Recherche par mot-clé
    uv run python list_all_models.py --export models.json # Exporte le catalogue complet en JSON
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict
from typing import Any

# Assure le support UTF-8 sous Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from nexusai_client import AIGateway, ModelInfo

PROVIDERS = [
    ("deepseek", "1. DeepSeek (API Payante)"),
    ("gemini_free", "2. Google Gemini (Tier Gratuit AI Studio)"),
    ("gemini_pro", "3. Google Gemini (Tier Payant Pro)"),
    ("mistral", "4. Mistral AI (La Plateforme)"),
    ("nvidia_free", "5. Nvidia NIM (Tier Gratuit NGC)"),
    ("openrouter", "6. OpenRouter (Multi-Fournisseurs)"),
]


async def fetch_provider_models(provider_id: str, free_only: bool) -> tuple[str, list[ModelInfo]]:
    """Récupère la liste des modèles pour un fournisseur donné."""
    try:
        async with AIGateway(provider=provider_id) as client:
            models = await client.list_models(free_only=free_only)
            return provider_id, models
    except Exception as e:
        print(f"⚠️ Erreur lors de la récupération pour '{provider_id}' : {e}", file=sys.stderr)
        return provider_id, []


async def main() -> None:
    parser = argparse.ArgumentParser(description="NexusAI-Client - Explorateur de Modèles")
    parser.add_argument("--free", action="store_true", help="Afficher uniquement les modèles gratuits")
    parser.add_argument("--provider", type=str, default=None, help="Filtrer par fournisseur (ex: deepseek, gemini_free, mistral, nvidia_free, openrouter)")
    parser.add_argument("--search", type=str, default=None, help="Rechercher un modèle par mot-clé (ex: llama, r1, flash, codestral)")
    parser.add_argument("--export", type=str, default=None, help="Exporter les résultats dans un fichier JSON")
    parser.add_argument("--limit", type=int, default=15, help="Nombre max de modèles affichés par fournisseur (défaut: 15)")
    args = parser.parse_args()

    target_providers = [p for p in PROVIDERS if not args.provider or p[0] == args.provider.lower()]

    if not target_providers:
        print(f"❌ Fournisseur inconnu : '{args.provider}'. Disponibles : {[p[0] for p in PROVIDERS]}")
        return

    print("=" * 80)
    print("🤖 NEXUSAI-CLIENT - CATALOGUE DES MODÈLES DISPONIBLES")
    if args.free:
        print("🎯 Filtre : Modèles Gratuits Uniquement")
    if args.search:
        print(f"🔍 Recherche : '{args.search}'")
    print("=" * 80)

    tasks = [fetch_provider_models(p_id, args.free) for p_id, _ in target_providers]
    results = await asyncio.gather(*tasks)

    total_models_found = 0
    export_data: dict[str, Any] = {}

    for (p_id, label), (_, models) in zip(target_providers, results, strict=False):
        # Filtre de recherche par mot-clé
        if args.search:
            s = args.search.lower()
            models = [m for m in models if s in m.id.lower() or s in m.name.lower() or (m.description and s in m.description.lower())]

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

        free_count = sum(1 for m in models if m.is_free)
        paid_count = len(models) - free_count

        print(f"\n📁 {label}")
        print(f"   Total disponible : {len(models)} ({free_count} gratuits, {paid_count} payants)")
        print("-" * 80)

        if not models:
            print("   (Aucun modèle correspondant aux critères)")
            continue

        # Affichage tabulaire
        print(f"   {'Identifiant (Model ID)':<40} | {'Contexte':<10} | {'Tarif / Statut'}")
        print("   " + "-" * 75)

        for m in models[:args.limit]:
            ctx = f"{m.context_length // 1000}k" if m.context_length else "N/A"
            if m.pricing:
                cost = m.pricing.format_pricing()
            else:
                cost = "🟢 Gratuit" if m.is_free else "💳 Payant standard"

            badge = "🟢 " if m.is_free else "💳 "
            print(f"   {badge}{m.id:<38} | {ctx:<10} | {cost}")

        if len(models) > args.limit:
            print(f"   ... et {len(models) - args.limit} autres modèles (utilisez --limit 50 ou --search pour affiner)")

    print("\n" + "=" * 80)
    print(f"✅ Total : {total_models_found} modèles disponibles répertoriés.")
    print("=" * 80)

    if args.export:
        with open(args.export, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        print(f"💾 Catalogue exporté avec succès vers '{args.export}' !")


if __name__ == "__main__":
    asyncio.run(main())
