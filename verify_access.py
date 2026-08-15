"""Script de validation et de test pour tous les fournisseurs NexusAI-Client.

Vérifie la configuration du .env, interroge le budget / solde restant,
liste les modèles disponibles (gratuits / payants et coûts associés)
et effectue un appel de test réel pour valider la connectivité.
"""

from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import dataclass
from typing import Any

# Assure l'encodage UTF-8 sous Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from nexusai_client import (
    AccountInfo,
    AIGateway,
    AuthenticationError,
    MissingAPIKeyError,
    NexusAIError,
    RateLimitError,
)

PROVIDERS_TO_TEST = [
    {
        "id": "deepseek",
        "name": "1. DeepSeek (API Payante)",
        "env_key": "DEEPSEEK_API_KEY",
        "is_paid": True,
    },
    {
        "id": "gemini_pro",
        "name": "2. Google Gemini Pro (Tier Payant)",
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
        "id": "mistral",
        "name": "4. Mistral AI (Tier Gratuit / La Plateforme)",
        "env_key": "MISTRAL_API_KEY",
        "is_paid": False,
    },
    {
        "id": "nvidia_free",
        "name": "5. Nvidia NIM (Tier Gratuit / NGC)",
        "env_key": "NVIDIA_API_KEY",
        "is_paid": False,
    },
    {
        "id": "openrouter",
        "name": "6. OpenRouter (Tier Gratuit / Modèles :free)",
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
    """Exécute la vérification complète pour un fournisseur donné."""
    p_id = item["id"]
    p_name = item["name"]

    print("\n" + "=" * 75)
    print(f"🔍 Test de connexion : {p_name}")
    print("=" * 75)

    try:
        async with AIGateway(provider=p_id) as client:
            print(f"🔑 Clé API détectée pour '{p_id}'.")

            # 1. Vérification du budget / solde restant / quotas
            print("💰 Récupération des informations de budget et quotas...")
            budget_str = "N/A"
            try:
                acc_info = await client.get_account_info()
                budget_str = acc_info.format_summary()
                print(f"  💵 Statut Compte : {budget_str}")
            except Exception as e:
                print(f"  ⚠️ Impossible de récupérer les crédits : {e}")

            # 2. Récupération du catalogue des modèles
            print("\n📋 Récupération du catalogue de modèles...")
            try:
                models = await client.list_models()
                free_models = [m for m in models if m.is_free]
                paid_models = [m for m in models if not m.is_free]

                print(f"  📊 Modèles disponibles : {len(models)} total ({len(free_models)} gratuits, {len(paid_models)} payants)")

                # Affichage des modèles gratuits
                if free_models:
                    print("  🟢 Exemples de modèles gratuits :")
                    for m in free_models[:4]:
                        ctx = f"{m.context_length // 1000}k" if m.context_length else "N/A"
                        print(f"     • {m.id:<35} | Contexte: {ctx:<6} | {m.name}")

                # Affichage des modèles payants avec coûts
                if paid_models:
                    print("  💳 Exemples de modèles payants & coûts :")
                    for m in paid_models[:4]:
                        cost = m.pricing.format_pricing() if m.pricing else "Tarif standard"
                        ctx = f"{m.context_length // 1000}k" if m.context_length else "N/A"
                        print(f"     • {m.id:<30} | Contexte: {ctx:<6} | {cost}")

            except Exception as e:
                print(f"  ⚠️ Impossible de lister les modèles : {e}")
                models = []

            # 3. Test d'inférence réel
            print(f"\n🚀 Test d'inférence en direct ({client.provider.default_model})...")
            start_t = time.perf_counter()
            response = await client.generate_text(
                prompt="Réponds uniquement 'Test réussi' en 2 mots.",
                temperature=0.1,
                max_tokens=20,
            )
            elapsed_ms = (time.perf_counter() - start_t) * 1000

            print(f"  ✅ Succès ! Latence : {elapsed_ms:.1f}ms")
            print(f"  💬 Réponse : \"{response.text.strip()}\"")
            if response.usage:
                print(f"  🔢 Tokens : {response.usage.total_tokens} (prompt: {response.usage.prompt_tokens}, completion: {response.usage.completion_tokens})")

            return ProviderTestResult(
                provider_id=p_id,
                name=p_name,
                status="✅ OPÉRATIONNEL",
                budget_info=budget_str,
                latency_ms=elapsed_ms,
                models_count=len(models),
                test_response=response.text.strip()[:30],
            )

    except MissingAPIKeyError as e:
        print(f"⚠️  Clé manquante : {e.message}")
        return ProviderTestResult(
            provider_id=p_id,
            name=p_name,
            status="⚠️ CLÉ MANQUANTE",
            error_message=f"Renseignez {item['env_key']} dans .env",
        )
    except AuthenticationError as e:
        print(f"❌ Authentification échouée (HTTP {e.status_code}) : {e.message}")
        return ProviderTestResult(
            provider_id=p_id,
            name=p_name,
            status="❌ AUTH INVALIDE",
            error_message="Clé API expirée ou rejetée",
        )
    except RateLimitError as e:
        print(f"⏳ Quota dépassé (HTTP 429) : {e.message}")
        return ProviderTestResult(
            provider_id=p_id,
            name=p_name,
            status="⏳ QUOTA DÉPASSÉ",
            error_message="Limite de requêtes atteinte",
        )
    except NexusAIError as e:
        print(f"❌ Erreur API : {e.message}")
        return ProviderTestResult(
            provider_id=p_id,
            name=p_name,
            status="❌ ERREUR API",
            error_message=str(e)[:40],
        )
    except Exception as e:
        print(f"❌ Erreur inattendue : {e}")
        return ProviderTestResult(
            provider_id=p_id,
            name=p_name,
            status="❌ ERREUR",
            error_message=str(e)[:40],
        )


async def main() -> None:
    """Exécution globale de la validation."""
    print("=" * 75)
    print("🤖 NEXUSAI-CLIENT - VÉRIFICATION DES ACCÈS, BUDGETS & MODÈLES")
    print("=" * 75)
    print("Ce script teste la configuration de votre fichier .env pour les 6 fournisseurs.")

    results: list[ProviderTestResult] = []

    for item in PROVIDERS_TO_TEST:
        res = await test_single_provider(item)
        results.append(res)

    # Tableau Récapitulatif Final
    print("\n\n" + "=" * 90)
    print("📊 TABLEAU RÉCAPITULATIF DES ACCÈS & BUDGETS")
    print("=" * 90)
    print(f"{'Fournisseur':<28} | {'Statut':<16} | {'Budget / Quota':<30} | {'Latence'}")
    print("-" * 90)

    for r in results:
        lat_str = f"{r.latency_ms:.0f}ms" if r.latency_ms > 0 else "-"
        budget_disp = (r.budget_info or r.error_message)[:28]
        print(f"{r.name:<28} | {r.status:<16} | {budget_disp:<30} | {lat_str}")

    print("=" * 90)
    print("💡 Conseil : Pour activer un fournisseur, ajoutez sa clé dans le fichier .env")


if __name__ == "__main__":
    asyncio.run(main())
