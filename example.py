"""Exemple d'utilisation de NexusAI-Client.

Démontre comment :
1. Consulter le budget / solde restant / quotas par provider
2. Lister les modèles disponibles (gratuits vs payants et coûts)
3. Exécuter un appel asynchrone à OpenRouter et Gemini Free avec gestion d'erreurs.
"""

from __future__ import annotations

import asyncio
import sys

# Assure le support de l'encodage UTF-8 pour l'affichage console sous Windows
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
    """Démonstration de la consultation des budgets et catalogues de modèles."""
    print("\n" + "=" * 65)
    print("💳 1. VÉRIFICATION DU BUDGET & DES MODÈLES (OpenRouter & DeepSeek)")
    print("=" * 65)

    # Consultation du compte OpenRouter
    try:
        async with AIGateway(provider="openrouter") as client:
            info = await client.get_account_info()
            print(f"📊 OpenRouter Statut : {info.format_summary()}")

            # Lister les modèles gratuits
            free_models = await client.list_models(free_only=True)
            print(f"🟢 Modèles gratuits trouvés ({len(free_models)}) :")
            for m in free_models[:3]:
                print(f"   • {m.id} (Contexte: {m.context_length} tokens)")

    except MissingAPIKeyError as e:
        print(f"ℹ️ OpenRouter non configuré : {e.message}")
    except NexusAIError as e:
        print(f"⚠️ Erreur OpenRouter : {e}")


async def demonstrate_openrouter_call() -> None:
    """Démonstration d'un appel asynchrone vers OpenRouter (Tier Gratuit)."""
    print("\n" + "=" * 65)
    print("🤖 2. TEST APPEL IA OPENROUTER (Tier Gratuit)")
    print("=" * 65)

    prompt = "Explique en 2 phrases simples le principe de la mécanique quantique."

    try:
        async with AIGateway(provider="openrouter") as client:
            print(f"👉 Envoi de la requête à OpenRouter ({client.provider.default_model})...")
            response = await client.generate_text(
                prompt=prompt,
                system_prompt="Tu es un professeur de vulgarisation scientifique bienveillant et concis.",
                temperature=0.5,
            )

            print("\n✅ Réponse reçue :")
            print(f"Modèle utilisé  : {response.model}")
            print(f"Provider        : {response.provider}")
            if response.usage:
                print(
                    f"Tokens consommés: {response.usage.total_tokens} "
                    f"(prompt: {response.usage.prompt_tokens}, completion: {response.usage.completion_tokens})"
                )
            print("-" * 65)
            print(response.text)
            print("-" * 65)

    except MissingAPIKeyError as e:
        print(f"⚠️  Clé manquante : {e.message}")
        print("💡 Astuce : Renseignez 'OPENROUTER_API_KEY' dans votre fichier .env")
    except NexusAIError as e:
        print(f"❌ Erreur NexusAI : {e}")


async def demonstrate_gemini_free() -> None:
    """Démonstration d'un appel asynchrone vers Google Gemini Free (Google AI Studio)."""
    print("\n" + "=" * 65)
    print("✨ 3. TEST APPEL IA GEMINI FREE (Google AI Studio)")
    print("=" * 65)

    conversation = [
        ChatMessage(role="system", content="Tu es un assistant expert en architecture logicielle Python."),
        ChatMessage(role="user", content="Quels sont les 3 avantages principaux de l'asynchronisme avec httpx ?"),
    ]

    try:
        async with AIGateway(provider="gemini_free") as client:
            # Récupérer les quotas du compte
            info = await client.get_account_info()
            print(f"📊 Gemini Quotas : {info.format_summary()}")

            print(f"👉 Envoi de la conversation à Gemini Free ({client.provider.default_model})...")
            response = await client.chat(
                messages=conversation,
                temperature=0.3,
                max_tokens=500,
            )

            print("\n✅ Réponse reçue :")
            print(f"Modèle utilisé  : {response.model}")
            print(f"Provider        : {response.provider}")
            if response.usage:
                print(
                    f"Tokens consommés: {response.usage.total_tokens} "
                    f"(prompt: {response.usage.prompt_tokens}, completion: {response.usage.completion_tokens})"
                )
            print("-" * 65)
            print(response.text)
            print("-" * 65)

    except MissingAPIKeyError as e:
        print(f"⚠️  Clé manquante : {e.message}")
        print("💡 Astuce : Renseignez 'GEMINI_FREE_API_KEY' ou 'GEMINI_API_KEY' dans votre .env")
    except NexusAIError as e:
        print(f"❌ Erreur NexusAI : {e}")


async def main() -> None:
    """Point d'entrée principal de l'exemple."""
    print("🚀 Démarrage de la démonstration NexusAI-Client")
    print(f"Python Version: {sys.version.split()[0]}")

    await demonstrate_account_info_and_models()
    await demonstrate_openrouter_call()
    await demonstrate_gemini_free()

    print("\n🎉 Démonstration terminée !")


if __name__ == "__main__":
    asyncio.run(main())
