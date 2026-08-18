"""Real-time Multi-Turn Tool Calling demonstration for NexusAI-Client.

Demonstrates:
1. Declaring JSON-schema tool definitions (functions).
2. Sending a prompt with multiple tasks to the AI model.
3. Intercepting and executing the requested tools in Python.
4. Feeding tool execution results back into the conversation history.
5. Receiving the final synthesized response from the model.
"""

from __future__ import annotations

import asyncio
import json
import sys

# Ensure UTF-8 output encoding on Windows platforms
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from nexusai_client import (
    AIGateway,
    ChatMessage,
    FunctionDefinition,
    MissingAPIKeyError,
    NexusAIError,
    ToolDefinition,
)

# 1. Définition des outils (schémas JSON universels)
TOOLS = [
    ToolDefinition(
        function=FunctionDefinition(
            name="get_weather",
            description="Obtenir la météo et température actuelle pour une ville donnée.",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "Nom de la ville"},
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Unité de température",
                    },
                },
                "required": ["city"],
            },
        )
    ),
    ToolDefinition(
        function=FunctionDefinition(
            name="calculate_vat",
            description="Calculer le montant TTC et le montant de la TVA à partir d'un prix Hors Taxe (HT).",
            parameters={
                "type": "object",
                "properties": {
                    "amount_ht": {
                        "type": "number",
                        "description": "Montant Hors Taxe en Euros",
                    },
                    "vat_rate": {
                        "type": "number",
                        "description": "Taux de TVA en pourcentage (ex: 20 pour 20%)",
                    },
                },
                "required": ["amount_ht", "vat_rate"],
            },
        )
    ),
]


# 2. Fonctions Python réelles exécutées localement
def execute_python_tool(name: str, args: dict) -> dict:
    """Simulate or execute real business logic in Python."""
    if name == "get_weather":
        city = args.get("city", "Inconnue")
        unit = args.get("unit", "celsius")
        return {
            "city": city,
            "temperature": 19.5,
            "unit": unit,
            "condition": "Ensoleillé avec quelques nuages",
        }

    if name == "calculate_vat":
        ht = float(args.get("amount_ht", 0.0))
        rate = float(args.get("vat_rate", 20.0))
        vat = round(ht * (rate / 100.0), 2)
        ttc = round(ht + vat, 2)
        return {
            "amount_ht": ht,
            "vat_rate": rate,
            "vat_amount": vat,
            "amount_ttc": ttc,
        }

    return {"error": f"Outil '{name}' inconnu"}


async def run_tool_calling_loop(provider_id: str) -> None:
    """Execute complete 2-turn tool calling loop for a given provider."""
    print("\n" + "=" * 70)
    print(f"🛠️  TEST TOOL CALLING EN DIRECT SUR : [{provider_id.upper()}]")
    print("=" * 70)

    try:
        async with AIGateway(provider=provider_id) as client:
            prompt = (
                "Bonjour ! Calcule la TVA pour un produit de 250€ HT avec une TVA à 20%, "
                "et donne-moi la météo actuelle à Tokyo."
            )
            messages = [ChatMessage(role="user", content=prompt)]

            print(f"👉 1. Envoi du prompt à [{client.provider.default_model}]...")
            print(f'   Prompt : "{prompt}"')

            # Turn 1 : Modèle détecte le besoin d'outils et renvoie tool_calls
            response_turn1 = await client.chat(messages=messages, tools=TOOLS)

            if not response_turn1.has_tool_calls:
                print(
                    f"⚠️ Le modèle a répondu en texte direct sans appeler d'outils :\n{response_turn1.text}"
                )
                return

            print(
                f"\n✅ 2. Le modèle a généré {len(response_turn1.tool_calls)} appel(s) d'outil :"
            )

            # Conserver la réponse de l'assistant (avec ses tool_calls) dans l'historique
            messages.append(
                ChatMessage(
                    role="assistant",
                    content=response_turn1.text,
                    tool_calls=response_turn1.tool_calls,
                )
            )

            # Exécuter chaque outil et ajouter le résultat à l'historique
            for call in response_turn1.tool_calls:
                print(f"   🔧 Exécution : {call.name}({call.arguments}) [id={call.id}]")
                tool_result = execute_python_tool(call.name, call.arguments)
                print(f"      📦 Résultat renvoyé : {tool_result}")

                messages.append(
                    ChatMessage(
                        role="tool",
                        name=call.name,
                        tool_call_id=call.id,
                        content=json.dumps(tool_result, ensure_ascii=False),
                    )
                )

            # Turn 2 : Renvoi de l'historique avec les résultats d'exécution des outils
            print(
                "\n👉 3. Renvoi des résultats d'outils au modèle pour synthèse finale..."
            )
            response_turn2 = await client.chat(messages=messages, tools=TOOLS)

            print("\n🎉 4. Réponse finale synthétisée par l'IA :")
            print("-" * 70)
            print(response_turn2.text.strip())
            print("-" * 70)

    except MissingAPIKeyError as e:
        print(f"⚠️ Clé API manquante pour {provider_id} : {e.message}")
    except NexusAIError as e:
        print(f"❌ Erreur API sur {provider_id} : {e}")


async def main() -> None:
    """Execute live tool tests against active providers."""
    print("🚀 Démarrage du banc d'essai Live Tool Calling (NexusAI-Client)")

    # Testons sur plusieurs familles d'adaptateurs réels :
    # 1. Google Gemini (Format FunctionCall / FunctionResponse)
    # 2. Mistral AI (Format OpenAI-compatible)
    # 3. DeepSeek (Format OpenAI-compatible)
    providers_to_test = ["gemini_free", "mistral", "deepseek"]

    for provider in providers_to_test:
        await run_tool_calling_loop(provider)


if __name__ == "__main__":
    asyncio.run(main())
