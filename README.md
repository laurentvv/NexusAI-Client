# NexusAI-Client ⚡

[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Code style: black](https://img.shields.io/badge/code%20style-pep8-000000.svg)](https://pep8.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**NexusAI-Client** est une passerelle unifiée (gateway) open-source conçue pour les développeurs Python. Elle permet d'intégrer et de basculer facilement entre les meilleurs modèles d'Intelligence Artificielle du marché via une interface commune et unifiée.

Construit en **Python 3.14** et géré avec **uv**, ce projet se veut minimaliste, asynchrone (via `httpx`) et exempt de dépendances lourdes. Il gère de manière sécurisée les clés API via un fichier `.env` pour des fournisseurs tels que **Deepseek**, **Google Gemini (Pro & Free)**, **Mistral**, **Nvidia NIM** et **OpenRouter**.

👉 **[Consulter le Guide d'Intégration Complet dans vos Projets (FastAPI, Fallback, Chat)](./INTEGRATION_GUIDE.md)**

---

## 🎯 Fournisseurs d'IA supportés

| Fournisseur | Identifiant (`provider`) | Tier | Modèle par défaut | Solde / Quotas détectés |
| :--- | :--- | :--- | :--- | :--- |
| **DeepSeek** | `"deepseek"` | Payant | `deepseek-chat` | Solde USD en temps réel (`GET /user/balance`) |
| **Gemini Pro** | `"gemini_pro"` | Payant | `gemini-2.5-pro` | Pay-as-you-go Google Cloud |
| **Gemini Free** | `"gemini_free"` | Gratuit (AI Studio) | `gemini-2.5-flash` | 15 RPM \| 1M TPM \| 1 500 RPD |
| **Mistral** | `"mistral"` | Gratuit / Plateforme | `mistral-small-latest` | Modèles dev gratuits (`codestral-latest`, etc.) |
| **Nvidia NIM** | `"nvidia"` ou `"nvidia_free"` | Gratuit (1000 crédits) | `meta/llama-3.1-8b-instruct` | 1 000 crédits d'inférence gratuits NGC |
| **OpenRouter** | `"openrouter"` | Gratuit (modèles `:free`) | `openrouter/free` | 19 modèles gratuits + 390 payants |

---

## 🚀 Utilisation comme Fonction dans un Projet Tiers

### 1. Installation

```bash
# Dans le projet consommateur avec uv
uv add --editable /chemin/vers/NexusAI-Client

# Ou avec pip
pip install -e /chemin/vers/NexusAI-Client
```

### 2. Exemple de Fonction Réutilisable (Copier-Coller)

```python
import asyncio
from nexusai_client import AIGateway

async def generer_texte(prompt: str, provider: str = "gemini_free") -> str:
    """Fonction générique pour appeler n'importe quel provider IA."""
    async with AIGateway(provider=provider) as client:
        response = await client.generate_text(
            prompt=prompt,
            system_prompt="Tu es un assistant concis et précis.",
            temperature=0.3,
        )
        return response.text

# Utilisation directe :
async def main():
    # Appel gratuit via Google AI Studio
    texte1 = await generer_texte("Explique Docker en 2 phrases.", provider="gemini_free")
    print(f"Gemini: {texte1}\n")

    # Appel gratuit via Nvidia NIM
    texte2 = await generer_texte("Explique Kubernetes en 2 phrases.", provider="nvidia_free")
    print(f"Nvidia: {texte2}\n")

    # Appel via DeepSeek (Payant)
    texte3 = await generer_texte("Explique les microservices en 2 phrases.", provider="deepseek")
    print(f"DeepSeek: {texte3}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 💡 Autres Cas d'Usage Avancés

### A. Fallback Automatique (Tolérance aux Pannes)

Bascule automatiquement sur le fournisseur suivant si un service est saturé :

```python
from nexusai_client import AIGateway, NexusAIError

async def call_with_fallback(prompt: str) -> str:
    for provider in ["gemini_free", "nvidia_free", "openrouter", "deepseek"]:
        try:
            async with AIGateway(provider) as client:
                res = await client.generate_text(prompt)
                return res.text
        except NexusAIError:
            continue
    raise RuntimeError("Tous les providers sont indisponibles.")
```

### B. Chat Conversationnel avec Historique

```python
from nexusai_client import AIGateway, ChatMessage

async def conversation():
    messages = [
        ChatMessage(role="system", content="Tu es un expert Python."),
        ChatMessage(role="user", content="Comment déclarer une coroutine ?"),
    ]
    async with AIGateway("openrouter") as client:
        res = await client.chat(messages)
        print(res.text)
```

### C. Vérifier les Budgets et Soldes Restants

```python
async with AIGateway("deepseek") as client:
    info = await client.get_account_info()
    print(f"Solde restant : ${info.total_balance:.2f}")
```

---

## 🛠️ Outils CLI Inclus

```bash
# Valider les clés API et les temps de réponse
uv run python verify_access.py

# Explorer le catalogue complet de 648 modèles
uv run python list_all_models.py --free
uv run python list_all_models.py --search llama
uv run python list_all_models.py --export catalogue.json

# Lancer la suite de tests unitaires automatisés
uv run pytest -v
```

---

## 📄 Licence

Projet distribué sous licence MIT.
