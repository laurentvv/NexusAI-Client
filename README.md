# NexusAI-Client ⚡

<p align="center">
  <strong>Passerelle IA asynchrone unifiée, ultra-légère et typée pour Python 3.14.</strong><br>
  <em>Unifiez DeepSeek, Google Gemini (Free & Pro), Mistral, Nvidia NIM et OpenRouter sous une interface unique sans SDKs lourds.</em>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.14+-3776AB.svg?style=flat-square&logo=python&logoColor=white" alt="Python 3.14+"></a>
  <a href="https://github.com/astral-sh/uv"><img src="https://img.shields.io/badge/package_manager-uv-DE5FE9.svg?style=flat-square" alt="uv"></a>
  <a href="https://www.python-httpx.org/"><img src="https://img.shields.io/badge/engine-httpx_async-009688.svg?style=flat-square" alt="httpx"></a>
  <a href="https://peps.python.org/pep-0561/"><img src="https://img.shields.io/badge/typing-PEP_561_Strict-blue.svg?style=flat-square" alt="Typing"></a>
  <a href="https://github.com/laurentvv/NexusAI-Client/actions"><img src="https://img.shields.io/badge/tests-15%2F15_passing-brightgreen.svg?style=flat-square" alt="Tests"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square" alt="License MIT"></a>
</p>

---

## 💡 Pourquoi NexusAI-Client ?

Intégrer plusieurs fournisseurs d'IA dans une application moderne impose généralement d'installer **5 ou 6 SDKs propriétaires distincts** (`google-genai`, `openai`, `mistralai`, etc.). Cela engendre des dizaines de dépendances transitives, des conflits de versions, une empreinte mémoire lourde et du code hétérogène difficile à maintenir.

**NexusAI-Client** résout ce problème à la racine :
- 🪶 **Zéro Dépendance Lourde** : Uniquement propulsé par `httpx` et `python-dotenv`.
- ⚡ **Asynchronisme Natif** : Conçu dès le départ pour `asyncio` avec pooling de connexions HTTP/2.
- 🎯 **Interface Commune & Standardisée** : Même signature (`generate_text`, `chat`, `list_models`, `get_account_info`) quel que soit le fournisseur.
- 💰 **Découverte du Budget & des Quotas en Direct** : Interroge en temps réel les soldes restants (USD, crédits NGC) et les limites (RPM, TPM, RPD).
- 🔍 **Catalogue de 640+ Modèles en Direct** : Découverte automatique des modèles gratuits (`:free`, tiers gratuits) et tarification au million de tokens.
- 🛡️ **Tolérance aux Pannes & Fallback** : Basculez automatiquement d'un fournisseur saturé à un autre sans rupture de service.

---

## 🏛️ Architecture

```
                             ┌──► 🟣 DeepSeek (V3 / R1 Reasoner - Solde USD en direct)
                             ├──► 🟡 Google Gemini Free (Google AI Studio - 1M tokens de contexte)
                             ├──► 🔵 Google Gemini Pro (Tier Payant GCP)
     [Votre Application] ──► AIGateway ─┼──► 🟠 Mistral AI (Codestral / Small / Large)
                             ├──► 🟢 Nvidia NIM (1 000 crédits offerts / Llama 3.3 70B)
                             └──► ⚪ OpenRouter (Routeur auto openrouter/free & 400+ modèles)
```

---

## 🎯 Matrice des Fournisseurs Supportés

| Fournisseur | Identifiant (`provider`) | Tier | Protocole | Modèle par défaut | Détection Budget & Quotas |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DeepSeek** | `"deepseek"` | Payant | OpenAI Chat API | `deepseek-chat` | Solde USD en temps réel (`GET /user/balance`) |
| **Gemini Free** | `"gemini_free"` | Gratuit (AI Studio) | Gemini REST | `gemini-2.5-flash` | Quotas : 15 RPM \| 1M TPM \| 1 500 RPD |
| **Gemini Pro** | `"gemini_pro"` | Payant | Gemini REST | `gemini-2.5-pro` | Facturation Google Cloud Pay-as-you-go |
| **Mistral AI** | `"mistral"` | Gratuit / Plateforme | OpenAI Chat API | `mistral-small-latest` | Modèles gratuits (`codestral-latest`, etc.) |
| **Nvidia NIM** | `"nvidia_free"` | Gratuit (NGC) | OpenAI Chat API | `meta/llama-3.1-8b-instruct` | 1 000 crédits d'inférence gratuits NGC |
| **OpenRouter** | `"openrouter"` | Gratuit & Payant | OpenAI Chat API | `openrouter/free` | 19 modèles gratuits en direct + 390 payants |

---

## 🚀 Démarrage Rapide (1 Minute)

### 1. Installation

```bash
# Avec uv (Recommandé)
uv add git+https://github.com/laurentvv/NexusAI-Client.git

# En mode local / éditable
uv add --editable /chemin/vers/NexusAI-Client
```

### 2. Configuration des Clés (`.env`)

Créez un fichier `.env` à la racine de votre projet :

```env
# Fournisseurs Gratuits
GEMINI_FREE_API_KEY=votre_cle_google_ai_studio
MISTRAL_API_KEY=votre_cle_mistral_la_plateforme
NVIDIA_API_KEY=nvapi-votre_cle_nvidia_nim
OPENROUTER_API_KEY=sk-or-v1-votre_cle_openrouter

# Fournisseurs Payants (Optionnels)
DEEPSEEK_API_KEY=sk-votre_cle_deepseek
GEMINI_PRO_API_KEY=votre_cle_gemini_pro
```

### 3. Exécuter un Appel IA en 3 Lignes de Code

```python
import asyncio
from nexusai_client import AIGateway

async def main():
    async with AIGateway("gemini_free") as client:
        response = await client.generate_text("Explique la théorie de la relativité en 2 phrases.")
        print(f"[{response.provider} | {response.model}]")
        print(response.text)

if __name__ == "__main__":
    asyncio.run(main())
```

---

##  cookbook : Recettes & Cas d'Usage

### 1. Fonction Réutilisable Générique

```python
from nexusai_client import AIGateway

async def ask_ai(prompt: str, provider: str = "gemini_free") -> str:
    """Fonction prête à l'emploi pour vos applications."""
    async with AIGateway(provider) as client:
        res = await client.generate_text(prompt, temperature=0.2)
        return res.text
```

### 2. Fallback Automatique (Tolérance aux Pannes)

Bascule automatiquement sur le fournisseur suivant en cas de saturation de quota (HTTP 429) ou de panne :

```python
from nexusai_client import AIGateway, NexusAIError

async def ask_with_fallback(prompt: str) -> tuple[str, str]:
    # Chaîne de secours : Gratuit -> Gratuit -> Gratuit -> Payant
    for prov in ["gemini_free", "nvidia_free", "openrouter", "deepseek"]:
        try:
            async with AIGateway(prov, timeout=10.0) as client:
                res = await client.generate_text(prompt)
                return res.text, prov
        except NexusAIError:
            continue
    raise RuntimeError("Tous les fournisseurs d'IA sont temporairement indisponibles.")
```

### 3. Chat Multi-Tours avec Historique

```python
from nexusai_client import AIGateway, ChatMessage

async def conversation():
    history = [
        ChatMessage(role="system", content="Tu es un tuteur expert en algorithmique."),
        ChatMessage(role="user", content="Comment fonctionne le tri rapide (QuickSort) ?"),
    ]
    async with AIGateway("openrouter") as client:
        reponse = await client.chat(history)
        print(reponse.text)
```

### 4. Vérifier les Soldes, Budgets et Quotas

```python
async with AIGateway("deepseek") as client:
    info = await client.get_account_info()
    print(info.format_summary())
    # Sortie : "Solde restant: $5.00 | (Offert: $0.00)"
```

### 5. Génération Spécialisée de Code (avec Codestral)

```python
async with AIGateway("mistral") as client:
    code = await client.generate_text(
        prompt="Écris une fonction asynchrone de retry avec backoff exponentiel.",
        model="codestral-latest",
        temperature=0.1,
    )
    print(code.text)
```

👉 **[Voir le Guide d'Intégration Complet (FastAPI, Workers, Sessions de Chat)](./INTEGRATION_GUIDE.md)**

---

## 🛠️ Outils CLI Inclus

Le package intègre des utilitaires en ligne de commande pour auditer vos accès et explorer les modèles :

### 1. Test et Diagnostic des Accès en Direct
```bash
uv run python verify_access.py
```
*Vérifie chaque clé du `.env`, extrait le solde/quota en direct, liste les modèles et mesure la latence d'inférence en millisecondes.*

### 2. Explorateur de Catalogue (640+ Modèles)
```bash
# Lister uniquement les modèles gratuits
uv run python list_all_models.py --free

# Rechercher un modèle par mot-clé (ex: llama, r1, codestral, sonnet)
uv run python list_all_models.py --search llama

# Exporter tout le catalogue avec tarifs en JSON
uv run python list_all_models.py --export catalogue_ia.json
```

### 3. Suite de Tests Unitaires Automatisés
```bash
uv run pytest -v
```

---

## 🛡️ Gestion Typée des Exceptions

Toutes les exceptions dérivent de `NexusAIError` pour un filtrage strict :

```python
from nexusai_client import (
    AIGateway,
    NexusAIError,
    MissingAPIKeyError,    # Clé API absente dans le .env
    AuthenticationError,   # Clé rejetée (HTTP 401/403)
    RateLimitError,        # Quota dépassé (HTTP 429)
    APITimeoutError,       # Délai d'attente réseau expiré
    APIConnectionError,    # Serveur injoignable
    ProviderNotFoundError, # Fournisseur non reconnu
)
```

---

## 📄 Licence

Ce projet est distribué sous licence **MIT**. Vous êtes libre de l'utiliser, le modifier et l'intégrer dans vos projets commerciaux et personnels.
