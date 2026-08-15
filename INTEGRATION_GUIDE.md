# 📦 Guide d'Intégration de NexusAI-Client dans vos Projets

Ce guide explique étape par étape comment intégrer et consommer **NexusAI-Client** en tant que bibliothèque / module centralisé dans un autre projet Python (Application Web, API FastAPI, Agent IA, Bot, Worker, Script d'automatisation).

---

## 📑 Sommaire

1. [Installation dans votre Projet](#1-installation-dans-votre-projet)
2. [Configuration des Variables d'Environnement](#2-configuration-des-variables-denvironnement)
3. [Exemples de Fonctions Métier Prêtes à l'Emploi](#3-exemples-de-fonctions-métier-prêtes-à-lemploi)
   - [A. Fonction de génération de texte simple](#a-fonction-de-génération-de-texte-simple)
   - [B. Fonction avec Fallback automatique Multi-Providers](#b-fonction-avec-fallback-automatique-multi-providers)
   - [C. Fonction de Chat Conversationnel avec Historique](#c-fonction-de-chat-conversationnel-avec-historique)
   - [D. Fonction de Génération de Code (Spécialisée)](#d-fonction-de-génération-de-code-spécialisée)
   - [E. Fonction d'Audit et de Découverte des Modèles Gratuits](#e-fonction-daudit-et-de-découverte-des-modèles-gratuits)
4. [Intégration dans une API FastAPI](#4-intégration-dans-une-api-fastapi)
5. [Gestion des Exceptions et Bonnes Pratiques](#5-gestion-des-exceptions-et-bonnes-pratiques)

---

## 1. Installation dans votre Projet

### Option A : Depuis le dossier local (Mode Développement / Monorepo)

Si `NexusAI-Client` est sur votre machine :

```bash
# Avec uv (Recommandé)
uv add --editable /chemin/vers/NexusAI-Client

# Avec pip
pip install -e /chemin/vers/NexusAI-Client
```

### Option B : Depuis un dépôt Git

```bash
# Avec uv
uv add git+https://github.com/votre-compte/NexusAI-Client.git

# Avec pip
pip install git+https://github.com/votre-compte/NexusAI-Client.git
```

### Option C : Dans votre `pyproject.toml`

```toml
[project]
dependencies = [
    "nexusai-client @ file:///D:/GIT/NexusAI-Client",
]
```

---

## 2. Configuration des Variables d'Environnement

Dans le fichier `.env` de votre application principale, renseignez les clés des fournisseurs que vous souhaitez utiliser :

```env
# Fournisseurs Gratuits
GEMINI_FREE_API_KEY=votre_cle_google_ai_studio
MISTRAL_API_KEY=votre_cle_mistral
NVIDIA_API_KEY=nvapi-votre_cle_nvidia
OPENROUTER_API_KEY=sk-or-v1-votre_cle_openrouter

# Fournisseurs Payants (Optionnels)
DEEPSEEK_API_KEY=sk-votre_cle_deepseek
GEMINI_PRO_API_KEY=votre_cle_gemini_pro
```

---

## 3. Exemples de Fonctions Métier Prêtes à l'Emploi

### A. Fonction de génération de texte simple

```python
import asyncio
from nexusai_client import AIGateway

async def ask_ai(
    prompt: str,
    provider: str = "gemini_free",
    system_prompt: str | None = None,
    temperature: float = 0.3,
) -> str:
    """Interroge un modèle d'IA et retourne le texte généré."""
    async with AIGateway(provider=provider) as client:
        response = await client.generate_text(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
        )
        return response.text

# --- Exemple d'appel ---
async def main():
    reponse = await ask_ai(
        prompt="Donne-moi 3 idées de noms pour une application de météo.",
        system_prompt="Tu es un expert en branding créatif.",
        provider="gemini_free", # ou "nvidia_free", "mistral", "openrouter", "deepseek"
    )
    print(reponse)

if __name__ == "__main__":
    asyncio.run(main())
```

---

### B. Fonction avec Fallback automatique Multi-Providers

Si un fournisseur gratuit est saturé (HTTP 429) ou indisponible, la requête bascule instantanément sur le suivant :

```python
import asyncio
import logging
from nexusai_client import AIGateway, NexusAIError, RateLimitError, APITimeoutError

logger = logging.getLogger(__name__)

# Chaîne de secours : Gemini Free -> Nvidia NIM -> OpenRouter -> DeepSeek (Payant)
FALLBACK_PROVIDERS = ["gemini_free", "nvidia_free", "openrouter", "deepseek"]

async def ask_ai_with_fallback(
    prompt: str,
    system_prompt: str | None = None,
    providers: list[str] = FALLBACK_PROVIDERS,
) -> tuple[str, str]:
    """Tente d'appeler les providers dans l'ordre de la liste jusqu'à réussite.
    
    Retourne (texte_genere, provider_ayant_reussi).
    """
    for provider in providers:
        try:
            async with AIGateway(provider=provider, timeout=15.0) as client:
                response = await client.generate_text(
                    prompt=prompt,
                    system_prompt=system_prompt,
                )
                return response.text, provider
        except (RateLimitError, APITimeoutError, NexusAIError) as err:
            logger.warning(f"⚠️ Échec sur '{provider}' ({err}). Bascule sur le suivant...")
            continue

    raise RuntimeError("❌ Tous les fournisseurs configurés ont échoué.")

# --- Exemple d'appel ---
async def main():
    texte, provider = await ask_ai_with_fallback("Résume la théorie du Big Bang.")
    print(f"✅ Réponse obtenue via [{provider}] :\n{texte}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

### C. Fonction de Chat Conversationnel avec Historique

```python
import asyncio
from nexusai_client import AIGateway, ChatMessage

class AIChatSession:
    """Gestionnaire de session de chat avec mémoire conversationnelle."""

    def __init__(self, provider: str = "openrouter", system_prompt: str = "Tu es un assistant utile.") -> None:
        self.provider = provider
        self.history: list[ChatMessage] = [
            ChatMessage(role="system", content=system_prompt)
        ]

    async def send_message(self, user_text: str) -> str:
        """Ajoute le message utilisateur, interroge l'IA et mémorise la réponse."""
        self.history.append(ChatMessage(role="user", content=user_text))

        async with AIGateway(self.provider) as client:
            response = await client.chat(messages=self.history)
            
            # Sauvegarder la réponse de l'assistant dans l'historique
            self.history.append(ChatMessage(role="assistant", content=response.text))
            return response.text

# --- Exemple de dialogue interactif ---
async def main():
    chat = AIChatSession(provider="mistral", system_prompt="Tu es un tuteur Python.")
    
    rep1 = await chat.send_message("Comment créer un dictionnaire en Python ?")
    print(f"Assistant: {rep1}\n")

    rep2 = await chat.send_message("Comment y ajouter une nouvelle clé ?")
    print(f"Assistant (avec contexte): {rep2}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

### D. Fonction de Génération de Code (Spécialisée)

```python
import asyncio
from nexusai_client import AIGateway

async def generate_python_code(task_description: str) -> str:
    """Utilise Codestral (Mistral) pour générer du code propre et documenté."""
    system = "Tu es un ingénieur logiciel Senior. Réponds uniquement avec du code Python propre, typé et testable."
    
    async with AIGateway("mistral") as client:
        response = await client.generate_text(
            prompt=f"Écris le code pour : {task_description}",
            system_prompt=system,
            model="codestral-latest", # Modèle spécialisé code
            temperature=0.1,
        )
        return response.text

# --- Exemple d'appel ---
async def main():
    code = await generate_python_code("un décorateur de retry avec backoff exponentiel")
    print(code)

if __name__ == "__main__":
    asyncio.run(main())
```

---

### E. Fonction d'Audit et de Découverte des Modèles Gratuits

```python
import asyncio
from nexusai_client import AIGateway

async def get_free_models_summary() -> list[dict[str, str]]:
    """Récupère la liste de tous les modèles gratuits disponibles en temps réel."""
    free_models_list: list[dict[str, str]] = []
    
    for prov in ["gemini_free", "mistral", "nvidia_free", "openrouter"]:
        try:
            async with AIGateway(prov) as client:
                models = await client.list_models(free_only=True)
                for m in models:
                    free_models_list.append({
                        "provider": prov,
                        "id": m.id,
                        "name": m.name,
                        "context": f"{m.context_length // 1000}k" if m.context_length else "N/A",
                    })
        except Exception:
            continue
            
    return free_models_list

# --- Exemple d'appel ---
async def main():
    models = await get_free_models_summary()
    print(f"🔍 {len(models)} modèles gratuits disponibles :")
    for m in models[:5]:
        print(f" - [{m['provider']}] {m['id']} (Contexte: {m['context']})")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 4. Intégration dans une API FastAPI

Voici comment brancher `NexusAI-Client` dans une route HTTP asynchrone :

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from nexusai_client import AIGateway, NexusAIError, RateLimitError

app = FastAPI(title="Mon API d'IA")

class GenerationRequest(BaseModel):
    prompt: str
    provider: str = "gemini_free"
    system_prompt: str | None = None
    temperature: float = 0.5

class GenerationResponse(BaseModel):
    result: str
    provider: str
    model: str
    tokens_used: int | None

@app.post("/api/generate", response_model=GenerationResponse)
async def generate_text_endpoint(req: GenerationRequest):
    try:
        async with AIGateway(provider=req.provider) as client:
            res = await client.generate_text(
                prompt=req.prompt,
                system_prompt=req.system_prompt,
                temperature=req.temperature,
            )
            return GenerationResponse(
                result=res.text,
                provider=res.provider,
                model=res.model,
                tokens_used=res.usage.total_tokens if res.usage else None,
            )
    except RateLimitError as e:
        raise HTTPException(status_code=429, detail=f"Quota dépassé : {e.message}")
    except NexusAIError as e:
        raise HTTPException(status_code=500, detail=f"Erreur IA : {e.message}")

# Lancement : uvicorn main:app --reload
```

---

## 5. Gestion des Exceptions et Bonnes Pratiques

Toutes les exceptions sont fortement typées :

```python
from nexusai_client import (
    AIGateway,
    NexusAIError,
    MissingAPIKeyError,    # Clé absente dans le .env
    AuthenticationError,   # Clé invalide (HTTP 401 / 403)
    RateLimitError,        # Quota ou débit dépassé (HTTP 429)
    APITimeoutError,       # Timeout réseau
    APIConnectionError,    # Serveur distant injoignable
    ProviderNotFoundError, # Nom de provider inconnu
)

try:
    async with AIGateway("deepseek") as client:
        response = await client.generate_text("Bonjour !")
except MissingAPIKeyError as e:
    print(f"⚠️ Configuration manquante : {e.env_var}")
except RateLimitError:
    print("⏳ Trop de requêtes, basculer sur un autre fournisseur !")
except NexusAIError as e:
    print(f"❌ Erreur générique : {e.message}")
```

---

## 💡 Tableau Récapitulatif des Identifiants Providers

| Nom dans `AIGateway("...")` | Fournisseur | Tier | Modèle par défaut |
| :--- | :--- | :--- | :--- |
| `"gemini_free"` | Google AI Studio | Gratuit | `gemini-2.5-flash` |
| `"gemini_pro"` | Google AI Studio Pro | Payant | `gemini-2.5-pro` |
| `"nvidia_free"` (ou `"nvidia"`) | Nvidia NIM | Gratuit (1 000 crédits) | `meta/llama-3.1-8b-instruct` |
| `"openrouter"` | OpenRouter | Gratuit (`:free`) & Payant | `openrouter/free` |
| `"mistral"` | Mistral AI | Gratuit & Payant | `mistral-small-latest` |
| `"deepseek"` | DeepSeek | Payant | `deepseek-chat` |
