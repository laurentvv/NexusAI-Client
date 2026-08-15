<p align="center">
  <img src="./assets/banner.jpg" alt="NexusAI-Client - Unified Multi-Provider AI Gateway" width="100%">
</p>

# NexusAI-Client ⚡

<p align="center">
  <strong>An ultra-lightweight, strictly-typed, asynchronous Python 3.14 gateway for multi-provider AI APIs.</strong><br>
  <em>Unify DeepSeek, Google Gemini (Free & Pro), Mistral, Nvidia NIM, and OpenRouter behind a single, elegant interface with zero heavy SDK dependencies.</em>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.14+-3776AB.svg?style=flat-square&logo=python&logoColor=white" alt="Python 3.14+"></a>
  <a href="https://github.com/astral-sh/uv"><img src="https://img.shields.io/badge/package_manager-uv-DE5FE9.svg?style=flat-square" alt="uv"></a>
  <a href="https://www.python-httpx.org/"><img src="https://img.shields.io/badge/engine-httpx_async-009688.svg?style=flat-square" alt="httpx"></a>
  <a href="https://peps.python.org/pep-0561/"><img src="https://img.shields.io/badge/typing-PEP_561_Strict-blue.svg?style=flat-square" alt="Typing"></a>
  <a href="https://github.com/laurentvv/NexusAI-Client/actions"><img src="https://img.shields.io/badge/tests-18%2F18_passing-brightgreen.svg?style=flat-square" alt="Tests"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square" alt="License MIT"></a>
</p>

---

## 💡 Why NexusAI-Client?

Integrating multiple AI providers in modern Python applications usually requires installing **5 or 6 separate proprietary SDKs** (`google-genai`, `openai`, `mistralai`, etc.). This creates dozens of transitive dependencies, version conflicts, memory overhead, and fragmented codebases.

**NexusAI-Client** solves this at the core:
- 🪶 **Zero Heavyweight Dependencies**: Powered purely by `httpx` and `python-dotenv`.
- ⚡ **Native Asynchronous & SSE Streaming**: Stream responses token-by-token in real time via `stream_text()` and `stream_chat()`.
- 🔄 **Zero-Cost-First Smart Fallback**: Automatic progression from 100% Free Tiers to Paid Backups (`AIGateway.auto_fallback()`).
- 🎯 **Guaranteed JSON Outputs**: Native `json_mode=True` across all supported providers.
- 💰 **Live Account & Budget Inspection**: Inspect real-time balances (USD, NGC credits) and rate limits (RPM, TPM, RPD).
- 🔍 **640+ Models Discovered Live**: Automatic detection of free-tier models (`:free`) and accurate per-million-token pricing.

---

## 🌟 Spotlight: Zero-Cost-First Smart Fallback Routing

Why pay for AI calls when you can leverage high-throughput free tiers first, with seamless automatic fallback to paid commercial models?

**`NexusAI-Client` automatically prioritizes zero-cost models before touching your wallet:**

```
  ┌──────────────────────────────────────────────────────────────────────────┐
  │                        100% FREE ZERO-COST TIERS                         │
  ├───────────────────┬───────────────────┬──────────────────┬───────────────┤
  │ 1. Gemini Free    │ 2. Nvidia NIM     │ 3. OpenRouter    │ 4. Mistral    │
  │ (1M Token Context)│ (1,000 Credits)   │ (Auto Free Hub)  │ (Small/Code)  │
  └─────────┬─────────┴─────────┬─────────┴────────┬─────────┴───────┬───────┘
            │                   │                  │                 │
            ▼ (If Rate-Limited / 429 Quota Exceeded / Network Outage)▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │                    ULTRA-LOW-COST PAID BACKUP TIERS                      │
  ├───────────────────────────────────────┬──────────────────────────────────┤
  │ 5. DeepSeek ($0.27 / 1M tokens)       │ 6. Gemini Pro (Enterprise GCP)   │
  └───────────────────────────────────────┴──────────────────────────────────┘
```

### 1-Line Zero-Cost Failover in Your Code:

```python
import asyncio
from nexusai_client import AIGateway

async def main():
    # Automatically discovers active keys in .env and routes: Free -> Free -> Paid
    async with AIGateway.auto_fallback() as client:
        response = await client.generate_text("Explain quantum computing in 2 sentences.")
        print(f"✅ Served by [{response.provider}] with zero downtime:")
        print(response.text)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 🎯 Supported Providers Matrix

| Provider | Identifier (`provider`) | Tier | Protocol | Default Model | Live Budget & Quota Detection |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DeepSeek** | `"deepseek"` | Paid | OpenAI Chat API | `deepseek-chat` | Real-time USD Balance (`GET /user/balance`) |
| **Gemini Free** | `"gemini_free"` | Free (AI Studio) | Gemini REST | `gemini-2.5-flash` | Quotas: 15 RPM \| 1M TPM \| 1,500 RPD |
| **Gemini Pro** | `"gemini_pro"` | Paid | Gemini REST | `gemini-2.5-pro` | Google Cloud Pay-as-you-go Billing |
| **Mistral AI** | `"mistral"` | Free / Platform | OpenAI Chat API | `mistral-small-latest` | Free Dev Models (`codestral-latest`, etc.) |
| **Nvidia NIM** | `"nvidia_free"` | Free (NGC) | OpenAI Chat API | `meta/llama-3.1-8b-instruct` | 1,000 Free GPU Inference Credits (NGC) |
| **OpenRouter** | `"openrouter"` | Free & Paid | OpenAI Chat API | `openrouter/free` | 19 Free models live + 390 Commercial models |

---

## 🚀 Quickstart (1 Minute)

### 1. Installation

```bash
# With uv (Recommended)
uv add git+https://github.com/laurentvv/NexusAI-Client.git

# Local / Editable mode
uv add --editable /path/to/NexusAI-Client
```

### 2. Configure API Keys (`.env`)

Create a `.env` file at the root of your project:

```env
# Free Tiers (Priority 1)
GEMINI_FREE_API_KEY=your_google_ai_studio_key
NVIDIA_API_KEY=nvapi-your_nvidia_nim_key
OPENROUTER_API_KEY=sk-or-v1-your_openrouter_key
MISTRAL_API_KEY=your_mistral_api_key

# Paid Tiers (Backup Priority 2)
DEEPSEEK_API_KEY=sk-your_deepseek_key
GEMINI_PRO_API_KEY=your_gemini_pro_key
```

### 3. Basic Generation

```python
import asyncio
from nexusai_client import AIGateway

async def main():
    async with AIGateway("gemini_free") as client:
        response = await client.generate_text("Explain the theory of relativity in 2 sentences.")
        print(response.text)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 🍳 Cookbooks & Common Patterns

### 1. Real-Time Token Streaming (SSE)

```python
import asyncio
from nexusai_client import AIGateway

async def main():
    async with AIGateway("openrouter") as client:
        async for chunk in client.stream_text("Write a short poem about space exploration."):
            print(chunk, end="", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
```

### 2. Custom Fallback Chain (Fine-Grained Strategy)

Define your own explicit priority list:

```python
import asyncio
from nexusai_client import AIGateway

async def main():
    # Priority: Free Gemini -> Free Nvidia NIM -> Free OpenRouter -> Paid DeepSeek
    custom_chain = ["gemini_free", "nvidia_free", "openrouter", "deepseek"]
    async with AIGateway.with_fallback(custom_chain) as client:
        res = await client.generate_text("Summarize the key advantages of Python 3.14.")
        print(f"[{res.provider}] {res.text}")

if __name__ == "__main__":
    asyncio.run(main())
```

### 3. Multi-Turn Conversation (Chat)

```python
import asyncio
from nexusai_client import AIGateway, ChatMessage

async def main():
    history = [
        ChatMessage(role="system", content="You are a senior algorithms instructor."),
        ChatMessage(role="user", content="How does QuickSort work?"),
    ]
    async with AIGateway("mistral") as client:
        response = await client.chat(history)
        print(response.text)

if __name__ == "__main__":
    asyncio.run(main())
```

### 4. Guaranteed Structured JSON Output

```python
import asyncio, json
from nexusai_client import AIGateway

async def main():
    async with AIGateway("gemini_free") as client:
        res = await client.generate_text(
            prompt="Extract profile data: Alice, 28 years old, Software Engineer.",
            json_mode=True,
        )
        data = json.loads(res.text)
        print("Parsed JSON:", data)

if __name__ == "__main__":
    asyncio.run(main())
```

### 5. Inspect Real-Time Account Balances & Quotas

```python
import asyncio
from nexusai_client import AIGateway

async def main():
    async with AIGateway("deepseek") as client:
        account = await client.get_account_info()
        print(account.format_summary())
        # Output: "Solde restant: $4.99 | (Offert: $0.00)"

if __name__ == "__main__":
    asyncio.run(main())
```

👉 **[Read the Full Integration Guide (FastAPI, Background Workers, Chat Sessions)](./INTEGRATION_GUIDE.md)**

---

## 🛠️ CLI Utilities Included

The package includes CLI diagnostic tools to audit your accesses and explore live models:

### 1. Test & Benchmark Access in Real-Time
```bash
uv run python verify_access.py
```
*Validates `.env` keys, inspects real-time balances, tests inference, and measures network latency in milliseconds.*

### 2. Live Catalog Explorer (640+ Models)
```bash
# List free-tier models only
uv run python list_all_models.py --free

# Search by keyword (e.g., llama, r1, codestral, sonnet)
uv run python list_all_models.py --search llama

# Export complete catalog with pricing to JSON
uv run python list_all_models.py --export models_catalog.json
```

### 3. Automated Unit Test Suite
```bash
uv run pytest -v
```

---

## 🛡️ Strongly-Typed Exceptions

All exceptions inherit from `NexusAIError` for clean error handling:

```python
from nexusai_client import (
    AIGateway,
    NexusAIError,
    MissingAPIKeyError,    # Missing API key in environment
    AuthenticationError,   # Invalid key (HTTP 401/403)
    RateLimitError,        # Quota exceeded (HTTP 429)
    APITimeoutError,       # Network timeout
    APIConnectionError,    # Unreachable provider host
    ProviderNotFoundError, # Unknown provider requested
)
```

---

## 📄 License

This project is licensed under the **MIT License**. Free for personal and commercial use.
