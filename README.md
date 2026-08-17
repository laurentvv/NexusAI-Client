<p align="center">
  <img src="./assets/banner.jpg" alt="NexusAI-Client - Unified Multi-Provider AI Gateway" width="100%">
</p>

# NexusAI-Client ⚡

<p align="center">
  <strong>An ultra-lightweight, strictly-typed, asynchronous Python 3.12+ gateway for multi-provider AI APIs.</strong><br>
  <em>Unify Cerebras, Cohere, DeepSeek, Google Gemini (Free & Pro), Groq, Mistral, Nvidia NIM, OpenRouter, and OrcaRouter behind a single, elegant interface with zero heavy SDK dependencies.</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/nexusai-client/"><img src="https://img.shields.io/pypi/v/nexusai-client.svg?style=flat-square&logo=pypi&logoColor=white" alt="PyPI version"></a>
  <a href="https://nexus-ai-client-doc.vercel.app/"><img src="https://img.shields.io/badge/docs-nexus--ai--client--doc.vercel.app-00DC82.svg?style=flat-square&logo=vercel&logoColor=white" alt="Documentation Website"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.12+-3776AB.svg?style=flat-square&logo=python&logoColor=white" alt="Python 3.12+"></a>
  <a href="https://github.com/astral-sh/uv"><img src="https://img.shields.io/badge/package_manager-uv-DE5FE9.svg?style=flat-square" alt="uv"></a>
  <a href="https://www.python-httpx.org/"><img src="https://img.shields.io/badge/engine-httpx_async-009688.svg?style=flat-square" alt="httpx"></a>
  <a href="https://peps.python.org/pep-0561/"><img src="https://img.shields.io/badge/typing-PEP_561_Strict-blue.svg?style=flat-square" alt="Typing"></a>
  <a href="https://github.com/laurentvv/NexusAI-Client/actions"><img src="https://img.shields.io/badge/tests-32%2F32_passing-brightgreen.svg?style=flat-square" alt="Tests"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square" alt="License MIT"></a>
</p>

> 🌐 **Interactive Documentation Website:** [https://nexus-ai-client-doc.vercel.app/](https://nexus-ai-client-doc.vercel.app/)

---

## 💡 Why NexusAI-Client?

Integrating multiple AI providers in modern Python applications usually requires installing **9 or 10 separate proprietary SDKs** (`google-genai`, `openai`, `groq`, `cohere`, `mistralai`, etc.). This creates dozens of transitive dependencies, version conflicts, memory overhead, and fragmented codebases.

**NexusAI-Client** solves this at the core:
- 🪶 **Zero Heavyweight Dependencies**: Powered purely by `httpx` and `python-dotenv`.
- ⚡ **Native Asynchronous & SSE Streaming**: Stream responses token-by-token in real time via `stream_text()` and `stream_chat()`.
- 🔄 **Zero-Cost-First Smart Fallback**: Automatic progression from 100% Free Tiers (Gemini, Groq, Cerebras, Cohere, Nvidia, OpenRouter, OrcaRouter, Mistral) to Paid Backups (`AIGateway.auto_fallback()`).
- 🛠️ **Universal Tool Calling / Function Calling**: Seamless tool definitions, structured function arguments parsing, and multi-turn agent loops across Groq, Cerebras, Mistral, DeepSeek, Gemini REST, Cohere V2, and Nvidia NIM.
- 🚀 **World-Record Hardware Accelerators**: Native support for Groq LPUs and Cerebras CS-3 Wafer-Scale engines (2,000+ tokens/sec).
- 🧠 **Enterprise Reasoning & Search Models**: Native Cohere Command R+, DeepSeek R1, and Qwen 3.8 models.
- 🎯 **Guaranteed JSON Outputs**: Native `json_mode=True` across all supported providers.
- 💰 **Live Account & Budget Inspection**: Inspect real-time balances (USD, NGC credits) and rate limits (RPM, TPM, RPD).
- 🔍 **670+ Models Discovered Live**: Automatic detection of free-tier models (`:free`, `-free`) and accurate per-million-token pricing.

---

## 🌟 Spotlight: Zero-Cost-First Smart Fallback Routing

Why pay for AI calls when you can leverage high-throughput free tiers first, with seamless automatic fallback to paid commercial models?

**`NexusAI-Client` automatically prioritizes zero-cost models before touching your wallet:**

```
  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │                                               100% FREE ZERO-COST TIERS                                                │
  ├──────────────┬──────────────┬──────────────────┬──────────────┬──────────────┬──────────────┬──────────────┬───────────┤
  │ 1. Gemini    │ 2. Groq LPU  │ 3. Cerebras CS-3 │ 4. Nvidia    │ 5. OpenRouter│ 6. OrcaRouter│ 7. Cohere    │ 8. Mistral│
  │ (1M Context) │ (Ultra-Fast) │ (2000+ tok/s)    │ (1k Credits) │ (Free Hub)   │ (Qwen/DeepS) │ (Command R+) │ (Dev Free)│
  └──────┬───────┴──────┬───────┴────────┬─────────┴──────┬───────┴──────┬───────┴──────┬───────┴──────┬───────┴─────┬─────┘
         │              │                │                │              │              │              │             │
         ▼ (If Rate-Limited / 429 Quota Exceeded / Network Outage / Timeout) ────────────────────────────────────────▼
  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │                                           ULTRA-LOW-COST PAID BACKUP TIERS                                             │
  ├────────────────────────────────────────────────────────────┬───────────────────────────────────────────────────────────┤
  │ 9. DeepSeek ($0.27 / 1M tokens)                            │ 10. Gemini Pro (Enterprise GCP)                           │
  └────────────────────────────────────────────────────────────┴───────────────────────────────────────────────────────────┘
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
| **Cerebras** | `"cerebras"` (or `"cerebras_free"`) | Free (CS-3) | OpenAI Chat API | `gpt-oss-120b` | Quotas: 30 RPM \| 60k TPM \| 1M tok/day |
| **Cohere** | `"cohere"` (or `"cohere_free"`) | Free Trial | Cohere V2 REST | `command-r-plus-08-2024` | Quotas: 20 RPM \| 1,000 calls/month |
| **DeepSeek** | `"deepseek"` | Paid | OpenAI Chat API | `deepseek-chat` | Real-time USD Balance (`GET /user/balance`) |
| **Gemini Free** | `"gemini_free"` | Free (AI Studio) | Gemini REST | `gemini-2.5-flash` | Quotas: 15 RPM \| 1M TPM \| 1,500 RPD |
| **Gemini Pro** | `"gemini_pro"` | Paid | Gemini REST | `gemini-2.5-pro` | Google Cloud Pay-as-you-go Billing |
| **Groq** | `"groq"` (or `"groq_free"`) | Free (LPU) | OpenAI Chat API | `llama-3.3-70b-versatile` | Quotas: 30 RPM \| 14,400 RPD \| 30k TPM |
| **Mistral AI** | `"mistral"` | Free / Platform | OpenAI Chat API | `mistral-small-latest` | Free Dev Models (`codestral-latest`, etc.) |
| **Nvidia NIM** | `"nvidia_free"` | Free (NGC) | OpenAI Chat API | `meta/llama-3.1-8b-instruct` | 1,000 Free GPU Inference Credits (NGC) |
| **OpenRouter** | `"openrouter"` | Free & Paid | OpenAI Chat API | `openrouter/free` | 19 Free models live + 390 Commercial models |
| **OrcaRouter** | `"orcarouter"` (or `"orcarouter_free"`) | Free & Paid | OpenAI Chat API | `qwen/qwen3.8-27b-free` | Zero-margin routing + Free tier models (`-free`) |

---

## 🚀 Quickstart (1 Minute)

### 1. Installation

```bash
# With pip
pip install nexusai-client

# With uv (Recommended)
uv add nexusai-client

# With poetry
poetry add nexusai-client

# Local / Editable mode (development)
uv add --editable /path/to/NexusAI-Client
```

### 2. Configure API Keys (`.env`)

Create a `.env` file at the root of your project:

```env
# Free Tiers (Priority 1)
GEMINI_FREE_API_KEY=your_google_ai_studio_key
GROQ_API_KEY=gsk_your_groq_key
CEREBRAS_API_KEY=csk-your_cerebras_key
COHERE_API_KEY=your_cohere_key
NVIDIA_API_KEY=nvapi-your_nvidia_nim_key
OPENROUTER_API_KEY=sk-or-v1-your_openrouter_key
ORCAROUTER_API_KEY=sk-orca-your_orcarouter_key
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
    async with AIGateway("cerebras") as client:
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
    async with AIGateway("groq") as client:
        async for chunk in client.stream_text("Write a short poem about lightning fast LPUs."):
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
    # Priority: Free Gemini -> Free Groq -> Free Cerebras -> Free Cohere -> Paid DeepSeek
    custom_chain = ["gemini_free", "groq", "cerebras", "cohere", "nvidia_free", "openrouter", "deepseek"]
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
    async with AIGateway("cohere") as client:
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
    async with AIGateway("groq") as client:
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

### 6. Multimodal Vision Analysis (Images, Charts, PDFs)

Pass a local file path (`Path` or `str`), raw `bytes`, or web URL:

```python
import asyncio
from nexusai_client import AIGateway

async def main():
    # Automatically selects the best Vision model (Gemini 2.5 Flash, Llama 3.2 Vision, Aya Vision, Pixtral)
    async with AIGateway.auto_fallback_vision() as client:
        res = await client.analyze_image(
            prompt="Extract the invoice total and line items formatted as JSON.",
            image="invoice.png", # or "https://example.com/chart.jpg" or raw bytes
            json_mode=True,
        )
        print(f"[{res.provider} / {res.model}]:")
        print(res.text)

if __name__ == "__main__":
    asyncio.run(main())
```

### 7. Universal Tool Calling (Autonomous AI Agents)

Equip AI models with callable tools across all providers (Groq, Cerebras, Mistral, DeepSeek, Gemini, Cohere, etc.):

```python
import asyncio
from nexusai_client import AIGateway, ChatMessage, FunctionDefinition, ToolDefinition

# Define tool schema
weather_tool = ToolDefinition(
    function=FunctionDefinition(
        name="get_current_weather",
        description="Get current temperature and conditions for a given city.",
        parameters={
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name, e.g. Tokyo, Paris"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    )
)

async def main():
    async with AIGateway.auto_fallback() as client:
        messages = [ChatMessage(role="user", content="What is the weather in Tokyo?")]
        response = await client.chat(messages=messages, tools=[weather_tool])

        if response.has_tool_calls:
            for call in response.tool_calls:
                print(f"🔧 Tool Requested: {call.name}")
                print(f"📦 Arguments: {call.arguments}")
                # Execute your local Python function and return result back to agent loop!

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

### 2. Live Catalog Explorer (670+ Models)
```bash
# List free-tier models only
uv run python list_all_models.py --free

# Search by keyword (e.g., llama, r1, command, sonnet)
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
