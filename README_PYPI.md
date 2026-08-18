<p align="center">
  <img src="https://raw.githubusercontent.com/laurentvv/NexusAI-Client/main/assets/banner.jpg" alt="NexusAI-Client - Unified Multi-Provider AI Gateway" width="100%">
</p>

# NexusAI-Client ⚡

<p align="center">
  <strong>An ultra-lightweight, strictly-typed, asynchronous Python gateway for multi-provider AI APIs.</strong><br>
  <em>Unify Cerebras, Cohere, DeepSeek, Google Gemini (Free & Pro), Groq, Mistral, Nvidia NIM, OpenRouter, and OrcaRouter behind a single, elegant interface with zero heavy SDK dependencies.</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/nexusai-client/"><img src="https://img.shields.io/pypi/v/nexusai-client.svg?style=flat-square&logo=pypi&logoColor=white" alt="PyPI version"></a>
  <a href="https://nexus-ai-client-doc.vercel.app/"><img src="https://img.shields.io/badge/docs-nexus--ai--client--doc.vercel.app-00DC82.svg?style=flat-square&logo=vercel&logoColor=white" alt="Documentation Website"></a>
  <a href="https://pypi.org/project/nexusai-client/"><img src="https://img.shields.io/pypi/pyversions/nexusai-client.svg?style=flat-square&logo=python&logoColor=white" alt="Python versions"></a>
  <a href="https://pypi.org/project/nexusai-client/"><img src="https://img.shields.io/pypi/l/nexusai-client.svg?style=flat-square" alt="License MIT"></a>
  <a href="https://www.python-httpx.org/"><img src="https://img.shields.io/badge/engine-httpx_async-009688.svg?style=flat-square" alt="httpx"></a>
  <a href="https://peps.python.org/pep-0561/"><img src="https://img.shields.io/badge/typing-PEP_561_Strict-blue.svg?style=flat-square" alt="Typing"></a>
</p>

> 🌐 **Interactive Documentation Website:** [https://nexus-ai-client-doc.vercel.app/](https://nexus-ai-client-doc.vercel.app/)

---

## 💡 Why NexusAI-Client?

Integrating multiple AI providers in modern Python applications usually requires installing **9 or 10 separate proprietary SDKs** (`google-genai`, `openai`, `groq`, `cohere`, `mistralai`, etc.). This creates dozens of transitive dependencies, version conflicts, memory overhead, and fragmented codebases.

**NexusAI-Client** solves this at the core:

- 🪶 **Zero Heavyweight Dependencies** — powered purely by `httpx` and `python-dotenv`.
- ⚡ **Native Asynchronous & SSE Streaming** — stream responses token-by-token in real time via `stream_text()` and `stream_chat()`.
- 🔄 **Zero-Cost-First Smart Fallback** — automatic progression from 100% free tiers (Gemini, Groq, Cerebras, Cohere, Nvidia, OpenRouter, OrcaRouter, Mistral) to paid backups with `AIGateway.auto_fallback()`.
- 🛠️ **Universal Tool Calling / Function Calling** — define tools once (`ToolDefinition`, `FunctionDefinition`), parse structured function calls, and run multi-turn agent loops across all providers.
- 🚀 **World-Record Hardware Accelerators** — native support for Groq LPUs and Cerebras CS-3 wafer-scale engines (2,000+ tokens/sec).
- 🧠 **Enterprise Reasoning & Search Models** — native Cohere Command R+, DeepSeek R1, and Qwen 3.8 models.
- 🎯 **Guaranteed JSON Outputs** — native `json_mode=True` across all supported providers.
- 💰 **Live Account & Budget Inspection** — inspect real-time balances (USD, NGC credits) and rate limits (RPM, TPM, RPD).
- 🔍 **670+ Models Discovered Live** — automatic detection of free-tier models (`:free`, `-free`) and accurate per-million-token pricing.
- 👁️ **Multimodal Vision** — analyze images, charts, and documents with automatic vision-model resolution via `analyze_image()` and `AIGateway.auto_fallback_vision()`.

---

## 🌟 Spotlight: Zero-Cost-First Smart Fallback Routing

Why pay for AI calls when you can leverage high-throughput free tiers first, with seamless automatic fallback to paid commercial models?

**NexusAI-Client automatically prioritizes zero-cost models before touching your wallet:**

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

### 1-Line Zero-Cost Failover in Your Code

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
| **Gemini Free** | `"gemini_free"` | Free (AI Studio) | Gemini REST | `gemini-3.5-flash-lite` | Auto-rotation 429 \| 15 RPM \| 500 RPD (Lite) / 20 RPD (Flash) |
| **Gemini Pro** | `"gemini_pro"` | Paid | Gemini REST | `gemini-3.1-pro-preview` | Google Cloud Pay-as-you-go Billing |
| **Groq** | `"groq"` (or `"groq_free"`) | Free (LPU) | OpenAI Chat API | `llama-3.3-70b-versatile` | Quotas: 30 RPM \| 14,400 RPD \| 30k TPM |
| **Mistral AI** | `"mistral"` | Free / Platform | OpenAI Chat API | `mistral-small-latest` | Free Dev Models (`codestral-latest`, etc.) |
| **Nvidia NIM** | `"nvidia_free"` | Free (NGC) | OpenAI Chat API | `meta/llama-3.1-8b-instruct` | 1,000 Free GPU Inference Credits (NGC) |
| **OpenRouter** | `"openrouter"` | Free & Paid | OpenAI Chat API | `openrouter/free` | 19 Free models live + 390 Commercial models |
| **OrcaRouter** | `"orcarouter"` (or `"orcarouter_free"`) | Free & Paid | OpenAI Chat API | `qwen/qwen3.8-27b-free` | Zero-margin gateway + Free tier models (`-free`) |

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
```

### 2. Configure API Keys (`.env`)

**No configuration code needed**: as soon as you `import nexusai_client`, the package automatically loads the `.env` file found in your current working directory (via `python-dotenv`). Real environment variables always take precedence.

Create a `.env` file at the root of your project with **only the keys you have** — every provider is optional:

```env
# ── Free Tiers (Priority 1) ────────────────────────────────────────────────
GEMINI_FREE_API_KEY=your_google_ai_studio_key
GROQ_API_KEY=gsk_your_groq_key
CEREBRAS_API_KEY=csk-your_cerebras_key
COHERE_API_KEY=your_cohere_key
NVIDIA_API_KEY=nvapi-your_nvidia_nim_key
OPENROUTER_API_KEY=sk-or-v1-your_openrouter_key
ORCAROUTER_API_KEY=sk-orca-your_orcarouter_key
MISTRAL_API_KEY=your_mistral_api_key

# ── Paid Tiers (Backup Priority 2) ─────────────────────────────────────────
DEEPSEEK_API_KEY=sk-your_deepseek_key
GEMINI_PRO_API_KEY=your_gemini_pro_key
```

#### Where to get each API key

| Provider | Environment Variable | Get a key |
| :--- | :--- | :--- |
| Gemini Free | `GEMINI_FREE_API_KEY` | [Google AI Studio](https://aistudio.google.com/apikey) |
| Gemini Pro | `GEMINI_PRO_API_KEY` | [Google AI Studio](https://aistudio.google.com/apikey) / GCP |
| Groq | `GROQ_API_KEY` | [console.groq.com/keys](https://console.groq.com/keys) |
| Cerebras | `CEREBRAS_API_KEY` | [cloud.cerebras.ai](https://cloud.cerebras.ai) |
| Cohere | `COHERE_API_KEY` | [dashboard.cohere.com](https://dashboard.cohere.com/api-keys) |
| Nvidia NIM | `NVIDIA_API_KEY` | [build.nvidia.com](https://build.nvidia.com) |
| OpenRouter | `OPENROUTER_API_KEY` | [openrouter.ai/keys](https://openrouter.ai/keys) |
| OrcaRouter | `ORCAROUTER_API_KEY` | [www.orcarouter.ai/console](https://www.orcarouter.ai/console) |
| Mistral | `MISTRAL_API_KEY` | [console.mistral.ai](https://console.mistral.ai/api-keys) |
| DeepSeek | `DEEPSEEK_API_KEY` | [platform.deepseek.com](https://platform.deepseek.com) |

> **Notes:**
> - `GEMINI_API_KEY` is accepted as a fallback alias for both `GEMINI_FREE_API_KEY` and `GEMINI_PRO_API_KEY`.
> - You can also pass a key directly in code: `AIGateway("groq", api_key="gsk_...")` — useful for CI/CD or key rotation without touching `.env`.

#### Optional advanced environment variables

| Variable | Purpose | Default |
| :--- | :--- | :--- |
| `DEEPSEEK_DEFAULT_MODEL`, `GROQ_DEFAULT_MODEL`, `CEREBRAS_DEFAULT_MODEL`, ... | Override the default model of a provider | Provider defaults (see matrix above) |
| `DEEPSEEK_BASE_URL`, `GROQ_BASE_URL`, `MISTRAL_BASE_URL`, ... | Point a provider at a custom endpoint or proxy | Official provider API URL |
| `NEXUS_DEFAULT_TIMEOUT` | Global request timeout in seconds (all providers) | `60` |
| `OPENROUTER_SITE_URL` / `OPENROUTER_APP_NAME` | App attribution headers sent to OpenRouter | `https://github.com/NexusAI-Client` / `NexusAI-Client` |

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
    # Automatically selects the best Vision model (Gemini 2.5 Flash, Llama 3.2 Vision, Qwen 3.8 Vision, Aya Vision, Pixtral)
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

if __name__ == "__main__":
    asyncio.run(main())
```

### 8. Intelligent Gemini Free Model Rotation (Auto 429 Quota Failover)

Google AI Studio Free Tier offers world-class models (`gemini-3.5-flash-lite`, `gemini-3.7-flash`, etc.) with massive context windows (up to 1M tokens) at zero cost. However, Google enforces **strict per-model rate limits and daily quota pools**:
- **Flash-Lite Models**: ~500 Requests/day (RPD), 15 RPM, 250k TPM
- **Flash Models**: ~20 Requests/day (RPD), 15 RPM, 1M TPM
- **Gemma Open Models**: ~14,400 Requests/day (RPD), 30 RPM, 30k TPM

When a single model hits its quota limit (`HTTP 429 RESOURCE_EXHAUSTED`), traditional SDKs fail immediately. **NexusAI-Client solves this natively with a 2-Tier Fallback Hierarchy**:

```
 ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │                                   LEVEL 1: INTRA-PROVIDER SMART MODEL ROTATION                             │
 │                                                                                                             │
 │  1. gemini-3.5-flash-lite (500 RPD) ──► 2. gemini-3.1-flash-lite (500 RPD) ──► 3. gemini-flash-lite-latest   │
 │                                                                                                             │
 │       ▼ (If 429 Quota Exceeded)                                                                             │
 │  4. gemini-3.7-flash (20 RPD)       ──► 5. gemini-3.6-flash (20 RPD)       ──► 6. gemini-3.5-flash (20 RPD)   │
 │                                                                                                             │
 │       ▼ (If 429 Quota Exceeded)                                                                             │
 │  7. gemini-flash-latest             ──► 8. gemini-2.5-flash-lite (500 RPD) ──► 9. gemini-2.5-flash (20 RPD)   │
 │                                                                                                             │
 │       ▼ (If 429 Quota Exceeded)                                                                             │
 │  10. gemma-4-31b-it (14.4k RPD)     ──► 11. gemma-4-26b-a4b-it (14.4k RPD)                                  │
 └──────────────────────────────────────────────────────┬──────────────────────────────────────────────────────┘
                                                        │ (Only if ALL 11 Gemini models exhausted)
                                                        ▼
 ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │                               LEVEL 2: INTER-PROVIDER AUTO-FALLBACK FAILOVER                                │
 │   Groq LPU ──► Cerebras CS-3 ──► Nvidia NIM ──► OrcaRouter ──► Mistral ──► Cohere ──► OpenRouter ──► DeepSeek│
 └─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

#### 📋 Complete 11-Model Cascade Chain

| Step | Model Identifier | Quotas (Free Tier) | Context Window | Primary Use Case |
| :---: | :--- | :--- | :---: | :--- |
| **1** | `gemini-3.5-flash-lite` *(Default)* | 500 RPD \| 15 RPM \| 250k TPM | 1,048,576 tokens | Ultra-fast default general queries & tool calling |
| **2** | `gemini-3.1-flash-lite` | 500 RPD \| 15 RPM \| 250k TPM | 1,048,576 tokens | Fast secondary lightweight failover |
| **3** | `gemini-flash-lite-latest` | 500 RPD \| 15 RPM \| 250k TPM | 1,048,576 tokens | Latest stable flash-lite pointer alias |
| **4** | `gemini-3.7-flash` | 20 RPD \| 15 RPM \| 1,000,000 TPM | 1,048,576 tokens | High-reasoning & complex logic tasks |
| **5** | `gemini-3.6-flash` | 20 RPD \| 15 RPM \| 1,000,000 TPM | 1,048,576 tokens | Advanced multimodal & code synthesis |
| **6** | `gemini-3.5-flash` | 20 RPD \| 15 RPM \| 1,000,000 TPM | 1,048,576 tokens | General multimodal & vision failover |
| **7** | `gemini-flash-latest` | 20 RPD \| 15 RPM \| 1,000,000 TPM | 1,048,576 tokens | Latest stable flash pointer alias |
| **8** | `gemini-2.5-flash-lite` | 500 RPD \| 15 RPM \| 250k TPM | 1,048,576 tokens | Previous-generation fast fallback |
| **9** | `gemini-2.5-flash` | 20 RPD \| 15 RPM \| 1,000,000 TPM | 1,048,576 tokens | Previous-generation robust fallback |
| **10** | `gemma-4-31b-it` | 14,400 RPD \| 30 RPM \| 30k TPM | 131,072 tokens | High-volume open-weight model with massive RPD |
| **11** | `gemma-4-26b-a4b-it` | 14,400 RPD \| 30 RPM \| 30k TPM | 131,072 tokens | Final emergency high-RPD free tier model |

#### 👁️ Multimodal Vision Rotation Sequence
For `analyze_image()`, the rotation automatically restricts itself to the 7 vision-capable Gemini models:
`gemini-3.5-flash-lite` ➔ `gemini-3.1-flash-lite` ➔ `gemini-3.7-flash` ➔ `gemini-3.6-flash` ➔ `gemini-3.5-flash` ➔ `gemini-2.5-flash-lite` ➔ `gemini-2.5-flash`.

#### ⏱️ Stateful Cooldown & Zero Latency Penalty
- **Automatic Cooldowns**: When a model returns `HTTP 429`, it is immediately placed in cooldown (`cooldown_seconds=60.0` by default). Deprecated/404 models are cooled down for 1 hour.
- **Zero Retrial Overhead**: Subsequent requests during the same process lifecycle instantly skip cooled-down models, routing directly to the first available operational model with **0ms penalty**.
- **Real-Time Visibility**: Inspect live model statuses and cooldown timers with `get_account_info()`.

#### 💻 Code Examples

**1. Transparent Auto-Failover (Zero Configuration):**
```python
import asyncio
from nexusai_client import AIGateway

async def main():
    # If the primary model (gemini-3.5-flash-lite) hits a 429 limit,
    # it immediately retries with gemini-3.1-flash-lite, gemini-3.7-flash, etc.
    async with AIGateway("gemini_free") as client:
        res = await client.generate_text("Explain quantum entanglement simply.")
        print(f"✅ Served by model [{res.model}] (Provider: {res.provider}):")
        print(res.text)

if __name__ == "__main__":
    asyncio.run(main())
```

**2. Live Rotation & Cooldown Diagnostics:**
```python
import asyncio
from nexusai_client import AIGateway

async def main():
    async with AIGateway("gemini_free") as client:
        info = await client.get_account_info()
        print(f"Platform: {info.extra_details['platform']}")
        print(f"Current Active Model: {info.extra_details['current_active_model']}")
        print(f"Models in Cooldown: {info.extra_details['models_in_cooldown']}")
        print(f"Quota Limits: {info.rate_limit_info}")

if __name__ == "__main__":
    asyncio.run(main())
```

**3. Customizing Fallback Sequence & Cooldowns:**
```python
import asyncio
from nexusai_client import AIGateway

async def main():
    # Define custom priority models and a 120s cooldown period
    custom_models = ["gemini-3.7-flash", "gemini-3.5-flash-lite", "gemma-4-31b-it"]
    async with AIGateway(
        "gemini_free",
        fallback_models=custom_models,
        cooldown_seconds=120.0,
        auto_rotate_models=True,
    ) as client:
        res = await client.generate_text("Write an async Python pipeline.")
        print(f"[{res.model}]: {res.text}")

if __name__ == "__main__":
    asyncio.run(main())
```

**4. Real-Time Token Streaming with Model Rotation:**
```python
import asyncio
from nexusai_client import AIGateway

async def main():
    # Streaming seamlessly falls back to candidate models if 429 is encountered at stream start
    async with AIGateway("gemini_free") as client:
        async for chunk in client.stream_text("Explain distributed systems in 3 bullet points."):
            print(chunk, end="", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
```

### 9. 100% Free Multi-Provider Fallback (`auto_fallback_free`)

Build zero-cost resilient workflows by cascading across all free providers present in your `.env`:

```python
import asyncio
from nexusai_client import AIGateway

async def main():
    # Cascade: Gemini Free (11 models) -> Groq LPU -> Cerebras CS-3 -> Nvidia NIM -> OrcaRouter -> Mistral -> Cohere -> OpenRouter
    async with AIGateway.auto_fallback_free() as client:
        res = await client.generate_text("Write a concise summary of AI agents architecture.")
        print(f"✅ Served by [{res.provider} / {res.model}]:\n{res.text}")

if __name__ == "__main__":
    asyncio.run(main())
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

The package ships a `py.typed` marker (PEP 561): all types are available to your IDE and type checker out of the box.

---

## 📚 Resources

- **Interactive Documentation Website:** [https://nexus-ai-client-doc.vercel.app/](https://nexus-ai-client-doc.vercel.app/)
- **Source code:** [github.com/laurentvv/NexusAI-Client](https://github.com/laurentvv/NexusAI-Client)
- **Full Integration Guide** (FastAPI, background workers, chat sessions): [INTEGRATION_GUIDE.md](https://github.com/laurentvv/NexusAI-Client/blob/main/INTEGRATION_GUIDE.md)
- **Issue tracker:** [github.com/laurentvv/NexusAI-Client/issues](https://github.com/laurentvv/NexusAI-Client/issues)

---

## 📄 License

This project is licensed under the **MIT License**. Free for personal and commercial use.
