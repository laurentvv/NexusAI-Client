# 📦 NexusAI-Client Integration Guide

This guide explains how to integrate and consume **NexusAI-Client** as a centralized library within your Python projects (Web Applications, FastAPI / Flask backends, Autonomous AI Agents, Chatbots, Background Workers, and CLI tools).

---

## 📑 Table of Contents

1. [Installation in Your Project](#1-installation-in-your-project)
2. [Environment Configuration](#2-environment-configuration)
3. [Ready-to-Use Functional Recipes](#3-ready-to-use-functional-recipes)
   - [A. Generic Single-Prompt Function](#a-generic-single-prompt-function)
   - [B. Resilient Automatic Multi-Provider Fallback](#b-resilient-automatic-multi-provider-fallback)
   - [C. Stateful Multi-Turn Chat Session](#c-stateful-multi-turn-chat-session)
   - [D. Specialized Code Generation](#d-specialized-code-generation)
   - [E. Real-Time Free Models Discovery](#e-real-time-free-models-discovery)
   - [F. Autonomous AI Agent with Tool Calling](#f-autonomous-ai-agent-with-tool-calling)
4. [FastAPI Endpoint Integration](#4-fastapi-endpoint-integration)
5. [Exception Handling & Best Practices](#5-exception-handling--best-practices)

---

## 1. Installation in Your Project

### Option A: From local path (Development / Monorepo)

```bash
# With uv (Recommended)
uv add --editable /path/to/NexusAI-Client

# With pip
pip install -e /path/to/NexusAI-Client
```

### Option B: From Git Repository

```bash
# With uv
uv add git+https://github.com/laurentvv/NexusAI-Client.git

# With pip
pip install git+https://github.com/laurentvv/NexusAI-Client.git
```

### Option C: In your `pyproject.toml`

```toml
[project]
dependencies = [
    "nexusai-client @ git+https://github.com/laurentvv/NexusAI-Client.git",
]
```

---

## 2. Environment Configuration

In your application's `.env` file, specify the API keys for the providers you want to use:

```env
# Free Tiers
GEMINI_FREE_API_KEY=your_google_ai_studio_key
GROQ_API_KEY=gsk_your_groq_key
CEREBRAS_API_KEY=csk-your_cerebras_key
COHERE_API_KEY=your_cohere_key
MISTRAL_API_KEY=your_mistral_api_key
NVIDIA_API_KEY=nvapi-your_nvidia_nim_key
OPENROUTER_API_KEY=sk-or-v1-your_openrouter_key
ORCAROUTER_API_KEY=sk-orca-your_orcarouter_key

# Paid Tiers (Optional)
DEEPSEEK_API_KEY=sk-your_deepseek_key
GEMINI_PRO_API_KEY=your_gemini_pro_key
```

---

## 3. Ready-to-Use Functional Recipes

### A. Generic Single-Prompt Function

```python
import asyncio
from nexusai_client import AIGateway

async def ask_ai(
    prompt: str,
    provider: str = "groq",
    system_prompt: str | None = None,
    temperature: float = 0.3,
) -> str:
    """Send a prompt to any supported AI provider and return the generated text."""
    async with AIGateway(provider=provider) as client:
        response = await client.generate_text(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
        )
        return response.text

# --- Example Usage ---
async def main():
    reply = await ask_ai(
        prompt="Suggest 3 catchy domain names for an AI analytics platform.",
        system_prompt="You are a creative branding expert.",
        provider="groq", # or "gemini_free", "nvidia_free", "mistral", "openrouter", "deepseek"
    )
    print(reply)

if __name__ == "__main__":
    asyncio.run(main())
```

---

### B. Resilient Automatic Multi-Provider Fallback

If a free provider is rate-limited (HTTP 429) or unreachable, automatically failover to the next provider in the chain:

```python
import asyncio
import logging
from nexusai_client import AIGateway, NexusAIError

logger = logging.getLogger(__name__)

# Fallback order: Gemini Free -> Groq LPU -> Nvidia NIM -> OpenRouter -> DeepSeek (Paid)
PROVIDER_CHAIN = ["gemini_free", "groq", "nvidia_free", "openrouter", "deepseek"]

async def ask_ai_with_fallback(prompt: str, system_prompt: str | None = None) -> tuple[str, str]:
    """Attempt generation across providers in priority order until success.
    
    Returns (generated_text, successful_provider_name).
    """
    async with AIGateway.with_fallback(PROVIDER_CHAIN, timeout=15.0) as client:
        response = await client.generate_text(
            prompt=prompt,
            system_prompt=system_prompt,
        )
        return response.text, response.provider

# --- Example Usage ---
async def main():
    text, provider = await ask_ai_with_fallback("Explain the concept of quantum entanglement.")
    print(f"✅ Succeeded via [{provider}]:\n{text}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

### C. Stateful Multi-Turn Chat Session

```python
import asyncio
from nexusai_client import AIGateway, ChatMessage

class AIChatSession:
    """Manages an ongoing multi-turn conversation with memory."""

    def __init__(self, provider: str = "openrouter", system_prompt: str = "You are a helpful assistant.") -> None:
        self.provider = provider
        self.history: list[ChatMessage] = [
            ChatMessage(role="system", content=system_prompt)
        ]

    async def send_message(self, user_text: str) -> str:
        """Append user message, query AI, and store assistant reply."""
        self.history.append(ChatMessage(role="user", content=user_text))

        async with AIGateway(self.provider) as client:
            response = await client.chat(messages=self.history)
            self.history.append(ChatMessage(role="assistant", content=response.text))
            return response.text

# --- Example Interactive Dialogue ---
async def main():
    chat = AIChatSession(provider="mistral", system_prompt="You are a Python programming tutor.")
    
    rep1 = await chat.send_message("How do I initialize an empty dictionary in Python?")
    print(f"Assistant: {rep1}\n")

    rep2 = await chat.send_message("How do I add a key to it?")
    print(f"Assistant (with context): {rep2}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

### D. Specialized Code Generation

```python
import asyncio
from nexusai_client import AIGateway

async def generate_python_code(task_description: str) -> str:
    """Use Codestral (Mistral) for production-grade, typed, and clean Python code."""
    system = "You are a Principal Software Engineer. Provide clean, typed, PEP 8 compliant code with docstrings."
    
    async with AIGateway("mistral") as client:
        response = await client.generate_text(
            prompt=f"Task: {task_description}",
            system_prompt=system,
            model="codestral-latest", # Specialized coding model
            temperature=0.1,
        )
        return response.text

# --- Example Usage ---
async def main():
    code = await generate_python_code("an asynchronous rate limiter with leaky bucket algorithm")
    print(code)

if __name__ == "__main__":
    asyncio.run(main())
```

---

### E. Real-Time Free Models Discovery

```python
import asyncio
from nexusai_client import AIGateway

async def get_all_free_models() -> list[dict[str, str]]:
    """Fetch live list of all active free-tier models across providers."""
    free_models: list[dict[str, str]] = []
    
    for prov in ["gemini_free", "mistral", "nvidia_free", "openrouter"]:
        try:
            async with AIGateway(prov) as client:
                models = await client.list_models(free_only=True)
                for m in models:
                    free_models.append({
                        "provider": prov,
                        "id": m.id,
                        "name": m.name,
                        "context": f"{m.context_length // 1000}k" if m.context_length else "N/A",
                    })
        except Exception:
            continue
            
    return free_models

# --- Example Usage ---
async def main():
    models = await get_all_free_models()
    print(f"🔍 Found {len(models)} active free models:")
    for m in models[:5]:
        print(f" - [{m['provider']}] {m['id']} (Context: {m['context']})")

if __name__ == "__main__":
    asyncio.run(main())
```

---

### F. Autonomous AI Agent with Tool Calling

Build autonomous ReAct agent loops that can query tools, databases, or external APIs seamlessly across any provider:

```python
import asyncio
import json
from nexusai_client import (
    AIGateway,
    ChatMessage,
    FunctionDefinition,
    ToolCall,
    ToolDefinition,
)

# 1. Define tools
def execute_database_query(sql_query: str) -> str:
    """Mock local database function."""
    return json.dumps({"status": "success", "rows": [{"id": 1, "user": "Alice", "balance": 1500}]})

sql_tool = ToolDefinition(
    function=FunctionDefinition(
        name="execute_database_query",
        description="Execute a read-only SQL query on the users database.",
        parameters={
            "type": "object",
            "properties": {
                "sql_query": {
                    "type": "string",
                    "description": "The SQL query to execute (e.g. SELECT * FROM users)",
                }
            },
            "required": ["sql_query"],
        },
    )
)

AVAILABLE_TOOLS = {
    "execute_database_query": execute_database_query,
}

# 2. Run Autonomous Tool Execution Loop
async def run_agent(user_question: str) -> str:
    messages: list[ChatMessage] = [
        ChatMessage(role="system", content="You are a data assistant with direct access to SQL tools."),
        ChatMessage(role="user", content=user_question),
    ]

    async with AIGateway.auto_fallback() as client:
        # Step 1: Query model with tools
        response = await client.chat(messages=messages, tools=[sql_tool])

        # Step 2: Check if model requested tools
        if response.has_tool_calls:
            # Append assistant's tool calling response to conversation history
            messages.append(ChatMessage(role="assistant", content=response.text, tool_calls=response.tool_calls))

            # Execute requested tools locally
            for tool_call in response.tool_calls:
                fn = AVAILABLE_TOOLS.get(tool_call.name)
                if fn:
                    tool_result = fn(**tool_call.arguments)
                    # Feed tool output back to agent
                    messages.append(
                        ChatMessage(
                            role="tool",
                            name=tool_call.name,
                            tool_call_id=tool_call.id,
                            content=tool_result,
                        )
                    )

            # Step 3: Get final model answer with tool observations
            final_response = await client.chat(messages=messages)
            return final_response.text

        return response.text

# --- Example Usage ---
async def main():
    answer = await run_agent("What is the account balance of user Alice?")
    print("🤖 Agent Final Answer:\n", answer)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 4. FastAPI Endpoint Integration

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from nexusai_client import AIGateway, NexusAIError, RateLimitError

app = FastAPI(title="AI Gateway Microservice")

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
        raise HTTPException(status_code=429, detail=f"Rate limit exceeded: {e.message}")
    except NexusAIError as e:
        raise HTTPException(status_code=500, detail=f"AI Provider error: {e.message}")

# Run with: uvicorn main:app --reload
```

---

## 5. Exception Handling & Best Practices

```python
from nexusai_client import (
    AIGateway,
    NexusAIError,
    MissingAPIKeyError,    # Missing API key in .env
    AuthenticationError,   # Invalid or expired key (HTTP 401/403)
    RateLimitError,        # Quota or rate limit reached (HTTP 429)
    APITimeoutError,       # Network timeout
    APIConnectionError,    # Remote host unreachable
    ProviderNotFoundError, # Unrecognized provider name
)

try:
    async with AIGateway("deepseek") as client:
        response = await client.generate_text("Hello!")
except MissingAPIKeyError as e:
    print(f"⚠️ Missing configuration: {e.env_var}")
except RateLimitError:
    print("⏳ Rate limit exceeded, fallback to secondary provider!")
except NexusAIError as e:
    print(f"❌ Generic NexusAI error: {e.message}")
```

---

## 💡 Provider Identifiers Quick Reference

| Identifier in `AIGateway("...")` | Provider | Tier | Default Model |
| :--- | :--- | :--- | :--- |
| `"gemini_free"` | Google AI Studio | Free | `gemini-2.5-flash` |
| `"gemini_pro"` | Google AI Studio Pro | Paid | `gemini-2.5-pro` |
| `"groq"` (or `"groq_free"`) | Groq Cloud LPU | Free (30 RPM) | `llama-3.3-70b-versatile` |
| `"nvidia_free"` (or `"nvidia"`) | Nvidia NIM | Free (1,000 credits) | `meta/llama-3.1-8b-instruct` |
| `"openrouter"` | OpenRouter | Free (`:free`) & Paid | `openrouter/free` |
| `"mistral"` | Mistral AI | Free & Paid | `mistral-small-latest` |
| `"deepseek"` | DeepSeek | Paid | `deepseek-chat` |
