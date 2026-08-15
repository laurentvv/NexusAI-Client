# Python Health Report — NexusAI-Client

Generated on 2026-08-15 21:32 by python-health-audit.

## 1. Executive Summary
- Global grade: B
- Reason: Grade B assigned: 0 Ruff dead code findings, 0 Vulture entries, 0 D/E/F complexity hotspots, and average Maintainability Index at 58.16.

## 2. Dead Code
### 2.1 Local — Ruff
*No finding (0 error - All checks passed!)*

### 2.2 Global — Vulture
*No finding (0 dead code detected at confidence >= 80%)*

> ⚠️ Vulture produces false positives by construction (global static
> detection). Verify each entry before removal.

## 3. Complexity Hotspots (Radon)
| Rank | Entity | Location | Complexity |
| :---: | :--- | :--- | :---: |
| **C** | Function `list_all_models:main` | `list_all_models.py:109` | 15 |
| **C** | Function `test_single_provider` | `verify_access.py:84` | 13 |
| **C** | Method `Config.get_provider_config` | `src/nexusai_client/config.py:208` | 11 |
| **C** | Class `AccountInfo` | `src/nexusai_client/models.py:97` | 12 |
| **C** | Method `GeminiBaseProvider.stream_chat` | `src/nexusai_client/providers/gemini.py:221` | 14 |
| **C** | Method `GeminiBaseProvider.chat` | `src/nexusai_client/providers/gemini.py:105` | 11 |
| **C** | Method `OpenAICompatibleProvider.stream_chat` | `src/nexusai_client/providers/openai_compat.py:182` | 13 |
| **C** | Method `OpenAICompatibleProvider.list_models` | `src/nexusai_client/providers/openai_compat.py:240` | 11 |
| **C** | Method `OpenRouterProvider.get_account_info` | `src/nexusai_client/providers/openrouter.py:144` | 11 |

*(Ranks A and B hidden — Zero D, E, or F critical hotspots detected)*

## 4. Code Duplication (Pylint)
- `src/nexusai_client/providers/gemini.py` & `src/nexusai_client/providers/openai_compat.py` : SSE streaming line parser and error mapping (28 lines).
- `src/nexusai_client/gateway.py` & `src/nexusai_client/providers/openai_compat.py` : Common signature parameter forwarding in `chat` and `stream_chat`.

## 5. Recommended Action Plan
1. **Factorize SSE Stream Decoding into a Shared Helper** : Extract the line iteration and JSON decode block into `BaseAIProvider._stream_sse_lines` to reduce remaining boilerplate duplication.
2. **Add CI Health Check Workflow** : Integrate `uvx ruff check` and `pytest` into a GitHub Action workflow to ensure 0-regression on future pull requests.
3. **Keep Model Registries Modular** : Maintain table-driven definitions for future providers to keep cyclomatic complexity at rank A/B.
