# Python Health Report — NexusAI-Client

Generated on 2026-08-15 21:58 by python-health-audit.

## 1. Executive Summary
- Global grade: B
- Reason: Grade B assigned: 0 Ruff findings, 0 Vulture dead code entries, 0 D/E/F complexity hotspots, and average Maintainability Index at 57.76 across 19 modules.

## 2. Dead Code
### 2.1 Local — Ruff
*No finding (0 error - All checks passed!)*

### 2.2 Global — Vulture
*No finding (0 dead code detected at confidence >= 80%)*

> ⚠️ Vulture produces false positives by construction (global static
> detection). Verify each entry before removal.

## 3. Complexity Hotspots (Radon)
| Rank | Entity | Location | Complexity |
| :---: | :--- | :--- | :--- |
| **C** | Function `list_all_models:main` | `list_all_models.py:109` | 15 |
| **C** | Function `test_single_provider` | `verify_access.py:102` | 13 |
| **C** | Method `Config.get_provider_config` | `src/nexusai_client/config.py:274` | 11 |
| **C** | Class `AccountInfo` | `src/nexusai_client/models.py:103` | 12 |
| **C** | Method `CohereProvider.stream_chat` | `src/nexusai_client/providers/cohere.py:198` | 14 |
| **C** | Method `CohereProvider.chat` | `src/nexusai_client/providers/cohere.py:98` | 11 |
| **C** | Method `GeminiBaseProvider.stream_chat` | `src/nexusai_client/providers/gemini.py:221` | 14 |
| **C** | Method `GeminiBaseProvider.chat` | `src/nexusai_client/providers/gemini.py:105` | 11 |
| **C** | Method `OpenAICompatibleProvider.stream_chat` | `src/nexusai_client/providers/openai_compat.py:182` | 13 |
| **C** | Method `OpenAICompatibleProvider.list_models` | `src/nexusai_client/providers/openai_compat.py:240` | 11 |
| **C** | Method `OpenRouterProvider.get_account_info` | `src/nexusai_client/providers/openrouter.py:144` | 11 |

*(Ranks A and B hidden — Zero D, E, or F critical hotspots detected across all 9 providers)*

## 4. Code Duplication (Pylint)
- `src/nexusai_client/providers/cohere.py` & `src/nexusai_client/providers/gemini.py` : SSE streaming line parser and error mapping blocks.
- `src/nexusai_client/gateway.py` & `src/nexusai_client/providers/cohere.py` : Forwarding parameter signatures for `chat` and `stream_chat`.

## 5. Recommended Action Plan
1. **Factorize SSE Line Streaming into Shared Base Utility** : Extract SSE parsing and error conversion into `BaseAIProvider._parse_sse_stream` to eliminate duplicated streaming logic.
2. **Setup Automated CI Matrix** : Add GitHub Actions workflow for Python 3.14 testing across Windows and Linux environments.
3. **Consolidate Common Rate Limit Structures** : Maintain unified dataclass models across all provider modules.
