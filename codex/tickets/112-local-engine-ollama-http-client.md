id: 112
slug: local-engine-ollama-http-client
title: Ticket 112 — Local engine (preferred): Ollama HTTP client
ticket_file: ./codex/tickets/112-local-engine-ollama-http-client.md
log_file: ./codex/logs/112.md

## Objective
- Implement an Ollama-backed `InferenceClient` using the local HTTP API with message and prompt modes and basic health checks.

## Scope
- New `backends/local/ollama_client.py` implementing `generate(messages|prompt, model, params)` returning `{text, finish_reason, usage?, request_id, latency_s}`.
- Parameter mapping: `temperature`, `top_p`, `top_k`, `num_predict→max_new_tokens`, `stop`.
- Health checks: GET `/api/tags`; clear error on server not running or model missing, with remediation (`ollama serve`, `ollama pull <model>`).

## Out of Scope
- Batch execution and outputs — Ticket 114.
- Token estimation — Ticket 118.

## Acceptance
- One-liner sanity:
  - `python - <<<'from backends.local.ollama_client import OllamaClient as C; print(C().generate(prompt="Hello", model="<model>")["text"][:10])'` prints text.
- When composed with Ticket 114 on a 20‑item slice, responses return without exceptions; finish reasons captured.
- Determinism & safety: Timeouts set; explicit, actionable errors; no network calls beyond localhost by default.

## Deliverables
- Files:
  - New: `backends/local/ollama_client.py`
  - (Optional) New: `backends/local/__init__.py`
- Log: ./codex/logs/112.md

## References
- docs/planning/PORT_WINDOWS_LOCAL_LLM_ACTION_PLAN.md (Ticket 2 — Ollama HTTP Client)
- docs/guides/gpt5-prompting-best-practices-guide.md
