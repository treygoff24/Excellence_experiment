id: 120
slug: local-example-configs
title: Ticket 120 — Local example configs (Ollama & llama.cpp)
ticket_file: ./codex/tickets/120-local-example-configs.md
log_file: ./codex/logs/120.md

## Objective
- Provide turnkey configs for Ollama and llama.cpp with validated fields and safe defaults for 16 GB VRAM.

## Scope
- New `config/eval_config.local.yaml` (Ollama) and `config/eval_config.local.llamacpp.yaml` (llama.cpp) with fields:
  - `backend: local`, `local_engine`, `local_endpoint` (Ollama), `local_model`, `temps`, `samples_per_item`, `max_new_tokens` per task type, `max_concurrent_requests`, `prompt_sets`, `default_prompt_set`.
- Validation hooks to check endpoint reachability or GGUF path existence where reasonable.

## Out of Scope
- Full docs (Ticket 121) and performance guide (Ticket 122).

## Acceptance
- Both configs validate via the repo’s config loader (or minimal loader if separate); actionable errors on typos or missing paths.
- Determinism & safety: Defaults match plan (context 4k, max_new_tokens ≤ 1024, concurrency 1).

## Deliverables
- Files:
  - New: `config/eval_config.local.yaml`
  - New: `config/eval_config.local.llamacpp.yaml`
- Log: ./codex/logs/120.md

## References
- docs/planning/PORT_WINDOWS_LOCAL_LLM_ACTION_PLAN.md (Ticket 10 — Local Example Configs)
- docs/guides/gpt5-prompting-best-practices-guide.md
