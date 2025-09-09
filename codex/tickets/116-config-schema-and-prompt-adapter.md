id: 116
slug: config-schema-and-prompt-adapter
title: Ticket 116 — Config schema & prompt adapter
ticket_file: ./codex/tickets/116-config-schema-and-prompt-adapter.md
log_file: ./codex/logs/116.md

## Objective
- Extend config to support local backend settings and centralize system prompt rendering for A/B experiments.

## Scope
- Update `config/schema.py` to add fields:
  - `backend: local|fireworks`, `local_engine: ollama|llama_cpp`, `local_endpoint`, `local_model`, `max_concurrent_requests`, optional `tokenizer`.
- New `scripts/prompt_adapter.py` to render either `messages=[{role:"system"|"user"}]` or a single `prompt` string per item, condition, prompt set, task type, temps, and max_new_tokens.
- Optional `scripts/audit_prompts.py` to length-audit prompts.

## Out of Scope
- Engine/network logic (Tickets 112–114).

## Acceptance
- Rendering a sample item for `control` vs `treatment` shows only the system prompt differs; user content and context handling are correct for closed/open-book.
- Config validation passes and fields are documented via Pydantic types/help.
- Determinism & safety: Single source of truth for prompts; unit check that control/treatment system prompts hash differently while user message hashes match per item.

## Deliverables
- Files:
  - Update: `config/schema.py`
  - New: `scripts/prompt_adapter.py`
  - (Optional) New: `scripts/audit_prompts.py`
- Log: ./codex/logs/116.md

## References
- docs/planning/PORT_WINDOWS_LOCAL_LLM_ACTION_PLAN.md (Ticket 6 — Config & Prompt Adapter)
- docs/guides/gpt5-prompting-best-practices-guide.md
