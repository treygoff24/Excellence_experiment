id: 125
slug: troubleshooting-playbook
title: Ticket 125 — Troubleshooting playbook
ticket_file: ./codex/tickets/125-troubleshooting-playbook.md
log_file: ./codex/logs/125.md

## Objective
- Provide fast, actionable guidance for common failures during local Windows runs.

## Scope
- Add a troubleshooting section (or new doc) covering:
  - Ollama connection refused → run `ollama serve`; verify with `/api/tags`.
  - OOM during generation → reduce context; quantize to Q4_K_M; set `max_concurrent_requests=1`; close other GPU apps.
  - High `finish_reason=length` → increase `max_new_tokens`; configure `stop` sequences.
  - Parser row mismatch → ensure outputs echo `custom_id` and metadata; rerun Ticket 114 validation.
  - Slow scoring → recommend `MKL_NUM_THREADS/OMP_NUM_THREADS/NUMEXPR_MAX_THREADS=8`; avoid concurrent inference.

## Out of Scope
- Deep engine debugging; keep guidance concise and task‑oriented.

## Acceptance
- Users resolve listed issues with one or two edits/commands; playbook linked from `docs/windows.md`.
- Determinism & safety: Provide exact commands and expected outputs; avoid ambiguous advice.

## Deliverables
- Files:
  - New: `docs/troubleshooting_windows_local.md` (or a section in `docs/windows.md`)
- Log: ./codex/logs/125.md

## References
- docs/planning/PORT_WINDOWS_LOCAL_LLM_ACTION_PLAN.md (Ticket 15 — Troubleshooting)
- docs/guides/gpt5-prompting-best-practices-guide.md
