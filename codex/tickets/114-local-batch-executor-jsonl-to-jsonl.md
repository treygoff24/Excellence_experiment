id: 114
slug: local-batch-executor-jsonl-to-jsonl
title: Ticket 114 — Local batch executor (JSONL in → JSONL out)
ticket_file: ./codex/tickets/114-local-batch-executor-jsonl-to-jsonl.md
log_file: ./codex/logs/114.md

## Objective
- Replace upload/submit/poll/download with a local work queue consuming per‑part JSONL inputs and writing per‑part results JSONL in the structure expected by the parser.

## Scope
- New `backends/local/local_batch.py`:
  - Read one `data/batch_inputs/*pXX.jsonl` at a time; render prompts via Prompt Adapter (Ticket 116).
  - Bounded async/thread pool with `max_concurrent_requests` from config.
  - Each output line must include `{custom_id, response_text, finish_reason, request_id, usage?, latency_s}` and echo needed input metadata.
  - Write per‑part JSONL to `results/raw/<trial>/<part>.jsonl` (or structure mirrored from cloud path used by existing parser).
- Idempotent resume: if output count == input count, skip part; record a small per‑part state file.
- Error handling: retries with backoff; errors file per part.

## Out of Scope
- CSV conversion — Ticket 115.
- Token estimation — Ticket 118.

## Acceptance
- On a 50‑item run, per‑part outputs are produced and combined without errors; rerun skips completed parts.
- Hard cap concurrency default ≤ 2 on 16 GB unless overridden.
- Determinism & safety: Stable ordering per input order; failures logged; no partial line corruption on crash.

## Deliverables
- Files:
  - New: `backends/local/local_batch.py`
  - (Optional) New: lightweight state and errors artifacts next to per‑part outputs
- Log: ./codex/logs/114.md

## References
- docs/planning/PORT_WINDOWS_LOCAL_LLM_ACTION_PLAN.md (Ticket 4 — Local Batch Executor)
- docs/guides/gpt5-prompting-best-practices-guide.md
