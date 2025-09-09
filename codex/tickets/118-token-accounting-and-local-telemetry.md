id: 118
slug: token-accounting-and-local-telemetry
title: Ticket 118 — Token accounting & optional local telemetry
ticket_file: ./codex/tickets/118-token-accounting-and-local-telemetry.md
log_file: ./codex/logs/118.md

## Objective
- Capture token counts when available; otherwise approximate via tokenizer; optionally record latency and GPU/VRAM telemetry.

## Scope
- Pass through engine‑reported usage to parser (Ticket 115) where available.
- New `scripts/estimate_tokens.py` with Llama/HF tokenizer support to estimate prompt vs completion tokens from rendered messages.
- Optional `telemetry/nvml.py` using `pynvml` to sample `gpu_util`, `mem_used`, `power_draw` during generation (feature‑flagged).
- Persist `local_costs.json` per trial with summary stats (avg latency, p50/p95, avg mem). Keep separate from core results.

## Out of Scope
- Changes to significance metrics; token estimates are best‑effort and informational.

## Acceptance
- `predictions.csv` includes token columns when the engine provides usage; otherwise blanks (or estimates if enabled).
- When telemetry flag enabled, a JSON summary is written; eval completes without telemetry as well.
- Determinism & safety: Clear labeling of estimates vs measured; telemetry is optional and off by default; robust fallback if NVML unavailable.

## Deliverables
- Files:
  - New: `scripts/estimate_tokens.py`
  - New: `telemetry/nvml.py` (optional module)
  - Update: wiring from batch executor to collect latency metrics
- Log: ./codex/logs/118.md

## References
- docs/planning/PORT_WINDOWS_LOCAL_LLM_ACTION_PLAN.md (Ticket 8 — Token Accounting & Telemetry)
- docs/guides/gpt5-prompting-best-practices-guide.md
