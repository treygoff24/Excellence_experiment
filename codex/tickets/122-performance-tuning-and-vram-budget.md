id: 122
slug: performance-tuning-and-vram-budget
title: Ticket 122 — Performance tuning & VRAM budget guide
ticket_file: ./codex/tickets/122-performance-tuning-and-vram-budget.md
log_file: ./codex/logs/122.md

## Objective
- Provide hardware‑aware tuning presets and an OOM/latency playbook tailored to RTX 5080 16 GB.

## Scope
- New `docs/performance.md` with presets for Throughput, Quality, and Stretch; symptoms→fixes; telemetry interpretation.
- Quick decision tree to resolve OOM and slow tokens.

## Out of Scope
- Engine code changes (informational guidance only).

## Acceptance
- Users can recover from common performance issues by following the guide; commands reference shipped scripts/configs.
- Determinism & safety: Emphasize conservative defaults; call out variability across models.

## Deliverables
- Files:
  - New: `docs/performance.md`
- Log: ./codex/logs/122.md

## References
- docs/planning/PORT_WINDOWS_LOCAL_LLM_ACTION_PLAN.md (Ticket 12 — Performance Tuning)
- docs/guides/gpt5-prompting-best-practices-guide.md
