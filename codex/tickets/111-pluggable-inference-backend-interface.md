id: 111
slug: pluggable-inference-backend-interface
title: Ticket 111 — Introduce pluggable inference backend interface
ticket_file: ./codex/tickets/111-pluggable-inference-backend-interface.md
log_file: ./codex/logs/111.md

## Objective
- Decouple inference from Fireworks by introducing a minimal backend interface and moving Fireworks code under `backends/fireworks/` without changing external behavior.

## Scope
- New `backends/interfaces.py` defining `InferenceClient` and `BatchExecutor` protocols.
- Relocate Fireworks-specific modules to `backends/fireworks/` with adapters implementing `BatchExecutor` surface (no functional change).
- Adapt orchestrator imports/dispatch to select backend via config while preserving current default to Fireworks.

## Out of Scope
- Local engine implementations (Ollama, llama.cpp) — handled by Tickets 112–113.
- Batch executor for local engines — Ticket 114.

## Acceptance
- Running existing Fireworks path behaves identically when `backend=fireworks` (no regressions in prepare→report flow on a smoke slice).
- `backends/interfaces.py` is PEP 8 compliant, fully typed for public APIs.
- Orchestrator can select backend based on config flag without altering downstream parsing/scoring.
- Determinism & safety: Imports remain absolute; default path stays Fireworks unless `backend=local` is explicitly configured.

## Deliverables
- Files:
  - New: `backends/interfaces.py`
  - New/Move: `backends/fireworks/*` (adapters)
  - Update: `scripts/run_all.py` (backend dispatch only)
- Log: ./codex/logs/111.md

## References
- docs/planning/PORT_WINDOWS_LOCAL_LLM_ACTION_PLAN.md (Ticket 1 — Pluggable Inference Backend)
- docs/guides/gpt5-prompting-best-practices-guide.md
