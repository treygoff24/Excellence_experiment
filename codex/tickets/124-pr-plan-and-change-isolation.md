id: 124
slug: pr-plan-and-change-isolation
title: Ticket 124 — PR plan & change isolation
ticket_file: ./codex/tickets/124-pr-plan-and-change-isolation.md
log_file: ./codex/logs/124.md

## Objective
- Stage changes into shippable PRs that minimize risk and keep Fireworks path working by default.

## Scope
- Define and follow PR sequence:
  - PR 1: Backend interface + Fireworks move + config flags + Windows bootstrap/tasks.
  - PR 2: Local engine clients (Ollama + llama.cpp) + Prompt Adapter.
  - PR 3: Local Batch Executor + Parser + token accounting.
  - PR 4: Docs (windows.md, performance), example configs, telemetry (optional).
  - PR 5 (opt): CI smoke on Windows.
- Ensure each PR is independently testable (`make smoke` or PowerShell equivalents) and does not break Fireworks.

## Out of Scope
- Implementations themselves (covered by previous tickets).

## Acceptance
- Each PR is mergeable, green on existing checks, and includes evidence: command outputs, artifact paths, and updated docs where relevant.
- Determinism & safety: Explicit mention of config diffs; no change to default behavior unless users opt into `backend=local`.

## Deliverables
- Files: PR descriptions/templates as needed
- Log: ./codex/logs/124.md

## References
- docs/planning/PORT_WINDOWS_LOCAL_LLM_ACTION_PLAN.md (Ticket 14 — PR Plan)
- docs/guides/gpt5-prompting-best-practices-guide.md
