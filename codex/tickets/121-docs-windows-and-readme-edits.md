id: 121
slug: docs-windows-and-readme-edits
title: Ticket 121 — Documentation: windows.md + README pointer
ticket_file: ./codex/tickets/121-docs-windows-and-readme-edits.md
log_file: ./codex/logs/121.md

## Objective
- Document Windows setup, local engines, run recipes, troubleshooting; add a concise README “Windows + Local” quick‑start pointer.

## Scope
- New `docs/windows.md` covering: prerequisites; bootstrap; Ollama vs llama.cpp; configs; commands; telemetry; OOM remediation; antivirus exclusions; known issues.
- Update README with a short section linking to `docs/windows.md` and highlighting `tools/bootstrap.ps1` and example configs.

## Out of Scope
- Performance presets (Ticket 122) and CI (Ticket 123).

## Acceptance
- A new Windows contributor can complete a small run end‑to‑end using the docs.
- Determinism & safety: All commands are copy‑pasteable and reflect the shipped scripts/configs; avoid side‑effects (no global env changes).

## Deliverables
- Files:
  - New/Update: `docs/windows.md`
  - Update: `README.md` (pointer section)
- Log: ./codex/logs/121.md

## References
- docs/planning/PORT_WINDOWS_LOCAL_LLM_ACTION_PLAN.md (Ticket 11 — Documentation)
- docs/guides/gpt5-prompting-best-practices-guide.md
