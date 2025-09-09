id: 110
slug: windows-bootstrap-and-dev-ergonomics
title: Ticket 110 — Windows bootstrap and developer ergonomics
ticket_file: ./codex/tickets/110-windows-bootstrap-and-dev-ergonomics.md
log_file: ./codex/logs/110.md

## Objective
- Provide a one‑command Windows setup and reproducible environment with PowerShell scripts that mirror Make targets, without changing Python package sets or pipeline semantics.

## Scope
- New `tools/bootstrap.ps1` to create/refresh venv and install deps via `py -3.11`.
- New `tools/tasks.ps1` exposing functions mapping to phases: Data, Build, Eval, Parse, Score, Stats, Report, Plan, Smoke.
- New `docs/windows.md` with copy/paste setup and usage.
- Ensure naming, paths, and outputs align with existing phases and artifacts.

## Out of Scope
- Any change to scoring/statistics logic or report generation.
- Engine/client implementations (handled by later tickets).

## Acceptance
- `powershell -ExecutionPolicy Bypass -File tools\bootstrap.ps1` creates `.venv`, upgrades pip, installs `requirements.txt` successfully.
- `tools\tasks.ps1 -List` prints available tasks; each task prints its underlying python command and exits 0 with `-WhatIf`.
- Documentation in `docs/windows.md` is sufficient for a fresh Windows 11 user to reach a working venv and list tasks.
- Determinism & safety: Scripts are idempotent; no global PATH edits; clear actionable errors if Python is missing (`py -3.11`).

## Deliverables
- Files:
  - New: `tools/bootstrap.ps1`
  - New: `tools/tasks.ps1`
  - New: `docs/windows.md` (initial skeleton; expanded in Ticket 121)
- Log: ./codex/logs/110.md

## References
- docs/planning/PORT_WINDOWS_LOCAL_LLM_ACTION_PLAN.md (Ticket 0 — Windows Bootstrap & Developer Ergonomics)
- docs/guides/gpt5-prompting-best-practices-guide.md
