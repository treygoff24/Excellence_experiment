id: 119
slug: windows-run-recipes-and-smoke-tests
title: Ticket 119 — Windows run recipes & smoke tests
ticket_file: ./codex/tickets/119-windows-run-recipes-and-smoke-tests.md
log_file: ./codex/logs/119.md

## Objective
- Provide one‑liners to validate orchestration locally before full runs; mirror README flow.

## Scope
- Extend `tools/tasks.ps1` with functions: `Invoke-Plan`, `Invoke-Smoke`, `Invoke-Data`, `Invoke-Build`, `Invoke-Eval`, `Invoke-Parse`, `Invoke-Score`, `Invoke-Stats`, `Invoke-Report`.
- Add three example flows (plan, tiny dry‑run, small end‑to‑end) using `config/eval_config.local.yaml` (Ticket 120).

## Out of Scope
- Engine implementation details (covered by earlier tickets).

## Acceptance
- The three documented commands complete; the end‑to‑end flow produces `predictions.csv`, `per_item_scores.csv`, `significance.json`, and a report under configured results dir.
- Determinism & safety: Keep `--limit_items` small; explicit docs for `--archive`, `--max_concurrent_jobs`.

## Deliverables
- Files:
  - Update: `tools/tasks.ps1`
  - Update: `docs/windows.md` (commands)
- Log: ./codex/logs/119.md

## References
- docs/planning/PORT_WINDOWS_LOCAL_LLM_ACTION_PLAN.md (Ticket 9 — Windows Run Recipes & Smoke Tests)
- docs/guides/gpt5-prompting-best-practices-guide.md
