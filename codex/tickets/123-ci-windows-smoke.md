id: 123
slug: ci-windows-smoke
title: Ticket 123 — CI smoke (Windows, optional)
ticket_file: ./codex/tickets/123-ci-windows-smoke.md
log_file: ./codex/logs/123.md

## Objective
- Add a small GitHub Actions workflow that lint‑checks and runs a dry‑run orchestration smoke on Windows to catch pathing regressions.

## Scope
- New `.github/workflows/windows-smoke.yml`: setup Python, cache pip, run `tools/bootstrap.ps1`, and `python -m scripts.run_all --config config/eval_config.local.yaml --dry_run --limit_items 10`.
- Skip model installs; test orchestration only.

## Out of Scope
- GPU tests or real inference.

## Acceptance
- Workflow passes on PRs; fails if Windows pathing or CLI regressions are introduced.
- Determinism & safety: Kept fast (<10 min); guarded behind conditional if repo does not have Windows runners.

## Deliverables
- Files:
  - New: `.github/workflows/windows-smoke.yml`
- Log: ./codex/logs/123.md

## References
- docs/planning/PORT_WINDOWS_LOCAL_LLM_ACTION_PLAN.md (Ticket 13 — CI Smoke)
- docs/guides/gpt5-prompting-best-practices-guide.md
