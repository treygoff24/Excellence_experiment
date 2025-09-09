id: 117
slug: windows-pathing-and-multiprocessing
title: Ticket 117 — Windows pathing & multiprocessing fixes
ticket_file: ./codex/tickets/117-windows-pathing-and-multiprocessing.md
log_file: ./codex/logs/117.md

## Objective
- Ensure all scripts run on Windows: use `pathlib`, correct spawn semantics, and Windows-friendly file/CSV operations.

## Scope
- Replace `os.path` with `pathlib.Path` in touched modules.
- Guard script entry points with `if __name__ == "__main__": mp.freeze_support(); main()`.
- Prefer thread/async pools for local HTTP; keep process pools for CPU-bound scoring only.
- Universal newline handling for CSV; temp file handling that avoids exclusive locks.

## Out of Scope
- Engine/client logic (Tickets 112–114).

## Acceptance
- Orchestrator completes a 50‑item trial on Windows with `backend=local`.
- No deadlocks from nested pools; documented guidance in `docs/windows.md` (to be finalized in Ticket 121).
- Determinism & safety: Behavior remains unchanged on Linux/macOS.

## Deliverables
- Files: selective updates under `scripts/` and any modules using multiprocessing/subprocess/paths
- Log: ./codex/logs/117.md

## References
- docs/planning/PORT_WINDOWS_LOCAL_LLM_ACTION_PLAN.md (Ticket 7 — Windows Pathing & Multiprocessing)
- docs/guides/gpt5-prompting-best-practices-guide.md
