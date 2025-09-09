# Stop Handling and Phase Tracking Fixes

Objective
- Prevent runs from halting early due to a stale `STOP_REQUESTED` file and fix misleading run-level phase status updates that were marked “completed” after the first trial.

Symptoms Observed
- Orchestrator stops submitting new jobs after finishing one condition/trial even though more trials remain.
- A `STOP_REQUESTED` file exists under `experiments/run_<RUN_ID>/`, often created in a prior attempt, causing silent early stopping on resume.
- `run_state.json` shows phases like `poll/parse/score/stats/costs/report` as “completed” after the first trial finishes, even though other trials haven’t run yet.

Root Causes
1) Stop sentinel handling
   - The queue honors any `STOP_REQUESTED` file found under the run directory. If a stale file remains from a previous run, new submissions are skipped.
2) Run-level phase accounting
   - Phases were marked “completed” inside the per-trial loop, so the run-level state flipped to completed as soon as the first trial advanced, which was misleading and masked partial progress.

Scope of Changes
- File: `scripts/state_utils.py` (StopToken)
- File: `scripts/run_all.py` (CLI flags, StopToken wiring, phase accounting)

Implementation Details

1) Harden StopToken against stale STOP files (scripts/state_utils.py)
- Add optional behavior controls to `StopToken`:
  - `ignore_file: bool = False` — ignore the presence of `STOP_REQUESTED` on disk (still honors SIGINT/SIGTERM for the current process).
  - `stale_minutes: Optional[int] = None` — if set, treat a `STOP_REQUESTED` file older than this many minutes as stale and ignore it. Also rename it to `STOP_REQUESTED.stale.<UTCSTAMP>` to avoid future false positives.

Code sketch (replace the StopToken in scripts/state_utils.py):
```
class StopToken:
    """Cooperative stop signal integrating OS signals and a STOP file.

    ignore_file: ignore STOP file on disk.
    stale_minutes: if set, ignore and rename STOP files older than N minutes.
    """
    def __init__(self, run_root: str, *, ignore_file: bool = False, stale_minutes: int | None = None):
        self._flag = False
        self.run_root = run_root
        self.ignore_file = bool(ignore_file)
        self.stale_minutes = stale_minutes

        def _handler(sig, frame):
            self.set()
            try:
                os.makedirs(self.run_root, exist_ok=True)
                with open(os.path.join(self.run_root, STOP_FILENAME), "w", encoding="utf-8") as f:
                    f.write(_utc_now_iso())
            except Exception:
                pass

        try:
            signal.signal(signal.SIGINT, _handler)
            signal.signal(signal.SIGTERM, _handler)
        except Exception:
            pass

    def set(self) -> None:
        self._flag = True

    def _stop_file_active(self) -> bool:
        if self.ignore_file:
            return False
        path = os.path.join(self.run_root, STOP_FILENAME)
        if not os.path.isfile(path):
            return False
        if self.stale_minutes is not None and self.stale_minutes >= 0:
            try:
                import time
                from datetime import datetime
                mtime = os.path.getmtime(path)
                age_sec = max(0.0, time.time() - mtime)
                if age_sec > (self.stale_minutes * 60):
                    try:
                        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
                        os.replace(path, os.path.join(self.run_root, f"{STOP_FILENAME}.stale.{ts}"))
                        print(f"Ignoring stale STOP (> {self.stale_minutes}m); renamed sentinel.")
                    except Exception:
                        pass
                    return False
            except Exception:
                pass
        return True

    def is_set(self) -> bool:
        if self._stop_file_active():
            self._flag = True
        return self._flag

    def check(self) -> None:
        if self.is_set():
            raise StopRequested("Stop requested via signal or STOP file")
```

2) Expose STOP handling controls via CLI (scripts/run_all.py)
- Add arguments:
  - `--ignore_stop` — ignore `STOP_REQUESTED` files (still honors Ctrl‑C for this process).
  - `--stop_stale_minutes <int>` — treat STOP files older than N minutes as stale and ignore (default 60 recommended).

Argparse additions:
```
ap.add_argument("--ignore_stop", action="store_true", help="Ignore STOP_REQUESTED file; still honors Ctrl-C")
ap.add_argument("--stop_stale_minutes", type=int, default=60, help="Treat STOP files older than N minutes as stale (default: 60)")
```

Wire into StopToken construction:
```
stop_token = StopToken(
    run_root or os.getcwd(),
    ignore_file=bool(args.ignore_stop),
    stale_minutes=int(args.stop_stale_minutes) if args.stop_stale_minutes is not None else None,
)
```

3) Fix run-level phase accounting (scripts/run_all.py)
- Problem: phases (`poll`, `parse`, `score`, `stats`, `costs`, `report`) were marked in_progress/completed inside the per-trial loop, causing the run-level state to flip to “completed” after the first trial.
- Fix: track “started once” booleans for each phase; mark in_progress only once (before the first trial enters that phase) and mark completed only after all trials have been processed.

Implementation pattern (inside the section that iterates trials for poll/parse/score/stats/costs/report):
```
all_trial_summaries = []
poll_started = parse_started = score_started = stats_started = costs_started = report_started = False

for trial in trials:
    ...
    # POLL
    if "poll" in selected_phases and not _poll_done():
        if state is not None and not poll_started:
            with RunStateLock(run_root):
                update_phase(state, "poll", status="in_progress")
                write_json_atomic(run_state_path(run_root), state)
            poll_started = True
        stop_token.check()
        ...  # run polling/download work
        # do NOT mark poll completed here

    # PARSE
    if "parse" in selected_phases and not _parse_done():
        if state is not None and not parse_started:
            ... set in_progress once ...
            parse_started = True
        stop_token.check()
        ...
        # do NOT mark parse completed here

    # SCORE/STATS/COSTS/REPORT follow the same pattern

# After the for-trials loop, mark completed once per phase that started
if state is not None:
    if poll_started:  update_phase(..., "poll",   status="completed")
    if parse_started: update_phase(..., "parse",  status="completed")
    if score_started: update_phase(..., "score",  status="completed")
    if stats_started: update_phase(..., "stats",  status="completed")
    if costs_started: update_phase(..., "costs",  status="completed")
    if report_started:update_phase(..., "report", status="completed")
```

Notes
- Queue behavior already respects `stop_event.is_set()` and pauses submissions; no changes needed there.
- We added a console notice when ignoring/renaming a stale STOP file to make behavior explicit in logs.

Operational Guidance
- If resuming a run that previously halted, delete any stale sentinel first:
  `rm experiments/run_<RUN_ID>/STOP_REQUESTED`
- Or run with the new flags to avoid manual deletion:
  - Ignore STOP file: `--ignore_stop`
  - Auto-ignore STOP older than 60 minutes (default): `--stop_stale_minutes 60`

Quick Test Plan
1) Create a test run; touch a STOP file before submission; verify that with `--ignore_stop` the orchestrator continues submissions.
2) Touch a STOP file with an old mtime (e.g., change timestamp to >60 minutes ago) and start without `--ignore_stop`; confirm it logs that the STOP is stale, renames it, and proceeds.
3) Run a sweep with at least 2 trials; observe `run_state.json` and ensure `poll/parse/score/stats/costs/report` are only marked completed after all trials finish, not after the first one.

Backward Compatibility
- Default retains safety: fresh STOP files still stop between phases; only stale ones (older than N minutes) are ignored.
- Ctrl‑C continues to work via OS signal handler regardless of `ignore_file`.

Implementation Checklist
- [ ] Update `scripts/state_utils.py` StopToken as above.
- [ ] Add CLI flags and StopToken wiring in `scripts/run_all.py`.
- [ ] Refactor run-level phase status updates as described.
- [ ] Smoke test resume behavior and phase reporting.

