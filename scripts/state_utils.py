from __future__ import annotations
import os
import json
import fcntl
import hashlib
import tempfile
import signal
from datetime import datetime
from typing import Any, Dict, Optional

RUN_STATE_FILENAME = "run_state.json"
STATE_LOCK_FILENAME = ".state.lock"
STOP_FILENAME = "STOP_REQUESTED"


def _utc_now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


class StopRequested(Exception):
    pass


class StopToken:
    """Cooperative stop signal that integrates with OS signals and a STOP file.

    Behavior customizations:
    - ignore_file: if True, do not honor STOP file presence (still honors SIGINT/SIGTERM)
    - stale_minutes: if set, treat a STOP file older than this many minutes as stale and ignore it
    """

    def __init__(self, run_root: str, *, ignore_file: bool = False, stale_minutes: int | None = None):
        self._flag = False
        self.run_root = run_root
        self.ignore_file = bool(ignore_file)
        self.stale_minutes = stale_minutes

        def _handler(sig, frame):
            self.set()
            # Ensure STOP file exists for other processes
            try:
                os.makedirs(self.run_root, exist_ok=True)
                with open(os.path.join(self.run_root, STOP_FILENAME), "w", encoding="utf-8") as f:
                    f.write(_utc_now_iso())
            except Exception:
                pass

        # Register best-effort handlers
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
        # If staleness window configured, ignore old STOP files
        if self.stale_minutes is not None and self.stale_minutes >= 0:
            try:
                import time
                mtime = os.path.getmtime(path)
                age_sec = max(0.0, time.time() - mtime)
                if age_sec > (self.stale_minutes * 60):
                    # Best-effort: rename the stale STOP so it won't keep tripping future runs
                    try:
                        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
                        new_name = os.path.join(self.run_root, f"{STOP_FILENAME}.stale.{ts}")
                        os.replace(path, new_name)
                        try:
                            print(f"Ignoring stale STOP file (> {self.stale_minutes}m old); renamed to {os.path.basename(new_name)}")
                        except Exception:
                            pass
                    except Exception:
                        # If rename fails, at least ignore this occurrence
                        pass
                    return False
            except Exception:
                # If we can't stat the file, fall back to honoring its presence
                pass
        return True

    def is_set(self) -> bool:
        # Also consider STOP file presence (subject to ignore/stale policy)
        if self._stop_file_active():
            self._flag = True
        return self._flag

    def check(self) -> None:
        if self.is_set():
            raise StopRequested("Stop requested via signal or STOP file")

