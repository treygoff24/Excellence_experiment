from __future__ import annotations
import os
import json
import fcntl
import hashlib
import tempfile
import signal
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional
import yaml


RUN_STATE_FILENAME = "run_state.json"
STATE_LOCK_FILENAME = ".state.lock"
STOP_FILENAME = "STOP_REQUESTED"


def _utc_now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def write_json_atomic(path: str, data: Dict[str, Any]) -> None:
    """Atomically write JSON to a file with fsync and os.replace.

    Uses a temporary file in the same directory and then replaces to guarantee
    readers never see a partially-written file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=os.path.dirname(path), delete=False,
            prefix=".tmp_state_", suffix=".json"
        ) as tf:
            tmp = tf.name
            # Exclusive lock on the temp file during write
            try:
                fcntl.flock(tf.fileno(), fcntl.LOCK_EX)
            except Exception:
                # Best effort; continue on non-POSIX
                pass
            json.dump(data, tf, indent=2)
            tf.flush()
            os.fsync(tf.fileno())
        # Atomic replace
        os.replace(tmp, path)
        tmp = None
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except Exception:
                pass


@dataclass
class RunStateLock:
    """Advisory lock for a run directory.

    On POSIX, uses fcntl.flock on a lockfile. On non-POSIX, falls back to best-effort
    exclusive open semantics.
    """
    run_root: str
    _fh: Optional[Any] = None

    def acquire(self) -> None:
        os.makedirs(self.run_root, exist_ok=True)
        lock_path = os.path.join(self.run_root, STATE_LOCK_FILENAME)
        # Open or create the lock file
        self._fh = open(lock_path, "a+")
        try:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX)
        except Exception:
            # Best effort on non-POSIX – nothing else to do
            pass

    def release(self) -> None:
        if not self._fh:
            return
        try:
            try:
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            self._fh.close()
        finally:
            self._fh = None

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()


def compute_config_hash(cfg: Dict[str, Any]) -> str:
    """Return a sha256 of the canonical YAML for the effective config."""
    # Keep it stable by sorting keys
    canonical = yaml.safe_dump(cfg, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def run_state_path(run_root: str) -> str:
    return os.path.join(run_root, RUN_STATE_FILENAME)


def load_run_state(run_root: str) -> Optional[Dict[str, Any]]:
    path = run_state_path(run_root)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def init_run_state(run_root: str, run_id: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Create a fresh run_state.json structure (not written)."""
    phases = [
        "prepare", "build", "submit", "poll", "parse", "score", "stats", "costs", "report",
    ]
    st = {
        "schema_version": 1,
        "run_id": run_id,
        "config_hash": compute_config_hash(cfg),
        "phases": {p: {"status": "not_started", "started_at": None, "updated_at": None, "last_error": None} for p in phases},
        "last_checkpoint": None,
        "stop_requested": False,
    }
    return st


def update_phase(state: Dict[str, Any], phase: str, *, status: str, error: Optional[str] = None) -> None:
    ph = state.setdefault("phases", {}).setdefault(phase, {"status": "not_started"})
    now = _utc_now_iso()
    if status == "in_progress" and not ph.get("started_at"):
        ph["started_at"] = now
    ph["status"] = status
    ph["updated_at"] = now
    if error is not None:
        ph["last_error"] = error


class StopRequested(Exception):
    pass


class StopToken:
    """Cooperative stop signal that integrates with OS signals and a STOP file."""

    def __init__(self, run_root: str):
        self._flag = False
        self.run_root = run_root

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

    def is_set(self) -> bool:
        # Also consider STOP file presence
        if os.path.isfile(os.path.join(self.run_root, STOP_FILENAME)):
            self._flag = True
        return self._flag

    def check(self) -> None:
        if self.is_set():
            raise StopRequested("Stop requested via signal or STOP file")

