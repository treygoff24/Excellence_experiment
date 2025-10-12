from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, Iterable, Iterator, Tuple, Any

from scripts.state_utils import (
    CONTROL_REGISTRY_FILENAME,
    control_registry_path,
    load_control_registry as _load_registry_state,
    write_control_registry as _write_registry_state,
    _ensure_controls_dict,
    _cleanup_registry,
)


SHARED_CONTROL_DIRNAME = "shared_controls"
CONTROL_RESULTS_FILENAME = "control_results.jsonl"
CONTROL_METADATA_FILENAME = "control_metadata.json"


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def find_run_root(start_dir: str) -> str | None:
    """Best-effort discovery of a run root containing control registry state."""

    if not start_dir:
        return None
    current = os.path.abspath(start_dir)
    while True:
        registry_path = control_registry_path(current)
        if os.path.isfile(registry_path):
            return current
        parent = os.path.dirname(current)
        if not parent or parent == current:
            return None
        current = parent


def shared_rel_default(key: str) -> str:
    return os.path.join(SHARED_CONTROL_DIRNAME, key)


def shared_control_dir(run_root: str, key: str) -> str:
    return os.path.join(run_root, SHARED_CONTROL_DIRNAME, key)


def resolve_shared_control_path(run_root: str | None, entry: dict, *, file_key: str = "results_jsonl") -> str | None:
    if not run_root or not entry:
        return None
    shared_rel = entry.get("shared_rel")
    if not shared_rel:
        return None
    files = entry.get("files") or {}
    fname = files.get(file_key)
    if not fname:
        if file_key == "results_jsonl":
            fname = CONTROL_RESULTS_FILENAME
        else:
            return None
    return os.path.join(run_root, shared_rel, fname)


def control_entry_files_exist(run_root: str | None, entry: dict) -> bool:
    if not run_root or not entry or entry.get("status") != "completed":
        return False
    shared_path = resolve_shared_control_path(run_root, entry)
    if not shared_path or not os.path.isfile(shared_path):
        return False
    for file_key in ("metadata",):
        aux = resolve_shared_control_path(run_root, entry, file_key=file_key)
        if aux and not os.path.isfile(aux):
            return False
    return True


def gather_control_entries(results_dir: str) -> Tuple[str | None, Dict[str, dict], dict]:
    manifest_path = os.path.join(results_dir, "trial_manifest.json")
    run_root = find_run_root(results_dir)
    if not os.path.isfile(manifest_path):
        return run_root, {}, {}
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception:
        return run_root, {}, {}
    ctrl = manifest.get("control_registry") or {}
    if not isinstance(ctrl, dict):
        ctrl = {}
    return run_root, ctrl, manifest


def iter_control_results(run_root: str | None, entries: Iterable[dict]) -> Iterator[dict]:
    if not run_root:
        return iter(())
    for entry in entries:
        shared_path = resolve_shared_control_path(run_root, entry)
        if not shared_path or not os.path.isfile(shared_path):
            continue
        try:
            with open(shared_path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    try:
                        yield json.loads(s)
                    except Exception:
                        continue
        except Exception:
            continue


def count_control_results(run_root: str | None, entries: Iterable[dict]) -> int:
    count = 0
    for entry in entries:
        try:
            cnt = entry.get("counts", {}).get("results_jsonl")
        except Exception:
            cnt = None
        if isinstance(cnt, int):
            count += cnt
        else:
            shared_path = resolve_shared_control_path(run_root, entry)
            if shared_path and os.path.isfile(shared_path):
                try:
                    with open(shared_path, "r", encoding="utf-8") as f:
                        count += sum(1 for line in f if line.strip())
                except Exception:
                    pass
    return count


def sanitize_control_registry(
    run_root: str | None,
    registry: dict,
    *,
    valid_keys: Iterable[str] | None = None,
) -> bool:
    if not registry or not isinstance(registry, dict):
        return False
    mutated = False
    controls = registry.get("controls")
    if not isinstance(controls, dict):
        registry["controls"] = {}
        controls = registry["controls"]
        mutated = True
    if valid_keys is not None:
        allow = set(valid_keys)
        removed = [key for key in controls.keys() if key not in allow]
        for key in removed:
            controls.pop(key, None)
        if removed:
            mutated = True
    for key, entry in list(controls.items()):
        if not isinstance(entry, dict):
            controls.pop(key, None)
            mutated = True
            continue
        entry.setdefault("shared_rel", shared_rel_default(key))
        if entry.get("status") == "completed" and not control_entry_files_exist(run_root, entry):
            entry.pop("files", None)
            entry.pop("counts", None)
            entry.pop("completed_at", None)
            entry["status"] = "pending"
            entry["repaired_at"] = _now_iso()
            mutated = True
        elif entry.get("status") not in {"pending", "completed", "in_progress"}:
            entry["status"] = "pending"
            mutated = True
    return mutated


def registry_keys(registry: dict) -> Iterable[str]:
    controls = registry.get("controls") if isinstance(registry, dict) else {}
    if isinstance(controls, dict):
        return controls.keys()
    return []


# Alias writable helper while keeping shared sanitation logic centralized here
def refresh_registry(run_root: str, *, valid_keys: Iterable[str] | None = None) -> dict:
    registry = _load_registry_state(run_root)
    mutate = sanitize_control_registry(run_root, registry, valid_keys=valid_keys)
    if mutate:
        _write_registry_state(run_root, registry)
    return registry


def write_control_registry(run_root: str, registry: Dict[str, Any], *, valid_keys: Iterable[str] | None = None) -> None:
    data = dict(registry or {})
    data = _ensure_controls_dict(data)
    sanitize_control_registry(run_root, data, valid_keys=valid_keys)
    data = _cleanup_registry(data, valid_keys=valid_keys)
    data["schema_version"] = int(data.get("schema_version") or 1)
    _write_registry_state(run_root, data)


