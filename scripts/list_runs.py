#!/usr/bin/env python3
"""
List experiment runs and show run_state status and archive metadata.
"""

from __future__ import annotations
import os
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from scripts.state_utils import load_run_state


def load_archive_manifest(experiment_path: str) -> dict:
    """Load archive manifest for an experiment."""
    manifest_path = os.path.join(experiment_path, "archive_manifest.json")
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def format_datetime(iso_string: str) -> str:
    """Format ISO datetime string for display."""
    try:
        dt = datetime.fromisoformat(iso_string.replace('Z', '+00:00'))
        return dt.strftime('%Y-%m-%d %H:%M')
    except Exception:
        return iso_string


def _overall_status(phases: Dict[str, Dict[str, Any]]) -> str:
    statuses = [str(v.get("status", "not_started")) for v in (phases or {}).values()]
    if not statuses:
        return "unknown"
    if any(s == "failed" for s in statuses):
        return "failed"
    if all(s == "completed" for s in statuses):
        return "completed"
    if any(s == "in_progress" for s in statuses):
        return "in_progress"
    if any(s == "stopped" for s in statuses):
        return "stopped"
    return "not_started"


def list_experiments(experiments_dir: str = "experiments", verbose: bool = False):
    """List all experiments under experiments_dir, reading run_state.json when present."""
    if not os.path.exists(experiments_dir):
        print(f"No experiments directory found at: {experiments_dir}")
        return

    experiments = []
    for item in os.listdir(experiments_dir):
        exp_path = os.path.join(experiments_dir, item)
        if not os.path.isdir(exp_path):
            continue
        # Consider any directory (run_*, archived experiment, etc.)
        manifest = load_archive_manifest(exp_path)
        state = load_run_state(exp_path)
        experiments.append((item, exp_path, manifest, state))

    if not experiments:
        print(f"No experiments found in {experiments_dir}")
        return

    # Sort by experiment name
    experiments.sort(key=lambda x: x[0])

    if verbose:
        print("EXPERIMENT RUNS:")
        print("=" * 80)
        for exp_name, exp_path, manifest, state in experiments:
            print(f"\n📁 {exp_name}")
            print(f"   Path: {exp_path}")
            if state:
                print("   Run State:")
                print(f"     Overall: {_overall_status(state.get('phases', {}))}")
                mig = bool(state.get("migrated"))
                print(f"     Migrated: {mig}")
                print(f"     Config Hash: {state.get('config_hash', 'unknown')}")
                # Brief phase summary
                phases = state.get("phases", {})
                if phases:
                    pkeys = ["prepare","build","submit","poll","parse","score","stats","costs","report"]
                    for p in pkeys:
                        if p in phases:
                            print(f"       - {p}: {phases[p].get('status', 'unknown')}")
            if manifest:
                config = manifest.get("config", {})
                print(f"   Temperature: {config.get('temperature', 'unknown')}")
                print(f"   Model: {config.get('model_id', 'unknown')}")
                print(f"   Run ID: {manifest.get('run_id', 'unknown')}")
                print(f"   Status: {manifest.get('status', 'unknown')}")
                print(f"   Archived: {format_datetime(manifest.get('archived_date', ''))}")

                if manifest.get('notes'):
                    print(f"   Notes: {manifest['notes']}")

                # File counts
                files = manifest.get('files', {})
                total_files = sum(len(file_list) for file_list in files.values())
                print(f"   Files: {total_files} total")
                for category, file_list in files.items():
                    if file_list:
                        print(f"     - {category}: {len(file_list)} files")
            else:
                print("   No manifest found")
    else:
        # Compact table format
        print("EXPERIMENT RUNS:")
        print("-" * 90)
        print(f"{'Name':<28} {'Overall':<12} {'Migrated':<9} {'CfgHash…':<10} {'Arch.Status':<12} {'Archived':<16}")
        print("-" * 90)

        for exp_name, exp_path, manifest, state in experiments:
            overall = _overall_status((state or {}).get("phases", {})) if state else "unknown"
            migrated = str(bool((state or {}).get("migrated", False))) if state else "-"
            ch = (state or {}).get("config_hash") or ""
            ch_short = (ch[:8] + "…") if ch else "-"
            arch_status = (manifest or {}).get("status", "-")
            archived = format_datetime((manifest or {}).get("archived_date", "")) if manifest else "-"
            print(f"{exp_name:<28} {overall:<12} {migrated:<9} {ch_short:<10} {arch_status:<12} {archived:<16}")


def main():
    ap = argparse.ArgumentParser(description="List archived experiment runs")
    ap.add_argument("--experiments_dir", default="experiments", help="Experiments directory")
    ap.add_argument("-v", "--verbose", action="store_true", help="Show detailed information")

    args = ap.parse_args()
    list_experiments(args.experiments_dir, args.verbose)


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    main()
