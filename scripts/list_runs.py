#!/usr/bin/env python3
"""
List all archived experiment runs in the experiments directory.
"""

from __future__ import annotations
import os
import json
import argparse
from datetime import datetime
from pathlib import Path


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


def list_experiments(experiments_dir: str = "experiments", verbose: bool = False):
    """List all experiments in the experiments directory."""
    if not os.path.exists(experiments_dir):
        print(f"No experiments directory found at: {experiments_dir}")
        return

    experiments = []
    for item in os.listdir(experiments_dir):
        exp_path = os.path.join(experiments_dir, item)
        if os.path.isdir(exp_path):
            manifest = load_archive_manifest(exp_path)
            experiments.append((item, exp_path, manifest))

    if not experiments:
        print(f"No experiments found in {experiments_dir}")
        return

    # Sort by experiment name
    experiments.sort(key=lambda x: x[0])

    if verbose:
        print("EXPERIMENT RUNS:")
        print("=" * 80)
        for exp_name, exp_path, manifest in experiments:
            print(f"\n📁 {exp_name}")
            print(f"   Path: {exp_path}")

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
        print("-" * 70)
        print(f"{'Name':<25} {'Temp':<6} {'Status':<12} {'Archived':<16}")
        print("-" * 70)

        for exp_name, exp_path, manifest in experiments:
            if manifest:
                config = manifest.get("config", {})
                temp = config.get("temperature", "?")
                status = manifest.get("status", "unknown")
                archived = format_datetime(manifest.get("archived_date", ""))
            else:
                temp = "?"
                status = "no-manifest"
                archived = "unknown"

            print(f"{exp_name:<25} {temp:<6} {status:<12} {archived:<16}")


def main():
    ap = argparse.ArgumentParser(description="List archived experiment runs")
    ap.add_argument("--experiments_dir", default="experiments", help="Experiments directory")
    ap.add_argument("-v", "--verbose", action="store_true", help="Show detailed information")

    args = ap.parse_args()
    list_experiments(args.experiments_dir, args.verbose)


if __name__ == "__main__":
    main()
