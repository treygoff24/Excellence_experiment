#!/usr/bin/env python3
"""
Archive a completed experiment run to the experiments directory.
"""

from __future__ import annotations
import os
import sys
import json
import shutil
import argparse
from datetime import datetime
from pathlib import Path


def get_run_info_from_manifest(manifest_path: str) -> dict:
    """Extract run information from run manifest."""
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        temps = manifest.get("temps", [0.0])
        temp_label = format_temp_label(temps[0]) if temps else "0"
        run_id = manifest.get("run_id", datetime.now().strftime("r%Y%m%d%H%M%S"))

        return {
            "temperature": temps[0] if temps else 0.0,
            "temp_label": temp_label,
            "run_id": run_id,
            "model_id": manifest.get("model_id", "unknown"),
            "samples_per_item": manifest.get("samples_per_item", {}),
            "manifest": manifest
        }
    except Exception as e:
        print(f"Warning: Could not read manifest {manifest_path}: {e}")
        return {}


def format_temp_label(t: float) -> str:
    """Format temperature for directory names."""
    s = f"{float(t):.1f}"
    return "0" if s == "0.0" else s.replace(".", "")


def create_experiment_archive(
    source_dirs: dict,
    run_info: dict,
    experiments_dir: str = "experiments",
    experiment_name: str = None
) -> str:
    """Archive experiment files to experiments directory."""

    if not experiment_name:
        experiment_name = f"t{run_info['temp_label']}_{run_info['run_id']}"

    archive_path = os.path.join(experiments_dir, experiment_name)
    os.makedirs(archive_path, exist_ok=True)

    # Create subdirectories
    subdirs = ["batch_inputs", "results", "reports"]
    for subdir in subdirs:
        os.makedirs(os.path.join(archive_path, subdir), exist_ok=True)

    archived_files = {"batch_inputs": [], "results": [], "reports": []}

    # Move files from each source directory
    for dest_subdir, source_dir in source_dirs.items():
        if not os.path.exists(source_dir):
            continue

        dest_dir = os.path.join(archive_path, dest_subdir)

        for item in os.listdir(source_dir):
            source_path = os.path.join(source_dir, item)
            dest_path = os.path.join(dest_dir, item)

            if os.path.isfile(source_path):
                shutil.move(source_path, dest_path)
                archived_files[dest_subdir].append(item)
                print(f"Moved: {source_path} -> {dest_path}")
            elif os.path.isdir(source_path):
                shutil.move(source_path, dest_path)
                archived_files[dest_subdir].append(f"{item}/")
                print(f"Moved directory: {source_path} -> {dest_path}")

    # Create archive manifest
    archive_manifest = {
        "experiment_id": experiment_name,
        "description": f"Temperature={run_info['temperature']} experiment run",
        "archived_date": datetime.utcnow().isoformat() + "Z",
        "config": {
            "temperature": run_info["temperature"],
            "model_id": run_info["model_id"],
            "samples_per_item": run_info["samples_per_item"]
        },
        "run_id": run_info["run_id"],
        "files": archived_files,
        "status": "archived",
        "notes": "Experiment archived using archive_run.py script"
    }

    manifest_path = os.path.join(archive_path, "archive_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(archive_manifest, f, indent=2)

    print(f"Archive manifest created: {manifest_path}")
    return archive_path


def main():
    ap = argparse.ArgumentParser(description="Archive completed experiment run")
    ap.add_argument("--results_dir", default="results", help="Results directory to archive")
    ap.add_argument("--batch_inputs_dir", default="data/batch_inputs", help="Batch inputs directory to archive")
    ap.add_argument("--reports_dir", default="reports", help="Reports directory to archive")
    ap.add_argument("--experiments_dir", default="experiments", help="Experiments archive directory")
    ap.add_argument("--experiment_name", help="Custom experiment name (auto-generated if not provided)")
    ap.add_argument("--manifest", default="results/run_manifest.json", help="Run manifest file")
    ap.add_argument("--dry_run", action="store_true", help="Show what would be archived without moving files")
    ap.add_argument("--yes", action="store_true", help="Proceed without interactive confirmation")

    args = ap.parse_args()

    # Get run information
    run_info = get_run_info_from_manifest(args.manifest)
    if not run_info:
        print("Error: Could not determine run information. Ensure run_manifest.json exists.")
        sys.exit(1)

    print(f"Found run: Temperature={run_info['temperature']}, Run ID={run_info['run_id']}")

    # Define source directories
    source_dirs = {
        "batch_inputs": args.batch_inputs_dir,
        "results": args.results_dir,
        "reports": args.reports_dir
    }

    if args.dry_run:
        print("DRY RUN - Files that would be archived:")
        for dest_subdir, source_dir in source_dirs.items():
            if os.path.exists(source_dir):
                files = os.listdir(source_dir)
                if files:
                    print(f"  {dest_subdir}/: {', '.join(files)}")
        return

    # Confirm archiving
    experiment_name = args.experiment_name or f"t{run_info['temp_label']}_{run_info['run_id']}"
    print(f"Archive experiment as: {experiment_name}")

    if not args.yes:
        try:
            response = input("Continue? (y/N): ").strip().lower()
            if response != 'y':
                print("Archiving cancelled.")
                return
        except KeyboardInterrupt:
            print("\nArchiving cancelled.")
            return

    # Create archive
    try:
        archive_path = create_experiment_archive(
            source_dirs=source_dirs,
            run_info=run_info,
            experiments_dir=args.experiments_dir,
            experiment_name=experiment_name
        )

        print(f"✅ Experiment archived successfully: {archive_path}")

        # Clean up empty source directories
        for source_dir in source_dirs.values():
            if os.path.exists(source_dir) and not os.listdir(source_dir):
                try:
                    os.rmdir(source_dir)
                    print(f"Removed empty directory: {source_dir}")
                except OSError:
                    pass

    except Exception as e:
        print(f"❌ Error archiving experiment: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
