#!/usr/bin/env python3
"""
Safely clean the workspace directories to prepare for a new experiment run.
"""

from __future__ import annotations
import os
import shutil
import argparse
from pathlib import Path


def check_directory_contents(directory: str) -> tuple[bool, list[str]]:
    """Check if directory exists and return its contents."""
    if not os.path.exists(directory):
        return False, []

    contents = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path):
            size = os.path.getsize(item_path)
            contents.append(f"  📄 {item} ({size:,} bytes)")
        elif os.path.isdir(item_path):
            file_count = len([f for f in os.listdir(item_path) if os.path.isfile(os.path.join(item_path, f))])
            contents.append(f"  📁 {item}/ ({file_count} files)")

    return True, contents


def clean_directory(directory: str, preserve_files: list[str] = None) -> bool:
    """Clean a directory, preserving specified files."""
    if not os.path.exists(directory):
        return True

    preserve_files = preserve_files or []
    removed_items = []

    for item in os.listdir(directory):
        if item in preserve_files:
            continue

        item_path = os.path.join(directory, item)
        try:
            if os.path.isfile(item_path):
                os.remove(item_path)
                removed_items.append(f"  🗑️  Removed file: {item}")
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
                removed_items.append(f"  🗑️  Removed directory: {item}/")
        except Exception as e:
            print(f"  ⚠️  Failed to remove {item}: {e}")
            return False

    for item in removed_items:
        print(item)

    return True


def clean_workspace(
    results_dir: str = "results",
    batch_inputs_dir: str = "data/batch_inputs",
    reports_dir: str = "reports",
    preserve_prepared: bool = True,
    dry_run: bool = False,
    assume_yes: bool = False
):
    """Clean workspace directories."""

    directories_to_clean = [
        (results_dir, []),
        (batch_inputs_dir, []),
        (reports_dir, [])
    ]

    # Files to preserve in prepared data directory
    prepared_preserve = ["closed_book.jsonl", "open_book.jsonl"] if preserve_prepared else []
    if preserve_prepared:
        directories_to_clean.append(("data/prepared", prepared_preserve))

    print("WORKSPACE CLEANUP")
    print("=" * 50)

    # Show what will be cleaned
    total_items = 0
    for directory, preserve in directories_to_clean:
        exists, contents = check_directory_contents(directory)
        if exists:
            print(f"\n📁 {directory}/:")
            if contents:
                for item in contents:
                    if preserve and any(p in item for p in preserve):
                        print(f"{item} (PRESERVED)")
                    else:
                        print(item)
                        total_items += 1
            else:
                print("  (empty)")
        else:
            print(f"\n📁 {directory}/: (does not exist)")

    if total_items == 0:
        print("\n✅ Workspace is already clean!")
        return

    print(f"\nTotal items to clean: {total_items}")

    if dry_run:
        print("\n🔍 DRY RUN - No files were actually removed")
        return

    # Confirm cleanup
    print(f"\n⚠️  This will permanently delete {total_items} items from your workspace.")
    print("Make sure you have archived any important results!")

    if not assume_yes:
        try:
            response = input("\nProceed with cleanup? (y/N): ").strip().lower()
            if response != 'y':
                print("Cleanup cancelled.")
                return
        except KeyboardInterrupt:
            print("\nCleanup cancelled.")
            return

    # Perform cleanup
    print("\nCleaning workspace...")
    success = True

    for directory, preserve in directories_to_clean:
        if os.path.exists(directory):
            print(f"\n🧹 Cleaning {directory}/:")
            if not clean_directory(directory, preserve):
                success = False

    if success:
        print("\n✅ Workspace cleaned successfully!")
        print("\nYou can now run a new experiment with clean directories.")
    else:
        print("\n❌ Some items could not be cleaned. Check permissions and try again.")


def main():
    ap = argparse.ArgumentParser(description="Clean workspace for new experiment")
    ap.add_argument("--results_dir", default="results", help="Results directory to clean")
    ap.add_argument("--batch_inputs_dir", default="data/batch_inputs", help="Batch inputs directory to clean")
    ap.add_argument("--reports_dir", default="reports", help="Reports directory to clean")
    ap.add_argument("--clean_prepared", action="store_true", help="Also clean prepared data (not recommended)")
    ap.add_argument("--dry_run", action="store_true", help="Show what would be cleaned without removing files")
    ap.add_argument("--yes", action="store_true", help="Proceed without interactive confirmation")

    args = ap.parse_args()

    clean_workspace(
        results_dir=args.results_dir,
        batch_inputs_dir=args.batch_inputs_dir,
        reports_dir=args.reports_dir,
        preserve_prepared=not args.clean_prepared,
        dry_run=args.dry_run,
        assume_yes=args.yes
    )


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    main()
