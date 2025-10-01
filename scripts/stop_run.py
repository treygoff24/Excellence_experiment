from __future__ import annotations
import os
import argparse
from datetime import datetime


def utc() -> str:
    return datetime.utcnow().isoformat() + "Z"


def main() -> None:
    ap = argparse.ArgumentParser(description="Request a cooperative stop for a run by creating STOP_REQUESTED")
    ap.add_argument("--run_id", required=True, help="Run identifier (e.g., r20240102030405)")
    ap.add_argument("--experiments_dir", default="experiments", help="Root experiments dir (default: experiments)")
    args = ap.parse_args()

    run_root = os.path.join(args.experiments_dir, f"run_{args.run_id}")
    if not os.path.isdir(run_root):
        raise SystemExit(f"Run root not found: {run_root}")

    path = os.path.join(run_root, "STOP_REQUESTED")
    with open(path, "w", encoding="utf-8") as f:
        f.write(utc())
    print(f"Stop requested: {path}")


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    main()
