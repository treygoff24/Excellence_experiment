#!/usr/bin/env python3
"""
Compare results between two experiment runs.
"""

from __future__ import annotations
import os
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Optional


def load_experiment_results(experiment_path: str) -> Dict:
    """Load results from an experiment directory."""
    results = {
        "manifest": {},
        "per_item_scores": [],
        "significance": {},
        "costs": {}
    }

    # Load archive manifest
    manifest_path = os.path.join(experiment_path, "archive_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            results["manifest"] = json.load(f)

    # Load per-item scores
    scores_path = os.path.join(experiment_path, "results", "per_item_scores.csv")
    if os.path.exists(scores_path):
        with open(scores_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            results["per_item_scores"] = list(reader)

    # Load significance results
    sig_path = os.path.join(experiment_path, "results", "significance.json")
    if os.path.exists(sig_path):
        with open(sig_path, "r", encoding="utf-8") as f:
            results["significance"] = json.load(f)

    # Load costs
    costs_path = os.path.join(experiment_path, "results", "costs.json")
    if os.path.exists(costs_path):
        with open(costs_path, "r", encoding="utf-8") as f:
            results["costs"] = json.load(f)

    return results


def calculate_metrics(per_item_scores: List[Dict]) -> Dict:
    """Calculate aggregate metrics from per-item scores."""
    if not per_item_scores:
        return {}

    metrics = {}

    # Group by condition and type
    groups = {}
    for row in per_item_scores:
        condition = row.get("condition", "unknown")
        typ = row.get("type", "unknown")
        key = (condition, typ)

        if key not in groups:
            groups[key] = {"em": [], "f1": [], "abstain_rate": [], "false_answer_rate": []}

        for metric in ["em", "f1", "abstain_rate", "false_answer_rate"]:
            try:
                value = float(row.get(metric, 0))
                groups[key][metric].append(value)
            except (ValueError, TypeError):
                pass

    # Calculate means
    for (condition, typ), values in groups.items():
        key = f"{condition}_{typ}"
        metrics[key] = {}
        for metric, metric_values in values.items():
            if metric_values:
                metrics[key][metric] = sum(metric_values) / len(metric_values)
            else:
                metrics[key][metric] = 0.0

    return metrics


def compare_experiments(exp1_path: str, exp2_path: str):
    """Compare two experiments and display results."""
    exp1_name = os.path.basename(exp1_path)
    exp2_name = os.path.basename(exp2_path)

    print(f"COMPARING EXPERIMENTS:")
    print(f"Experiment 1: {exp1_name}")
    print(f"Experiment 2: {exp2_name}")
    print("=" * 70)

    # Load results
    exp1_results = load_experiment_results(exp1_path)
    exp2_results = load_experiment_results(exp2_path)

    # Compare configurations
    exp1_config = exp1_results["manifest"].get("config", {})
    exp2_config = exp2_results["manifest"].get("config", {})

    print("\nCONFIGURATION:")
    print(f"{'Metric':<20} {'Experiment 1':<20} {'Experiment 2':<20}")
    print("-" * 60)
    print(f"{'Temperature':<20} {exp1_config.get('temperature', '?'):<20} {exp2_config.get('temperature', '?'):<20}")
    print(f"{'Model':<20} {exp1_config.get('model_id', '?').split('/')[-1] if '/' in str(exp1_config.get('model_id', '?')) else exp1_config.get('model_id', '?'):<20} {exp2_config.get('model_id', '?').split('/')[-1] if '/' in str(exp2_config.get('model_id', '?')) else exp2_config.get('model_id', '?'):<20}")

    # Compare metrics
    exp1_metrics = calculate_metrics(exp1_results["per_item_scores"])
    exp2_metrics = calculate_metrics(exp2_results["per_item_scores"])

    all_keys = set(exp1_metrics.keys()) | set(exp2_metrics.keys())

    if all_keys:
        print("\nPERFORMANCE METRICS:")
        for key in sorted(all_keys):
            condition, typ = key.split("_", 1)
            print(f"\n{condition.title()} - {typ.title()}:")
            print(f"{'Metric':<15} {'Exp1':<10} {'Exp2':<10} {'Diff':<10}")
            print("-" * 45)

            exp1_data = exp1_metrics.get(key, {})
            exp2_data = exp2_metrics.get(key, {})

            for metric in ["em", "f1", "abstain_rate", "false_answer_rate"]:
                val1 = exp1_data.get(metric, 0.0)
                val2 = exp2_data.get(metric, 0.0)
                diff = val2 - val1

                print(f"{metric:<15} {val1:<10.3f} {val2:<10.3f} {diff:+.3f}")

    # Compare costs
    exp1_costs = exp1_results["costs"]
    exp2_costs = exp2_results["costs"]

    if exp1_costs or exp2_costs:
        print("\nCOSTS:")
        print(f"{'Metric':<20} {'Experiment 1':<15} {'Experiment 2':<15}")
        print("-" * 50)

        for metric in ["prompt_tokens", "completion_tokens", "total_tokens", "usd"]:
            val1 = exp1_costs.get(metric, 0)
            val2 = exp2_costs.get(metric, 0)

            if metric == "usd":
                print(f"{metric:<20} ${val1:<14.4f} ${val2:<14.4f}")
            else:
                print(f"{metric:<20} {val1:<15,} {val2:<15,}")

    # Display significance if available
    exp1_sig = exp1_results["significance"]
    exp2_sig = exp2_results["significance"]

    if exp1_sig or exp2_sig:
        print("\nSTATISTICAL SIGNIFICANCE:")
        for temp, sig_data in exp1_sig.items():
            print(f"\nTemperature {temp} (Experiment 1):")
            mcnemar = sig_data.get("mcnemar", {})
            wilcoxon = sig_data.get("wilcoxon", {})
            if mcnemar:
                print(f"  McNemar: p={mcnemar.get('p_value', 'N/A')}")
            if wilcoxon:
                print(f"  Wilcoxon: p={wilcoxon.get('p_value', 'N/A')}")


def main():
    ap = argparse.ArgumentParser(description="Compare two experiment runs")
    ap.add_argument("--run1", required=True, help="First experiment name or path")
    ap.add_argument("--run2", required=True, help="Second experiment name or path")
    ap.add_argument("--experiments_dir", default="experiments", help="Experiments directory")

    args = ap.parse_args()

    # Resolve experiment paths
    if os.path.isabs(args.run1):
        exp1_path = args.run1
    else:
        exp1_path = os.path.join(args.experiments_dir, args.run1)

    if os.path.isabs(args.run2):
        exp2_path = args.run2
    else:
        exp2_path = os.path.join(args.experiments_dir, args.run2)

    # Validate paths
    if not os.path.exists(exp1_path):
        print(f"Error: Experiment not found: {exp1_path}")
        return

    if not os.path.exists(exp2_path):
        print(f"Error: Experiment not found: {exp2_path}")
        return

    compare_experiments(exp1_path, exp2_path)


if __name__ == "__main__":
    main()
