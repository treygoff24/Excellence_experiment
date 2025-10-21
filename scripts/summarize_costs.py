from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from config.schema import load_config
from scripts import manifest_v2 as mf

TOKEN_KEYS = {"prompt_tokens", "completion_tokens", "input_tokens", "output_tokens"}


@dataclass
class UsageTotals:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    records: int = 0
    source_type: str = "csv"
    source_path: Optional[str] = None
    mtime: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class PricingSelection:
    provider: str
    model_key: Optional[str]
    input_rate_per_mtok: float
    output_rate_per_mtok: float
    rate_type: str
    table_found: bool
    batch_applied: bool
    use_batch_api: bool


def _safe_int(value: Any) -> int:
    if value in (None, "", "None"):
        return 0
    try:
        return int(float(value))
    except Exception:
        return 0


def _safe_mtime(path: Optional[str]) -> float:
    if not path:
        return 0.0
    try:
        return os.path.getmtime(path)
    except Exception:
        return 0.0


def _extract_usage_dicts(obj: Any) -> List[Dict[str, Any]]:
    found: List[Dict[str, Any]] = []
    if isinstance(obj, dict):
        if TOKEN_KEYS.intersection(obj.keys()):
            found.append(obj)
        else:
            for value in obj.values():
                found.extend(_extract_usage_dicts(value))
    elif isinstance(obj, list):
        for item in obj:
            found.extend(_extract_usage_dicts(item))
    return found


def _load_usage_json(path: str) -> UsageTotals:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    usage_dicts = _extract_usage_dicts(data)
    prompt = completion = 0
    for usage in usage_dicts:
        prompt += _safe_int(usage.get("prompt_tokens", usage.get("input_tokens", 0)))
        completion += _safe_int(usage.get("completion_tokens", usage.get("output_tokens", 0)))
    return UsageTotals(
        prompt_tokens=prompt,
        completion_tokens=completion,
        records=len(usage_dicts),
        source_type="json",
        source_path=os.path.abspath(path),
        mtime=_safe_mtime(path),
    )


def _load_usage_csv(path: str) -> UsageTotals:
    if not os.path.isfile(path):
        return UsageTotals(
            prompt_tokens=0,
            completion_tokens=0,
            records=0,
            source_type="csv",
            source_path=os.path.abspath(path),
            mtime=0.0,
        )
    prompt = completion = 0
    records = 0
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records += 1
            prompt += _safe_int(row.get("prompt_tokens"))
            completion += _safe_int(row.get("completion_tokens"))
    return UsageTotals(
        prompt_tokens=prompt,
        completion_tokens=completion,
        records=records,
        source_type="csv",
        source_path=os.path.abspath(path),
        mtime=_safe_mtime(path),
    )


def _get_usage_totals(pred_csv: str, usage_json: Optional[str]) -> UsageTotals:
    if usage_json and os.path.isfile(usage_json):
        totals = _load_usage_json(usage_json)
        if totals.records > 0 or totals.total_tokens > 0:
            return totals
    return _load_usage_csv(pred_csv)


def _resolve_pricing(cfg: Dict[str, Any]) -> PricingSelection:
    pricing_cfg: Dict[str, Any] = cfg.get("pricing", {}) or {}
    provider_cfg: Dict[str, Any] = cfg.get("provider", {}) or {}
    provider_name = (provider_cfg.get("name") or cfg.get("backend") or "fireworks").lower()
    pricing_key = provider_cfg.get("pricing_key") or provider_cfg.get("model") or cfg.get("model_id")
    use_batch_api = bool(cfg.get("use_batch_api", True))

    providers_table = pricing_cfg.get("providers") or {}
    provider_table = providers_table.get(provider_name) if isinstance(providers_table, dict) else None
    entry: Optional[Dict[str, Any]] = None
    resolved_key = pricing_key
    if isinstance(provider_table, dict) and pricing_key:
        entry = provider_table.get(pricing_key)
        if entry is None:
            key_lower = str(pricing_key).lower()
            for candidate_key, candidate_value in provider_table.items():
                if isinstance(candidate_key, str) and candidate_key.lower() == key_lower:
                    entry = candidate_value
                    resolved_key = candidate_key
                    break
    table_found = entry is not None

    if entry is not None:
        input_rate = entry.get("input_per_mtok")
        output_rate = entry.get("output_per_mtok")
        input_rate_batch = entry.get("input_per_mtok_batch")
        output_rate_batch = entry.get("output_per_mtok_batch")
        discount = entry.get("batch_discount")
        if discount is None:
            discount = pricing_cfg.get("batch_discount", 1.0)
        rate_type = "standard"
        batch_applied = False
        if use_batch_api:
            if input_rate_batch is not None and output_rate_batch is not None:
                input_rate = input_rate_batch
                output_rate = output_rate_batch
                rate_type = "batch"
                batch_applied = True
            elif input_rate is not None and output_rate is not None and discount not in (None, 0, 1):
                input_rate = float(input_rate) * float(discount)
                output_rate = float(output_rate) * float(discount)
                rate_type = "batch_discount"
                batch_applied = True
        if input_rate is None or output_rate is None:
            raise SystemExit(f"Pricing for provider '{provider_name}' and model '{pricing_key}' is incomplete")
        return PricingSelection(
            provider=provider_name,
            model_key=resolved_key,
            input_rate_per_mtok=float(input_rate),
            output_rate_per_mtok=float(output_rate),
            rate_type=rate_type,
            table_found=table_found,
            batch_applied=batch_applied,
            use_batch_api=use_batch_api,
        )

    # Fallback to default pricing fields
    input_rate_default = float(pricing_cfg.get("input_per_million", 0.0))
    output_rate_default = float(pricing_cfg.get("output_per_million", 0.0))
    discount_default = float(pricing_cfg.get("batch_discount", 1.0))
    batch_applied = bool(use_batch_api and discount_default not in (0.0, 1.0))
    if use_batch_api:
        input_rate_default *= discount_default
        output_rate_default *= discount_default
        rate_type = "default_batch" if batch_applied else "default"
    else:
        rate_type = "default"
    return PricingSelection(
        provider=provider_name,
        model_key=resolved_key,
        input_rate_per_mtok=input_rate_default,
        output_rate_per_mtok=output_rate_default,
        rate_type=rate_type,
        table_found=False,
        batch_applied=batch_applied,
        use_batch_api=use_batch_api,
    )


def _build_cost_summary(usage: UsageTotals, pricing: PricingSelection) -> Dict[str, Any]:
    usd = 0.0
    if usage.prompt_tokens or usage.completion_tokens:
        usd = (
            (usage.prompt_tokens / 1_000_000.0) * pricing.input_rate_per_mtok
            + (usage.completion_tokens / 1_000_000.0) * pricing.output_rate_per_mtok
        )
    summary: Dict[str, Any] = {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "usd": round(usd, 6),
        "batch_discount_applied": pricing.batch_applied,
        "usage_records": usage.records,
        "usage_source": {
            "type": usage.source_type,
            "path": os.path.relpath(usage.source_path, os.getcwd()) if usage.source_path else None,
        },
        "pricing": {
            "provider": pricing.provider,
            "model_key": pricing.model_key,
            "input_rate_per_mtok": pricing.input_rate_per_mtok,
            "output_rate_per_mtok": pricing.output_rate_per_mtok,
            "rate_type": pricing.rate_type,
            "table_found": pricing.table_found,
            "use_batch_api": pricing.use_batch_api,
        },
    }
    return summary


def compute_cost_summary(
    config_path: str,
    pred_csv: str,
    usage_json: Optional[str] = None,
) -> Tuple[Dict[str, Any], UsageTotals, PricingSelection]:
    cfg = load_config(config_path)
    usage = _get_usage_totals(pred_csv, usage_json)
    pricing = _resolve_pricing(cfg)
    return _build_cost_summary(usage, pricing), usage, pricing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", default="results/predictions.csv")
    ap.add_argument("--usage_json", default=None)
    ap.add_argument("--config", default="config/eval_config.yaml")
    ap.add_argument("--out_path", default="results/costs.json")
    ap.add_argument("--dry_run", action="store_true", help="Emit summary to stdout without writing artifacts")
    args = ap.parse_args()

    summary, usage, pricing = compute_cost_summary(
        config_path=args.config,
        pred_csv=args.pred_csv,
        usage_json=args.usage_json,
    )

    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return

    latest_source_mtime = max(_safe_mtime(args.config), usage.mtime)
    if os.path.isfile(args.out_path) and _safe_mtime(args.out_path) >= latest_source_mtime:
        print("Idempotent skip: costs.json up-to-date")
        _update_manifest_costs(args.out_path)
        return

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Wrote", args.out_path)
    _update_manifest_costs(args.out_path, summary)


def _update_manifest_costs(costs_path: str, summary: Optional[Dict[str, Any]] = None) -> None:
    results_dir = os.path.dirname(costs_path)
    manifest_path = os.path.join(results_dir, "trial_manifest.json")
    if not os.path.isfile(manifest_path):
        return

    payload = summary
    if payload is None and os.path.isfile(costs_path):
        try:
            with open(costs_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            payload = None

    artifacts: Dict[str, Any] = {}
    if os.path.exists(costs_path):
        artifacts["costs_json"] = os.path.relpath(costs_path, results_dir)
    pricing_info = (payload or {}).get("pricing", {}) if isinstance(payload, dict) else {}
    if isinstance(pricing_info, dict):
        if pricing_info.get("provider"):
            artifacts["provider"] = pricing_info["provider"]
        if pricing_info.get("model_key"):
            artifacts["pricing_key"] = pricing_info["model_key"]
    if isinstance(payload, dict) and "usd" in payload:
        artifacts["usd"] = payload["usd"]

    status = "completed" if artifacts.get("costs_json") else "pending"
    try:
        mf.update_stage_status(manifest_path, "costs", status, artifacts)
    except Exception:
        pass


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    main()
