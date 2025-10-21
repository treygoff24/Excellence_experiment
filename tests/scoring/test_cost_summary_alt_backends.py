from __future__ import annotations

import json
from textwrap import dedent

from scripts.summarize_costs import compute_cost_summary


def _write_config(tmp_path, content: str) -> str:
    path = tmp_path / "config.yaml"
    path.write_text(dedent(content), encoding="utf-8")
    return str(path)


def _write_predictions_csv(tmp_path) -> str:
    path = tmp_path / "predictions.csv"
    path.write_text("prompt_tokens,completion_tokens\n", encoding="utf-8")
    return str(path)


def _write_usage_json(tmp_path, name: str, payload) -> str:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_openai_usage_applies_batch_discount(tmp_path):
    config_path = _write_config(
        tmp_path,
        """
        model_id: "gpt-5"
        backend: "alt"
        temps: [0.0]
        samples_per_item: {"0.0": 1}
        use_batch_api: true
        provider:
          name: openai
          model: gpt-5
          pricing_key: gpt-5
        pricing:
          input_per_million: 0.15
          output_per_million: 0.60
          batch_discount: 0.5
          openai:
            gpt-5:
              input_per_mtok: 1.25
              output_per_mtok: 5.00
              batch_discount: 0.5
        """,
    )
    usage_payload = [
        {"response": {"body": {"usage": {"prompt_tokens": 2000, "completion_tokens": 1000}}}},
        {"usage": {"prompt_tokens": 1000, "completion_tokens": 500}},
    ]
    usage_path = _write_usage_json(tmp_path, "usage.json", usage_payload)
    preds_path = _write_predictions_csv(tmp_path)

    summary, usage, _ = compute_cost_summary(
        config_path=config_path,
        pred_csv=preds_path,
        usage_json=usage_path,
    )

    assert usage.prompt_tokens == 3000
    assert usage.completion_tokens == 1500
    assert summary["batch_discount_applied"] is True
    assert summary["pricing"]["rate_type"] == "batch_discount"
    assert summary["pricing"]["provider"] == "openai"
    assert summary["pricing"]["input_rate_per_mtok"] == 0.625
    assert summary["pricing"]["output_rate_per_mtok"] == 2.5
    assert summary["usd"] == 0.005625


def test_anthropic_usage_prefers_batch_rates(tmp_path):
    config_path = _write_config(
        tmp_path,
        """
        model_id: "claude-sonnet"
        backend: "alt"
        temps: [0.0]
        samples_per_item: {"0.0": 1}
        use_batch_api: true
        provider:
          name: anthropic
          model: claude-sonnet
        pricing:
          input_per_million: 0.15
          output_per_million: 0.60
          batch_discount: 0.5
          anthropic:
            claude-sonnet:
              input_per_mtok_batch: 1.50
              output_per_mtok_batch: 7.50
        """,
    )
    usage_payload = [
        {"response": {"body": {"usage": {"input_tokens": 5000, "output_tokens": 2000}}}},
        {"usage": {"input_tokens": 3000, "output_tokens": 2000}},
    ]
    usage_path = _write_usage_json(tmp_path, "usage_anthropic.json", usage_payload)
    preds_path = _write_predictions_csv(tmp_path)

    summary, usage, _ = compute_cost_summary(
        config_path=config_path,
        pred_csv=preds_path,
        usage_json=usage_path,
    )

    assert usage.prompt_tokens == 8000
    assert usage.completion_tokens == 4000
    assert summary["pricing"]["rate_type"] == "batch"
    assert summary["pricing"]["input_rate_per_mtok"] == 1.5
    assert summary["pricing"]["output_rate_per_mtok"] == 7.5
    assert summary["pricing"]["provider"] == "anthropic"
    assert summary["batch_discount_applied"] is True
    assert summary["usd"] == 0.042
