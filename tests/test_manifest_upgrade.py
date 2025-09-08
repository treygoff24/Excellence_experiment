import os
import json
import tempfile
from scripts import manifest_v2 as mf


def test_manifest_upgrade_v1_to_v2_and_stage_status():
    with tempfile.TemporaryDirectory() as d:
        results_dir = os.path.join(d, "results")
        os.makedirs(results_dir, exist_ok=True)
        manifest_path = os.path.join(results_dir, "trial_manifest.json")

        # Write a minimal v1-like manifest
        v1 = {
            "schema_version": 1,
            "created_utc": "2020-01-01T00:00:00Z",
            "run_id": "rtest",
            "trial": {"model_id": "m", "prompt_set": "ps"},
            "temps": [0.0],
            "samples_per_item": {"0.0": 1},
            "prompts": {},
            "jobs": {},
            "job_status": {},
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(v1, f)

        # Create a tiny predictions.csv so parsed status can be inferred
        preds = os.path.join(results_dir, "predictions.csv")
        with open(preds, "w", encoding="utf-8") as f:
            f.write("custom_id,type,answer\n")
            f.write("id1,closed,foo\n")

        data, upgraded = mf.load_manifest(manifest_path)
        assert upgraded is True
        assert int(data.get("schema_version", 0)) == mf.SCHEMA_VERSION
        assert isinstance(data.get("stage_status"), dict)
        # parsed stage should reference predictions.csv
        parsed = data.get("stage_status", {}).get("parsed", {})
        assert parsed.get("status") in ("completed", "pending")

