import os
import json
import tempfile
from scripts.manifest_v2 import compute_stage_statuses


def _write_csv(path: str, header: str, rows: list[list[str]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")


def test_idempotency_predicates_via_stage_statuses():
    with tempfile.TemporaryDirectory() as d:
        results_dir = os.path.join(d, "trial", "results")
        os.makedirs(results_dir, exist_ok=True)

        # Create predictions.csv (parse done)
        _write_csv(
            os.path.join(results_dir, "predictions.csv"),
            "custom_id,type,answer",
            [["id1", "closed", "foo"], ["id2", "open", "bar"]],
        )
        # Create per_item_scores.csv (score done)
        _write_csv(
            os.path.join(results_dir, "per_item_scores.csv"),
            "custom_id,type,em,f1",
            [["id1", "closed", 1, 1.0], ["id2", "open", 0, 0.0]],
        )
        # Create significance.json (stats done) with schema_version>=2
        with open(os.path.join(results_dir, "significance.json"), "w", encoding="utf-8") as f:
            json.dump({"schema_version": 2, "metrics": {}}, f)

        st = compute_stage_statuses(results_dir)
        assert st.get("parsed", {}).get("status") == "completed"
        assert st.get("scored", {}).get("status") == "completed"
        assert st.get("stats", {}).get("status") == "completed"

