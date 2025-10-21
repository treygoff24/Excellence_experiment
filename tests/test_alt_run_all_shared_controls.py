import os
import tempfile

from scripts.alt_run_all import _ensure_control_entry  # type: ignore[attr-defined]
from scripts.run_all import _backend_tag  # type: ignore[attr-defined]


def test_control_registry_only_updates_once_for_shared_control_key():
    with tempfile.TemporaryDirectory() as tmpdir:
        run_root = tmpdir
        os.makedirs(os.path.join(run_root, "shared_controls"), exist_ok=True)

        control_registry: dict = {"controls": {}}
        manifest_first = {"prompts": {"control": {"sha256": "sha-a"}}}

        ctrl_key = "ctrl-key-123"
        entry1, mutated1 = _ensure_control_entry(
            run_root=run_root,
            control_registry=control_registry,
            manifest=manifest_first,
            label="t0",
            ctrl_key=ctrl_key,
            slug="trial-a",
            prompt_sha="sha-a",
            input_sha="input-a",
            backend_tag=_backend_tag(is_local_backend=False),
            model_id="model-a",
            temp=0.0,
        )

        assert mutated1 is True
        assert entry1["mode"] == "producer"
        assert control_registry["controls"][ctrl_key]["producer_trial"] == "trial-a"

        manifest_second = {"prompts": {"control": {"sha256": "sha-a"}}}
        entry2, mutated2 = _ensure_control_entry(
            run_root=run_root,
            control_registry=control_registry,
            manifest=manifest_second,
            label="t0",
            ctrl_key=ctrl_key,
            slug="trial-b",
            prompt_sha="sha-a",
            input_sha="input-a",
            backend_tag=_backend_tag(is_local_backend=False),
            model_id="model-a",
            temp=0.0,
        )

        assert mutated2 is False
        assert entry2["mode"] == "producer"
        assert control_registry["controls"][ctrl_key]["producer_trial"] == "trial-a"
        assert len(control_registry["controls"]) == 1
