from __future__ import annotations

from pathlib import Path

from scripts.prompt_adapter import render_payload


def _build_config(tmp_path: Path, *, provider_name: str = "anthropic", cache_cfg: dict | None = None) -> dict:
    control_path = tmp_path / "control.txt"
    treatment_path = tmp_path / "treatment.txt"
    control_path.write_text("Control prompt", encoding="utf-8")
    treatment_path.write_text("Treatment prompt", encoding="utf-8")
    provider: dict = {"name": provider_name, "model": "claude-test"}
    if provider_name == "anthropic":
        provider["cache_control"] = cache_cfg or {"enable_system_cache": True, "type": "ephemeral", "ttl": "1h"}
    cfg = {
        "prompt_sets": {"default": {"control": str(control_path), "treatment": str(treatment_path)}},
        "provider": provider,
        "max_new_tokens": {"open_book": 256, "closed_book": 128},
        "stop": [],
    }
    return cfg


def test_render_payload_emits_cache_control_blocks(tmp_path: Path) -> None:
    cfg = _build_config(tmp_path)
    payload = render_payload(
        cfg,
        condition="treatment",
        prompt_set="default",
        task_type="open_book",
        temp=1.0,
        question="What is caching?",
        context="Context text",
        out_format="messages",
    )
    messages = payload["body"]["messages"]
    system_content = messages[0]["content"]
    assert isinstance(system_content, list)
    assert system_content[0]["cache_control"]["type"] == "ephemeral"
    assert system_content[0]["cache_control"]["ttl"] == "1h"
    assert "What is caching?" in messages[1]["content"][0]["text"]


def test_render_payload_skips_cache_control_when_disabled(tmp_path: Path) -> None:
    cfg = _build_config(tmp_path, cache_cfg={"enable_system_cache": False})
    payload = render_payload(
        cfg,
        condition="control",
        prompt_set="default",
        task_type="closed_book",
        temp=0.0,
        question="Question?",
        context=None,
        out_format="messages",
    )
    system_content = payload["body"]["messages"][0]["content"]
    assert isinstance(system_content, list)
    assert "cache_control" not in system_content[0]


def test_render_payload_default_string_for_non_anthropic(tmp_path: Path) -> None:
    cfg = _build_config(tmp_path, provider_name="openai")
    payload = render_payload(
        cfg,
        condition="control",
        prompt_set="default",
        task_type="closed_book",
        temp=0.0,
        question="Hello?",
        context=None,
        out_format="messages",
    )
    system_content = payload["body"]["messages"][0]["content"]
    assert isinstance(system_content, str)
