from __future__ import annotations

from pathlib import Path

from scripts.build_batches import _ShardWriter


def test_shard_writer_rotates_when_max_lines_exceeded(tmp_path: Path) -> None:
    base_path = tmp_path / "shard.jsonl"
    writer = _ShardWriter(str(base_path), max_lines=3)

    for idx in range(8):
        writer.write_json(
            {
                "custom_id": f"dataset|item{idx}|condition|0.0|0|open",
                "body": {"messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]},
            }
        )

    entries = writer.finalize()
    assert len(entries) == 3

    first, second, third = entries
    assert Path(first["path"]).name == "shard.jsonl"
    assert first["lines"] == 3
    assert first["part_index"] == 0

    assert Path(second["path"]).name == "shard_part02.jsonl"
    assert second["lines"] == 3
    assert second["part_index"] == 1

    assert Path(third["path"]).name == "shard_part03.jsonl"
    assert third["lines"] == 2
    assert third["part_index"] == 2

    # Ensure files exist on disk with the expected number of lines
    for entry in entries:
        path = Path(entry["path"])
        assert path.exists()
        with path.open("r", encoding="utf-8") as fin:
            assert sum(1 for line in fin if line.strip()) == entry["lines"]
