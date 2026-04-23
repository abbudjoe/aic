import hashlib
import json
from pathlib import Path

import pytest

from aic_signal_harness import HarnessIOError, read_json, sha256_file, write_json


def test_write_read_json_and_overwrite_contract(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "artifact.json"

    write_json(path, {"z": 1, "a": {"b": 2}})

    assert read_json(path) == {"z": 1, "a": {"b": 2}}
    assert path.read_text(encoding="utf-8") == json.dumps(
        {"z": 1, "a": {"b": 2}},
        indent=2,
        sort_keys=True,
    ) + "\n"

    with pytest.raises(HarnessIOError, match="already exists"):
        write_json(path, {"z": 2})

    write_json(path, {"z": 2}, overwrite=True)

    assert read_json(path) == {"z": 2}


def test_read_json_requires_top_level_object(tmp_path: Path) -> None:
    path = tmp_path / "array.json"
    path.write_text("[1, 2, 3]\n", encoding="utf-8")

    with pytest.raises(HarnessIOError, match="top-level object"):
        read_json(path)


def test_write_json_requires_mapping(tmp_path: Path) -> None:
    with pytest.raises(HarnessIOError, match="requires a mapping"):
        write_json(tmp_path / "bad.json", ["not", "an", "object"])


def test_write_json_rejects_non_string_keys(tmp_path: Path) -> None:
    with pytest.raises(HarnessIOError, match="keys must be strings"):
        write_json(tmp_path / "bad.json", {"nested": {5: "would be coerced"}})


def test_write_json_wraps_parent_directory_creation_failures(tmp_path: Path) -> None:
    blocked_parent = tmp_path / "blocked"
    blocked_parent.write_text("not a directory\n", encoding="utf-8")

    with pytest.raises(HarnessIOError, match="failed to write JSON file"):
        write_json(blocked_parent / "child" / "artifact.json", {"ok": True})


def test_sha256_file(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    payload = b"aic signal harness\n"
    path.write_bytes(payload)

    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()
