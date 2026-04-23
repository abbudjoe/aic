"""Neutral artifact IO helpers for the AIC signal harness."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping


class HarnessIOError(RuntimeError):
    """Raised for expected harness IO or artifact contract failures."""


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON object from disk and reject non-object top-level payloads."""

    json_path = Path(path)
    try:
        with json_path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError as exc:
        raise HarnessIOError(f"JSON file not found: {json_path}") from exc
    except json.JSONDecodeError as exc:
        raise HarnessIOError(f"invalid JSON in {json_path}: {exc.msg}") from exc
    except OSError as exc:
        raise HarnessIOError(f"failed to read JSON file {json_path}: {exc}") from exc

    if not isinstance(value, dict):
        raise HarnessIOError(f"JSON file must contain a top-level object: {json_path}")
    return value


def write_json(path: str | Path, mapping: Mapping[str, Any], *, overwrite: bool = False) -> None:
    """Atomically write a JSON object with stable formatting."""

    if not isinstance(mapping, Mapping):
        raise HarnessIOError("write_json requires a mapping")
    _validate_json_object_keys(mapping, "mapping")

    json_path = Path(path)
    if json_path.exists() and not overwrite:
        raise HarnessIOError(f"{json_path} already exists; pass overwrite=True to replace it")

    tmp_name: str | None = None
    try:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=json_path.parent,
            prefix=f".{json_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_name = handle.name
            json.dump(mapping, handle, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
        if overwrite:
            os.replace(tmp_name, json_path)
        else:
            try:
                os.link(tmp_name, json_path)
            except FileExistsError as exc:
                raise HarnessIOError(
                    f"{json_path} already exists; pass overwrite=True to replace it"
                ) from exc
    except TypeError as exc:
        raise HarnessIOError(f"mapping is not JSON serializable: {exc}") from exc
    except ValueError as exc:
        raise HarnessIOError(f"mapping is not strict JSON: {exc}") from exc
    except OSError as exc:
        raise HarnessIOError(f"failed to write JSON file {json_path}: {exc}") from exc
    finally:
        if tmp_name is not None:
            try:
                Path(tmp_name).unlink(missing_ok=True)
            except OSError:
                pass


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()
    file_path = Path(path)
    try:
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except FileNotFoundError as exc:
        raise HarnessIOError(f"file not found: {file_path}") from exc
    except OSError as exc:
        raise HarnessIOError(f"failed to read file {file_path}: {exc}") from exc
    return digest.hexdigest()


def _validate_json_object_keys(value: Any, field_name: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise HarnessIOError(f"{field_name} keys must be strings")
            _validate_json_object_keys(item, f"{field_name}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_json_object_keys(item, f"{field_name}[{index}]")
