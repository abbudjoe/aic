"""Neutral artifact IO helpers for the AIC signal harness."""

from __future__ import annotations

import hashlib
import json
import os
import string
import tempfile
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import unquote, urlparse


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


def local_artifact_path(
    *,
    path: str | None,
    uri: str | None,
    field_name: str,
) -> Path | None:
    """Return the local artifact path and bind path/file:// URI when both are set."""

    uri_path = file_uri_artifact_path(uri, field_name=field_name)
    if path is not None:
        artifact_path = Path(path).expanduser()
        if uri_path is not None:
            try:
                artifact_resolved = artifact_path.resolve(strict=False)
                uri_resolved = uri_path.resolve(strict=False)
            except (OSError, ValueError) as exc:
                raise HarnessIOError(f"{field_name} local artifact path cannot be resolved") from exc
            if artifact_resolved != uri_resolved:
                raise HarnessIOError(
                    f"{field_name} path and file URI must refer to the same local artifact"
                )
        return artifact_path
    return uri_path


def file_uri_artifact_path(uri: str | None, *, field_name: str) -> Path | None:
    if uri is None:
        return None
    if any(char.isspace() for char in uri):
        raise HarnessIOError(f"{field_name} URI must not contain whitespace")
    parsed = urlparse(uri)
    if not parsed.scheme:
        raise HarnessIOError(f"{field_name} URI must include a scheme")
    if parsed.scheme != "file":
        if not parsed.netloc:
            raise HarnessIOError(f"{field_name} URI must include a network location")
        return None
    if not parsed.path:
        raise HarnessIOError(f"{field_name} file URI must include a file path")
    if parsed.netloc not in ("", "localhost"):
        raise HarnessIOError(f"{field_name} file URI host must be empty or localhost")
    decoded_path = _decode_file_uri_path(parsed.path, field_name)
    file_path = Path(decoded_path).expanduser()
    if not file_path.is_absolute():
        raise HarnessIOError(f"{field_name} file URI path must be absolute")
    return file_path


def _decode_file_uri_path(path: str, field_name: str) -> str:
    _validate_percent_escapes(path, field_name)
    try:
        decoded_path = unquote(path, errors="strict")
    except UnicodeDecodeError as exc:
        raise HarnessIOError(f"{field_name} file URI path has invalid percent-encoded UTF-8") from exc
    if "\x00" in decoded_path:
        raise HarnessIOError(f"{field_name} file URI path must not contain NUL bytes")
    return decoded_path


def _validate_percent_escapes(value: str, field_name: str) -> None:
    hex_digits = set(string.hexdigits)
    index = 0
    while True:
        index = value.find("%", index)
        if index == -1:
            return
        escape = value[index + 1 : index + 3]
        if len(escape) != 2 or any(char not in hex_digits for char in escape):
            raise HarnessIOError(f"{field_name} file URI path has invalid percent escape")
        index += 3


def _validate_json_object_keys(value: Any, field_name: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise HarnessIOError(f"{field_name} keys must be strings")
            _validate_json_object_keys(item, f"{field_name}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_json_object_keys(item, f"{field_name}[{index}]")
