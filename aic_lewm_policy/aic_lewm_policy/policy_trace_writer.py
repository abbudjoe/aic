"""Optional JSONL policy trace writer for live AIC eval runs."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, cast

from .actions import CartesianVelocityAction


SCHEMA_VERSION = 1


class PolicyTraceWriter:
    """Small runtime writer for the neutral policy trace JSONL contract."""

    def __init__(self, *, path: str | Path | None, run_id: str | None = None):
        self._path = None if path is None else Path(path).expanduser()
        self._run_id = "" if run_id is None else run_id.strip()
        self._event_index = 0
        if self._path is not None and not self._run_id:
            raise ValueError("AIC_LEWM_POLICY_TRACE_RUN_ID is required when policy tracing is enabled")
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text("", encoding="utf-8")

    @classmethod
    def disabled(cls) -> "PolicyTraceWriter":
        return cls(path=None)

    @classmethod
    def from_env(cls) -> "PolicyTraceWriter":
        trace_path = os.getenv("AIC_LEWM_POLICY_TRACE_PATH")
        if trace_path is None or not trace_path.strip():
            return cls.disabled()
        run_id = (
            os.getenv("AIC_LEWM_POLICY_TRACE_RUN_ID")
            or os.getenv("AIC_EVAL_RUN_ID")
            or os.getenv("AIC_EXPERIMENT_RUN_ID")
        )
        return cls(path=trace_path, run_id=run_id)

    @property
    def enabled(self) -> bool:
        return self._path is not None

    def emit(
        self,
        *,
        trial_id: str,
        event_type: str,
        elapsed_sec: float,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        if self._path is None:
            return
        trial = trial_id.strip()
        if not trial:
            raise ValueError("policy trace trial_id must be nonempty")
        elapsed = _nonnegative_finite_float(elapsed_sec, "elapsed_sec")
        event = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self._run_id,
            "trial_id": trial,
            "event_index": self._event_index,
            "event_type": event_type,
            "elapsed_sec": elapsed,
            "emitted_at_utc": _utc_now_iso(),
            "source": "aic_lewm_policy.LewmMpcPolicy",
            "leakage_class": "legal_policy_input",
            "payload": _jsonable(payload or {}),
        }
        with self._path.open("a", encoding="utf-8") as handle:
            json.dump(event, handle, allow_nan=False, separators=(",", ":"), sort_keys=True)
            handle.write("\n")
        self._event_index += 1


def task_payload(task_spec: Any) -> dict[str, Any]:
    """Return public task metadata suitable for legal policy trace payloads."""

    if is_dataclass(task_spec) and not isinstance(task_spec, type):
        payload = asdict(cast(Any, task_spec))
    else:
        payload = {
            key: getattr(task_spec, key)
            for key in (
                "task_id",
                "cable_type",
                "cable_name",
                "plug_type",
                "plug_name",
                "port_type",
                "port_name",
                "target_module_name",
                "time_limit_sec",
            )
            if hasattr(task_spec, key)
        }
    return _jsonable(payload)


def observation_payload(observation: Any) -> dict[str, Any]:
    """Summarize observation availability without recording raw images/state."""

    if observation is None:
        return {"available": False}
    timestamp_sec = getattr(observation, "timestamp_sec", None)
    payload: dict[str, Any] = {"available": True}
    if isinstance(timestamp_sec, (int, float)) and math.isfinite(float(timestamp_sec)):
        payload["timestamp_sec"] = float(timestamp_sec)
    images = getattr(observation, "images", None)
    state = getattr(observation, "state", None)
    if isinstance(images, Mapping):
        payload["image_names"] = sorted(str(name) for name in images)
    shape = getattr(state, "shape", None)
    if shape is not None:
        payload["state_shape"] = [int(value) for value in shape]
    return payload


def action_payload(action: CartesianVelocityAction) -> dict[str, Any]:
    return {
        "linear": [float(value) for value in action.linear],
        "angular": [float(value) for value in action.angular],
        "frame_id": action.frame_id,
    }


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _nonnegative_finite_float(value: Any, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{field_name} must be a nonnegative finite number")
    return float(value)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("policy trace payload cannot contain non-finite floats")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError("policy trace payload keys must be nonempty strings")
            result[key] = _jsonable(item)
        return result
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    raise ValueError(f"policy trace payload contains unsupported type {type(value).__name__}")
