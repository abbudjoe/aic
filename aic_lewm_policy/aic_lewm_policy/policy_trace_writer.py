"""Optional JSONL policy trace writer for live AIC eval runs."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Mapping, cast

from .actions import CartesianVelocityAction


SCHEMA_VERSION = 1
_OFFICIAL_TRIAL_ID_RE = re.compile(r"^trial_[1-9][0-9]*$")


class PolicyTraceWriter:
    """Small runtime writer for the neutral policy trace JSONL contract."""

    def __init__(
        self,
        *,
        path: str | Path | None,
        run_id: str | None = None,
        official_trial_ids_by_call: tuple[str, ...] = (),
        official_trial_ids_by_trace_trial: Mapping[str, str] | None = None,
        trust_task_official_trial_id: bool = False,
    ):
        self._path = None if path is None else Path(path).expanduser()
        self._run_id = "" if run_id is None else run_id.strip()
        self._event_index = 0
        if type(trust_task_official_trial_id) is not bool:
            raise ValueError("trust_task_official_trial_id must be a boolean")
        self._trust_task_official_trial_id = trust_task_official_trial_id
        self._official_trial_ids_by_call = tuple(
            _require_official_trial_id(trial_id)
            for trial_id in official_trial_ids_by_call
        )
        trace_trial_map = official_trial_ids_by_trace_trial or {}
        self._official_trial_ids_by_trace_trial = _normalize_official_trial_id_map(
            trace_trial_map,
            "policy trace official trial map",
        )
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
        return cls(
            path=trace_path,
            run_id=run_id,
            official_trial_ids_by_call=_official_trial_ids_from_env(
                "AIC_LEWM_POLICY_TRACE_OFFICIAL_TRIAL_IDS"
            ),
            official_trial_ids_by_trace_trial=_official_trial_id_map_from_env(
                "AIC_LEWM_POLICY_TRACE_OFFICIAL_TRIAL_ID_MAP"
            ),
            trust_task_official_trial_id=_bool_from_env(
                "AIC_LEWM_POLICY_TRACE_TRUST_TASK_OFFICIAL_TRIAL_ID",
                default=False,
            ),
        )

    @property
    def enabled(self) -> bool:
        return self._path is not None

    def official_trial_id_for(
        self,
        *,
        trace_trial_id: str,
        policy_call_index: int,
        task_official_trial_id: Any | None = None,
    ) -> str | None:
        """Resolve an official trial id from explicit runtime surfaces only."""

        if self._path is None:
            return None
        trace_trial = _nonempty_text(trace_trial_id, "policy trace trial_id")
        if type(policy_call_index) is not int or policy_call_index <= 0:
            raise ValueError("policy_call_index must be a positive integer")
        candidates: list[str] = []
        if self._trust_task_official_trial_id:
            task_trial = _optional_official_trial_id(
                task_official_trial_id,
                field_name="task.official_trial_id",
            )
            if task_trial is not None:
                candidates.append(task_trial)
        mapped_trial = self._official_trial_ids_by_trace_trial.get(trace_trial)
        if mapped_trial is not None:
            candidates.append(mapped_trial)
        if policy_call_index <= len(self._official_trial_ids_by_call):
            candidates.append(self._official_trial_ids_by_call[policy_call_index - 1])
        distinct_candidates = sorted(set(candidates))
        if len(distinct_candidates) > 1:
            raise ValueError(
                "conflicting explicit official_trial_id mappings for policy trace trial "
                f"{trace_trial!r}: " + ", ".join(distinct_candidates)
            )
        return None if not distinct_candidates else distinct_candidates[0]

    def emit(
        self,
        *,
        trial_id: str,
        event_type: str,
        elapsed_sec: float,
        payload: Mapping[str, Any] | None = None,
        official_trial_id: str | None = None,
    ) -> None:
        if self._path is None:
            return
        trial = trial_id.strip()
        if not trial:
            raise ValueError("policy trace trial_id must be nonempty")
        official_trial = _optional_official_trial_id(official_trial_id)
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
            "leakage_class": _leakage_class_for_event(event_type),
            "payload": _jsonable(payload or {}),
        }
        if official_trial is not None:
            event["official_trial_id"] = official_trial
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


def _leakage_class_for_event(event_type: str) -> str:
    if event_type in {"action_selected", "action_published"}:
        return "legal_policy_action_output"
    return "legal_policy_input"


def _nonempty_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a nonempty string")
    return value.strip()


def _optional_official_trial_id(
    value: Any | None,
    *,
    field_name: str = "policy trace official_trial_id",
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a nonempty string when set")
    trial_id = value.strip()
    if not _OFFICIAL_TRIAL_ID_RE.match(trial_id):
        raise ValueError(f"{field_name} must be an official trial_* id")
    return trial_id


def _require_official_trial_id(value: Any) -> str:
    trial_id = _optional_official_trial_id(value)
    if trial_id is None:
        raise ValueError("policy trace official_trial_id must be set")
    return trial_id


def _official_trial_ids_from_env(name: str) -> tuple[str, ...]:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return ()
    value = _json_or_csv_env(raw, name)
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a JSON list or comma-separated list of official trial ids")
    return tuple(_require_official_trial_id(item) for item in value)


def _official_trial_id_map_from_env(name: str) -> dict[str, str]:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return {}
    try:
        pairs = json.loads(raw, object_pairs_hook=list)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must be a JSON object mapping trace trial ids to official trial ids") from exc
    if not raw.strip().startswith("{") or not isinstance(pairs, list):
        raise ValueError(f"{name} must be a JSON object mapping trace trial ids to official trial ids")
    normalized_pairs: list[tuple[Any, Any]] = []
    for item in pairs:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"{name} must be a JSON object mapping trace trial ids to official trial ids")
        trace_trial_id, official_trial_id = item
        normalized_pairs.append((trace_trial_id, official_trial_id))
    return _normalize_official_trial_id_pairs(normalized_pairs, name)


def _bool_from_env(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true"}:
        return True
    if normalized in {"0", "false"}:
        return False
    raise ValueError(f"{name} must be 1, true, 0, or false")


def _json_or_csv_env(raw: str, name: str) -> Any:
    stripped = raw.strip()
    if stripped.startswith("["):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{name} must be valid JSON") from exc
    items = [item.strip() for item in stripped.split(",")]
    if any(not item for item in items):
        raise ValueError(f"{name} CSV list must not contain empty entries")
    return items


def _normalize_official_trial_id_map(
    value: Mapping[str, str],
    field_name: str,
) -> dict[str, str]:
    return _normalize_official_trial_id_pairs(tuple(value.items()), field_name)


def _normalize_official_trial_id_pairs(
    pairs: tuple[tuple[Any, Any], ...] | list[tuple[Any, Any]],
    field_name: str,
) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for trace_trial_id, official_trial_id in pairs:
        normalized_key = _nonempty_text(trace_trial_id, f"{field_name} key")
        if normalized_key in normalized:
            raise ValueError(
                f"{field_name} must not contain duplicate trace trial ids: {normalized_key}"
            )
        normalized[normalized_key] = _require_official_trial_id(official_trial_id)
    return normalized


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
