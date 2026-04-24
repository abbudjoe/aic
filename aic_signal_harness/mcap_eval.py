"""Typed contracts for official AIC MCAP evaluation bundle analysis."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any, Mapping
from urllib.parse import urlparse

from aic_signal_harness.artifacts import HarnessIOError
from aic_signal_harness.schemas import SCHEMA_VERSION


_MCAP_TASK_HINTS_KEYS = frozenset(
    {"task_id", "plug_type", "port_type", "target_module_name"}
)
_MCAP_FIRST_CONTACT_KEYS = frozenset(
    {
        "log_time_ns",
        "collision1",
        "collision2",
        "log_elapsed_sec",
        "nearest_controller_elapsed_sec",
        "nearest_command_elapsed_sec",
        "nearest_tcp_position",
        "nearest_tcp_error",
        "nearest_command_linear",
        "nearest_command_angular",
        "recommended_stop_sec",
    }
)
_MCAP_TRIAL_KEYS = frozenset(
    {
        "trial_id",
        "source",
        "size_bytes",
        "sha256",
        "controller_state_count",
        "pose_command_count",
        "off_limit_contact_count",
        "controller_stamp_start_sec",
        "controller_stamp_end_sec",
        "controller_duration_sec",
        "final_tcp_position",
        "final_tcp_error",
        "task_hints",
        "first_off_limit_contact",
    }
)
_MCAP_BUNDLE_KEYS = frozenset(
    {
        "schema_version",
        "source",
        "analyzed_at_utc",
        "contact_margin_sec",
        "stop_step_sec",
        "recommended_env",
        "trials",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_TRIAL_ID_RE = re.compile(r"^trial_([1-9][0-9]*)$")
_URI_SOURCE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_SC_RECOMMENDATION_ENV = "AIC_LEWM_SC_REPLAY_STOP_SEC"
_OFFICIAL_TRIAL_INDEXES = (1, 2, 3)


@dataclass(frozen=True)
class McapEvalTaskHints:
    """Task hints aligned to one official evaluation trial."""

    task_id: str | None = None
    plug_type: str | None = None
    port_type: str | None = None
    target_module_name: str | None = None

    def __post_init__(self) -> None:
        errors: list[str] = []
        present = False
        for field_name in _MCAP_TASK_HINTS_KEYS:
            try:
                value = _optional_nonempty_text(
                    getattr(self, field_name),
                    f"mcap task_hints.{field_name}",
                )
                object.__setattr__(self, field_name, value)
                if value is not None:
                    present = True
            except HarnessIOError as exc:
                errors.append(str(exc))
        if not present:
            errors.append("mcap task_hints must set at least one field")
        if errors:
            raise HarnessIOError("; ".join(errors))

    def to_dict(self) -> dict[str, str]:
        value: dict[str, str] = {}
        for field_name in sorted(_MCAP_TASK_HINTS_KEYS):
            field_value = getattr(self, field_name)
            if field_value is not None:
                value[field_name] = field_value
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "McapEvalTaskHints":
        if not isinstance(value, Mapping):
            raise HarnessIOError("mcap task_hints must be a mapping")
        _reject_unknown_keys(value, _MCAP_TASK_HINTS_KEYS, "mcap task_hints")
        return cls(
            task_id=value.get("task_id"),
            plug_type=value.get("plug_type"),
            port_type=value.get("port_type"),
            target_module_name=value.get("target_module_name"),
        )


@dataclass(frozen=True)
class McapEvalFirstContact:
    """First off-limit contact context for one evaluation trial."""

    log_time_ns: int
    collision1: str
    collision2: str
    log_elapsed_sec: float | None = None
    nearest_controller_elapsed_sec: float | None = None
    nearest_command_elapsed_sec: float | None = None
    nearest_tcp_position: tuple[float, float, float] | None = None
    nearest_tcp_error: tuple[float, float, float] | None = None
    nearest_command_linear: tuple[float, float, float] | None = None
    nearest_command_angular: tuple[float, float, float] | None = None
    recommended_stop_sec: float | None = None

    def __post_init__(self) -> None:
        errors: list[str] = []
        try:
            object.__setattr__(
                self,
                "log_time_ns",
                _require_nonnegative_int(
                    self.log_time_ns,
                    "mcap first_off_limit_contact.log_time_ns",
                ),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        for field_name in ("collision1", "collision2"):
            try:
                object.__setattr__(
                    self,
                    field_name,
                    _require_nonempty_text(
                        getattr(self, field_name),
                        f"mcap first_off_limit_contact.{field_name}",
                    ),
                )
            except HarnessIOError as exc:
                errors.append(str(exc))
        for field_name in (
            "log_elapsed_sec",
            "nearest_controller_elapsed_sec",
            "nearest_command_elapsed_sec",
            "recommended_stop_sec",
        ):
            try:
                object.__setattr__(
                    self,
                    field_name,
                    _optional_nonnegative_float(
                        getattr(self, field_name),
                        f"mcap first_off_limit_contact.{field_name}",
                    ),
                )
            except HarnessIOError as exc:
                errors.append(str(exc))
        for field_name in (
            "nearest_tcp_position",
            "nearest_tcp_error",
            "nearest_command_linear",
            "nearest_command_angular",
        ):
            try:
                object.__setattr__(
                    self,
                    field_name,
                    _optional_vec3(
                        getattr(self, field_name),
                        f"mcap first_off_limit_contact.{field_name}",
                    ),
                )
            except HarnessIOError as exc:
                errors.append(str(exc))

        if _any_set(
            self.nearest_controller_elapsed_sec,
            self.nearest_tcp_position,
            self.nearest_tcp_error,
        ) and not _all_set(
            self.nearest_controller_elapsed_sec,
            self.nearest_tcp_position,
            self.nearest_tcp_error,
        ):
            errors.append(
                "mcap first_off_limit_contact nearest controller context must set "
                "nearest_controller_elapsed_sec, nearest_tcp_position, and nearest_tcp_error together"
            )
        if _any_set(
            self.nearest_command_elapsed_sec,
            self.nearest_command_linear,
            self.nearest_command_angular,
        ) and not _all_set(
            self.nearest_command_elapsed_sec,
            self.nearest_command_linear,
            self.nearest_command_angular,
        ):
            errors.append(
                "mcap first_off_limit_contact nearest command context must set "
                "nearest_command_elapsed_sec, nearest_command_linear, and nearest_command_angular together"
            )
        if self.recommended_stop_sec is not None and self.nearest_controller_elapsed_sec is None:
            errors.append(
                "mcap first_off_limit_contact.recommended_stop_sec requires nearest_controller_elapsed_sec"
            )
        elif (
            self.recommended_stop_sec is not None
            and self.nearest_controller_elapsed_sec is not None
            and self.recommended_stop_sec > self.nearest_controller_elapsed_sec + 1e-9
        ):
            errors.append(
                "mcap first_off_limit_contact.recommended_stop_sec must not exceed nearest_controller_elapsed_sec"
            )
        if errors:
            raise HarnessIOError("; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "log_time_ns": self.log_time_ns,
            "collision1": self.collision1,
            "collision2": self.collision2,
        }
        for field_name in (
            "log_elapsed_sec",
            "nearest_controller_elapsed_sec",
            "nearest_command_elapsed_sec",
            "recommended_stop_sec",
        ):
            field_value = getattr(self, field_name)
            if field_value is not None:
                value[field_name] = field_value
        for field_name in (
            "nearest_tcp_position",
            "nearest_tcp_error",
            "nearest_command_linear",
            "nearest_command_angular",
        ):
            field_value = getattr(self, field_name)
            if field_value is not None:
                value[field_name] = list(field_value)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "McapEvalFirstContact":
        if not isinstance(value, Mapping):
            raise HarnessIOError("mcap first_off_limit_contact must be a mapping")
        _reject_unknown_keys(
            value,
            _MCAP_FIRST_CONTACT_KEYS,
            "mcap first_off_limit_contact",
        )
        return cls(
            log_time_ns=value.get("log_time_ns"),
            collision1=value.get("collision1"),
            collision2=value.get("collision2"),
            log_elapsed_sec=value.get("log_elapsed_sec"),
            nearest_controller_elapsed_sec=value.get("nearest_controller_elapsed_sec"),
            nearest_command_elapsed_sec=value.get("nearest_command_elapsed_sec"),
            nearest_tcp_position=value.get("nearest_tcp_position"),
            nearest_tcp_error=value.get("nearest_tcp_error"),
            nearest_command_linear=value.get("nearest_command_linear"),
            nearest_command_angular=value.get("nearest_command_angular"),
            recommended_stop_sec=value.get("recommended_stop_sec"),
        )


@dataclass(frozen=True)
class McapEvalTrialReport:
    """Typed MCAP evidence for one official AIC evaluation trial."""

    trial_id: str
    source: str
    size_bytes: int
    sha256: str
    controller_state_count: int
    pose_command_count: int
    off_limit_contact_count: int
    controller_stamp_start_sec: float | None = None
    controller_stamp_end_sec: float | None = None
    controller_duration_sec: float | None = None
    final_tcp_position: tuple[float, float, float] | None = None
    final_tcp_error: tuple[float, float, float] | None = None
    task_hints: McapEvalTaskHints | None = None
    first_off_limit_contact: McapEvalFirstContact | None = None

    def __post_init__(self) -> None:
        errors: list[str] = []
        try:
            trial_id = _require_nonempty_text(self.trial_id, "mcap trial.trial_id")
            if _TRIAL_ID_RE.fullmatch(trial_id) is None:
                raise HarnessIOError(
                    "mcap trial.trial_id must match the pattern 'trial_<positive integer>'"
                )
            object.__setattr__(self, "trial_id", trial_id)
        except HarnessIOError as exc:
            errors.append(str(exc))

        try:
            object.__setattr__(
                self,
                "source",
                _validate_source_identity(self.source, "mcap trial.source"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "size_bytes",
                _require_nonnegative_int(self.size_bytes, "mcap trial.size_bytes"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        if not isinstance(self.sha256, str) or not _SHA256_RE.match(self.sha256):
            errors.append("mcap trial.sha256 must be 64 hexadecimal characters")
        for field_name in (
            "controller_state_count",
            "pose_command_count",
            "off_limit_contact_count",
        ):
            try:
                object.__setattr__(
                    self,
                    field_name,
                    _require_nonnegative_int(
                        getattr(self, field_name),
                        f"mcap trial.{field_name}",
                    ),
                )
            except HarnessIOError as exc:
                errors.append(str(exc))
        for field_name in (
            "controller_stamp_start_sec",
            "controller_stamp_end_sec",
            "controller_duration_sec",
        ):
            try:
                object.__setattr__(
                    self,
                    field_name,
                    _optional_nonnegative_float(
                        getattr(self, field_name),
                        f"mcap trial.{field_name}",
                    ),
                )
            except HarnessIOError as exc:
                errors.append(str(exc))
        for field_name in ("final_tcp_position", "final_tcp_error"):
            try:
                object.__setattr__(
                    self,
                    field_name,
                    _optional_vec3(getattr(self, field_name), f"mcap trial.{field_name}"),
                )
            except HarnessIOError as exc:
                errors.append(str(exc))

        if self.task_hints is not None and not isinstance(self.task_hints, McapEvalTaskHints):
            try:
                object.__setattr__(
                    self,
                    "task_hints",
                    McapEvalTaskHints.from_dict(self.task_hints),
                )
            except HarnessIOError as exc:
                errors.append(str(exc))
        if self.first_off_limit_contact is not None and not isinstance(
            self.first_off_limit_contact,
            McapEvalFirstContact,
        ):
            try:
                object.__setattr__(
                    self,
                    "first_off_limit_contact",
                    McapEvalFirstContact.from_dict(self.first_off_limit_contact),
                )
            except HarnessIOError as exc:
                errors.append(str(exc))

        if self.controller_state_count == 0:
            if _any_set(
                self.controller_stamp_start_sec,
                self.controller_stamp_end_sec,
                self.controller_duration_sec,
                self.final_tcp_position,
                self.final_tcp_error,
            ):
                errors.append(
                    "mcap trial with controller_state_count == 0 must not set controller timing or final tcp evidence"
                )
        elif not _all_set(
            self.controller_stamp_start_sec,
            self.controller_stamp_end_sec,
            self.controller_duration_sec,
            self.final_tcp_position,
            self.final_tcp_error,
        ):
            errors.append(
                "mcap trial with controller_state_count > 0 must set controller timing and final tcp evidence"
            )
        elif (
            self.controller_stamp_start_sec is not None
            and self.controller_stamp_end_sec is not None
            and self.controller_duration_sec is not None
        ):
            if self.controller_stamp_end_sec < self.controller_stamp_start_sec:
                errors.append(
                    "mcap trial.controller_stamp_end_sec must be >= controller_stamp_start_sec"
                )
            expected_duration = self.controller_stamp_end_sec - self.controller_stamp_start_sec
            if not math.isclose(
                self.controller_duration_sec,
                expected_duration,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                errors.append(
                    "mcap trial.controller_duration_sec must match "
                    "controller_stamp_end_sec - controller_stamp_start_sec"
                )

        if self.off_limit_contact_count == 0 and self.first_off_limit_contact is not None:
            errors.append(
                "mcap trial.first_off_limit_contact must be null when off_limit_contact_count is 0"
            )
        if self.off_limit_contact_count > 0 and self.first_off_limit_contact is None:
            errors.append(
                "mcap trial.first_off_limit_contact must be set when off_limit_contact_count is positive"
            )
        if (
            self.first_off_limit_contact is not None
            and self.first_off_limit_contact.recommended_stop_sec is not None
        ):
            if self.task_hints is None or not _same(self.task_hints.port_type, "sc"):
                errors.append(
                    "mcap trial.first_off_limit_contact.recommended_stop_sec requires task_hints.port_type == 'sc'"
                )
        if self.first_off_limit_contact is not None:
            contact = self.first_off_limit_contact
            if self.controller_state_count == 0:
                if _any_set(
                    contact.log_elapsed_sec,
                    contact.nearest_controller_elapsed_sec,
                    contact.nearest_tcp_position,
                    contact.nearest_tcp_error,
                    contact.recommended_stop_sec,
                ):
                    errors.append(
                        "mcap trial.first_off_limit_contact controller context requires controller_state_count > 0"
                    )
            elif not _all_set(
                contact.log_elapsed_sec,
                contact.nearest_controller_elapsed_sec,
                contact.nearest_tcp_position,
                contact.nearest_tcp_error,
            ):
                errors.append(
                    "mcap trial.first_off_limit_contact controller context must be complete when controller_state_count is positive"
                )

            if self.pose_command_count == 0:
                if _any_set(
                    contact.nearest_command_elapsed_sec,
                    contact.nearest_command_linear,
                    contact.nearest_command_angular,
                ):
                    errors.append(
                        "mcap trial.first_off_limit_contact command context requires pose_command_count > 0"
                    )
            elif not _all_set(
                contact.nearest_command_elapsed_sec,
                contact.nearest_command_linear,
                contact.nearest_command_angular,
            ):
                errors.append(
                    "mcap trial.first_off_limit_contact command context must be complete when pose_command_count is positive"
                )

        if errors:
            raise HarnessIOError("; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "trial_id": self.trial_id,
            "source": self.source,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "controller_state_count": self.controller_state_count,
            "pose_command_count": self.pose_command_count,
            "off_limit_contact_count": self.off_limit_contact_count,
        }
        for field_name in (
            "controller_stamp_start_sec",
            "controller_stamp_end_sec",
            "controller_duration_sec",
        ):
            field_value = getattr(self, field_name)
            if field_value is not None:
                value[field_name] = field_value
        for field_name in ("final_tcp_position", "final_tcp_error"):
            field_value = getattr(self, field_name)
            if field_value is not None:
                value[field_name] = list(field_value)
        if self.task_hints is not None:
            value["task_hints"] = self.task_hints.to_dict()
        if self.first_off_limit_contact is not None:
            value["first_off_limit_contact"] = self.first_off_limit_contact.to_dict()
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "McapEvalTrialReport":
        if not isinstance(value, Mapping):
            raise HarnessIOError("mcap trial must be a mapping")
        _reject_unknown_keys(value, _MCAP_TRIAL_KEYS, "mcap trial")
        return cls(
            trial_id=value.get("trial_id"),
            source=value.get("source"),
            size_bytes=value.get("size_bytes"),
            sha256=value.get("sha256"),
            controller_state_count=value.get("controller_state_count"),
            pose_command_count=value.get("pose_command_count"),
            off_limit_contact_count=value.get("off_limit_contact_count"),
            controller_stamp_start_sec=value.get("controller_stamp_start_sec"),
            controller_stamp_end_sec=value.get("controller_stamp_end_sec"),
            controller_duration_sec=value.get("controller_duration_sec"),
            final_tcp_position=value.get("final_tcp_position"),
            final_tcp_error=value.get("final_tcp_error"),
            task_hints=value.get("task_hints"),
            first_off_limit_contact=value.get("first_off_limit_contact"),
        )


@dataclass(frozen=True)
class McapEvalBundleReport:
    """Typed analysis report for one official AIC MCAP eval bundle."""

    source: str
    analyzed_at_utc: str
    contact_margin_sec: float
    stop_step_sec: float
    trials: tuple[McapEvalTrialReport, ...]
    recommended_env: Mapping[str, str] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        errors: list[str] = []
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            errors.append(f"schema_version must be {SCHEMA_VERSION}")
        try:
            object.__setattr__(
                self,
                "source",
                _validate_source_identity(self.source, "mcap bundle.source"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "analyzed_at_utc",
                _require_nonempty_text(
                    self.analyzed_at_utc,
                    "mcap bundle.analyzed_at_utc",
                ),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        for field_name in ("contact_margin_sec", "stop_step_sec"):
            try:
                object.__setattr__(
                    self,
                    field_name,
                    _require_nonnegative_float(
                        getattr(self, field_name),
                        f"mcap bundle.{field_name}",
                    ),
                )
            except HarnessIOError as exc:
                errors.append(str(exc))

        trials: list[McapEvalTrialReport] = []
        if isinstance(self.trials, (str, bytes, bytearray)) or not isinstance(
            self.trials,
            (list, tuple),
        ):
            errors.append("mcap bundle.trials must be a list or tuple")
        else:
            for index, trial in enumerate(self.trials):
                try:
                    trials.append(
                        trial
                        if isinstance(trial, McapEvalTrialReport)
                        else McapEvalTrialReport.from_dict(trial)
                    )
                except HarnessIOError as exc:
                    errors.append(f"mcap bundle.trials[{index}]: {exc}")
        object.__setattr__(self, "trials", tuple(trials))
        if not trials:
            errors.append("mcap bundle.trials must not be empty")

        recommended_env, env_errors = _copy_recommended_env(self.recommended_env)
        object.__setattr__(self, "recommended_env", recommended_env)
        errors.extend(env_errors)

        trial_ids = tuple(trial.trial_id for trial in trials)
        if len(set(trial_ids)) != len(trial_ids):
            errors.append("mcap bundle.trials must not contain duplicate trial_id values")
        trial_indexes = tuple(_trial_index(trial_id) for trial_id in trial_ids)
        if tuple(sorted(trial_indexes)) != trial_indexes:
            errors.append("mcap bundle.trials must be ordered by ascending trial_id")
        if trial_indexes and trial_indexes != _OFFICIAL_TRIAL_INDEXES:
            errors.append(
                "mcap bundle.trials must encode the official three-trial sequence: trial_1, trial_2, trial_3"
            )

        if _is_uri_source(self.source):
            uri_trial_sources: set[str] = set()
            for trial in trials:
                if not _is_uri_source(trial.source):
                    errors.append("mcap bundle with uri source must not contain local trial sources")
                    continue
                if not _uri_source_is_within_bundle(self.source, trial.source):
                    errors.append(
                        "mcap trial.source must be rooted under mcap bundle.source when bundle.source is a uri"
                    )
                uri_trial_sources.add(trial.source)
            if len(uri_trial_sources) != len(trials):
                errors.append("mcap bundle.trials must not contain duplicate source paths")
        else:
            bundle_root = _local_source_path(self.source)
            local_trial_sources: set[str] = set()
            for trial in trials:
                if _is_uri_source(trial.source):
                    errors.append("mcap bundle with local source must not contain uri trial sources")
                    continue
                try:
                    trial_path = _local_source_path(trial.source)
                    trial_path.relative_to(bundle_root)
                except ValueError:
                    errors.append(
                        "mcap trial.source must be rooted under mcap bundle.source when bundle.source is local"
                    )
                else:
                    local_trial_sources.add(str(trial_path))
            if len(local_trial_sources) != len(trials):
                errors.append("mcap bundle.trials must not contain duplicate source paths")

        recommended_stops: list[float] = []
        for trial in trials:
            actual_stop = (
                None
                if trial.first_off_limit_contact is None
                else trial.first_off_limit_contact.recommended_stop_sec
            )
            expected_stop = _expected_sc_recommended_stop_sec(
                trial,
                contact_margin_sec=self.contact_margin_sec,
                stop_step_sec=self.stop_step_sec,
            )
            if expected_stop is None:
                if actual_stop is not None:
                    errors.append(
                        "mcap trial.first_off_limit_contact.recommended_stop_sec must be null unless the bundle configuration derives an sc stop recommendation"
                    )
                continue
            if actual_stop is None or not math.isclose(
                actual_stop,
                expected_stop,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                errors.append(
                    "mcap trial.first_off_limit_contact.recommended_stop_sec must match the bundle-derived sc stop recommendation"
                )
                continue
            recommended_stops.append(expected_stop)
        unique_recommended_env = {
            _SC_RECOMMENDATION_ENV: f"{recommended_stops[0]:.2f}"
        } if recommended_stops else {}
        if len({f"{value:.2f}" for value in recommended_stops}) > 1:
            errors.append(
                "mcap bundle.trials contain conflicting sc stop recommendations"
            )
        if dict(recommended_env) != unique_recommended_env:
            errors.append(
                "mcap bundle.recommended_env must exactly match trial-derived sc stop recommendations"
            )

        if errors:
            raise HarnessIOError("; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": self.schema_version,
            "source": self.source,
            "analyzed_at_utc": self.analyzed_at_utc,
            "contact_margin_sec": self.contact_margin_sec,
            "stop_step_sec": self.stop_step_sec,
            "recommended_env": dict(self.recommended_env),
            "trials": [trial.to_dict() for trial in self.trials],
        }
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "McapEvalBundleReport":
        if not isinstance(value, Mapping):
            raise HarnessIOError("mcap bundle report must be a mapping")
        _reject_unknown_keys(value, _MCAP_BUNDLE_KEYS, "mcap bundle")
        return cls(
            schema_version=value.get("schema_version"),
            source=value.get("source"),
            analyzed_at_utc=value.get("analyzed_at_utc"),
            contact_margin_sec=value.get("contact_margin_sec"),
            stop_step_sec=value.get("stop_step_sec"),
            recommended_env=value.get("recommended_env", {}),
            trials=value.get("trials", ()),
        )


def _reject_unknown_keys(
    value: Mapping[str, Any],
    allowed_keys: frozenset[str],
    field_name: str,
) -> None:
    unknown_keys = sorted(repr(key) for key in value if key not in allowed_keys)
    if unknown_keys:
        raise HarnessIOError(f"{field_name} has unknown fields: {', '.join(unknown_keys)}")


def _require_nonempty_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HarnessIOError(f"{field_name} must be a nonempty string")
    return value


def _optional_nonempty_text(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_nonempty_text(value, field_name)


def _require_nonnegative_int(value: Any, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise HarnessIOError(f"{field_name} must be a nonnegative integer")
    return value


def _require_nonnegative_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise HarnessIOError(f"{field_name} must be a finite nonnegative number")
    normalized = float(value)
    if normalized < 0.0:
        raise HarnessIOError(f"{field_name} must be a finite nonnegative number")
    return normalized


def _optional_nonnegative_float(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    return _require_nonnegative_float(value, field_name)


def _optional_vec3(value: Any, field_name: str) -> tuple[float, float, float] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, (list, tuple)):
        raise HarnessIOError(f"{field_name} must be a list or tuple of length 3")
    if len(value) != 3:
        raise HarnessIOError(f"{field_name} must be a list or tuple of length 3")
    normalized: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
            raise HarnessIOError(
                f"{field_name}[{index}] must be a finite number"
            )
        normalized.append(float(item))
    return tuple(normalized)  # type: ignore[return-value]


def _copy_recommended_env(
    value: Mapping[str, Any] | None,
) -> tuple[Mapping[str, str], list[str]]:
    if value is None:
        return MappingProxyType({}), []
    if not isinstance(value, Mapping):
        return MappingProxyType({}), ["mcap bundle.recommended_env must be a mapping"]

    normalized: dict[str, str] = {}
    errors: list[str] = []
    for key, item in value.items():
        if key != _SC_RECOMMENDATION_ENV:
            errors.append(
                "mcap bundle.recommended_env only supports "
                f"{_SC_RECOMMENDATION_ENV!r} in schema_version {SCHEMA_VERSION}"
            )
            continue
        try:
            string_value = _require_nonempty_text(item, f"mcap bundle.recommended_env[{key!r}]")
            parsed_value = float(string_value)
            if not math.isfinite(parsed_value) or parsed_value < 0.0:
                raise ValueError("nonfinite")
            normalized[key] = f"{parsed_value:.2f}"
        except (HarnessIOError, ValueError):
            errors.append(
                f"mcap bundle.recommended_env[{key!r}] must be a nonnegative decimal string"
            )
    return MappingProxyType(normalized), errors


def _expected_sc_recommended_stop_sec(
    trial: McapEvalTrialReport,
    *,
    contact_margin_sec: float,
    stop_step_sec: float,
) -> float | None:
    contact = trial.first_off_limit_contact
    if contact is None or contact.nearest_controller_elapsed_sec is None:
        return None
    if trial.task_hints is None or not _same(trial.task_hints.port_type, "sc"):
        return None
    return _floor_step(
        max(stop_step_sec, contact.nearest_controller_elapsed_sec - contact_margin_sec),
        stop_step_sec,
    )


def _floor_step(value: float, step: float) -> float:
    if step <= 0.0:
        return value
    if value <= 0.0:
        return 0.0
    return math.floor((value / step) + 1e-9) * step


def _validate_source_identity(value: Any, field_name: str) -> str:
    source = _require_nonempty_text(value, field_name)
    if _is_uri_source(source):
        parsed = urlparse(source)
        if parsed.scheme == "file":
            if not parsed.path:
                raise HarnessIOError(f"{field_name} must include a file path")
        elif not parsed.netloc:
            raise HarnessIOError(f"{field_name} must include a network location")
        if _has_dot_path_segments(_uri_path_segments(parsed.path)):
            raise HarnessIOError(f"{field_name} must not contain '.' or '..' path segments")
        return source
    if not _is_absolute_local_source(source):
        raise HarnessIOError(f"{field_name} must be an absolute local path or uri")
    if _has_dot_path_segments(_local_source_path(source).parts):
        raise HarnessIOError(f"{field_name} must not contain '.' or '..' path segments")
    return source


def _trial_index(trial_id: str) -> int:
    match = _TRIAL_ID_RE.fullmatch(trial_id)
    if match is None:
        raise HarnessIOError("mcap trial.trial_id must match the pattern 'trial_<positive integer>'")
    return int(match.group(1))


def _any_set(*values: Any) -> bool:
    return any(value is not None for value in values)


def _all_set(*values: Any) -> bool:
    return all(value is not None for value in values)


def _is_uri_source(source: str) -> bool:
    return bool(_URI_SOURCE_RE.match(source))


def _is_absolute_local_source(source: str) -> bool:
    return Path(source).is_absolute() or _is_windows_absolute_local_source(source)


def _is_windows_absolute_local_source(source: str) -> bool:
    return PureWindowsPath(source).is_absolute()


def _local_source_path(source: str) -> PurePosixPath | PureWindowsPath:
    if _is_windows_absolute_local_source(source):
        return PureWindowsPath(source)
    return PurePosixPath(source)


def _uri_source_is_within_bundle(bundle_source: str, trial_source: str) -> bool:
    bundle_uri = urlparse(bundle_source)
    trial_uri = urlparse(trial_source)
    if bundle_uri.scheme != trial_uri.scheme or bundle_uri.netloc != trial_uri.netloc:
        return False
    bundle_segments = _uri_path_segments(bundle_uri.path)
    trial_segments = _uri_path_segments(trial_uri.path)
    return (
        len(trial_segments) > len(bundle_segments)
        and trial_segments[: len(bundle_segments)] == bundle_segments
    )


def _uri_path_segments(path: str) -> tuple[str, ...]:
    return tuple(segment for segment in path.split("/") if segment)


def _has_dot_path_segments(parts: tuple[str, ...]) -> bool:
    return any(part in (".", "..") for part in parts)


def _same(left: Any, right: str) -> bool:
    if left is None:
        return False
    return str(left).strip().lower() == right
