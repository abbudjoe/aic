"""Typed promotion and baseline gate contracts for the backend-neutral harness."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, TypeVar

from aic_signal_harness.artifacts import HarnessIOError, read_json, write_json
from aic_signal_harness.schemas import ArtifactRef, SCHEMA_VERSION


_PROMOTION_METRIC_KEYS = frozenset(
    {"name", "goal", "value", "baseline_value", "min_improvement"}
)
_PROMOTION_DECISION_KEYS = frozenset(
    {
        "schema_version",
        "decided_at_utc",
        "decision",
        "accepted",
        "eligible_for_submission",
        "reason",
        "metric",
        "run_id",
        "manifest",
        "baseline_run_id",
        "baseline_manifest",
        "notes",
    }
)


class _StrEnum(str, Enum):
    @classmethod
    def parse(cls: type["_EnumT"], value: str | "_EnumT") -> "_EnumT":
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise HarnessIOError(f"{cls.__name__} must be a string")
        try:
            return cls(value)
        except ValueError as exc:
            allowed = ", ".join(item.value for item in cls)
            raise HarnessIOError(
                f"unknown {cls.__name__} {value!r}; expected one of: {allowed}"
            ) from exc


_EnumT = TypeVar("_EnumT", bound=_StrEnum)


class PromotionGoal(_StrEnum):
    max = "max"
    min = "min"


class PromotionDecisionKind(_StrEnum):
    bootstrap = "bootstrap"
    accepted = "accepted"
    rejected = "rejected"


@dataclass(frozen=True)
class PromotionMetric:
    """Metric comparison snapshot recorded by the promotion gate."""

    name: str
    goal: PromotionGoal
    value: float
    baseline_value: float | None = None
    min_improvement: float = 0.0

    def __post_init__(self) -> None:
        errors: list[str] = []
        try:
            object.__setattr__(
                self,
                "name",
                _require_nonempty_text(self.name, "promotion metric.name"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(self, "goal", PromotionGoal.parse(self.goal))
        except HarnessIOError as exc:
            errors.append(str(exc))
        for field_name in ("value", "baseline_value", "min_improvement"):
            raw_value = getattr(self, field_name)
            try:
                if field_name == "baseline_value":
                    normalized = _optional_finite_float(
                        raw_value,
                        f"promotion metric.{field_name}",
                    )
                else:
                    normalized = _require_finite_float(
                        raw_value,
                        f"promotion metric.{field_name}",
                    )
                object.__setattr__(self, field_name, normalized)
            except HarnessIOError as exc:
                errors.append(str(exc))
        if not errors and self.min_improvement < 0.0:
            errors.append("promotion metric.min_improvement must be nonnegative")
        if errors:
            raise HarnessIOError("; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "goal": self.goal.value,
            "value": self.value,
            "baseline_value": self.baseline_value,
            "min_improvement": self.min_improvement,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PromotionMetric":
        if not isinstance(value, Mapping):
            raise HarnessIOError("promotion metric must be a mapping")
        _reject_unknown_keys(value, _PROMOTION_METRIC_KEYS, "promotion metric")
        return cls(
            name=value.get("name"),
            goal=value.get("goal"),
            value=value.get("value"),
            baseline_value=value.get("baseline_value"),
            min_improvement=value.get("min_improvement", 0.0),
        )


@dataclass(frozen=True)
class PromotionDecision:
    """Typed result of comparing one candidate against the current baseline."""

    decided_at_utc: str
    decision: PromotionDecisionKind
    accepted: bool
    eligible_for_submission: bool
    reason: str
    metric: PromotionMetric
    run_id: str
    manifest: ArtifactRef
    baseline_run_id: str | None = None
    baseline_manifest: ArtifactRef | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        errors: list[str] = []
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            errors.append(f"schema_version must be {SCHEMA_VERSION}")
        try:
            object.__setattr__(
                self,
                "decided_at_utc",
                _require_nonempty_text(self.decided_at_utc, "promotion decision.decided_at_utc"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(self, "decision", PromotionDecisionKind.parse(self.decision))
        except HarnessIOError as exc:
            errors.append(str(exc))
        if type(self.accepted) is not bool:
            errors.append("promotion decision.accepted must be a boolean")
        if type(self.eligible_for_submission) is not bool:
            errors.append("promotion decision.eligible_for_submission must be a boolean")
        try:
            object.__setattr__(
                self,
                "reason",
                _require_nonempty_text(self.reason, "promotion decision.reason"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(self, "run_id", _require_nonempty_text(self.run_id, "promotion decision.run_id"))
        except HarnessIOError as exc:
            errors.append(str(exc))

        if not isinstance(self.metric, PromotionMetric):
            try:
                object.__setattr__(self, "metric", PromotionMetric.from_dict(self.metric))
            except HarnessIOError as exc:
                errors.append(str(exc))

        if not isinstance(self.manifest, ArtifactRef):
            try:
                object.__setattr__(self, "manifest", ArtifactRef.from_dict(self.manifest))
            except Exception as exc:  # pragma: no cover - ArtifactRef owns detail
                errors.append(str(exc))
        if self.baseline_manifest is not None and not isinstance(self.baseline_manifest, ArtifactRef):
            try:
                object.__setattr__(
                    self,
                    "baseline_manifest",
                    ArtifactRef.from_dict(self.baseline_manifest),
                )
            except Exception as exc:  # pragma: no cover - ArtifactRef owns detail
                errors.append(str(exc))

        for field_name in ("manifest", "baseline_manifest"):
            artifact = getattr(self, field_name)
            if artifact is None or not isinstance(artifact, ArtifactRef):
                continue
            if artifact.kind != "run_manifest":
                errors.append(f"promotion decision.{field_name}.kind must be 'run_manifest'")
            if artifact.sha256 is None:
                errors.append(f"promotion decision.{field_name}.sha256 must be set")

        try:
            object.__setattr__(
                self,
                "baseline_run_id",
                _optional_nonempty_text(
                    self.baseline_run_id,
                    "promotion decision.baseline_run_id",
                ),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))

        try:
            object.__setattr__(
                self,
                "notes",
                _as_text_sequence(self.notes, "promotion decision.notes"),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))

        if (
            (self.baseline_run_id is None) != (self.baseline_manifest is None)
        ):
            errors.append(
                "promotion decision.baseline_run_id and baseline_manifest must be set together"
            )
        metric = self.metric if isinstance(self.metric, PromotionMetric) else None
        manifest = self.manifest if isinstance(self.manifest, ArtifactRef) else None
        baseline_manifest = (
            self.baseline_manifest
            if isinstance(self.baseline_manifest, ArtifactRef)
            else None
        )
        if (
            self.decision is PromotionDecisionKind.bootstrap
            and metric is not None
            and metric.baseline_value is not None
        ):
            errors.append("bootstrap promotion decisions must not set metric.baseline_value")
        if self.decision is PromotionDecisionKind.bootstrap and baseline_manifest is not None:
            errors.append("bootstrap promotion decisions must not set a baseline manifest")
        if (
            self.decision is not PromotionDecisionKind.bootstrap
            and metric is not None
            and metric.baseline_value is None
        ):
            errors.append(
                "accepted/rejected promotion decisions must set metric.baseline_value"
            )
        if (
            self.decision is not PromotionDecisionKind.bootstrap
            and baseline_manifest is None
        ):
            errors.append(
                "accepted/rejected promotion decisions must set baseline_run_id and baseline_manifest"
            )
        if (
            self.baseline_run_id is not None
            and self.baseline_run_id == self.run_id
        ):
            errors.append(
                "promotion decisions must not compare a run against itself"
            )
        if (
            manifest is not None
            and baseline_manifest is not None
            and manifest.sha256 == baseline_manifest.sha256
        ):
            errors.append(
                "promotion decisions must not compare a run against itself"
            )
        if self.decision is PromotionDecisionKind.rejected and self.accepted:
            errors.append("rejected promotion decisions must set accepted=false")
        if self.decision in (PromotionDecisionKind.accepted, PromotionDecisionKind.bootstrap) and not self.accepted:
            errors.append("accepted/bootstrap promotion decisions must set accepted=true")
        if not self.accepted and self.eligible_for_submission:
            errors.append(
                "promotion decisions with accepted=false must set eligible_for_submission=false"
            )
        if (
            self.decision is not PromotionDecisionKind.bootstrap
            and metric is not None
            and metric.baseline_value is not None
        ):
            expected_accepted = _promotion_metric_accepts(metric)
            if expected_accepted != self.accepted:
                errors.append(
                    "promotion decision acceptance does not match metric comparison"
                )
        if errors:
            raise HarnessIOError("; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "decided_at_utc": self.decided_at_utc,
            "decision": self.decision.value,
            "accepted": self.accepted,
            "eligible_for_submission": self.eligible_for_submission,
            "reason": self.reason,
            "metric": self.metric.to_dict(),
            "run_id": self.run_id,
            "manifest": self.manifest.to_dict(),
            "baseline_run_id": self.baseline_run_id,
            "baseline_manifest": None if self.baseline_manifest is None else self.baseline_manifest.to_dict(),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PromotionDecision":
        if not isinstance(value, Mapping):
            raise HarnessIOError("promotion decision must be a mapping")
        _reject_unknown_keys(value, _PROMOTION_DECISION_KEYS, "promotion decision")
        return cls(
            schema_version=value.get("schema_version"),
            decided_at_utc=value.get("decided_at_utc"),
            decision=value.get("decision"),
            accepted=value.get("accepted"),
            eligible_for_submission=value.get("eligible_for_submission"),
            reason=value.get("reason"),
            metric=value.get("metric"),
            run_id=value.get("run_id"),
            manifest=value.get("manifest"),
            baseline_run_id=value.get("baseline_run_id"),
            baseline_manifest=value.get("baseline_manifest"),
            notes=value.get("notes", ()),
        )


def read_promotion_decision(path: str) -> PromotionDecision:
    """Read one persisted promotion decision from JSON."""

    return PromotionDecision.from_dict(read_json(path))


def write_promotion_decision(
    path: str,
    decision: PromotionDecision | Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    """Persist one promotion decision as JSON."""

    typed_decision = (
        decision if isinstance(decision, PromotionDecision) else PromotionDecision.from_dict(decision)
    )
    write_json(path, typed_decision.to_dict(), overwrite=overwrite)


def write_baseline_decision(
    path: str,
    decision: PromotionDecision | Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    """Persist an accepted promotion decision as the current baseline."""

    typed_decision = (
        decision if isinstance(decision, PromotionDecision) else PromotionDecision.from_dict(decision)
    )
    if not typed_decision.accepted:
        raise HarnessIOError("baseline decisions must be accepted")
    write_promotion_decision(path, typed_decision, overwrite=overwrite)


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


def _require_finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise HarnessIOError(f"{field_name} must be a finite number")
    return float(value)


def _optional_finite_float(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    return _require_finite_float(value, field_name)


def _as_text_sequence(value: Any, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, (list, tuple)):
        raise HarnessIOError(f"{field_name} must be a list or tuple")
    return tuple(
        _require_nonempty_text(item, f"{field_name}[{index}]")
        for index, item in enumerate(value)
    )


def _promotion_metric_accepts(metric: PromotionMetric) -> bool:
    if metric.goal is PromotionGoal.max:
        return metric.value >= metric.baseline_value + metric.min_improvement
    return metric.value <= metric.baseline_value - metric.min_improvement
