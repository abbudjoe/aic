"""Typed parser for official AIC ``scoring.yaml`` artifacts."""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from aic_signal_harness.artifacts import HarnessIOError
from aic_signal_harness.schemas import SCHEMA_VERSION


_SCORE_REPORT_KEYS = frozenset(
    {"schema_version", "source", "parsed_at_utc", "total", "trial_count", "trials"}
)
_TRIAL_SCORE_KEYS = frozenset({"total", "tier_1", "tier_2", "tier_3"})
_TIER_NAMES = ("tier_1", "tier_2", "tier_3")
_TOTAL_TOLERANCE = 1e-9
_OFFICIAL_TRIAL_ID_RE = re.compile(r"^trial_[1-9][0-9]*$")


@dataclass(frozen=True)
class TrialScore:
    """Normalized official score terms for one trial."""

    total: float
    tier_1: float | None = None
    tier_2: float | None = None
    tier_3: float | None = None
    _allow_partial: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        errors: list[str] = []
        total: float | None = None
        try:
            total = _required_float(self.total, "trial.total")
            object.__setattr__(self, "total", total)
        except HarnessIOError as exc:
            errors.append(str(exc))
        tier_scores: list[float | None] = []
        for tier_name in _TIER_NAMES:
            try:
                tier_score = _optional_float(getattr(self, tier_name), f"trial.{tier_name}")
                object.__setattr__(
                    self,
                    tier_name,
                    tier_score,
                )
                tier_scores.append(tier_score)
            except HarnessIOError as exc:
                errors.append(str(exc))
                object.__setattr__(self, tier_name, None)
                tier_scores.append(None)
        if not errors and any(score is None for score in tier_scores):
            if not self._allow_partial:
                errors.append("trial scores must include tier_1, tier_2, and tier_3")
            elif total is not None and not _totals_match(
                total,
                sum(score for score in tier_scores if score is not None),
            ):
                errors.append("trial.total does not match sum of provided tier scores")
        elif not errors and total is not None and not _totals_match(
            total,
            sum(score for score in tier_scores if score is not None),
        ):
            errors.append("trial.total does not match sum of tier scores")
        if errors:
            raise HarnessIOError("; ".join(errors))

    def to_dict(self) -> dict[str, float | None]:
        return {
            "total": self.total,
            "tier_1": self.tier_1,
            "tier_2": self.tier_2,
            "tier_3": self.tier_3,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrialScore":
        if not isinstance(value, Mapping):
            raise HarnessIOError("trial score must be a mapping")
        _reject_unknown_keys(value, _TRIAL_SCORE_KEYS, "trial score")
        return cls(
            total=value.get("total"),
            tier_1=value.get("tier_1"),
            tier_2=value.get("tier_2"),
            tier_3=value.get("tier_3"),
        )

    @classmethod
    def from_legacy_v1_dict(cls, value: Mapping[str, Any]) -> "TrialScore":
        if not isinstance(value, Mapping):
            raise HarnessIOError("trial score must be a mapping")
        _reject_unknown_keys(value, _TRIAL_SCORE_KEYS, "trial score")
        normalized_tiers = {
            tier_name: 0.0 if value.get(tier_name) is None else value.get(tier_name)
            for tier_name in _TIER_NAMES
        }
        return cls(
            total=value.get("total"),
            tier_1=normalized_tiers["tier_1"],
            tier_2=normalized_tiers["tier_2"],
            tier_3=normalized_tiers["tier_3"],
        )


@dataclass(frozen=True)
class ScoreReport:
    """Normalized report parsed from official AIC scoring output."""

    source: str
    total: float
    trials: Mapping[str, TrialScore] = field(default_factory=dict)
    parsed_at_utc: str | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        errors: list[str] = []
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            errors.append(f"schema_version must be {SCHEMA_VERSION}")
        if not isinstance(self.source, str) or not self.source.strip():
            errors.append("score.source must be a nonempty string")
        if self.parsed_at_utc is not None and (
            not isinstance(self.parsed_at_utc, str) or not self.parsed_at_utc.strip()
        ):
            errors.append("score.parsed_at_utc must be a nonempty string when set")
        try:
            object.__setattr__(self, "total", _required_float(self.total, "score.total"))
        except HarnessIOError as exc:
            errors.append(str(exc))

        trials: dict[str, TrialScore] = {}
        if not isinstance(self.trials, Mapping):
            errors.append("score.trials must be a mapping")
        else:
            if not self.trials:
                errors.append("score.trials must contain at least one trial")
            for trial_name, trial_score in self.trials.items():
                try:
                    official_trial_name = _official_trial_name(trial_name)
                    trials[official_trial_name] = (
                        trial_score
                        if isinstance(trial_score, TrialScore)
                        else TrialScore.from_dict(trial_score)
                    )
                except HarnessIOError as exc:
                    errors.append(str(exc))
        if not errors:
            trial_total = sum(trial_score.total for trial_score in trials.values())
            if not _totals_match(self.total, trial_total):
                errors.append(
                    "score.total does not match sum of trial totals: "
                    f"{self.total} != {trial_total}"
                )
        if errors:
            raise HarnessIOError("; ".join(errors))
        object.__setattr__(self, "trials", MappingProxyType(trials))

    @property
    def trial_count(self) -> int:
        return len(self.trials)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source": self.source,
            "parsed_at_utc": self.parsed_at_utc,
            "total": self.total,
            "trial_count": self.trial_count,
            "trials": {
                trial_name: trial_score.to_dict()
                for trial_name, trial_score in self.trials.items()
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ScoreReport":
        if not isinstance(value, Mapping):
            raise HarnessIOError("score report must be a mapping")
        _reject_unknown_keys(value, _SCORE_REPORT_KEYS, "score report")
        report = cls(
            schema_version=value.get("schema_version"),
            source=value.get("source"),
            parsed_at_utc=value.get("parsed_at_utc"),
            total=value.get("total"),
            trials=value.get("trials", {}),
        )
        declared_trial_count = value.get("trial_count")
        if type(declared_trial_count) is not int:
            raise HarnessIOError("score report trial_count must be an integer")
        if declared_trial_count != report.trial_count:
            raise HarnessIOError("score report trial_count does not match trials")
        return report

    @classmethod
    def from_legacy_v1_dict(cls, value: Mapping[str, Any]) -> "ScoreReport":
        if not isinstance(value, Mapping):
            raise HarnessIOError("score report must be a mapping")
        _reject_unknown_keys(value, _SCORE_REPORT_KEYS, "score report")
        raw_trials = value.get("trials", {})
        if not isinstance(raw_trials, Mapping):
            raise HarnessIOError("score.trials must be a mapping")
        report = cls(
            schema_version=value.get("schema_version"),
            source=value.get("source"),
            parsed_at_utc=value.get("parsed_at_utc"),
            total=value.get("total"),
            trials={
                _official_trial_name(trial_name): TrialScore.from_legacy_v1_dict(trial_score)
                for trial_name, trial_score in raw_trials.items()
            },
        )
        declared_trial_count = value.get("trial_count")
        if type(declared_trial_count) is not int:
            raise HarnessIOError("score report trial_count must be an integer")
        if declared_trial_count != report.trial_count:
            raise HarnessIOError("score report trial_count does not match trials")
        return report

    def equivalent_to(
        self,
        other: object,
        *,
        source_aliases: Iterable[str] = (),
    ) -> bool:
        """Return whether two reports encode the same score state."""

        if not isinstance(other, ScoreReport):
            return False
        if self.schema_version != other.schema_version:
            return False
        if not _score_sources_match(self.source, other.source, source_aliases):
            return False
        if not _totals_match(self.total, other.total):
            return False
        if self.trials.keys() != other.trials.keys():
            return False
        return all(
            _trial_scores_equivalent(self.trials[trial_name], other.trials[trial_name])
            for trial_name in self.trials
        )


def parse_scoring_yaml(path: str | Path, *, parsed_at_utc: str | None = None) -> ScoreReport:
    """Parse official AIC ``scoring.yaml`` into a typed score report."""

    raw_bytes, source = read_scoring_yaml_snapshot(path)
    return parse_scoring_yaml_bytes(
        raw_bytes,
        source=source,
        parsed_at_utc=parsed_at_utc,
    )


def read_scoring_yaml_bytes(path: str | Path) -> bytes:
    """Read official ``scoring.yaml`` bytes from disk."""

    return read_scoring_yaml_snapshot(path)[0]


def read_scoring_yaml_snapshot(path: str | Path) -> tuple[bytes, str]:
    """Read scoring bytes and bind them to a stable canonical local path."""

    scoring_path = Path(path)
    try:
        resolved_path = scoring_path.resolve()
    except FileNotFoundError as exc:
        raise HarnessIOError(f"scoring.yaml not found: {scoring_path}") from exc
    except OSError as exc:
        raise HarnessIOError(f"failed to resolve scoring.yaml {scoring_path}: {exc}") from exc

    try:
        with scoring_path.open("rb") as handle:
            raw_bytes = handle.read()
            opened_stat = os.fstat(handle.fileno())
    except FileNotFoundError as exc:
        raise HarnessIOError(f"scoring.yaml not found: {scoring_path}") from exc
    except OSError as exc:
        raise HarnessIOError(f"failed to read scoring.yaml {scoring_path}: {exc}") from exc
    try:
        resolved_stat = resolved_path.stat()
    except OSError as exc:
        raise HarnessIOError(f"scoring.yaml changed while reading {scoring_path}: {exc}") from exc

    if (
        opened_stat.st_dev != resolved_stat.st_dev
        or opened_stat.st_ino != resolved_stat.st_ino
    ):
        raise HarnessIOError(f"scoring.yaml changed while reading: {scoring_path}")
    return raw_bytes, str(resolved_path)


def parse_scoring_yaml_bytes(
    raw_bytes: bytes,
    *,
    source: str,
    parsed_at_utc: str | None = None,
) -> ScoreReport:
    """Parse loaded official ``scoring.yaml`` bytes into a typed score report."""

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on runtime env
        raise HarnessIOError("parse_scoring_yaml requires PyYAML to read scoring.yaml") from exc

    class _UniqueKeySafeLoader(yaml.SafeLoader):
        pass

    def _construct_unique_mapping(loader, node, deep=False):
        mapping: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            try:
                key_seen = key in mapping
            except TypeError as exc:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found unhashable key {key!r}",
                    key_node.start_mark,
                ) from exc
            if key_seen:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key {key!r}",
                    key_node.start_mark,
                )
            try:
                mapping[key] = loader.construct_object(value_node, deep=deep)
            except TypeError as exc:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found unhashable key {key!r}",
                    key_node.start_mark,
                ) from exc
        return mapping

    _UniqueKeySafeLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        _construct_unique_mapping,
    )

    try:
        scoring = yaml.load(raw_bytes, Loader=_UniqueKeySafeLoader)
    except UnicodeDecodeError as exc:
        raise HarnessIOError(f"failed to decode scoring.yaml {source}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise HarnessIOError(f"invalid scoring.yaml {source}: {exc}") from exc

    if not isinstance(scoring, Mapping):
        raise HarnessIOError(f"scoring.yaml must contain a top-level mapping: {source}")
    if "total" not in scoring:
        raise HarnessIOError("scoring.yaml is missing required key: total")

    trials: dict[str, TrialScore] = {}
    for key, value in scoring.items():
        if key == "total":
            continue
        trial_name = _official_trial_name(key)
        if not isinstance(value, Mapping):
            raise HarnessIOError(f"trial {trial_name!r} must be a mapping")
        _reject_unknown_keys(value, frozenset(_TIER_NAMES), f"trial {trial_name!r}")
        tier_scores = {
            tier_name: _tier_score(value, tier_name, trial_name) for tier_name in _TIER_NAMES
        }
        trials[trial_name] = TrialScore(
            total=sum(tier_scores.values()),
            **tier_scores,
        )

    total = _required_float(scoring["total"], "score.total")
    parsed_trial_total = sum(trial.total for trial in trials.values())
    if not _totals_match(total, parsed_trial_total):
        raise HarnessIOError(
            "score.total does not match sum of trial totals: "
            f"{total} != {parsed_trial_total}"
        )

    return ScoreReport(
        source=source,
        total=total,
        trials=trials,
        parsed_at_utc=parsed_at_utc,
    )


def _official_trial_name(key: Any) -> str:
    if not isinstance(key, str) or not key.strip():
        raise HarnessIOError("scoring.yaml trial keys must be nonempty trial_* strings")
    if not _OFFICIAL_TRIAL_ID_RE.match(key):
        raise HarnessIOError("scoring.yaml trial keys must be official trial_* ids")
    return key


def _tier_score(trial: Mapping[str, Any], tier_name: str, trial_name: str) -> float:
    if tier_name not in trial:
        raise HarnessIOError(f"trial {trial_name!r} must contain {tier_name}")
    value = trial[tier_name]
    field_name = f"trial {trial_name!r} {tier_name}"
    if not isinstance(value, Mapping):
        raise HarnessIOError(f"{field_name} must be a mapping")
    if "score" not in value:
        raise HarnessIOError(f"{field_name} must contain score")
    return _required_float(value["score"], f"{field_name}.score")


def _totals_match(total: float, trial_total: float) -> bool:
    return math.isclose(
        total,
        trial_total,
        rel_tol=_TOTAL_TOLERANCE,
        abs_tol=_TOTAL_TOLERANCE,
    )


def _score_sources_match(
    left_source: str,
    right_source: str,
    source_aliases: Iterable[str],
) -> bool:
    if left_source == right_source:
        return True
    aliases = frozenset(
        alias for alias in source_aliases if isinstance(alias, str) and alias.strip()
    )
    return left_source in aliases and right_source in aliases


def _trial_scores_equivalent(left: TrialScore, right: TrialScore) -> bool:
    return (
        _totals_match(left.total, right.total)
        and _optional_scores_match(left.tier_1, right.tier_1)
        and _optional_scores_match(left.tier_2, right.tier_2)
        and _optional_scores_match(left.tier_3, right.tier_3)
    )


def _optional_scores_match(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left in (None, 0.0) and right in (None, 0.0)
    return _totals_match(left, right)


def _required_float(value: Any, field_name: str) -> float:
    score = _optional_float(value, field_name)
    if score is None:
        raise HarnessIOError(f"{field_name} must be a finite number")
    return score


def _optional_float(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HarnessIOError(f"{field_name} must be a finite number")
    score = float(value)
    if not math.isfinite(score):
        raise HarnessIOError(f"{field_name} must be a finite number")
    return score


def _reject_unknown_keys(
    value: Mapping[str, Any],
    allowed_keys: frozenset[str],
    field_name: str,
) -> None:
    unknown_keys = sorted(repr(key) for key in value if key not in allowed_keys)
    if unknown_keys:
        raise HarnessIOError(f"{field_name} has unknown fields: {', '.join(unknown_keys)}")
