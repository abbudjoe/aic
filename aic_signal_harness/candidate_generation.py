"""Generate executable candidate registries from next-experiment plans."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, cast

from aic_signal_harness.artifacts import (
    HarnessIOError,
    local_artifact_path,
    read_json,
    sha256_file,
)
from aic_signal_harness.candidate_registry import (
    CandidateCheckpointExpectation,
    CandidateEvalMode,
    CandidateEvalSpec,
    CandidatePolicyRecord,
    CandidatePolicyRegistry,
    CandidateSafetyContract,
)
from aic_signal_harness.next_experiment_plan import (
    NextExperimentExecutionMode,
    NextExperimentPlan,
    NextExperimentPlanCandidate,
    NextExperimentPlanStatus,
)
from aic_signal_harness.policy_training import TrainerInvocation
from aic_signal_harness.policy_training_runner import CandidatePolicyTemplate
from aic_signal_harness.reducers.policy_training import read_training_dataset_report_artifact
from aic_signal_harness.schemas import ArtifactRef


NEXT_EXPERIMENT_PLAN_KIND = "next_experiment_plan"

_GENERATION_CONFIG_KEYS = frozenset(
    {
        "registry_id",
        "generated_at_utc",
        "selected_plan_candidate_id",
        "eval_mode",
        "min_improvement",
        "eligible_for_submission",
        "notes",
    }
)


@dataclass(frozen=True)
class CandidateGenerationConfig:
    """Operator-chosen knobs for one candidate-registry generation."""

    registry_id: str
    generated_at_utc: str
    selected_plan_candidate_id: str
    eval_mode: CandidateEvalMode = CandidateEvalMode.local_live_eval
    min_improvement: float = 0.0
    eligible_for_submission: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        errors: list[str] = []
        for field_name in ("registry_id", "generated_at_utc", "selected_plan_candidate_id"):
            try:
                object.__setattr__(
                    self,
                    field_name,
                    _require_nonempty_text(
                        getattr(self, field_name),
                        f"candidate generation.{field_name}",
                    ),
                )
            except HarnessIOError as exc:
                errors.append(str(exc))
        try:
            object.__setattr__(self, "eval_mode", CandidateEvalMode.parse(self.eval_mode))
        except HarnessIOError as exc:
            errors.append(str(exc))
        try:
            object.__setattr__(
                self,
                "min_improvement",
                _require_nonnegative_finite_float(
                    self.min_improvement,
                    "candidate generation.min_improvement",
                ),
            )
        except HarnessIOError as exc:
            errors.append(str(exc))
        if type(self.eligible_for_submission) is not bool:
            errors.append("candidate generation.eligible_for_submission must be a boolean")
        try:
            object.__setattr__(self, "notes", _as_text_sequence(self.notes, "candidate generation.notes"))
        except HarnessIOError as exc:
            errors.append(str(exc))
        if errors:
            raise HarnessIOError("; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        return {
            "registry_id": self.registry_id,
            "generated_at_utc": self.generated_at_utc,
            "selected_plan_candidate_id": self.selected_plan_candidate_id,
            "eval_mode": self.eval_mode.value,
            "min_improvement": self.min_improvement,
            "eligible_for_submission": self.eligible_for_submission,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CandidateGenerationConfig":
        if not isinstance(value, Mapping):
            raise HarnessIOError("candidate generation config must be a mapping")
        _reject_unknown_keys(value, _GENERATION_CONFIG_KEYS, "candidate generation config")
        _require_keys(
            value,
            _GENERATION_CONFIG_KEYS
            - {"eligible_for_submission", "eval_mode", "min_improvement", "notes"},
            "candidate generation config",
        )
        return cls(
            registry_id=cast(Any, value.get("registry_id")),
            generated_at_utc=cast(Any, value.get("generated_at_utc")),
            selected_plan_candidate_id=cast(Any, value.get("selected_plan_candidate_id")),
            eval_mode=cast(Any, value.get("eval_mode", CandidateEvalMode.local_live_eval.value)),
            min_improvement=cast(Any, value.get("min_improvement", 0.0)),
            eligible_for_submission=cast(Any, value.get("eligible_for_submission", False)),
            notes=cast(Any, value.get("notes", ())),
        )


def derive_candidate_policy_registry_from_plan(
    *,
    source_next_experiment_plan: ArtifactRef | Mapping[str, Any],
    source_training_dataset_report: ArtifactRef | Mapping[str, Any],
    trainer: TrainerInvocation | Mapping[str, Any],
    policy_template: CandidatePolicyTemplate | Mapping[str, Any],
    config: CandidateGenerationConfig | Mapping[str, Any],
    checkpoint: CandidateCheckpointExpectation | Mapping[str, Any] | None = None,
    safety: CandidateSafetyContract | Mapping[str, Any] | None = None,
) -> CandidatePolicyRegistry:
    """Create one selected candidate registry from one executable plan candidate."""

    typed_config = (
        config if isinstance(config, CandidateGenerationConfig) else CandidateGenerationConfig.from_dict(config)
    )
    plan_artifact = (
        source_next_experiment_plan
        if isinstance(source_next_experiment_plan, ArtifactRef)
        else ArtifactRef.from_dict(source_next_experiment_plan)
    )
    dataset_artifact = (
        source_training_dataset_report
        if isinstance(source_training_dataset_report, ArtifactRef)
        else ArtifactRef.from_dict(source_training_dataset_report)
    )
    typed_trainer = trainer if isinstance(trainer, TrainerInvocation) else TrainerInvocation.from_dict(trainer)
    typed_template = (
        policy_template
        if isinstance(policy_template, CandidatePolicyTemplate)
        else CandidatePolicyTemplate.from_dict(policy_template)
    )
    typed_checkpoint = (
        CandidateCheckpointExpectation()
        if checkpoint is None
        else checkpoint
        if isinstance(checkpoint, CandidateCheckpointExpectation)
        else CandidateCheckpointExpectation.from_dict(checkpoint)
    )
    typed_safety = (
        CandidateSafetyContract()
        if safety is None
        else safety
        if isinstance(safety, CandidateSafetyContract)
        else CandidateSafetyContract.from_dict(safety)
    )
    plan = read_next_experiment_plan_artifact(plan_artifact)
    plan_candidate = _selected_plan_candidate(
        plan,
        selected_plan_candidate_id=typed_config.selected_plan_candidate_id,
    )
    read_training_dataset_report_artifact(dataset_artifact)
    eval_spec = CandidateEvalSpec(
        mode=typed_config.eval_mode,
        gate_id=plan.gate_id,
        planner_mode=_planner_mode(typed_template),
        min_improvement=typed_config.min_improvement,
        eligible_for_submission=typed_config.eligible_for_submission,
        notes=(
            f"Derived from next experiment plan candidate {plan_candidate.candidate_id}.",
        ),
    )
    candidate = CandidatePolicyRecord(
        generated_at_utc=typed_config.generated_at_utc,
        source_training_dataset_report=dataset_artifact,
        trainer=typed_trainer,
        policy_template=typed_template,
        checkpoint=typed_checkpoint,
        eval=eval_spec,
        safety=typed_safety,
        source_next_experiment_plan=plan_artifact,
        source_plan_candidate_id=plan_candidate.candidate_id,
        notes=typed_config.notes
        + (
            f"Generated from next experiment plan {plan_candidate.plan_id}.",
            "This record is executable training identity only; it does not launch training or eval.",
        ),
    )
    assert candidate.candidate_id is not None
    return CandidatePolicyRegistry(
        registry_id=typed_config.registry_id,
        generated_at_utc=typed_config.generated_at_utc,
        candidates=(candidate,),
        selected_candidate_id=candidate.candidate_id,
        notes=typed_config.notes
        + (
            f"Selected plan candidate {plan_candidate.candidate_id}.",
        ),
    )


def read_next_experiment_plan_artifact(
    artifact: ArtifactRef | Mapping[str, Any],
) -> NextExperimentPlan:
    """Read and validate a local next-experiment-plan artifact."""

    typed_artifact = artifact if isinstance(artifact, ArtifactRef) else ArtifactRef.from_dict(artifact)
    errors: list[str] = []
    if typed_artifact.kind != NEXT_EXPERIMENT_PLAN_KIND:
        errors.append(f"next experiment plan artifact.kind must be {NEXT_EXPERIMENT_PLAN_KIND!r}")
    if typed_artifact.sha256 is None:
        errors.append("next experiment plan artifact.sha256 must be set")
    for provenance_key in ("producer", "derivation"):
        if provenance_key not in typed_artifact.provenance:
            errors.append(f"next experiment plan artifact provenance must set {provenance_key}")
        else:
            try:
                _require_nonempty_text(
                    typed_artifact.provenance.get(provenance_key),
                    f"next experiment plan artifact provenance.{provenance_key}",
                )
            except HarnessIOError as exc:
                errors.append(str(exc))
    try:
        plan_path = local_artifact_path(
            path=typed_artifact.path,
            uri=typed_artifact.uri,
            field_name="next experiment plan artifact",
        )
    except HarnessIOError as exc:
        errors.append(str(exc))
        plan_path = None
    if plan_path is None:
        errors.append("next experiment plan artifact must be local byte-verifiable")
    elif not plan_path.exists():
        errors.append("next experiment plan artifact.path must exist")
    elif not plan_path.is_file():
        errors.append("next experiment plan artifact.path must be a file")
    elif typed_artifact.sha256 is not None and sha256_file(plan_path) != typed_artifact.sha256:
        errors.append("next experiment plan artifact.sha256 must match path")
    if errors:
        raise HarnessIOError("; ".join(errors))
    assert plan_path is not None
    plan = NextExperimentPlan.from_dict(read_json(plan_path))
    if typed_artifact.provenance.get("run_id") not in (None, plan.run_id):
        raise HarnessIOError("next experiment plan artifact provenance run_id must match plan.run_id")
    if typed_artifact.provenance.get("gate_id") not in (None, plan.gate_id):
        raise HarnessIOError("next experiment plan artifact provenance gate_id must match plan.gate_id")
    return plan


def _selected_plan_candidate(
    plan: NextExperimentPlan,
    *,
    selected_plan_candidate_id: str,
) -> NextExperimentPlanCandidate:
    for candidate in plan.candidates:
        if candidate.candidate_id == selected_plan_candidate_id:
            errors = _plan_candidate_errors(candidate)
            if errors:
                raise HarnessIOError("; ".join(errors))
            return candidate
    raise HarnessIOError("selected_plan_candidate_id must match a next experiment plan candidate")


def _plan_candidate_errors(candidate: NextExperimentPlanCandidate) -> list[str]:
    errors: list[str] = []
    if candidate.status is not NextExperimentPlanStatus.requires_implementation:
        errors.append(
            "candidate generation requires a plan candidate with status requires_implementation"
        )
    if candidate.execution_mode is not NextExperimentExecutionMode.policy_development:
        errors.append("candidate generation requires execution_mode policy_development")
    if candidate.launchable:
        errors.append("candidate generation plan candidate must not already be launchable")
    if candidate.autonomous_launch_allowed:
        errors.append("candidate generation plan candidate must not allow autonomous launch")
    return errors


def _planner_mode(template: CandidatePolicyTemplate) -> str:
    planner_mode = template.config.get("planner_mode")
    return _require_nonempty_text(planner_mode, "candidate policy template.config.planner_mode")


def _reject_unknown_keys(
    value: Mapping[str, Any],
    allowed_keys: frozenset[str],
    field_name: str,
) -> None:
    unknown_keys = sorted(repr(key) for key in value if key not in allowed_keys)
    if unknown_keys:
        raise HarnessIOError(f"{field_name} has unknown fields: {', '.join(unknown_keys)}")


def _require_keys(
    value: Mapping[str, Any],
    required_keys: frozenset[str],
    field_name: str,
) -> None:
    missing = sorted(key for key in required_keys if key not in value)
    if missing:
        raise HarnessIOError(f"{field_name} missing required fields: {', '.join(missing)}")


def _require_nonempty_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HarnessIOError(f"{field_name} must be a nonempty string")
    return value.strip()


def _require_nonnegative_finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HarnessIOError(f"{field_name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise HarnessIOError(f"{field_name} must be a finite number")
    if number < 0.0:
        raise HarnessIOError(f"{field_name} must be >= 0")
    return number


def _as_text_sequence(value: Any, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, (list, tuple)):
        raise HarnessIOError(f"{field_name} must be a list or tuple")
    return tuple(_require_nonempty_text(item, f"{field_name}[{index}]") for index, item in enumerate(value))
