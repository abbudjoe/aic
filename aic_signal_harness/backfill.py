"""Backfill adapters for legacy AIC LEWM experiment runs."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from aic_signal_harness.artifacts import (
    HarnessIOError,
    read_json,
    sha256_file,
    write_json,
)
from aic_signal_harness.ledger import LedgerEntry
from aic_signal_harness.manifest import RunManifest, RunStatus
from aic_signal_harness.promotion import (
    PromotionDecision,
    PromotionDecisionKind,
    PromotionGoal,
    PromotionMetric,
)
from aic_signal_harness.reducers import (
    attach_scoring_yaml_reduction,
    build_ledger_entry,
    reduce_hdf5_dataset,
    reduce_scoring_yaml,
)
from aic_signal_harness.reducers.hdf5_dataset import Hdf5DatasetReduction
from aic_signal_harness.reducers.scoring_yaml import ScoringYamlReduction
from aic_signal_harness.schemas import (
    ArtifactRef,
    BackendKind,
    LeakageClass,
    PolicyBackendSpec,
    RuntimeBoundaryProof,
    RuntimeRole,
    SCHEMA_VERSION,
    SchemaValidationError,
    SimulatorKind,
    TrainingSourceKind,
)
from aic_signal_harness.scoring import ScoreReport


_CANONICAL_REPO_ROOT = Path(__file__).resolve().parents[1]
_CANONICAL_EXPERIMENTS_ROOT = _CANONICAL_REPO_ROOT / "aic_lewm_policy" / "experiments"


@dataclass(frozen=True)
class LegacyRunBackfill:
    """Neutral harness objects reconstructed from one legacy run directory."""

    manifest: RunManifest
    manifest_artifact: ArtifactRef
    ledger_entry: LedgerEntry
    scoring_reduction: ScoringYamlReduction
    dataset_reduction: Hdf5DatasetReduction
    promotion_decision: PromotionDecision | None = None

    def __post_init__(self) -> None:
        errors: list[str] = []
        if not isinstance(self.manifest, RunManifest):
            errors.append("legacy run backfill manifest must be a RunManifest")
        if not isinstance(self.manifest_artifact, ArtifactRef):
            errors.append("legacy run backfill manifest_artifact must be an ArtifactRef")
        if not isinstance(self.ledger_entry, LedgerEntry):
            errors.append("legacy run backfill ledger_entry must be a LedgerEntry")
        if not isinstance(self.scoring_reduction, ScoringYamlReduction):
            errors.append(
                "legacy run backfill scoring_reduction must be a ScoringYamlReduction"
            )
        if not isinstance(self.dataset_reduction, Hdf5DatasetReduction):
            errors.append(
                "legacy run backfill dataset_reduction must be an Hdf5DatasetReduction"
            )
        if self.promotion_decision is not None and not isinstance(
            self.promotion_decision, PromotionDecision
        ):
            errors.append(
                "legacy run backfill promotion_decision must be a PromotionDecision when set"
            )
        if errors:
            raise HarnessIOError("; ".join(errors))


def backfill_legacy_replay_policy_eval(
    run_dir: str | Path,
    *,
    manifest_output_path: str | Path,
    baseline: PromotionDecision | Mapping[str, Any] | None = None,
    bootstrap: bool = False,
    repo_root: str | Path | None = None,
) -> LegacyRunBackfill:
    """Backfill one legacy ``replay_policy_eval`` run into neutral harness types."""

    run_path = _resolve_existing_dir(run_dir, "run_dir")
    repo_root_path = _resolve_repo_root(run_path, repo_root)
    legacy_manifest_path = run_path / "manifest.json"
    legacy_manifest = read_json(legacy_manifest_path)
    _require_schema_version(legacy_manifest, "legacy manifest")

    if legacy_manifest.get("kind") != "replay_policy_eval":
        raise HarnessIOError(
            "backfill_legacy_replay_policy_eval only supports legacy kind='replay_policy_eval'"
        )
    append_only_ledger = _current_legacy_ledger_entry(
        legacy_manifest,
        repo_root=repo_root_path,
    )
    canonical_legacy_manifest_path = _validate_current_legacy_manifest_binding(
        legacy_manifest,
        append_only_ledger=append_only_ledger,
        repo_root=repo_root_path,
    )

    backend = _legacy_backend_spec(legacy_manifest, canonical_legacy_manifest_path)
    scoring_reduction = _legacy_scoring_reduction(
        legacy_manifest,
        run_path=run_path,
        repo_root=repo_root_path,
        canonical_legacy_manifest_path=canonical_legacy_manifest_path,
    )
    dataset_reduction = _legacy_dataset_reduction(
        legacy_manifest,
        repo_root=repo_root_path,
    )
    generic_artifacts = _legacy_generic_artifacts(
        legacy_manifest,
        run_path=run_path,
        repo_root=repo_root_path,
        canonical_legacy_manifest_path=canonical_legacy_manifest_path,
    )

    staged_manifest = attach_scoring_yaml_reduction(
        RunManifest(
            run_id=_require_text(legacy_manifest.get("run_id"), "legacy manifest.run_id"),
            status=RunStatus.running,
            backend=backend,
            created_at_utc=_require_text(
                legacy_manifest.get("created_at_utc"),
                "legacy manifest.created_at_utc",
            ),
            updated_at_utc=_require_text(
                legacy_manifest.get("updated_at_utc"),
                "legacy manifest.updated_at_utc",
            ),
            artifacts=generic_artifacts + (dataset_reduction.artifact,),
            score=None,
            experiment_id=_legacy_experiment_id(legacy_manifest),
            hypothesis=_optional_text(legacy_manifest.get("hypothesis")),
            notes=_legacy_notes(legacy_manifest),
        ),
        scoring_reduction,
        updated_at_utc=_require_text(
            legacy_manifest.get("updated_at_utc"),
            "legacy manifest.updated_at_utc",
        ),
    )
    manifest = RunManifest(
        run_id=staged_manifest.run_id,
        status=RunStatus.parse(
            _require_text(legacy_manifest.get("status"), "legacy manifest.status")
        ),
        backend=staged_manifest.backend,
        created_at_utc=staged_manifest.created_at_utc,
        updated_at_utc=staged_manifest.updated_at_utc,
        artifacts=staged_manifest.artifacts,
        score=staged_manifest.score,
        experiment_id=staged_manifest.experiment_id,
        hypothesis=staged_manifest.hypothesis,
        notes=staged_manifest.notes,
    )

    manifest_output = Path(manifest_output_path).expanduser().resolve()
    write_json(manifest_output, manifest.to_dict())
    manifest_artifact = ArtifactRef(
        kind="run_manifest",
        path=str(manifest_output),
        sha256=sha256_file(manifest_output),
    )

    legacy_ledger = _read_optional_json(run_path / "ledger_entry.json")
    ledger_recorded_at = _legacy_recorded_at_utc(
        append_only_ledger,
        legacy_ledger,
    )
    promotion_decision = _legacy_promotion_decision(
        legacy_manifest,
        run_path=run_path,
        repo_root=repo_root_path,
        manifest_artifact=manifest_artifact,
        baseline=baseline,
        bootstrap=bootstrap,
        current_recorded_at_utc=ledger_recorded_at,
        canonical_legacy_manifest_path=canonical_legacy_manifest_path,
    )
    ledger_entry = build_ledger_entry(
        manifest,
        manifest_artifact=manifest_artifact,
        dataset_reduction=dataset_reduction,
        promotion=(
            None if promotion_decision is None else promotion_decision.to_dict()
        ),
        recorded_at_utc=ledger_recorded_at,
    )
    _validate_legacy_ledger_equivalence(
        append_only_ledger,
        legacy_manifest=legacy_manifest,
        ledger_entry=ledger_entry,
        promotion_decision=promotion_decision,
        run_path=run_path,
        repo_root=repo_root_path,
        expected_legacy_manifest_path=canonical_legacy_manifest_path,
    )
    _validate_legacy_ledger_equivalence(
        legacy_ledger,
        legacy_manifest=legacy_manifest,
        ledger_entry=ledger_entry,
        promotion_decision=promotion_decision,
        run_path=run_path,
        repo_root=repo_root_path,
        expected_legacy_manifest_path=canonical_legacy_manifest_path,
    )

    return LegacyRunBackfill(
        manifest=manifest,
        manifest_artifact=manifest_artifact,
        ledger_entry=ledger_entry,
        scoring_reduction=scoring_reduction,
        dataset_reduction=dataset_reduction,
        promotion_decision=promotion_decision,
    )


def _legacy_backend_spec(
    legacy_manifest: Mapping[str, Any],
    legacy_manifest_path: Path,
) -> PolicyBackendSpec:
    evaluation = _require_mapping(legacy_manifest.get("evaluation"), "legacy manifest.evaluation")
    planner_mode = _require_text(
        evaluation.get("planner_mode"),
        "legacy manifest.evaluation.planner_mode",
    )
    policy_name = _require_text(
        evaluation.get("policy"),
        "legacy manifest.evaluation.policy",
    )
    legacy_description = _require_text(
        legacy_manifest.get("description"),
        "legacy manifest.description",
    )
    runtime_env = _require_mapping(
        evaluation.get("runtime_env"),
        "legacy manifest.evaluation.runtime_env",
    )
    code = _require_mapping(legacy_manifest.get("code"), "legacy manifest.code")
    return PolicyBackendSpec(
        backend_kind=BackendKind.replay_servo,
        name=f"{planner_mode}-legacy-replay-servo",
        runtime_role=RuntimeRole.live_policy,
        training_sources=(TrainingSourceKind.official_demo,),
        simulator_sources=(SimulatorKind.offline_replay,),
        runtime_allowed=True,
        leakage_class=LeakageClass.legal_policy_input,
        runtime_boundary=RuntimeBoundaryProof(
            deterministic=True,
            uses_online_language_model_control=False,
            legal_observation_contract="official aic_model observations only",
            notes="Backfilled from a legacy replay_policy_eval manifest.",
        ),
        description=legacy_description,
        config={
            "legacy_kind": legacy_manifest.get("kind"),
            "planner_mode": planner_mode,
            "policy": policy_name,
            "participant_image": evaluation.get("participant_image"),
        },
        provenance={
            "legacy_manifest_path": str(legacy_manifest_path),
            "legacy_code_commit": code.get("commit"),
            "legacy_code_dirty": code.get("dirty"),
            "legacy_runtime_env": runtime_env,
        },
    )


def _legacy_scoring_reduction(
    legacy_manifest: Mapping[str, Any],
    *,
    run_path: Path,
    repo_root: Path,
    canonical_legacy_manifest_path: Path,
) -> ScoringYamlReduction:
    evaluation = _require_mapping(legacy_manifest.get("evaluation"), "legacy manifest.evaluation")
    scoring_source = _require_text(
        evaluation.get("official_result_path"),
        "legacy manifest.evaluation.official_result_path",
    )
    scoring_path = Path(
        _resolve_canonical_legacy_existing_path(
            scoring_source,
            run_path=canonical_legacy_manifest_path.parent,
            field_name="legacy manifest.evaluation.official_result_path",
        )
    )
    score_report_payload = _legacy_score_report_payload(
        legacy_manifest,
        run_path=run_path,
        repo_root=repo_root,
        canonical_legacy_manifest_path=canonical_legacy_manifest_path,
    )
    scoring_uri = _legacy_scoring_uri(legacy_manifest)
    reduction = reduce_scoring_yaml(
        scoring_path,
        parsed_at_utc=_require_utc_timestamp(
            score_report_payload.get("parsed_at_utc"),
            "legacy score_report.parsed_at_utc",
        ),
        uri=scoring_uri,
        provenance={"backfill_source": "legacy replay_policy_eval manifest"},
    )
    _validate_legacy_score_report(
        score_report_payload,
        scoring_source=scoring_source,
        scoring_path=scoring_path,
        scoring_reduction=reduction,
        run_path=run_path,
        repo_root=repo_root,
    )
    return reduction


def _legacy_dataset_reduction(
    legacy_manifest: Mapping[str, Any],
    *,
    repo_root: Path,
) -> Hdf5DatasetReduction:
    dataset = _require_mapping(legacy_manifest.get("dataset"), "legacy manifest.dataset")
    dataset_source = _require_text(dataset.get("uri"), "legacy manifest.dataset.uri")
    dataset_path = _resolve_legacy_existing_path(
        dataset_source,
        run_path=repo_root,
        repo_root=repo_root,
    )
    return reduce_hdf5_dataset(
        dataset_path,
        validated_at_utc=_legacy_dataset_validated_at_utc(legacy_manifest),
    )


def _legacy_generic_artifacts(
    legacy_manifest: Mapping[str, Any],
    *,
    run_path: Path,
    repo_root: Path,
    canonical_legacy_manifest_path: Path,
) -> tuple[ArtifactRef, ...]:
    raw_artifacts = legacy_manifest.get("artifacts", ())
    if not isinstance(raw_artifacts, list):
        raise HarnessIOError("legacy manifest.artifacts must be a list")
    artifacts: list[ArtifactRef] = []
    for index, raw_artifact in enumerate(raw_artifacts):
        artifact = _require_mapping(raw_artifact, f"legacy manifest.artifacts[{index}]")
        kind = _require_text(artifact.get("kind"), f"legacy manifest.artifacts[{index}].kind")
        raw_path = artifact.get("path")
        resolved_path = None
        if raw_path is not None:
            raw_path_text = _require_text(raw_path, f"legacy manifest.artifacts[{index}].path")
            consumed_path = Path(
                _resolve_legacy_existing_path(
                    raw_path_text,
                    run_path=run_path,
                    repo_root=repo_root,
                )
            )
            canonical_path = Path(
                _resolve_canonical_legacy_existing_path(
                    raw_path_text,
                    run_path=canonical_legacy_manifest_path.parent,
                    field_name=f"legacy manifest.artifacts[{index}].path",
                )
            )
            _validate_legacy_artifact_bytes(
                consumed_path,
                canonical_path,
                field_name=f"legacy manifest.artifacts[{index}].path",
            )
            resolved_path = str(canonical_path)
        raw_uri = artifact.get("uri")
        uri = None if raw_uri is None else _require_text(
            raw_uri,
            f"legacy manifest.artifacts[{index}].uri",
        )
        sha256 = _artifact_file_sha256(resolved_path)
        artifacts.append(
            ArtifactRef(
                kind=kind,
                path=resolved_path,
                uri=uri,
                sha256=sha256,
                provenance={"legacy_relative_path": raw_path} if raw_path is not None else {},
            )
        )
    return tuple(artifacts)


def _require_declared_legacy_artifact(
    legacy_manifest: Mapping[str, Any],
    *,
    kind: str,
    expected_path: Path,
    run_path: Path,
    repo_root: Path,
) -> None:
    raw_artifacts = legacy_manifest.get("artifacts", ())
    if not isinstance(raw_artifacts, list):
        raise HarnessIOError("legacy manifest.artifacts must be a list")
    matches: list[Path] = []
    for index, raw_artifact in enumerate(raw_artifacts):
        artifact = _require_mapping(raw_artifact, f"legacy manifest.artifacts[{index}]")
        if artifact.get("kind") != kind:
            continue
        raw_path = _require_text(
            artifact.get("path"),
            f"legacy manifest.artifacts[{index}].path",
        )
        matches.append(
            Path(
                _resolve_legacy_existing_path(
                    raw_path,
                    run_path=run_path,
                    repo_root=repo_root,
                )
            )
        )
    if len(matches) != 1:
        raise HarnessIOError(
            f"legacy {kind} sidecar must be declared exactly once in manifest.artifacts"
        )
    if matches[0] != expected_path.resolve():
        raise HarnessIOError(
            f"legacy {kind} artifact path does not match consumed sidecar"
        )


def _legacy_promotion_decision(
    legacy_manifest: Mapping[str, Any],
    *,
    run_path: Path,
    repo_root: Path,
    manifest_artifact: ArtifactRef,
    baseline: PromotionDecision | Mapping[str, Any] | None,
    bootstrap: bool,
    current_recorded_at_utc: str,
    canonical_legacy_manifest_path: Path,
) -> PromotionDecision | None:
    payload = _legacy_promotion_payload(
        legacy_manifest,
        run_path,
        repo_root=repo_root,
    )
    if payload is None:
        if baseline is not None or bootstrap:
            raise HarnessIOError(
                "baseline/bootstrap inputs are not allowed when legacy promotion evidence is absent"
            )
        return None

    decision_kind = PromotionDecisionKind.parse(
        _require_text(payload.get("decision"), "legacy promotion.decision")
    )
    typed_baseline = (
        None
        if baseline is None
        else baseline
        if isinstance(baseline, PromotionDecision)
        else PromotionDecision.from_dict(baseline)
    )
    current_run_id = _require_text(legacy_manifest.get("run_id"), "legacy manifest.run_id")
    _validate_legacy_promotion_binding(
        payload,
        expected_run_id=current_run_id,
        expected_manifest_path=canonical_legacy_manifest_path,
        run_path=run_path,
        repo_root=repo_root,
    )

    if bootstrap and decision_kind is not PromotionDecisionKind.bootstrap:
        raise HarnessIOError(
            "bootstrap=True conflicts with recorded non-bootstrap legacy promotion"
        )
    if typed_baseline is not None and decision_kind is PromotionDecisionKind.bootstrap:
        raise HarnessIOError(
            "baseline input conflicts with recorded bootstrap legacy promotion"
        )

    baseline_run_id: str | None = None
    baseline_manifest: ArtifactRef | None = None
    metric = _require_mapping(payload.get("metric"), "legacy promotion.metric")
    baseline_value = metric.get("baseline_value")
    if decision_kind is not PromotionDecisionKind.bootstrap:
        if typed_baseline is None:
            raise HarnessIOError(
                "recorded non-bootstrap legacy promotion requires a baseline decision"
            )
        expected_baseline_entry = _expected_legacy_baseline_entry(
            repo_root=repo_root,
            current_run_id=current_run_id,
            current_recorded_at_utc=current_recorded_at_utc,
            payload=payload,
        )
        _validate_legacy_baseline_binding(
            payload,
            typed_baseline,
            expected_baseline_entry=expected_baseline_entry,
            repo_root=repo_root,
        )
        baseline_run_id = typed_baseline.run_id
        baseline_manifest = typed_baseline.manifest
    elif baseline_value is not None:
        raise HarnessIOError(
            "recorded bootstrap legacy promotion must not set metric.baseline_value"
        )

    return PromotionDecision(
        decided_at_utc=_require_text(
            payload.get("decided_at_utc"),
            "legacy promotion.decided_at_utc",
        ),
        decision=decision_kind,
        accepted=_require_bool(payload.get("accepted"), "legacy promotion.accepted"),
        eligible_for_submission=_require_bool(
            payload.get("eligible_for_submission"),
            "legacy promotion.eligible_for_submission",
        ),
        reason=_require_text(payload.get("reason"), "legacy promotion.reason"),
        metric=PromotionMetric(
            name=_require_text(metric.get("name"), "legacy promotion.metric.name"),
            goal=PromotionGoal.parse(
                _require_text(metric.get("goal"), "legacy promotion.metric.goal")
            ),
            value=_require_number(metric.get("value"), "legacy promotion.metric.value"),
            baseline_value=(
                None
                if baseline_value is None
                else _require_number(
                    baseline_value,
                    "legacy promotion.metric.baseline_value",
                )
            ),
            min_improvement=_legacy_min_improvement(payload),
        ),
        run_id=current_run_id,
        manifest=manifest_artifact,
        baseline_run_id=baseline_run_id,
        baseline_manifest=baseline_manifest,
        notes=_legacy_promotion_notes(payload),
    )


def _legacy_promotion_payload(
    legacy_manifest: Mapping[str, Any],
    run_path: Path,
    *,
    repo_root: Path,
) -> Mapping[str, Any] | None:
    manifest_payload = legacy_manifest.get("promotion")
    manifest_promotion = None
    if manifest_payload is not None:
        manifest_promotion = _require_mapping(
            manifest_payload,
            "legacy manifest.promotion",
        )
        _require_schema_version(manifest_promotion, "legacy manifest.promotion")
    sidecar = _read_optional_json(run_path / "promotion_report.json")
    if sidecar is None:
        return manifest_promotion
    _require_declared_legacy_artifact(
        legacy_manifest,
        kind="promotion_report",
        expected_path=run_path / "promotion_report.json",
        run_path=run_path,
        repo_root=repo_root,
    )
    _require_schema_version(sidecar, "legacy promotion_report")
    if manifest_promotion is not None and not _json_mappings_equal(manifest_promotion, sidecar):
        raise HarnessIOError(
            "legacy manifest.promotion does not match promotion_report.json"
        )
    return sidecar


def _validate_legacy_baseline_binding(
    payload: Mapping[str, Any],
    baseline: PromotionDecision,
    *,
    expected_baseline_entry: Mapping[str, Any],
    repo_root: Path,
) -> None:
    expected_baseline_run_id = _require_text(
        expected_baseline_entry.get("run_id"),
        "legacy baseline ledger.run_id",
    )
    if not baseline.accepted:
        raise HarnessIOError("legacy promotion baseline must be accepted")
    if baseline.run_id != expected_baseline_run_id:
        raise HarnessIOError(
            "legacy promotion baseline run_id does not match recorded historical ledger"
        )
    metric = _require_mapping(payload.get("metric"), "legacy promotion.metric")
    legacy_goal = _require_text(metric.get("goal"), "legacy promotion.metric.goal")
    legacy_name = _require_text(metric.get("name"), "legacy promotion.metric.name")
    legacy_baseline_value = _require_number(
        metric.get("baseline_value"),
        "legacy promotion.metric.baseline_value",
    )
    if baseline.metric.name != legacy_name:
        raise HarnessIOError(
            "legacy promotion baseline metric.name does not match recorded promotion metric.name"
        )
    if baseline.metric.goal.value != legacy_goal:
        raise HarnessIOError(
            "legacy promotion baseline metric.goal does not match recorded promotion metric.goal"
        )
    if not math.isclose(
        baseline.metric.value,
        legacy_baseline_value,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise HarnessIOError(
            "legacy promotion baseline metric.value does not match recorded baseline_value"
        )
    _validate_legacy_baseline_manifest_artifact(
        baseline.manifest,
        expected_baseline_entry=expected_baseline_entry,
        repo_root=repo_root,
    )


def _validate_legacy_baseline_manifest_artifact(
    artifact: ArtifactRef,
    *,
    expected_baseline_entry: Mapping[str, Any],
    repo_root: Path,
) -> None:
    if artifact.path is None:
        raise HarnessIOError("legacy promotion baseline manifest.path must be set")
    manifest_path = Path(artifact.path).expanduser()
    if not manifest_path.is_absolute():
        raise HarnessIOError("legacy promotion baseline manifest.path must be absolute")
    resolved_manifest_path = manifest_path.resolve()
    if not resolved_manifest_path.is_file():
        raise HarnessIOError("legacy promotion baseline manifest.path does not exist")
    if artifact.sha256 != sha256_file(resolved_manifest_path):
        raise HarnessIOError("legacy promotion baseline manifest.sha256 does not match file")

    expected_run_id = _require_text(
        expected_baseline_entry.get("run_id"),
        "legacy baseline ledger.run_id",
    )
    expected_metric = _require_mapping(
        expected_baseline_entry.get("metric"),
        "legacy baseline ledger.metric",
    )
    expected_score = _require_number(
        expected_metric.get("value"),
        "legacy baseline ledger.metric.value",
    )
    expected_legacy_manifest_path = _legacy_ledger_manifest_path(
        expected_baseline_entry,
    )

    manifest_payload = read_json(resolved_manifest_path)
    if _looks_like_neutral_run_manifest(manifest_payload):
        try:
            neutral_manifest = RunManifest.from_dict(manifest_payload)
        except SchemaValidationError as exc:
            raise HarnessIOError(
                "legacy promotion baseline neutral manifest must be a valid RunManifest"
            ) from exc
        if neutral_manifest.status not in (RunStatus.completed, RunStatus.promoted):
            raise HarnessIOError(
                "legacy promotion baseline neutral manifest must be completed or promoted"
            )
        if neutral_manifest.score is None:
            raise HarnessIOError(
                "legacy promotion baseline neutral manifest must include a score"
            )
        manifest_run_id = neutral_manifest.run_id
        manifest_score = neutral_manifest.score.total
        provenance = neutral_manifest.backend.provenance
        raw_legacy_path = _require_text(
            provenance.get("legacy_manifest_path"),
            "legacy promotion baseline neutral manifest.backend.provenance.legacy_manifest_path",
        )
        baseline_legacy_manifest_path = Path(
            _resolve_canonical_legacy_existing_path(
                raw_legacy_path,
                run_path=_CANONICAL_EXPERIMENTS_ROOT,
                field_name=(
                    "legacy promotion baseline neutral "
                    "manifest.backend.provenance.legacy_manifest_path"
                ),
            )
        )
        expected_neutral_manifest = _reconstruct_neutral_legacy_manifest(
            expected_legacy_manifest_path
        )
        if not _json_mappings_equal(
            neutral_manifest.to_dict(),
            expected_neutral_manifest.to_dict(),
        ):
            raise HarnessIOError(
                "legacy promotion baseline neutral manifest does not match reconstructed historical ledger"
            )
    else:
        manifest_run_id = _require_text(
            manifest_payload.get("run_id"),
            "legacy promotion baseline manifest.run_id",
        )
        evaluation = _require_mapping(
            manifest_payload.get("evaluation"),
            "legacy promotion baseline manifest.evaluation",
        )
        score = _require_mapping(
            evaluation.get("score"),
            "legacy promotion baseline manifest.evaluation.score",
        )
        manifest_score = _require_number(
            score.get("total"),
            "legacy promotion baseline manifest.evaluation.score.total",
        )
        baseline_legacy_manifest_path = resolved_manifest_path

    if manifest_run_id != expected_run_id:
        raise HarnessIOError(
            "legacy promotion baseline manifest.run_id does not match recorded historical ledger"
        )
    if not math.isclose(
        manifest_score,
        expected_score,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise HarnessIOError(
            "legacy promotion baseline manifest score does not match recorded historical ledger"
        )
    if baseline_legacy_manifest_path != expected_legacy_manifest_path:
        raise HarnessIOError(
            "legacy promotion baseline manifest provenance does not match recorded historical ledger"
        )


def _reconstruct_neutral_legacy_manifest(
    legacy_manifest_path: Path,
) -> RunManifest:
    legacy_manifest = read_json(legacy_manifest_path)
    _require_schema_version(legacy_manifest, "legacy baseline manifest")
    if legacy_manifest.get("kind") != "replay_policy_eval":
        raise HarnessIOError(
            "legacy promotion baseline neutral manifest can only reconstruct replay_policy_eval"
        )

    backend = _legacy_backend_spec(legacy_manifest, legacy_manifest_path)
    scoring_reduction = _legacy_scoring_reduction(
        legacy_manifest,
        run_path=legacy_manifest_path.parent,
        repo_root=_CANONICAL_REPO_ROOT,
        canonical_legacy_manifest_path=legacy_manifest_path,
    )
    dataset_reduction = _legacy_dataset_reduction(
        legacy_manifest,
        repo_root=_CANONICAL_REPO_ROOT,
    )
    generic_artifacts = _legacy_generic_artifacts(
        legacy_manifest,
        run_path=legacy_manifest_path.parent,
        repo_root=_CANONICAL_REPO_ROOT,
        canonical_legacy_manifest_path=legacy_manifest_path,
    )
    staged_manifest = attach_scoring_yaml_reduction(
        RunManifest(
            run_id=_require_text(legacy_manifest.get("run_id"), "legacy baseline manifest.run_id"),
            status=RunStatus.running,
            backend=backend,
            created_at_utc=_require_text(
                legacy_manifest.get("created_at_utc"),
                "legacy baseline manifest.created_at_utc",
            ),
            updated_at_utc=_require_text(
                legacy_manifest.get("updated_at_utc"),
                "legacy baseline manifest.updated_at_utc",
            ),
            artifacts=generic_artifacts + (dataset_reduction.artifact,),
            score=None,
            experiment_id=_legacy_experiment_id(legacy_manifest),
            hypothesis=_optional_text(legacy_manifest.get("hypothesis")),
            notes=_legacy_notes(legacy_manifest),
        ),
        scoring_reduction,
        updated_at_utc=_require_text(
            legacy_manifest.get("updated_at_utc"),
            "legacy baseline manifest.updated_at_utc",
        ),
    )
    return RunManifest(
        run_id=staged_manifest.run_id,
        status=RunStatus.parse(
            _require_text(legacy_manifest.get("status"), "legacy baseline manifest.status")
        ),
        backend=staged_manifest.backend,
        created_at_utc=staged_manifest.created_at_utc,
        updated_at_utc=staged_manifest.updated_at_utc,
        artifacts=staged_manifest.artifacts,
        score=staged_manifest.score,
        experiment_id=staged_manifest.experiment_id,
        hypothesis=staged_manifest.hypothesis,
        notes=staged_manifest.notes,
    )


def _validate_legacy_promotion_binding(
    payload: Mapping[str, Any],
    *,
    expected_run_id: str,
    expected_manifest_path: Path,
    run_path: Path,
    repo_root: Path,
) -> None:
    payload_run_id = _require_text(payload.get("run_id"), "legacy promotion.run_id")
    if payload_run_id != expected_run_id:
        raise HarnessIOError(
            "legacy promotion.run_id does not match the current run manifest"
        )
    raw_manifest_path = _require_text(
        payload.get("manifest_path"),
        "legacy promotion.manifest_path",
    )
    resolved_manifest_path = _resolve_canonical_legacy_existing_path(
        raw_manifest_path,
        run_path=_CANONICAL_EXPERIMENTS_ROOT,
    )
    if Path(resolved_manifest_path) != expected_manifest_path.resolve():
        raise HarnessIOError(
            "legacy promotion.manifest_path does not match the current run manifest path"
        )


def _expected_legacy_baseline_entry(
    *,
    repo_root: Path,
    current_run_id: str,
    current_recorded_at_utc: str,
    payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    metric = _require_mapping(payload.get("metric"), "legacy promotion.metric")
    target_name = _require_text(metric.get("name"), "legacy promotion.metric.name")
    target_goal = _require_text(metric.get("goal"), "legacy promotion.metric.goal")
    target_value = _require_number(
        metric.get("baseline_value"),
        "legacy promotion.metric.baseline_value",
    )
    ledger_entries = _read_canonical_legacy_ledger_entries()
    matches: list[Mapping[str, Any]] = []
    current_recorded_at = _require_utc_timestamp(
        current_recorded_at_utc,
        "legacy current recorded_at_utc",
    )
    for entry in ledger_entries:
        if _require_text(entry.get("run_id"), "legacy ledger entry.run_id") == current_run_id:
            continue
        recorded_at = _require_utc_timestamp(
            entry.get("recorded_at_utc"),
            "legacy ledger entry.recorded_at_utc",
        )
        if recorded_at >= current_recorded_at:
            continue
        promotion = entry.get("promotion", {})
        if not isinstance(promotion, Mapping) or promotion.get("accepted") is not True:
            continue
        entry_metric = _require_mapping(entry.get("metric"), "legacy ledger entry.metric")
        if _require_text(entry_metric.get("name"), "legacy ledger entry.metric.name") != target_name:
            continue
        if _require_text(entry_metric.get("goal"), "legacy ledger entry.metric.goal") != target_goal:
            continue
        if not math.isclose(
            _require_number(
                entry_metric.get("value"),
                "legacy ledger entry.metric.value",
            ),
            target_value,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            continue
        _validate_accepted_legacy_baseline_entry(
            entry,
            metric_name=target_name,
            metric_goal=target_goal,
            metric_value=target_value,
        )
        matches.append(entry)
    if len(matches) != 1:
        raise HarnessIOError(
            "could not derive a unique accepted historical baseline identity from legacy ledger"
        )
    _validate_supplied_ledger_entry_matches_canonical(
        repo_root=repo_root,
        canonical_entry=matches[0],
        field_name="legacy baseline ledger",
    )
    return matches[0]


def _current_legacy_ledger_entry(
    legacy_manifest: Mapping[str, Any],
    *,
    repo_root: Path,
) -> Mapping[str, Any]:
    current_run_id = _require_text(legacy_manifest.get("run_id"), "legacy manifest.run_id")
    current_kind = _require_text(legacy_manifest.get("kind"), "legacy manifest.kind")
    canonical_entry = _canonical_legacy_ledger_entry(
        current_run_id,
        field_name="legacy append-only ledger",
    )
    _validate_supplied_ledger_entry_matches_canonical(
        repo_root=repo_root,
        canonical_entry=canonical_entry,
        field_name="legacy append-only ledger",
    )
    _require_schema_version(canonical_entry, "legacy append-only ledger")
    _require_utc_timestamp(
        canonical_entry.get("recorded_at_utc"),
        "legacy append-only ledger.recorded_at_utc",
    )
    ledger_kind = _require_text(canonical_entry.get("kind"), "legacy ledger entry.kind")
    if ledger_kind != current_kind:
        raise HarnessIOError(
            "legacy append-only ledger kind does not match the current run manifest"
        )
    return canonical_entry


def _validate_current_legacy_manifest_binding(
    legacy_manifest: Mapping[str, Any],
    *,
    append_only_ledger: Mapping[str, Any],
    repo_root: Path,
) -> Path:
    del repo_root
    canonical_manifest_path = _legacy_ledger_manifest_path(
        append_only_ledger,
        field_name="legacy append-only ledger.manifest_path",
    )
    canonical_manifest = read_json(canonical_manifest_path)
    if not _json_mappings_equal(legacy_manifest, canonical_manifest):
        raise HarnessIOError(
            "legacy manifest does not match append-only ledger manifest evidence"
        )

    promotion = append_only_ledger.get("promotion", {})
    if isinstance(promotion, Mapping) and promotion.get("manifest_path") is not None:
        promotion_manifest_path = _legacy_ledger_manifest_path(
            promotion,
            field_name="legacy append-only ledger.promotion.manifest_path",
        )
        if promotion_manifest_path != canonical_manifest_path:
            raise HarnessIOError(
                "legacy append-only ledger.promotion.manifest_path does not match ledger manifest_path"
            )
    return canonical_manifest_path


def _validate_accepted_legacy_baseline_entry(
    entry: Mapping[str, Any],
    *,
    metric_name: str,
    metric_goal: str,
    metric_value: float,
) -> None:
    _require_schema_version(entry, "legacy baseline ledger")
    status = _require_text(entry.get("status"), "legacy baseline ledger.status")
    if status not in (RunStatus.completed.value, RunStatus.promoted.value):
        raise HarnessIOError("legacy baseline ledger.status must be completed or promoted")
    promotion = _require_mapping(
        entry.get("promotion"),
        "legacy baseline ledger.promotion",
    )
    _require_schema_version(promotion, "legacy baseline ledger.promotion")
    if _require_bool(
        promotion.get("accepted"),
        "legacy baseline ledger.promotion.accepted",
    ) is not True:
        raise HarnessIOError("legacy baseline ledger.promotion.accepted must be true")
    if _require_text(
        promotion.get("decision"),
        "legacy baseline ledger.promotion.decision",
    ) != PromotionDecisionKind.accepted.value:
        raise HarnessIOError("legacy baseline ledger.promotion.decision must be accepted")
    if _require_text(
        promotion.get("run_id"),
        "legacy baseline ledger.promotion.run_id",
    ) != _require_text(entry.get("run_id"), "legacy baseline ledger.run_id"):
        raise HarnessIOError(
            "legacy baseline ledger.promotion.run_id does not match ledger run_id"
        )
    baseline_manifest_path = _legacy_ledger_manifest_path(
        entry,
        field_name="legacy baseline ledger.manifest_path",
    )
    baseline_promotion_manifest_path = _legacy_ledger_manifest_path(
        promotion,
        field_name="legacy baseline ledger.promotion.manifest_path",
    )
    if baseline_promotion_manifest_path != baseline_manifest_path:
        raise HarnessIOError(
            "legacy baseline ledger.promotion.manifest_path does not match ledger manifest_path"
        )
    baseline_manifest = read_json(baseline_manifest_path)
    baseline_manifest_promotion = _require_mapping(
        baseline_manifest.get("promotion"),
        "legacy baseline manifest.promotion",
    )
    _validate_legacy_ledger_promotion_provenance(
        promotion,
        expected_promotion=baseline_manifest_promotion,
        expected_legacy_manifest_path=baseline_manifest_path,
        field_prefix="legacy baseline ledger.promotion",
    )
    promotion_metric = _require_mapping(
        promotion.get("metric"),
        "legacy baseline ledger.promotion.metric",
    )
    if _require_text(
        promotion_metric.get("name"),
        "legacy baseline ledger.promotion.metric.name",
    ) != metric_name:
        raise HarnessIOError(
            "legacy baseline ledger.promotion.metric.name does not match ledger metric"
        )
    if _require_text(
        promotion_metric.get("goal"),
        "legacy baseline ledger.promotion.metric.goal",
    ) != metric_goal:
        raise HarnessIOError(
            "legacy baseline ledger.promotion.metric.goal does not match ledger metric"
        )
    if not math.isclose(
        _require_number(
            promotion_metric.get("value"),
            "legacy baseline ledger.promotion.metric.value",
        ),
        metric_value,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise HarnessIOError(
            "legacy baseline ledger.promotion.metric.value does not match ledger metric"
        )


def _legacy_ledger_manifest_path(
    ledger_entry: Mapping[str, Any],
    *,
    field_name: str = "legacy baseline ledger.manifest_path",
) -> Path:
    raw_manifest_path = _require_text(
        ledger_entry.get("manifest_path"),
        field_name,
    )
    return Path(
        _resolve_canonical_legacy_existing_path(
            raw_manifest_path,
            run_path=_CANONICAL_EXPERIMENTS_ROOT,
            field_name=field_name,
        )
    )


def _looks_like_neutral_run_manifest(payload: Mapping[str, Any]) -> bool:
    return "backend" in payload and "score" in payload and "artifacts" in payload


def _legacy_score_report_payload(
    legacy_manifest: Mapping[str, Any],
    *,
    run_path: Path,
    repo_root: Path,
    canonical_legacy_manifest_path: Path,
) -> Mapping[str, Any]:
    evaluation = _require_mapping(legacy_manifest.get("evaluation"), "legacy manifest.evaluation")
    raw_score_report_path = _require_text(
        evaluation.get("score_report_path"),
        "legacy manifest.evaluation.score_report_path",
    )
    consumed_score_report_path = Path(
        _resolve_legacy_existing_path(
            raw_score_report_path,
            run_path=run_path,
            repo_root=repo_root,
        )
    )
    _require_declared_legacy_artifact(
        legacy_manifest,
        kind="score_report",
        expected_path=consumed_score_report_path,
        run_path=run_path,
        repo_root=repo_root,
    )
    payload = read_json(consumed_score_report_path)
    canonical_score_report_path = Path(
        _resolve_legacy_existing_path(
            raw_score_report_path,
            run_path=canonical_legacy_manifest_path.parent,
            repo_root=_CANONICAL_REPO_ROOT,
        )
    )
    canonical_payload = read_json(canonical_score_report_path)
    if not _json_mappings_equal(payload, canonical_payload):
        raise HarnessIOError(
            "legacy score_report does not match canonical score_report evidence"
        )
    _require_schema_version(payload, "legacy score_report")
    return payload


def _validate_legacy_score_report(
    payload: Mapping[str, Any],
    *,
    scoring_source: str,
    scoring_path: Path,
    scoring_reduction: ScoringYamlReduction,
    run_path: Path,
    repo_root: Path,
) -> None:
    _require_schema_version(payload, "legacy score_report")
    declared_path = _require_text(payload.get("path"), "legacy score_report.path")
    if declared_path != scoring_source:
        raise HarnessIOError(
            "legacy score_report.path does not match manifest official_result_path"
        )
    resolved_declared_path = Path(
        _resolve_legacy_existing_path(
            declared_path,
            run_path=run_path,
            repo_root=repo_root,
        )
    )
    if resolved_declared_path != scoring_path:
        raise HarnessIOError("legacy score_report.path does not match scoring artifact")

    score = _require_mapping(payload.get("score"), "legacy score_report.score")
    score_source = _require_text(
        score.get("source"),
        "legacy score_report.score.source",
    )
    if score_source != scoring_source:
        raise HarnessIOError(
            "legacy score_report.score.source does not match manifest official_result_path"
        )
    resolved_score_source = Path(
        _resolve_legacy_existing_path(
            score_source,
            run_path=run_path,
            repo_root=repo_root,
        )
    )
    if resolved_score_source != scoring_path:
        raise HarnessIOError(
            "legacy score_report.score.source does not match scoring artifact"
        )

    parsed_at_utc = _require_utc_timestamp(
        payload.get("parsed_at_utc"),
        "legacy score_report.parsed_at_utc",
    )
    sidecar_score = ScoreReport.from_legacy_v1_dict(
        {
            "schema_version": payload.get("schema_version"),
            "source": score_source,
            "parsed_at_utc": parsed_at_utc,
            "total": score.get("total"),
            "trial_count": score.get("trial_count"),
            "trials": score.get("trials"),
        }
    )
    aliases = {
        declared_path,
        score_source,
        scoring_source,
        str(scoring_path),
        scoring_reduction.score.source,
    }
    if not sidecar_score.equivalent_to(
        scoring_reduction.score,
        source_aliases=aliases,
    ):
        raise HarnessIOError("legacy score_report.score does not match scoring artifact")
    if sidecar_score.parsed_at_utc != scoring_reduction.score.parsed_at_utc:
        raise HarnessIOError(
            "legacy score_report.parsed_at_utc does not match scoring artifact"
        )


def _legacy_scoring_uri(legacy_manifest: Mapping[str, Any]) -> str | None:
    raw_artifacts = legacy_manifest.get("artifacts", ())
    if not isinstance(raw_artifacts, list):
        return None
    for raw_artifact in raw_artifacts:
        if not isinstance(raw_artifact, Mapping):
            continue
        if raw_artifact.get("kind") != "eval_bundle":
            continue
        raw_uri = raw_artifact.get("uri")
        if not isinstance(raw_uri, str) or not raw_uri.strip():
            continue
        return raw_uri.rstrip("/") + "/eval/scoring.yaml"
    return None


def _legacy_dataset_validated_at_utc(legacy_manifest: Mapping[str, Any]) -> str:
    value = legacy_manifest.get("created_at_utc")
    return _require_text(value, "legacy manifest.created_at_utc")


def _legacy_recorded_at_utc(
    append_only_ledger: Mapping[str, Any],
    legacy_ledger: Mapping[str, Any] | None,
) -> str:
    append_only_recorded_at = _require_utc_timestamp(
        append_only_ledger.get("recorded_at_utc"),
        "legacy append-only ledger.recorded_at_utc",
    )
    if legacy_ledger is not None and legacy_ledger.get("recorded_at_utc") is not None:
        sidecar_recorded_at = _require_utc_timestamp(
            legacy_ledger.get("recorded_at_utc"),
            "legacy ledger.recorded_at_utc",
        )
        if sidecar_recorded_at != append_only_recorded_at:
            raise HarnessIOError(
                "legacy ledger.recorded_at_utc does not match append-only ledger"
            )
    return append_only_recorded_at


def _validate_legacy_ledger_equivalence(
    legacy_ledger: Mapping[str, Any] | None,
    *,
    legacy_manifest: Mapping[str, Any],
    ledger_entry: LedgerEntry,
    promotion_decision: PromotionDecision | None,
    run_path: Path,
    repo_root: Path,
    expected_legacy_manifest_path: Path,
) -> None:
    if legacy_ledger is None:
        return
    _require_schema_version(legacy_ledger, "legacy ledger")
    if _require_text(legacy_ledger.get("kind"), "legacy ledger.kind") != _require_text(
        legacy_manifest.get("kind"),
        "legacy manifest.kind",
    ):
        raise HarnessIOError("legacy ledger.kind does not match legacy manifest")
    _validate_legacy_ledger_code_metadata(legacy_ledger, legacy_manifest)
    _validate_legacy_ledger_training_metadata(legacy_ledger, legacy_manifest)
    if _require_utc_timestamp(
        legacy_ledger.get("recorded_at_utc"),
        "legacy ledger.recorded_at_utc",
    ) != ledger_entry.recorded_at_utc:
        raise HarnessIOError(
            "legacy ledger.recorded_at_utc does not match reconstructed ledger entry"
        )
    if _require_text(legacy_ledger.get("run_id"), "legacy ledger.run_id") != ledger_entry.run_id:
        raise HarnessIOError("legacy ledger.run_id does not match reconstructed ledger entry")
    if _require_text(legacy_ledger.get("status"), "legacy ledger.status") != ledger_entry.status.value:
        raise HarnessIOError("legacy ledger.status does not match reconstructed ledger entry")
    raw_manifest_path = _require_text(
        legacy_ledger.get("manifest_path"),
        "legacy ledger.manifest_path",
    )
    resolved_manifest_path = _resolve_canonical_legacy_existing_path(
        raw_manifest_path,
        run_path=_CANONICAL_EXPERIMENTS_ROOT,
    )
    if Path(resolved_manifest_path) != expected_legacy_manifest_path:
        raise HarnessIOError(
            "legacy ledger.manifest_path does not match append-only ledger manifest evidence"
        )
    metric = _require_mapping(legacy_ledger.get("metric"), "legacy ledger.metric")
    if _require_text(metric.get("name"), "legacy ledger.metric.name") != ledger_entry.metric.name:
        raise HarnessIOError("legacy ledger.metric.name does not match reconstructed ledger entry")
    if _require_text(metric.get("goal"), "legacy ledger.metric.goal") != ledger_entry.metric.goal.value:
        raise HarnessIOError("legacy ledger.metric.goal does not match reconstructed ledger entry")
    if not math.isclose(
        _require_number(metric.get("value"), "legacy ledger.metric.value"),
        _require_number(ledger_entry.metric.value, "reconstructed ledger.metric.value"),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise HarnessIOError("legacy ledger.metric.value does not match reconstructed ledger entry")
    dataset = _require_mapping(legacy_ledger.get("dataset"), "legacy ledger.dataset")
    if ledger_entry.dataset is None:
        raise HarnessIOError("reconstructed ledger entry is missing dataset summary")
    if _require_number(dataset.get("episode_count"), "legacy ledger.dataset.episode_count") != float(
        ledger_entry.dataset.episode_count
    ):
        raise HarnessIOError(
            "legacy ledger.dataset.episode_count does not match reconstructed ledger entry"
        )
    if _require_number(dataset.get("step_count"), "legacy ledger.dataset.step_count") != float(
        ledger_entry.dataset.step_count
    ):
        raise HarnessIOError(
            "legacy ledger.dataset.step_count does not match reconstructed ledger entry"
        )
    _validate_legacy_ledger_dataset_identity(
        dataset,
        ledger_entry.dataset.artifact,
        repo_root=repo_root,
    )
    legacy_promotion = legacy_ledger.get("promotion", {})
    if promotion_decision is None:
        if legacy_promotion not in ({}, None):
            raise HarnessIOError(
                "legacy ledger.promotion is present but reconstructed promotion_decision is missing"
            )
        return
    if not isinstance(legacy_promotion, Mapping):
        raise HarnessIOError("legacy ledger.promotion must be a mapping when set")
    _validate_legacy_ledger_promotion_equivalence(
        legacy_promotion,
        legacy_manifest=legacy_manifest,
        promotion_decision=promotion_decision,
        run_path=run_path,
        repo_root=repo_root,
        expected_legacy_manifest_path=expected_legacy_manifest_path,
    )


def _validate_legacy_ledger_dataset_identity(
    dataset: Mapping[str, Any],
    artifact: ArtifactRef,
    *,
    repo_root: Path,
) -> None:
    if _require_text(dataset.get("sha256"), "legacy ledger.dataset.sha256") != artifact.sha256:
        raise HarnessIOError(
            "legacy ledger.dataset.sha256 does not match reconstructed ledger entry"
        )
    raw_dataset_source = _require_text(dataset.get("uri"), "legacy ledger.dataset.uri")
    if _is_uri_source(raw_dataset_source):
        declared_uri = artifact.provenance.get("declared_uri")
        if artifact.uri != raw_dataset_source and declared_uri != raw_dataset_source:
            raise HarnessIOError(
                "legacy ledger.dataset.uri does not match reconstructed ledger entry"
            )
        return
    if artifact.path is None:
        raise HarnessIOError(
            "legacy ledger.dataset.uri is local but reconstructed dataset artifact.path is unset"
        )
    resolved_dataset_path = Path(
        _resolve_legacy_existing_path(
            raw_dataset_source,
            run_path=repo_root,
            repo_root=repo_root,
        )
    )
    if resolved_dataset_path != Path(artifact.path).expanduser().resolve():
        raise HarnessIOError(
            "legacy ledger.dataset.uri does not match reconstructed ledger entry"
        )


def _validate_legacy_ledger_code_metadata(
    legacy_ledger: Mapping[str, Any],
    legacy_manifest: Mapping[str, Any],
) -> None:
    code = _require_mapping(legacy_manifest.get("code"), "legacy manifest.code")
    if _require_text(legacy_ledger.get("git_commit"), "legacy ledger.git_commit") != _require_text(
        code.get("commit"),
        "legacy manifest.code.commit",
    ):
        raise HarnessIOError("legacy ledger.git_commit does not match legacy manifest")
    if _require_bool(legacy_ledger.get("git_dirty"), "legacy ledger.git_dirty") != _require_bool(
        code.get("dirty"),
        "legacy manifest.code.dirty",
    ):
        raise HarnessIOError("legacy ledger.git_dirty does not match legacy manifest")


def _validate_legacy_ledger_training_metadata(
    legacy_ledger: Mapping[str, Any],
    legacy_manifest: Mapping[str, Any],
) -> None:
    ledger_training = _require_mapping(
        legacy_ledger.get("training"),
        "legacy ledger.training",
    )
    manifest_training = _require_mapping(
        legacy_manifest.get("training"),
        "legacy manifest.training",
    )
    expected_training = {
        "checkpoint_uri": manifest_training.get("checkpoint_uri"),
        "max_epochs": manifest_training.get("max_epochs"),
        "run_name": manifest_training.get("run_name"),
    }
    if dict(ledger_training) != expected_training:
        raise HarnessIOError("legacy ledger.training does not match legacy manifest")


def _validate_legacy_ledger_promotion_equivalence(
    legacy_promotion: Mapping[str, Any],
    *,
    legacy_manifest: Mapping[str, Any],
    promotion_decision: PromotionDecision,
    run_path: Path,
    repo_root: Path,
    expected_legacy_manifest_path: Path,
) -> None:
    del repo_root
    _require_schema_version(legacy_promotion, "legacy ledger.promotion")
    expected_promotion = _require_mapping(
        legacy_manifest.get("promotion"),
        "legacy manifest.promotion",
    )
    _validate_legacy_ledger_promotion_provenance(
        legacy_promotion,
        expected_promotion=expected_promotion,
        expected_legacy_manifest_path=expected_legacy_manifest_path,
        field_prefix="legacy ledger.promotion",
    )
    if _require_text(legacy_promotion.get("run_id"), "legacy ledger.promotion.run_id") != promotion_decision.run_id:
        raise HarnessIOError(
            "legacy ledger.promotion.run_id does not match reconstructed promotion decision"
        )
    raw_manifest_path = _require_text(
        legacy_promotion.get("manifest_path"),
        "legacy ledger.promotion.manifest_path",
    )
    resolved_manifest_path = _resolve_canonical_legacy_existing_path(
        raw_manifest_path,
        run_path=_CANONICAL_EXPERIMENTS_ROOT,
    )
    if Path(resolved_manifest_path) != expected_legacy_manifest_path:
        raise HarnessIOError(
            "legacy ledger.promotion.manifest_path does not match append-only ledger manifest evidence"
        )
    if _require_bool(legacy_promotion.get("accepted"), "legacy ledger.promotion.accepted") != promotion_decision.accepted:
        raise HarnessIOError(
            "legacy ledger.promotion.accepted does not match reconstructed promotion decision"
        )
    if _require_text(legacy_promotion.get("decision"), "legacy ledger.promotion.decision") != promotion_decision.decision.value:
        raise HarnessIOError(
            "legacy ledger.promotion.decision does not match reconstructed promotion decision"
        )
    if _require_bool(
        legacy_promotion.get("eligible_for_submission"),
        "legacy ledger.promotion.eligible_for_submission",
    ) != promotion_decision.eligible_for_submission:
        raise HarnessIOError(
            "legacy ledger.promotion.eligible_for_submission does not match reconstructed promotion decision"
        )
    if _require_text(legacy_promotion.get("reason"), "legacy ledger.promotion.reason") != promotion_decision.reason:
        raise HarnessIOError(
            "legacy ledger.promotion.reason does not match reconstructed promotion decision"
        )
    if _require_text_tuple(legacy_promotion.get("notes"), "legacy ledger.promotion.notes") != promotion_decision.notes:
        raise HarnessIOError(
            "legacy ledger.promotion.notes does not match reconstructed promotion decision"
        )
    legacy_promotion_metric = _require_mapping(
        legacy_promotion.get("metric"),
        "legacy ledger.promotion.metric",
    )
    if _require_text(
        legacy_promotion_metric.get("name"),
        "legacy ledger.promotion.metric.name",
    ) != promotion_decision.metric.name:
        raise HarnessIOError(
            "legacy ledger.promotion.metric.name does not match reconstructed promotion decision"
        )
    if _require_text(
        legacy_promotion_metric.get("goal"),
        "legacy ledger.promotion.metric.goal",
    ) != promotion_decision.metric.goal.value:
        raise HarnessIOError(
            "legacy ledger.promotion.metric.goal does not match reconstructed promotion decision"
        )
    if not math.isclose(
        _require_number(
            legacy_promotion_metric.get("value"),
            "legacy ledger.promotion.metric.value",
        ),
        promotion_decision.metric.value,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise HarnessIOError(
            "legacy ledger.promotion.metric.value does not match reconstructed promotion decision"
        )
    if not math.isclose(
        _require_number(
            legacy_promotion_metric.get("baseline_value"),
            "legacy ledger.promotion.metric.baseline_value",
        ),
        _require_number(
            promotion_decision.metric.baseline_value,
            "reconstructed promotion metric.baseline_value",
        ),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise HarnessIOError(
            "legacy ledger.promotion.metric.baseline_value does not match reconstructed promotion decision"
        )
    if not math.isclose(
        _require_number(
            legacy_promotion_metric.get("min_improvement"),
            "legacy ledger.promotion.metric.min_improvement",
        ),
        promotion_decision.metric.min_improvement,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise HarnessIOError(
            "legacy ledger.promotion.metric.min_improvement does not match reconstructed promotion decision"
        )


def _validate_legacy_ledger_promotion_provenance(
    legacy_promotion: Mapping[str, Any],
    *,
    expected_promotion: Mapping[str, Any],
    expected_legacy_manifest_path: Path,
    field_prefix: str,
) -> None:
    legacy_report_path = _require_optional_text_field(
        legacy_promotion,
        "report_path",
        f"{field_prefix}.report_path",
    )
    expected_report_path = _require_optional_text_field(
        expected_promotion,
        "report_path",
        "legacy manifest.promotion.report_path",
    )
    if legacy_report_path != expected_report_path:
        raise HarnessIOError(
            f"{field_prefix}.report_path does not match legacy manifest.promotion"
        )
    if legacy_report_path is not None:
        _resolve_canonical_legacy_existing_path(
            legacy_report_path,
            run_path=expected_legacy_manifest_path.parent,
            field_name=f"{field_prefix}.report_path",
        )

    if _require_optional_text_field(
        legacy_promotion,
        "checkpoint_uri",
        f"{field_prefix}.checkpoint_uri",
    ) != _require_optional_text_field(
        expected_promotion,
        "checkpoint_uri",
        "legacy manifest.promotion.checkpoint_uri",
    ):
        raise HarnessIOError(
            f"{field_prefix}.checkpoint_uri does not match legacy manifest.promotion"
        )


def _legacy_min_improvement(legacy_promotion: Mapping[str, Any]) -> float:
    metric = _require_mapping(legacy_promotion.get("metric"), "legacy promotion.metric")
    value = metric.get("min_improvement", 0.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HarnessIOError("legacy promotion.metric.min_improvement must be numeric")
    return float(value)


def _legacy_promotion_notes(legacy_promotion: Mapping[str, Any]) -> tuple[str, ...]:
    raw_notes = legacy_promotion.get("notes", ())
    if not isinstance(raw_notes, list):
        raise HarnessIOError("legacy promotion.notes must be a list")
    return tuple(
        _require_text(note, f"legacy promotion.notes[{index}]")
        for index, note in enumerate(raw_notes)
    )


def _legacy_notes(legacy_manifest: Mapping[str, Any]) -> tuple[str, ...]:
    raw_notes = legacy_manifest.get("notes", ())
    if not isinstance(raw_notes, list):
        raise HarnessIOError("legacy manifest.notes must be a list")
    return tuple(
        _require_text(note, f"legacy manifest.notes[{index}]")
        for index, note in enumerate(raw_notes)
    )


def _legacy_experiment_id(legacy_manifest: Mapping[str, Any]) -> str:
    run_id = _require_text(legacy_manifest.get("run_id"), "legacy manifest.run_id")
    prefix = run_id.split("-", 1)[0]
    return prefix if prefix else run_id


def _resolve_repo_root(run_path: Path, repo_root: str | Path | None) -> Path:
    if repo_root is not None:
        return _resolve_existing_dir(repo_root, "repo_root")
    for candidate in (run_path, *run_path.parents):
        if (candidate / "aic_signal_harness").is_dir() and (candidate / "aic_lewm_policy").is_dir():
            return candidate
    raise HarnessIOError("could not infer repo_root from run_dir")


def _resolve_existing_dir(path: str | Path, field_name: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise HarnessIOError(f"{field_name} must be an existing directory: {resolved}")
    return resolved


def _resolve_legacy_existing_path(
    raw_path: str,
    *,
    run_path: Path,
    repo_root: Path,
) -> str:
    path = Path(raw_path)
    if path.is_absolute():
        resolved = path.resolve()
        if not resolved.exists():
            raise HarnessIOError(f"legacy artifact path does not exist: {resolved}")
        return str(resolved)
    run_candidate = (run_path / path).resolve()
    repo_candidate = (repo_root / path).resolve()
    run_exists = run_candidate.exists()
    repo_exists = repo_candidate.exists()
    if run_exists and repo_exists and run_candidate != repo_candidate:
        raise HarnessIOError(f"ambiguous legacy path {raw_path!r}")
    if run_exists:
        return str(run_candidate)
    if repo_exists:
        return str(repo_candidate)
    raise HarnessIOError(f"legacy artifact path does not exist: {raw_path!r}")


def _resolve_canonical_legacy_existing_path(
    raw_path: str,
    *,
    run_path: Path,
    field_name: str = "legacy path",
) -> str:
    path = Path(raw_path)
    if path.is_absolute():
        resolved = path.expanduser().resolve()
        if not resolved.exists():
            raise HarnessIOError(f"legacy artifact path does not exist: {resolved}")
        if not resolved.is_relative_to(_CANONICAL_REPO_ROOT):
            raise HarnessIOError(f"{field_name} must point inside canonical legacy evidence")
        return str(resolved)
    return _resolve_legacy_existing_path(
        raw_path,
        run_path=run_path,
        repo_root=_CANONICAL_REPO_ROOT,
    )


def _validate_legacy_artifact_bytes(
    consumed_path: Path,
    canonical_path: Path,
    *,
    field_name: str,
) -> None:
    if consumed_path == canonical_path:
        return
    if consumed_path.is_file() and canonical_path.is_file():
        if sha256_file(consumed_path) == sha256_file(canonical_path):
            return
        raise HarnessIOError(
            f"{field_name} does not match canonical legacy artifact evidence"
        )
    raise HarnessIOError(f"{field_name} must point at canonical legacy artifact evidence")


def _artifact_file_sha256(path: str | None) -> str | None:
    if path is None:
        return None
    candidate = Path(path)
    if candidate.is_file():
        return hashlib.sha256(candidate.read_bytes()).hexdigest()
    return None


def _read_optional_json(path: Path) -> Mapping[str, Any] | None:
    if not path.exists():
        return None
    return read_json(path)


def _read_legacy_jsonl(path: Path) -> tuple[Mapping[str, Any], ...]:
    if not path.exists():
        raise HarnessIOError(f"legacy jsonl path does not exist: {path}")
    entries: list[Mapping[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise HarnessIOError(
                        f"invalid legacy JSONL in {path}:{line_number}: {exc.msg}"
                    ) from exc
                if not isinstance(payload, Mapping):
                    raise HarnessIOError(
                        f"legacy jsonl line {line_number} in {path} must be a JSON object"
                    )
                entries.append(payload)
    except OSError as exc:
        raise HarnessIOError(f"failed to read legacy jsonl {path}: {exc}") from exc
    return tuple(entries)


def _read_canonical_legacy_ledger_entries() -> tuple[Mapping[str, Any], ...]:
    return _read_legacy_jsonl(_CANONICAL_EXPERIMENTS_ROOT / "ledger.jsonl")


def _canonical_legacy_ledger_entry(
    run_id: str,
    *,
    field_name: str,
) -> Mapping[str, Any]:
    return _unique_legacy_ledger_entry(
        _read_canonical_legacy_ledger_entries(),
        run_id=run_id,
        field_name=f"canonical {field_name}",
    )


def _supplied_legacy_ledger_entry(
    repo_root: Path,
    *,
    run_id: str,
    field_name: str,
) -> Mapping[str, Any]:
    return _unique_legacy_ledger_entry(
        _read_legacy_jsonl(repo_root / "aic_lewm_policy" / "experiments" / "ledger.jsonl"),
        run_id=run_id,
        field_name=field_name,
    )


def _unique_legacy_ledger_entry(
    entries: tuple[Mapping[str, Any], ...],
    *,
    run_id: str,
    field_name: str,
) -> Mapping[str, Any]:
    matches = tuple(
        entry
        for entry in entries
        if _require_text(entry.get("run_id"), f"{field_name}.run_id") == run_id
    )
    if len(matches) != 1:
        raise HarnessIOError(f"could not derive a unique {field_name} row")
    return matches[0]


def _validate_supplied_ledger_entry_matches_canonical(
    *,
    repo_root: Path,
    canonical_entry: Mapping[str, Any],
    field_name: str,
) -> None:
    run_id = _require_text(canonical_entry.get("run_id"), f"canonical {field_name}.run_id")
    supplied_entry = _supplied_legacy_ledger_entry(
        repo_root,
        run_id=run_id,
        field_name=field_name,
    )
    if not _json_mappings_equal(supplied_entry, canonical_entry):
        raise HarnessIOError(f"{field_name} does not match canonical ledger evidence")


def _json_mappings_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def _require_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HarnessIOError(f"{field_name} must be a mapping")
    return value


def _require_schema_version(value: Mapping[str, Any], field_name: str) -> None:
    schema_version = value.get("schema_version")
    if type(schema_version) is not int or schema_version != SCHEMA_VERSION:
        raise HarnessIOError(f"{field_name}.schema_version must be {SCHEMA_VERSION}")


def _require_utc_timestamp(value: Any, field_name: str) -> str:
    timestamp = _require_text(value, field_name)
    try:
        parsed = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise HarnessIOError(f"{field_name} must be a UTC timestamp ending in Z") from exc
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != timestamp:
        raise HarnessIOError(f"{field_name} must be a canonical UTC timestamp")
    return timestamp


def _require_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HarnessIOError(f"{field_name} must be a nonempty string")
    return value


def _require_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise HarnessIOError(f"{field_name} must be a finite number")
    return float(value)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    return _require_text(value, "legacy optional text")


def _require_optional_text_field(
    payload: Mapping[str, Any],
    key: str,
    field_name: str,
) -> str | None:
    if key not in payload:
        raise HarnessIOError(f"{field_name} must be present")
    value = payload[key]
    return None if value is None else _require_text(value, field_name)


def _require_bool(value: Any, field_name: str) -> bool:
    if type(value) is not bool:
        raise HarnessIOError(f"{field_name} must be a boolean")
    return value


def _require_text_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise HarnessIOError(f"{field_name} must be a list")
    return tuple(
        _require_text(item, f"{field_name}[{index}]")
        for index, item in enumerate(value)
    )


def _is_uri_source(value: str) -> bool:
    return "://" in value
