from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, cast

import h5py
import pytest

from aic_signal_harness import (
    ArtifactRef,
    EpisodeTrace,
    FailureKind,
    FailureLabel,
    FailureReport,
    FailureSeverity,
    HarnessIOError,
    Hdf5DatasetReport,
    Hdf5DatasetStats,
    Hdf5DatasetThresholds,
    LeakageClass,
    McapEvalBundleReport,
    McapEvalFirstContact,
    McapEvalTaskHints,
    McapEvalTrialReport,
    PolicyTraceEvent,
    PolicyTraceEventType,
    REQUIRED_HDF5_DATASET_KEYS,
    RewardReport,
    RewardSignalKind,
    RewardTerm,
    ScoreReport,
    SchemaValidationError,
    TimelineEvent,
    TimelineEventKind,
    TrainingSignal,
    TrainingSignalKind,
    TrainingSignalReport,
    TrialTrace,
    TrialScore,
    derive_episode_trace,
    derive_training_signal_report,
    sha256_file,
    write_json,
)

_SCORE_SOURCE = "memory://pytest/scoring.yaml"


def _policy_events(run_id: str = "run-a", official_trial_id: str | None = "trial_1"):
    return (
        PolicyTraceEvent(
            run_id=run_id,
            trial_id="task_1__policy_call_0001",
            event_index=0,
            event_type=PolicyTraceEventType.task_started,
            elapsed_sec=0.0,
            emitted_at_utc="2026-04-24T00:00:00Z",
            source="pytest",
            leakage_class=LeakageClass.legal_policy_input,
            payload={"task_id": "task_1"},
            official_trial_id=official_trial_id,
        ),
        PolicyTraceEvent(
            run_id=run_id,
            trial_id="task_1__policy_call_0001",
            event_index=1,
            event_type=PolicyTraceEventType.action_published,
            elapsed_sec=0.25,
            emitted_at_utc="2026-04-24T00:00:01Z",
            source="pytest",
            leakage_class=LeakageClass.legal_policy_action_output,
            official_trial_id=official_trial_id,
            payload={
                "linear": [0.1, 0.0, 0.0],
                "angular": [0.0, 0.0, 0.0],
                "frame_id": "base_link",
            },
        ),
        PolicyTraceEvent(
            run_id=run_id,
            trial_id="task_1__policy_call_0001",
            event_index=2,
            event_type=PolicyTraceEventType.safety_guard,
            elapsed_sec=0.5,
            emitted_at_utc="2026-04-24T00:00:02Z",
            source="pytest",
            leakage_class=LeakageClass.legal_policy_input,
            payload={"guard": "delta_force", "stopped": True},
            official_trial_id=official_trial_id,
        ),
    )


def _three_trial_policy_events(run_id: str = "run-a") -> tuple[PolicyTraceEvent, ...]:
    events: list[PolicyTraceEvent] = []
    event_index = 0
    for trial_index in (1, 2, 3):
        policy_trial_id = f"task_{trial_index}__policy_call_0001"
        official_trial_id = f"trial_{trial_index}"
        trial_events = [
            PolicyTraceEvent(
                run_id=run_id,
                trial_id=policy_trial_id,
                event_index=event_index,
                event_type=PolicyTraceEventType.task_started,
                elapsed_sec=float(trial_index - 1),
                emitted_at_utc=f"2026-04-24T00:00:0{event_index}Z",
                source="pytest",
                leakage_class=LeakageClass.legal_policy_input,
                payload={"task_id": f"task_{trial_index}"},
                official_trial_id=official_trial_id,
            ),
            PolicyTraceEvent(
                run_id=run_id,
                trial_id=policy_trial_id,
                event_index=event_index + 1,
                event_type=PolicyTraceEventType.observation,
                elapsed_sec=float(trial_index - 1) + 0.1,
                emitted_at_utc=f"2026-04-24T00:00:0{event_index + 1}Z",
                source="pytest",
                leakage_class=LeakageClass.legal_policy_input,
                payload={"camera": "left_pixels", "state_shape": [32]},
                official_trial_id=official_trial_id,
            ),
            PolicyTraceEvent(
                run_id=run_id,
                trial_id=policy_trial_id,
                event_index=event_index + 2,
                event_type=PolicyTraceEventType.action_published,
                elapsed_sec=float(trial_index - 1) + 0.2,
                emitted_at_utc=f"2026-04-24T00:00:0{event_index + 2}Z",
                source="pytest",
                leakage_class=LeakageClass.legal_policy_action_output,
                payload={
                    "linear": [0.01 * trial_index, 0.0, 0.0],
                    "angular": [0.0, 0.0, 0.0],
                },
                official_trial_id=official_trial_id,
            ),
        ]
        events.extend(trial_events)
        event_index += len(trial_events)
    return tuple(events)


def _policy_events_with_duplicate_official_trial() -> tuple[PolicyTraceEvent, ...]:
    events = []
    for event in _three_trial_policy_events():
        official_trial_id = "trial_1" if event.trial_id == "task_2__policy_call_0001" else event.official_trial_id
        events.append(
            PolicyTraceEvent(
                run_id=event.run_id,
                trial_id=event.trial_id,
                event_index=event.event_index,
                event_type=event.event_type,
                elapsed_sec=event.elapsed_sec,
                emitted_at_utc=event.emitted_at_utc,
                source=event.source,
                leakage_class=event.leakage_class,
                payload=event.payload,
                official_trial_id=official_trial_id,
            )
        )
    return tuple(events)


def _scoring_yaml_source(total: float, trials: dict[str, TrialScore]) -> str:
    identity = json.dumps(
        {"total": total, "trials": {key: value.to_dict() for key, value in trials.items()}},
        allow_nan=False,
        sort_keys=True,
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    scoring_root = Path(tempfile.gettempdir()) / "aic_signal_harness_episode_trace_scoring"
    scoring_root.mkdir(parents=True, exist_ok=True)
    scoring_path = scoring_root / f"scoring-{digest}.yaml"
    lines = [f"total: {total}"]
    for trial_id, trial_score in trials.items():
        lines.extend(
            [
                f"{trial_id}:",
                "  tier_1:",
                f"    score: {trial_score.tier_1}",
                "  tier_2:",
                f"    score: {trial_score.tier_2}",
                "  tier_3:",
                f"    score: {trial_score.tier_3}",
            ]
        )
    scoring_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(scoring_path.resolve())


def _score_report() -> ScoreReport:
    trials = {"trial_1": TrialScore(total=7.5, tier_1=1.0, tier_2=2.5, tier_3=4.0)}
    return ScoreReport(
        source=_scoring_yaml_source(7.5, trials),
        parsed_at_utc="2026-04-24T00:00:03Z",
        total=7.5,
        trials=trials,
    )


def _three_trial_score_report() -> ScoreReport:
    trials = {
        "trial_1": TrialScore(total=7.5, tier_1=1.0, tier_2=2.5, tier_3=4.0),
        "trial_2": TrialScore(total=2.5, tier_1=1.0, tier_2=1.5, tier_3=0.0),
        "trial_3": TrialScore(total=2.0, tier_1=1.0, tier_2=1.0, tier_3=0.0),
    }
    return ScoreReport(
        source=_scoring_yaml_source(12.0, trials),
        parsed_at_utc="2026-04-24T00:00:09Z",
        total=12.0,
        trials=trials,
    )


def _mcap_bundle_report() -> McapEvalBundleReport:
    bundle_root = Path(tempfile.gettempdir()) / "aic_signal_harness_episode_trace_mcap" / "run-a"
    bundle_root.mkdir(parents=True, exist_ok=True)

    def trial_file(trial_index: int) -> tuple[str, int, str]:
        trial_dir = bundle_root / f"bag_trial_{trial_index}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        trial_path = trial_dir / f"bag_trial_{trial_index}_0.mcap"
        content = f"mcap-trial-{trial_index}\n".encode("utf-8")
        trial_path.write_bytes(content)
        return str(trial_path.resolve()), len(content), hashlib.sha256(content).hexdigest()

    trial_1_source, trial_1_size, trial_1_sha256 = trial_file(1)
    trial_2_source, trial_2_size, trial_2_sha256 = trial_file(2)
    trial_3_source, trial_3_size, trial_3_sha256 = trial_file(3)
    return McapEvalBundleReport(
        source=str(bundle_root.resolve()),
        analyzed_at_utc="2026-04-24T00:00:10Z",
        contact_margin_sec=0.25,
        stop_step_sec=0.05,
        recommended_env={"AIC_LEWM_SC_REPLAY_STOP_SEC": "0.15"},
        trials=(
            McapEvalTrialReport(
                trial_id="trial_1",
                source=trial_1_source,
                size_bytes=trial_1_size,
                sha256=trial_1_sha256,
                controller_state_count=2,
                pose_command_count=1,
                off_limit_contact_count=1,
                controller_stamp_start_sec=10.0,
                controller_stamp_end_sec=10.4,
                controller_duration_sec=0.4,
                final_tcp_position=(1.0, 2.0, 3.0),
                final_tcp_error=(0.1, 0.2, 0.3),
                task_hints=McapEvalTaskHints(port_type="sc", task_id="task_1"),
                first_off_limit_contact=McapEvalFirstContact(
                    log_time_ns=1_300_000_000,
                    collision1="plug",
                    collision2="enclosure",
                    log_elapsed_sec=0.3,
                    nearest_controller_elapsed_sec=0.4,
                    nearest_command_elapsed_sec=0.15,
                    nearest_tcp_position=(1.0, 2.0, 3.0),
                    nearest_tcp_error=(0.1, 0.2, 0.3),
                    nearest_command_linear=(0.01, 0.02, 0.03),
                    nearest_command_angular=(0.04, 0.05, 0.06),
                    recommended_stop_sec=0.15,
                ),
            ),
            McapEvalTrialReport(
                trial_id="trial_2",
                source=trial_2_source,
                size_bytes=trial_2_size,
                sha256=trial_2_sha256,
                controller_state_count=0,
                pose_command_count=0,
                off_limit_contact_count=0,
            ),
            McapEvalTrialReport(
                trial_id="trial_3",
                source=trial_3_source,
                size_bytes=trial_3_size,
                sha256=trial_3_sha256,
                controller_state_count=0,
                pose_command_count=0,
                off_limit_contact_count=0,
            ),
        ),
    )


def _hdf5_dataset_report() -> Hdf5DatasetReport:
    dataset_root = Path(tempfile.gettempdir()) / "aic_signal_harness_episode_trace_hdf5"
    dataset_root.mkdir(parents=True, exist_ok=True)
    dataset_path = dataset_root / "demo.hdf5"
    _write_hdf5_dataset_fixture(dataset_path)
    datasets = {
        "ep_len": Hdf5DatasetStats(shape=(3,), dtype="int32"),
        "ep_offset": Hdf5DatasetStats(shape=(3,), dtype="int64"),
        "ep_idx": Hdf5DatasetStats(shape=(6,), dtype="int32"),
        "episode_idx": Hdf5DatasetStats(shape=(6,), dtype="int32"),
        "step_idx": Hdf5DatasetStats(shape=(6,), dtype="int32"),
        "pixels": Hdf5DatasetStats(shape=(6, 8, 8, 3), dtype="uint8"),
        "left_pixels": Hdf5DatasetStats(shape=(6, 8, 8, 3), dtype="uint8"),
        "right_pixels": Hdf5DatasetStats(shape=(6, 8, 8, 3), dtype="uint8"),
        "proprio": Hdf5DatasetStats(shape=(6, 32), dtype="float32"),
        "state": Hdf5DatasetStats(shape=(6, 32), dtype="float32"),
        "action": Hdf5DatasetStats(shape=(6, 6), dtype="float32"),
        "task_id": Hdf5DatasetStats(shape=(6,), dtype="int32"),
        "plug_type": Hdf5DatasetStats(shape=(6,), dtype="int32"),
        "port_type": Hdf5DatasetStats(shape=(6,), dtype="int32"),
        "target_module_name": Hdf5DatasetStats(shape=(6,), dtype="int32"),
    }
    return Hdf5DatasetReport(
        source=str(dataset_path.resolve()),
        size_bytes=dataset_path.stat().st_size,
        sha256=sha256_file(dataset_path),
        validated_at_utc="2026-04-24T00:00:11Z",
        thresholds=Hdf5DatasetThresholds(min_episodes=3, min_steps=6),
        required_datasets=REQUIRED_HDF5_DATASET_KEYS,
        missing_datasets=(),
        datasets=datasets,
        episode_count=3,
        step_count=6,
        episode_lengths=(2, 2, 2),
        episode_offsets=(0, 2, 4),
        errors=(),
        ok=True,
    )


def _write_hdf5_dataset_fixture(dataset_path: Path) -> None:
    with h5py.File(dataset_path, "w") as handle:
        handle.create_dataset("ep_len", data=[2, 2, 2], dtype="int32")
        handle.create_dataset("ep_offset", data=[0, 2, 4], dtype="int64")
        handle.create_dataset("ep_idx", data=[0, 0, 1, 1, 2, 2], dtype="int32")
        handle.create_dataset("episode_idx", data=[0, 0, 1, 1, 2, 2], dtype="int32")
        handle.create_dataset("step_idx", data=[0, 1, 0, 1, 0, 1], dtype="int32")
        pixels = [[[[0, 0, 0] for _ in range(8)] for _ in range(8)] for _ in range(6)]
        handle.create_dataset("pixels", data=pixels, dtype="uint8")
        handle.create_dataset("left_pixels", data=pixels, dtype="uint8")
        handle.create_dataset("right_pixels", data=pixels, dtype="uint8")
        proprio = [[0.0 for _ in range(32)] for _ in range(6)]
        handle.create_dataset("proprio", data=proprio, dtype="float32")
        handle.create_dataset("state", data=proprio, dtype="float32")
        action = [[0.0 for _ in range(6)] for _ in range(6)]
        handle.create_dataset("action", data=action, dtype="float32")
        task_ids = [0, 1, 2, 3, 4, 5]
        handle.create_dataset("task_id", data=task_ids, dtype="int32")
        handle.create_dataset("plug_type", data=task_ids, dtype="int32")
        handle.create_dataset("port_type", data=task_ids, dtype="int32")
        handle.create_dataset("target_module_name", data=task_ids, dtype="int32")


def _uri_mcap_bundle_report() -> McapEvalBundleReport:
    return McapEvalBundleReport(
        source="gs://bucket/eval",
        analyzed_at_utc="2026-04-24T00:00:10Z",
        contact_margin_sec=0.25,
        stop_step_sec=0.05,
        recommended_env={},
        trials=(
            McapEvalTrialReport(
                trial_id="trial_1",
                source="gs://bucket/eval/bag_trial_1/bag_trial_1_0.mcap",
                size_bytes=101,
                sha256="1" * 64,
                controller_state_count=0,
                pose_command_count=0,
                off_limit_contact_count=0,
            ),
            McapEvalTrialReport(
                trial_id="trial_2",
                source="gs://bucket/eval/bag_trial_2/bag_trial_2_0.mcap",
                size_bytes=102,
                sha256="2" * 64,
                controller_state_count=0,
                pose_command_count=0,
                off_limit_contact_count=0,
            ),
            McapEvalTrialReport(
                trial_id="trial_3",
                source="gs://bucket/eval/bag_trial_3/bag_trial_3_0.mcap",
                size_bytes=103,
                sha256="3" * 64,
                controller_state_count=0,
                pose_command_count=0,
                off_limit_contact_count=0,
            ),
        ),
    )


def _uri_hdf5_dataset_report() -> Hdf5DatasetReport:
    local_report = _hdf5_dataset_report()
    return Hdf5DatasetReport(
        source="gs://bucket/data/demo.hdf5",
        size_bytes=local_report.size_bytes,
        sha256=local_report.sha256,
        validated_at_utc=local_report.validated_at_utc,
        thresholds=local_report.thresholds,
        required_datasets=local_report.required_datasets,
        missing_datasets=local_report.missing_datasets,
        datasets=local_report.datasets,
        episode_count=local_report.episode_count,
        step_count=local_report.step_count,
        episode_lengths=local_report.episode_lengths,
        episode_offsets=local_report.episode_offsets,
        errors=local_report.errors,
        ok=local_report.ok,
    )


def _mcap_bundle_sha256(report: McapEvalBundleReport) -> str:
    digest = hashlib.sha256()
    bundle_root = None if "://" in report.source else Path(report.source)
    for trial in report.trials:
        trial_identity = (
            trial.source
            if bundle_root is None
            else Path(trial.source).relative_to(bundle_root).as_posix()
        )
        digest.update(trial.trial_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(trial_identity.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(trial.size_bytes).encode("utf-8"))
        digest.update(b"\0")
        digest.update(trial.sha256.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _report_sha256(mapping: dict[str, Any]) -> str:
    payload = (
        json.dumps(mapping, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _event_projection_sha256(events: list[dict[str, Any]]) -> str:
    payload = (
        json.dumps(events, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _mcap_artifact(
    report: McapEvalBundleReport,
    *,
    sha256: str | None = None,
    report_sha256: str | None = None,
    source_report: dict[str, Any] | None = None,
) -> ArtifactRef:
    provenance: dict[str, Any] = {
        "producer": "pytest",
        "run_id": "run-a",
        "report_sha256": (
            _report_sha256(report.to_dict())
            if report_sha256 is None
            else report_sha256
        ),
    }
    if source_report is not None:
        provenance["source_report"] = source_report
    source_key = "uri" if "://" in report.source else "path"
    return ArtifactRef(
        kind="mcap_eval_bundle",
        **{source_key: report.source},
        sha256=_mcap_bundle_sha256(report) if sha256 is None else sha256,
        provenance=provenance,
    )


def _hdf5_artifact(
    report: Hdf5DatasetReport,
    *,
    sha256: str | None = None,
    report_sha256: str | None = None,
    source_report: dict[str, Any] | None = None,
) -> ArtifactRef:
    provenance: dict[str, Any] = {
        "producer": "pytest",
        "run_id": "run-a",
        "report_sha256": (
            _report_sha256(report.to_dict())
            if report_sha256 is None
            else report_sha256
        ),
    }
    if source_report is not None:
        provenance["source_report"] = source_report
    source_key = "uri" if "://" in report.source else "path"
    return ArtifactRef(
        kind="hdf5_dataset",
        **{source_key: report.source},
        sha256=report.sha256 if sha256 is None else sha256,
        provenance=provenance,
    )


def _typed_report_artifact(kind: str, report: Any) -> ArtifactRef:
    report_payload = report.to_dict()
    report_digest = _report_sha256(report_payload)
    report_root = Path(tempfile.gettempdir()) / "aic_signal_harness_episode_trace_reports"
    report_path = report_root / f"{kind}-{report_digest}.json"
    write_json(report_path, report_payload, overwrite=True)
    derivation_by_kind = {
        "reward_report": "derive_reward_failure_reports",
        "failure_report": "derive_reward_failure_reports",
        "hdf5_dataset_report": "reduce_hdf5_dataset",
        "mcap_eval_report": "reduce_mcap_eval_bundle",
    }
    return ArtifactRef(
        kind=kind,
        path=str(report_path),
        sha256=sha256_file(report_path),
        provenance={
            "producer": "pytest",
            "run_id": "run-a",
            "derivation": derivation_by_kind.get(kind, f"derive_{kind}"),
        },
    )


def _reward_report() -> RewardReport:
    return RewardReport(
        run_id="run-a",
        generated_at_utc="2026-04-24T00:00:04Z",
        terms=(
            RewardTerm(
                name="official.score.total",
                value=7.5,
                signal_kind=RewardSignalKind.official_score,
                leakage_class=LeakageClass.privileged_eval_signal,
                source=_SCORE_SOURCE,
            ),
            RewardTerm(
                name="diagnostic.margin",
                value=123.0,
                signal_kind=RewardSignalKind.diagnostic,
                leakage_class=LeakageClass.privileged_eval_signal,
                source=_SCORE_SOURCE,
                trial_id="task_1__policy_call_0001",
            ),
        ),
    )


def _failure_report() -> FailureReport:
    return FailureReport(
        run_id="run-a",
        generated_at_utc="2026-04-24T00:00:04Z",
        labels=(
            FailureLabel(
                kind=FailureKind.no_partial_or_full_insertion,
                severity=FailureSeverity.blocker,
                summary="No insertion credit.",
                source=_SCORE_SOURCE,
                leakage_class=LeakageClass.privileged_eval_signal,
                trial_id="task_1__policy_call_0001",
                evidence={"max_tier_3": 4.0},
            ),
        ),
    )


def _source_artifacts(
    *,
    include_reports: bool = False,
    score_report: ScoreReport | None = None,
) -> tuple[ArtifactRef, ...]:
    typed_score_report = _score_report() if score_report is None else score_report
    policy_artifact = ArtifactRef(
        kind="policy_trace_jsonl",
        uri="memory://pytest/policy_trace.jsonl",
        sha256="a" * 64,
        provenance={"producer": "pytest", "run_id": "run-a"},
    )
    scoring_artifact = ArtifactRef(
        kind="scoring_yaml",
        path=typed_score_report.source,
        sha256=sha256_file(typed_score_report.source),
        provenance={
            "producer": "pytest",
            "run_id": "run-a",
            "report_sha256": _report_sha256(typed_score_report.to_dict()),
            "source_report": typed_score_report.to_dict(),
        },
    )
    artifacts: tuple[ArtifactRef, ...] = (scoring_artifact, policy_artifact)
    if include_reports:
        report_source_artifacts = [
            {"kind": scoring_artifact.kind, "sha256": scoring_artifact.sha256},
            {"kind": policy_artifact.kind, "sha256": policy_artifact.sha256},
        ]
        artifacts += (
            ArtifactRef(
                kind="reward_report",
                uri="memory://pytest/reward_report.json",
                sha256="d" * 64,
                provenance={
                    "producer": "pytest",
                    "run_id": "run-a",
                    "derivation": "derive_reward_failure_reports",
                    "source_artifacts": report_source_artifacts,
                },
            ),
            ArtifactRef(
                kind="failure_report",
                uri="memory://pytest/failure_report.json",
                sha256="e" * 64,
                provenance={
                    "producer": "pytest",
                    "run_id": "run-a",
                    "derivation": "derive_reward_failure_reports",
                    "source_artifacts": report_source_artifacts,
                },
            ),
        )
    return artifacts


def _episode_trace() -> EpisodeTrace:
    return derive_episode_trace(
        run_id="run-a",
        policy_events=_policy_events(),
        source_artifacts=_source_artifacts(include_reports=True),
        score_report=_score_report(),
        reward_report=_reward_report(),
        failure_report=_failure_report(),
        generated_at_utc="2026-04-24T00:00:05Z",
    )


def _hdf5_episode_trace() -> EpisodeTrace:
    hdf5_report = _hdf5_dataset_report()
    score_report = _three_trial_score_report()
    return derive_episode_trace(
        run_id="run-a",
        policy_events=_three_trial_policy_events(),
        source_artifacts=(
            *_source_artifacts(score_report=score_report),
            _hdf5_artifact(hdf5_report),
            _typed_report_artifact("hdf5_dataset_report", hdf5_report),
        ),
        score_report=score_report,
        hdf5_dataset_report=hdf5_report,
        generated_at_utc="2026-04-24T00:00:11Z",
    )


def _episode_trace_sha256(trace: EpisodeTrace) -> str:
    payload = (
        json.dumps(trace.to_dict(), allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _source_trace_ref(
    trace: EpisodeTrace,
    *,
    uri: str = "memory://pytest/episode_trace.json",
) -> ArtifactRef:
    return ArtifactRef(
        kind="episode_trace",
        uri=uri,
        sha256=_episode_trace_sha256(trace),
        provenance={
            "producer": "pytest",
            "run_id": trace.run_id,
            "derivation": "derive_episode_trace",
        },
    )


def _training_signal_source_reports(
    reward_report: RewardReport | None = None,
    failure_report: FailureReport | None = None,
) -> tuple[RewardReport, FailureReport, ArtifactRef, ArtifactRef]:
    typed_reward = _reward_report() if reward_report is None else reward_report
    typed_failure = _failure_report() if failure_report is None else failure_report
    return (
        typed_reward,
        typed_failure,
        _typed_report_artifact("reward_report", typed_reward),
        _typed_report_artifact("failure_report", typed_failure),
    )


def _reindex_episode_payload(payload: dict[str, Any]) -> None:
    ordered_events = [
        event
        for trial in payload["trials"]
        for event in trial["events"]
    ] + payload["run_events"]
    for index, event in enumerate(ordered_events):
        event["event_index"] = index


def test_episode_trace_round_trip_and_training_signals() -> None:
    reward_report, failure_report, reward_artifact, failure_artifact = _training_signal_source_reports()
    trace = derive_episode_trace(
        run_id="run-a",
        policy_events=_policy_events(),
        source_artifacts=_source_artifacts(include_reports=True),
        score_report=_score_report(),
        reward_report=reward_report,
        failure_report=failure_report,
        generated_at_utc="2026-04-24T00:00:05Z",
    )

    assert EpisodeTrace.from_dict(trace.to_dict()) == trace
    assert trace.trials[0].action_event_count == 1
    assert trace.trials[0].nonzero_action_event_count == 1
    assert trace.trials[0].safety_guard_event_count == 1
    assert trace.trials[0].score is not None
    assert {event.event_kind for event in trace.run_events} == {
        TimelineEventKind.official_score,
        TimelineEventKind.reward_term,
        TimelineEventKind.failure_label,
    }

    report = derive_training_signal_report(
        episode_trace=trace,
        source_trace=_source_trace_ref(trace),
        reward_report=reward_report,
        failure_report=failure_report,
        source_reward_report=reward_artifact,
        source_failure_report=failure_artifact,
        generated_at_utc="2026-04-24T00:00:06Z",
    )

    assert TrainingSignalReport.from_dict(report.to_dict()) == report
    assert report.schema_version == 2
    assert {signal.kind for signal in report.signals} == {
        TrainingSignalKind.behavior_clone_action,
        TrainingSignalKind.safety_guard_avoidance,
        TrainingSignalKind.official_score_term,
        TrainingSignalKind.reward_term,
        TrainingSignalKind.failure_label,
    }
    assert report.generated_at_utc == "2026-04-24T00:00:06Z"
    assert report.source_reward_report == reward_artifact
    assert report.source_failure_report == failure_artifact
    assert len({signal.signal_id for signal in report.signals}) == len(report.signals)
    for signal in report.signals:
        assert signal.offline_only is True
        assert signal.runtime_allowed is False
        assert signal.consumable_by_policy_runtime is False
        assert signal.source_event_indices
        assert signal.extraction_method.startswith("episode_trace.v1.")
    reward_signal = next(signal for signal in report.signals if signal.kind is TrainingSignalKind.reward_term)
    assert reward_signal.weight == 1.0
    assert reward_signal.evidence["value"] == 123.0
    action_signal = next(
        signal for signal in report.signals if signal.kind is TrainingSignalKind.behavior_clone_action
    )
    assert action_signal.source_event_indices == (1,)
    assert all(
        signal.kind is not TrainingSignalKind.reward_term
        or signal.evidence["signal_kind"] != "official_score"
        for signal in report.signals
    )
    assert all(event.elapsed_sec == 0.5 for event in trace.run_events)
    assert all(event.payload["event_scope"].startswith("post_hoc") for event in trace.run_events)
    trial_scoped = [event for event in trace.run_events if event.trial_id == "task_1__policy_call_0001"]
    assert trial_scoped
    assert all(event.payload["evidence_window"]["end_elapsed_sec"] == 0.5 for event in trial_scoped)


def test_episode_trace_fuses_hdf5_and_observation_evidence() -> None:
    trace = _hdf5_episode_trace()

    assert EpisodeTrace.from_dict(trace.to_dict()) == trace
    assert trace.generated_at_utc == "2026-04-24T00:00:11Z"
    assert trace.trials[0].observation_event_count == 1
    assert trace.trials[0].events[1].event_kind is TimelineEventKind.observation_event
    assert trace.trials[0].score is not None
    assert trace.trials[0].score.total == 7.5

    event_kinds = [event.event_kind for event in trace.run_events]
    assert event_kinds.count(TimelineEventKind.dataset_episode_summary) == 1
    assert event_kinds.count(TimelineEventKind.official_score) == 1

    dataset_event = next(
        event
        for event in trace.run_events
        if event.event_kind is TimelineEventKind.dataset_episode_summary
    )
    assert dataset_event.leakage_class is LeakageClass.privileged_training_signal
    dataset_payload = dataset_event.to_dict()["payload"]
    assert dataset_payload["episode_count"] == 3
    assert dataset_payload["step_count"] == 6
    assert dataset_payload["observation_datasets"]["pixels"]["shape"] == [6, 8, 8, 3]
    assert dataset_payload["action_datasets"]["action"]["shape"] == [6, 6]
    report = derive_training_signal_report(
        episode_trace=trace,
        source_trace=_source_trace_ref(trace),
    )
    dataset_signal = next(
        signal for signal in report.signals if signal.kind is TrainingSignalKind.dataset_episode_summary
    )
    assert dataset_signal.target == "training.dataset"
    assert dataset_signal.source_event_indices == (dataset_event.event_index,)
    action_signals = [
        signal for signal in report.signals if signal.kind is TrainingSignalKind.behavior_clone_action
    ]
    assert all(len(signal.source_event_indices) == 2 for signal in action_signals)


def test_training_signal_report_defaults_to_episode_trace_timestamp() -> None:
    trace = _episode_trace()
    reward_report, failure_report, reward_artifact, failure_artifact = _training_signal_source_reports()

    first_report = derive_training_signal_report(
        episode_trace=trace,
        source_trace=_source_trace_ref(trace),
        reward_report=reward_report,
        failure_report=failure_report,
        source_reward_report=reward_artifact,
        source_failure_report=failure_artifact,
    )
    second_report = derive_training_signal_report(
        episode_trace=trace,
        source_trace=_source_trace_ref(trace),
        reward_report=reward_report,
        failure_report=failure_report,
        source_reward_report=reward_artifact,
        source_failure_report=failure_artifact,
    )

    assert first_report == second_report
    assert first_report.generated_at_utc == trace.generated_at_utc


def test_episode_trace_defaults_to_latest_source_evidence_timestamp() -> None:
    first_trace = derive_episode_trace(
        run_id="run-a",
        policy_events=_policy_events(),
        source_artifacts=_source_artifacts(),
        score_report=_score_report(),
    )
    second_trace = derive_episode_trace(
        run_id="run-a",
        policy_events=_policy_events(),
        source_artifacts=_source_artifacts(),
        score_report=_score_report(),
    )

    assert first_trace == second_trace
    assert first_trace.generated_at_utc == "2026-04-24T00:00:03Z"

    hdf5_trace = _hdf5_episode_trace()
    assert hdf5_trace.generated_at_utc == "2026-04-24T00:00:11Z"


def test_episode_trace_rejects_policy_run_id_mismatch() -> None:
    with pytest.raises(HarnessIOError, match="policy trace events do not match"):
        derive_episode_trace(
            run_id="run-a",
            policy_events=_policy_events(run_id="other-run"),
            source_artifacts=(),
            generated_at_utc="2026-04-24T00:00:05Z",
        )


def test_episode_trace_rejects_non_policy_event_inputs() -> None:
    with pytest.raises(HarnessIOError, match="PolicyTraceEvent instances"):
        derive_episode_trace(
            run_id="run-a",
            policy_events=cast(Any, ("not-a-policy-event",)),
            source_artifacts=_source_artifacts(),
            score_report=_score_report(),
            generated_at_utc="2026-04-24T00:00:05Z",
        )


def test_episode_trace_rejects_reward_report_run_id_mismatch() -> None:
    bad_reward = RewardReport(
        run_id="other-run",
        generated_at_utc="2026-04-24T00:00:04Z",
        terms=(),
    )

    with pytest.raises(HarnessIOError, match="reward_report.run_id"):
        derive_episode_trace(
            run_id="run-a",
            policy_events=_policy_events(),
            source_artifacts=(),
            reward_report=bad_reward,
            generated_at_utc="2026-04-24T00:00:05Z",
        )


def test_episode_trace_rejects_ambiguous_or_mismatched_score_trial_mapping() -> None:
    second_policy_call = PolicyTraceEvent(
        run_id="run-a",
        trial_id="task_1__policy_call_0002",
        event_index=3,
        event_type=PolicyTraceEventType.task_started,
        elapsed_sec=0.0,
        emitted_at_utc="2026-04-24T00:00:03Z",
        source="pytest",
        leakage_class=LeakageClass.legal_policy_input,
        payload={"task_id": "task_1"},
        official_trial_id="trial_1",
    )

    with pytest.raises(HarnessIOError, match="ambiguously"):
        derive_episode_trace(
            run_id="run-a",
            policy_events=_policy_events() + (second_policy_call,),
            source_artifacts=_source_artifacts(),
            score_report=_score_report(),
            generated_at_utc="2026-04-24T00:00:05Z",
        )

    two_trial_score_trials = {
        "trial_1": TrialScore(total=7.5, tier_1=1.0, tier_2=2.5, tier_3=4.0),
        "trial_2": TrialScore(total=0.5, tier_1=0.5, tier_2=0.0, tier_3=0.0),
    }
    two_trial_score = ScoreReport(
        source=_scoring_yaml_source(8.0, two_trial_score_trials),
        parsed_at_utc="2026-04-24T00:00:03Z",
        total=8.0,
        trials=two_trial_score_trials,
    )
    with pytest.raises(HarnessIOError, match="unmatched official scores"):
        derive_episode_trace(
            run_id="run-a",
            policy_events=_policy_events(),
            source_artifacts=_source_artifacts(score_report=two_trial_score),
            score_report=two_trial_score,
            generated_at_utc="2026-04-24T00:00:05Z",
        )


def test_episode_trace_rejects_scored_policy_events_without_official_trial_id() -> None:
    with pytest.raises(HarnessIOError, match="official_trial_id"):
        derive_episode_trace(
            run_id="run-a",
            policy_events=_policy_events(official_trial_id=None),
            source_artifacts=_source_artifacts(),
            score_report=_score_report(),
            generated_at_utc="2026-04-24T00:00:05Z",
        )


def test_episode_trace_rejects_unbound_mcap_and_hdf5_evidence() -> None:
    mcap_report = _mcap_bundle_report()
    hdf5_report = _hdf5_dataset_report()
    score_report = _three_trial_score_report()

    forged_hdf5_payload = hdf5_report.to_dict()
    forged_hdf5_payload["datasets"]["pixels"]["shape"] = [6, 99, 8, 3]
    forged_hdf5_report = Hdf5DatasetReport.from_dict(forged_hdf5_payload)
    with pytest.raises(HarnessIOError, match="must match supplied hdf5_dataset_report bytes"):
        derive_episode_trace(
            run_id="run-a",
            policy_events=_three_trial_policy_events(),
            source_artifacts=(
                *_source_artifacts(score_report=score_report),
                _hdf5_artifact(forged_hdf5_report),
                _typed_report_artifact("hdf5_dataset_report", forged_hdf5_report),
            ),
            score_report=score_report,
            hdf5_dataset_report=forged_hdf5_report,
            generated_at_utc="2026-04-24T00:00:05Z",
        )

    with pytest.raises(
        HarnessIOError,
        match="MCAP evidence requires the dedicated byte rederive slice/analyzer injection",
        ):
            derive_episode_trace(
                run_id="run-a",
                policy_events=_three_trial_policy_events(),
                source_artifacts=_source_artifacts(score_report=score_report),
                score_report=score_report,
                mcap_eval_bundle=mcap_report,
                generated_at_utc="2026-04-24T00:00:05Z",
            )


def test_episode_trace_allows_fused_hdf5_with_attached_report_artifact() -> None:
    hdf5_report = _hdf5_dataset_report()

    trace = derive_episode_trace(
        run_id="run-a",
        policy_events=_policy_events(),
        source_artifacts=(
            *_source_artifacts(),
            _hdf5_artifact(hdf5_report),
            _typed_report_artifact("hdf5_dataset_report", hdf5_report),
        ),
        hdf5_dataset_report=hdf5_report,
        generated_at_utc="2026-04-24T00:00:05Z",
    )

    assert EpisodeTrace.from_dict(trace.to_dict()) == trace
    assert any(
        artifact.kind == "hdf5_dataset_report"
        for artifact in trace.source_artifacts
    )


def test_episode_trace_allows_quiet_uri_only_mcap_and_hdf5_artifacts() -> None:
    mcap_report = _uri_mcap_bundle_report()
    hdf5_report = _uri_hdf5_dataset_report()

    trace = derive_episode_trace(
        run_id="run-a",
        policy_events=_policy_events(),
        source_artifacts=(
            *_source_artifacts(),
            _mcap_artifact(mcap_report),
            _hdf5_artifact(hdf5_report),
        ),
        generated_at_utc="2026-04-24T00:00:05Z",
    )

    assert EpisodeTrace.from_dict(trace.to_dict()) == trace
    assert trace.run_events == ()
    assert {artifact.kind for artifact in trace.source_artifacts} == {
        "scoring_yaml",
        "policy_trace_jsonl",
        "mcap_eval_bundle",
        "hdf5_dataset",
    }
    quiet_mcap = next(artifact for artifact in trace.source_artifacts if artifact.kind == "mcap_eval_bundle")
    quiet_hdf5 = next(artifact for artifact in trace.source_artifacts if artifact.kind == "hdf5_dataset")
    assert quiet_mcap.path is None and quiet_mcap.uri == mcap_report.source
    assert quiet_hdf5.path is None and quiet_hdf5.uri == hdf5_report.source

    with pytest.raises(
        HarnessIOError,
        match="MCAP evidence requires the dedicated byte rederive slice/analyzer injection",
    ):
        derive_episode_trace(
            run_id="run-a",
            policy_events=_policy_events(),
            source_artifacts=(*_source_artifacts(), _mcap_artifact(mcap_report)),
            mcap_eval_bundle=mcap_report,
            generated_at_utc="2026-04-24T00:00:05Z",
        )

    with pytest.raises(
        HarnessIOError,
        match="hdf5_dataset source artifact must be a local byte-verifiable file",
    ):
        derive_episode_trace(
            run_id="run-a",
            policy_events=_policy_events(),
            source_artifacts=(
                *_source_artifacts(),
                _hdf5_artifact(hdf5_report),
                _typed_report_artifact("hdf5_dataset_report", hdf5_report),
            ),
            hdf5_dataset_report=hdf5_report,
            generated_at_utc="2026-04-24T00:00:05Z",
        )


def test_episode_trace_rejects_malformed_post_hoc_evidence_events() -> None:
    hdf5_report = _hdf5_dataset_report()
    trace = _hdf5_episode_trace()
    payload = trace.to_dict()

    missing_payload = dict(payload)
    missing_payload["run_events"] = [dict(event) for event in payload["run_events"]]
    missing_payload["run_events"][0] = {
        **missing_payload["run_events"][0],
        "payload": {},
    }
    with pytest.raises(HarnessIOError, match="source_report_sha256"):
        EpisodeTrace.from_dict(missing_payload)

    bad_leakage = dict(payload)
    bad_leakage["run_events"] = [dict(event) for event in payload["run_events"]]
    bad_leakage["run_events"][0] = {
        **bad_leakage["run_events"][0],
        "leakage_class": LeakageClass.legal_policy_input.value,
    }
    with pytest.raises(HarnessIOError, match="privileged_training_signal"):
        EpisodeTrace.from_dict(bad_leakage)

    extra_dataset_payload = dict(payload)
    extra_dataset_payload["run_events"] = [dict(event) for event in payload["run_events"]]
    extra_dataset_payload["run_events"][0] = {
        **extra_dataset_payload["run_events"][0],
        "payload": {
            **extra_dataset_payload["run_events"][0]["payload"],
            "extra_label": "must not be accepted",
        },
    }
    extra_dataset_events = [
        event
        for event in extra_dataset_payload["run_events"]
        if event["event_kind"] == TimelineEventKind.dataset_episode_summary.value
    ]
    extra_dataset_payload["source_artifacts"] = [dict(artifact) for artifact in payload["source_artifacts"]]
    for artifact in extra_dataset_payload["source_artifacts"]:
        if artifact["kind"] == "hdf5_dataset":
            artifact["provenance"] = {
                **artifact["provenance"],
                "source_report_event_projection_sha256": _event_projection_sha256(extra_dataset_events),
                "canonical_event_projection_sha256": _event_projection_sha256(extra_dataset_events),
            }
    with pytest.raises(HarnessIOError, match="payload must exactly match source_report"):
        EpisodeTrace.from_dict(extra_dataset_payload)

    bad_dataset_report_sha = dict(payload)
    bad_dataset_report_sha["source_artifacts"] = [dict(artifact) for artifact in payload["source_artifacts"]]
    for artifact in bad_dataset_report_sha["source_artifacts"]:
        if artifact["kind"] == "hdf5_dataset":
            artifact["provenance"] = {
                **artifact["provenance"],
                "report_sha256": "0" * 64,
            }
    with pytest.raises(HarnessIOError, match="provenance.report_sha256"):
        EpisodeTrace.from_dict(bad_dataset_report_sha)

    forged_hdf5_payload = hdf5_report.to_dict()
    forged_hdf5_payload["datasets"]["pixels"]["shape"] = [6, 99, 8, 3]
    forged_hdf5_report = Hdf5DatasetReport.from_dict(forged_hdf5_payload)
    forged_dataset_payload = dict(payload)
    forged_dataset_payload["run_events"] = [dict(event) for event in payload["run_events"]]
    forged_dataset_payload["run_events"][0] = {
        **forged_dataset_payload["run_events"][0],
        "payload": {
            **forged_dataset_payload["run_events"][0]["payload"],
            "source_report_sha256": _report_sha256(forged_hdf5_report.to_dict()),
            "observation_datasets": {
                **forged_dataset_payload["run_events"][0]["payload"]["observation_datasets"],
                "pixels": {
                    **forged_dataset_payload["run_events"][0]["payload"]["observation_datasets"]["pixels"],
                    "shape": [6, 99, 8, 3],
                },
            },
        },
    }
    forged_dataset_events = [
        event
        for event in forged_dataset_payload["run_events"]
        if event["event_kind"] == TimelineEventKind.dataset_episode_summary.value
    ]
    forged_dataset_payload["source_artifacts"] = [dict(artifact) for artifact in payload["source_artifacts"]]
    for artifact in forged_dataset_payload["source_artifacts"]:
        if artifact["kind"] == "hdf5_dataset":
            artifact["provenance"] = {
                **artifact["provenance"],
                "report_sha256": _report_sha256(forged_hdf5_report.to_dict()),
                "source_report_event_projection_sha256": _event_projection_sha256(forged_dataset_events),
                "canonical_event_projection_sha256": _event_projection_sha256(forged_dataset_events),
            }
        elif artifact["kind"] == "hdf5_dataset_report":
            artifact.update(_typed_report_artifact("hdf5_dataset_report", forged_hdf5_report).to_dict())
    with pytest.raises(HarnessIOError, match="must match supplied hdf5_dataset_report bytes"):
        EpisodeTrace.from_dict(forged_dataset_payload)

    for event_kind in (TimelineEventKind.controller_summary.value, TimelineEventKind.contact_evidence.value):
        bad_mcap_payload = dict(payload)
        bad_mcap_payload["run_events"] = [dict(event) for event in payload["run_events"]]
        bad_mcap_payload["run_events"][0] = {
            **bad_mcap_payload["run_events"][0],
            "event_kind": event_kind,
        }
        with pytest.raises(
            HarnessIOError,
            match="MCAP evidence requires the dedicated byte rederive slice/analyzer injection",
        ):
            EpisodeTrace.from_dict(bad_mcap_payload)


def test_episode_trace_revalidates_direct_policy_event_sequences() -> None:
    events = _policy_events()
    bad_index = PolicyTraceEvent(
        run_id="run-a",
        trial_id="task_1__policy_call_0001",
        event_index=4,
        event_type=PolicyTraceEventType.task_finished,
        elapsed_sec=0.75,
        emitted_at_utc="2026-04-24T00:00:03Z",
        source="pytest",
        leakage_class=LeakageClass.legal_policy_input,
        payload={},
    )
    with pytest.raises(HarnessIOError, match="contiguous"):
        derive_episode_trace(
            run_id="run-a",
            policy_events=events + (bad_index,),
            source_artifacts=_source_artifacts(),
            score_report=_score_report(),
            generated_at_utc="2026-04-24T00:00:05Z",
        )

    nonmonotonic = PolicyTraceEvent(
        run_id="run-a",
        trial_id="task_1__policy_call_0001",
        event_index=3,
        event_type=PolicyTraceEventType.task_finished,
        elapsed_sec=0.1,
        emitted_at_utc="2026-04-24T00:00:03Z",
        source="pytest",
        leakage_class=LeakageClass.legal_policy_input,
        payload={},
    )
    with pytest.raises(HarnessIOError, match="nondecreasing"):
        derive_episode_trace(
            run_id="run-a",
            policy_events=events + (nonmonotonic,),
            source_artifacts=_source_artifacts(),
            score_report=_score_report(),
            generated_at_utc="2026-04-24T00:00:05Z",
        )


def test_episode_trace_rejects_weak_source_artifacts() -> None:
    with pytest.raises(HarnessIOError, match="source_artifacts must not be empty"):
        EpisodeTrace(
            run_id="run-a",
            generated_at_utc="2026-04-24T00:00:05Z",
            trials=(_episode_trace().trials[0],),
        )
    with pytest.raises(HarnessIOError, match="must include exactly one scoring_yaml"):
        derive_episode_trace(
            run_id="run-a",
            policy_events=_policy_events(),
            source_artifacts=(_source_artifacts()[1],),
            score_report=_score_report(),
            generated_at_utc="2026-04-24T00:00:05Z",
        )
    with pytest.raises(HarnessIOError, match="must include exactly one reward_report"):
        derive_episode_trace(
            run_id="run-a",
            policy_events=_policy_events(),
            source_artifacts=_source_artifacts(),
            score_report=_score_report(),
            reward_report=_reward_report(),
            generated_at_utc="2026-04-24T00:00:05Z",
        )
    with pytest.raises(HarnessIOError, match="must include exactly one failure_report"):
        derive_episode_trace(
            run_id="run-a",
            policy_events=_policy_events(),
            source_artifacts=_source_artifacts(),
            score_report=_score_report(),
            failure_report=_failure_report(),
            generated_at_utc="2026-04-24T00:00:05Z",
        )
    scoring_artifact, policy_artifact = _source_artifacts()
    weak_reward_artifact = ArtifactRef(
        kind="reward_report",
        uri="memory://pytest/reward_report.json",
        sha256="d" * 64,
        provenance={"producer": "pytest", "run_id": "run-a"},
    )
    with pytest.raises(HarnessIOError, match="provenance.derivation"):
        derive_episode_trace(
            run_id="run-a",
            policy_events=_policy_events(),
            source_artifacts=(scoring_artifact, policy_artifact, weak_reward_artifact),
            score_report=_score_report(),
            reward_report=_reward_report(),
            generated_at_utc="2026-04-24T00:00:05Z",
        )
    malformed_reward_artifact = ArtifactRef(
        kind="reward_report",
        uri="memory://pytest/reward_report.json",
        sha256="d" * 64,
        provenance={
            "producer": "pytest",
            "run_id": "run-a",
            "derivation": "derive_reward_failure_reports",
            "source_artifacts": [{"kind": "scoring_yaml", "sha256": "not-a-digest"}],
        },
    )
    with pytest.raises(HarnessIOError, match=r"source_artifacts\[0\].sha256"):
        derive_episode_trace(
            run_id="run-a",
            policy_events=_policy_events(),
            source_artifacts=(scoring_artifact, policy_artifact, malformed_reward_artifact),
            score_report=_score_report(),
            reward_report=_reward_report(),
            generated_at_utc="2026-04-24T00:00:05Z",
        )
    forged_reward_artifact = ArtifactRef(
        kind="reward_report",
        uri="memory://pytest/reward_report.json",
        sha256="d" * 64,
        provenance={
            "producer": "pytest",
            "run_id": "run-a",
            "derivation": "derive_reward_failure_reports",
            "source_artifacts": [{"kind": "scoring_yaml", "sha256": "f" * 64}],
        },
    )
    with pytest.raises(HarnessIOError, match="must match an episode trace source artifact"):
        derive_episode_trace(
            run_id="run-a",
            policy_events=_policy_events(),
            source_artifacts=(scoring_artifact, policy_artifact, forged_reward_artifact),
            score_report=_score_report(),
            reward_report=_reward_report(),
            generated_at_utc="2026-04-24T00:00:05Z",
        )
    promotion_reward_report = RewardReport(
        run_id="run-a",
        generated_at_utc="2026-04-24T00:00:04Z",
        terms=(
            RewardTerm(
                name="promotion.metric.improvement",
                value=1.0,
                signal_kind=RewardSignalKind.diagnostic,
                leakage_class=LeakageClass.privileged_eval_signal,
                source="promotion_decision",
            ),
        ),
    )
    promotion_derived_reward_artifact = ArtifactRef(
        kind="reward_report",
        uri="memory://pytest/reward_report.json",
        sha256="d" * 64,
        provenance={
            "producer": "pytest",
            "run_id": "run-a",
            "derivation": "derive_reward_failure_reports",
            "source_artifacts": [
                {"kind": scoring_artifact.kind, "sha256": scoring_artifact.sha256},
            ],
        },
    )
    with pytest.raises(HarnessIOError, match="promotion_decision_snapshot"):
        derive_episode_trace(
            run_id="run-a",
            policy_events=_policy_events(),
            source_artifacts=(
                scoring_artifact,
                policy_artifact,
                promotion_derived_reward_artifact,
            ),
            score_report=_score_report(),
            reward_report=promotion_reward_report,
            generated_at_utc="2026-04-24T00:00:05Z",
        )


def test_episode_trace_rejects_foreign_or_unbound_source_artifacts(tmp_path: Path) -> None:
    scoring_artifact, policy_artifact = _source_artifacts()
    foreign_policy = ArtifactRef(
        kind=policy_artifact.kind,
        uri=policy_artifact.uri,
        sha256=policy_artifact.sha256,
        provenance={"producer": "pytest", "run_id": "other-run"},
    )
    with pytest.raises(HarnessIOError, match="provenance.run_id"):
        derive_episode_trace(
            run_id="run-a",
            policy_events=_policy_events(),
            source_artifacts=(scoring_artifact, foreign_policy),
            score_report=_score_report(),
            generated_at_utc="2026-04-24T00:00:05Z",
        )

    missing_policy = ArtifactRef(
        kind="policy_trace_jsonl",
        path=str(tmp_path / "missing_policy_trace.jsonl"),
        sha256="a" * 64,
        provenance={"producer": "pytest", "run_id": "run-a"},
    )
    with pytest.raises(HarnessIOError, match="policy_trace_jsonl source artifact path must exist"):
        derive_episode_trace(
            run_id="run-a",
            policy_events=_policy_events(),
            source_artifacts=(scoring_artifact, missing_policy),
            score_report=_score_report(),
            generated_at_utc="2026-04-24T00:00:05Z",
        )

    missing_policy_file_uri = ArtifactRef(
        kind="policy_trace_jsonl",
        uri=(tmp_path / "missing_policy_trace_uri.jsonl").as_uri(),
        sha256="a" * 64,
        provenance={"producer": "pytest", "run_id": "run-a"},
    )
    with pytest.raises(HarnessIOError, match="policy_trace_jsonl source artifact path must exist"):
        derive_episode_trace(
            run_id="run-a",
            policy_events=_policy_events(),
            source_artifacts=(scoring_artifact, missing_policy_file_uri),
            generated_at_utc="2026-04-24T00:00:05Z",
        )

    wrong_scoring = ArtifactRef(
        kind="scoring_yaml",
        path="/tmp/not_the_score.yaml",
        sha256="c" * 64,
        provenance={"producer": "pytest", "run_id": "run-a"},
    )
    with pytest.raises(HarnessIOError, match="score_report.source"):
        derive_episode_trace(
            run_id="run-a",
            policy_events=_policy_events(),
            source_artifacts=(wrong_scoring, policy_artifact),
            score_report=_score_report(),
            generated_at_utc="2026-04-24T00:00:05Z",
        )

    unbound_uri_scoring = ArtifactRef(
        kind="scoring_yaml",
        uri=_SCORE_SOURCE,
        sha256="c" * 64,
        provenance={"producer": "pytest", "run_id": "run-a"},
    )
    with pytest.raises(HarnessIOError, match="local scoring.yaml snapshot"):
        derive_episode_trace(
            run_id="run-a",
            policy_events=_policy_events(),
            source_artifacts=(unbound_uri_scoring, policy_artifact),
            score_report=ScoreReport(
                source=_SCORE_SOURCE,
                parsed_at_utc="2026-04-24T00:00:03Z",
                total=999.0,
                trials={
                    "trial_1": TrialScore(total=999.0, tier_1=1.0, tier_2=2.5, tier_3=995.5)
                },
            ),
            generated_at_utc="2026-04-24T00:00:05Z",
        )

    forged_score_report = ScoreReport(
        source=_SCORE_SOURCE,
        parsed_at_utc="2026-04-24T00:00:03Z",
        total=999.0,
        trials={
            "trial_1": TrialScore(total=999.0, tier_1=1.0, tier_2=2.5, tier_3=995.5)
        },
    )
    caller_bound_uri_scoring = ArtifactRef(
        kind="scoring_yaml",
        uri=_SCORE_SOURCE,
        sha256="0" * 64,
        provenance={
            "producer": "pytest",
            "run_id": "run-a",
            "report_sha256": _report_sha256(forged_score_report.to_dict()),
        },
    )
    with pytest.raises(HarnessIOError, match="local scoring.yaml snapshot"):
        derive_episode_trace(
            run_id="run-a",
            policy_events=_policy_events(),
            source_artifacts=(caller_bound_uri_scoring, policy_artifact),
            score_report=forged_score_report,
            generated_at_utc="2026-04-24T00:00:05Z",
        )

    scoreless_trace = _episode_trace().to_dict()
    scoreless_trace["run_events"] = [
        event
        for event in scoreless_trace["run_events"]
        if event["event_kind"] != TimelineEventKind.official_score.value
    ]
    _reindex_episode_payload(scoreless_trace)
    for artifact in scoreless_trace["source_artifacts"]:
        if artifact["kind"] == "scoring_yaml":
            artifact["provenance"] = {
                **artifact["provenance"],
                "report_sha256": _report_sha256(forged_score_report.to_dict()),
                "source_report": forged_score_report.to_dict(),
            }
    with pytest.raises(HarnessIOError, match="source_report must match parsed scoring.yaml"):
        EpisodeTrace.from_dict(scoreless_trace)

    scoring_path = tmp_path / "scoring.yaml"
    scoring_path.write_text(
        """
total: 7.5
trial_1:
  tier_1:
    score: 1.0
  tier_2:
    score: 2.5
  tier_3:
    score: 4.0
""",
        encoding="utf-8",
    )
    wrong_digest_scoring = ArtifactRef(
        kind="scoring_yaml",
        path=str(scoring_path),
        sha256="0" * 64,
        provenance={"producer": "pytest", "run_id": "run-a"},
    )
    with pytest.raises(HarnessIOError, match="scoring_yaml source artifact sha256"):
        derive_episode_trace(
            run_id="run-a",
            policy_events=_policy_events(),
            source_artifacts=(wrong_digest_scoring, policy_artifact),
            score_report=ScoreReport(
                source=str(scoring_path.resolve()),
                parsed_at_utc="2026-04-24T00:00:03Z",
                total=7.5,
                trials={
                    "trial_1": TrialScore(total=7.5, tier_1=1.0, tier_2=2.5, tier_3=4.0)
                },
            ),
            generated_at_utc="2026-04-24T00:00:05Z",
        )

    bound_scoring_path = tmp_path / "bound_scoring.yaml"
    bound_scoring_path.write_text(
        """
total: 7.5
trial_1:
  tier_1:
    score: 1.0
  tier_2:
    score: 2.5
  tier_3:
    score: 4.0
""",
        encoding="utf-8",
    )
    bound_scoring = ArtifactRef(
        kind="scoring_yaml",
        path=str(bound_scoring_path),
        sha256=sha256_file(bound_scoring_path),
        provenance={"producer": "pytest", "run_id": "run-a"},
    )
    with pytest.raises(HarnessIOError, match="score_report must match scoring_yaml"):
        derive_episode_trace(
            run_id="run-a",
            policy_events=_policy_events(),
            source_artifacts=(bound_scoring, policy_artifact),
            score_report=ScoreReport(
                source=str(bound_scoring_path.resolve()),
                parsed_at_utc="2026-04-24T00:00:03Z",
                total=999.0,
                trials={
                    "trial_1": TrialScore(total=999.0, tier_1=1.0, tier_2=2.5, tier_3=995.5)
                },
            ),
            generated_at_utc="2026-04-24T00:00:05Z",
        )

    localhost_scoring = ArtifactRef(
        kind="scoring_yaml",
        path=str(scoring_path),
        sha256=sha256_file(scoring_path),
        provenance={"producer": "pytest", "run_id": "run-a"},
    )
    localhost_trace = derive_episode_trace(
        run_id="run-a",
        policy_events=_policy_events(),
        source_artifacts=(localhost_scoring, policy_artifact),
        score_report=ScoreReport(
            source=f"file://localhost{scoring_path.resolve().as_posix()}",
            parsed_at_utc="2026-04-24T00:00:03Z",
            total=7.5,
            trials={
                "trial_1": TrialScore(total=7.5, tier_1=1.0, tier_2=2.5, tier_3=4.0)
            },
        ),
        generated_at_utc="2026-04-24T00:00:05Z",
    )
    assert localhost_trace.trials[0].score is not None

    conflicting_source_report = ScoreReport(
        source=str(scoring_path.resolve()),
        parsed_at_utc="2026-04-24T00:00:03Z",
        total=999.0,
        trials={
            "trial_1": TrialScore(total=999.0, tier_1=1.0, tier_2=2.5, tier_3=995.5)
        },
    )
    conflicting_source_report_scoring = ArtifactRef(
        kind="scoring_yaml",
        path=str(scoring_path),
        sha256=sha256_file(scoring_path),
        provenance={
            "producer": "pytest",
            "run_id": "run-a",
            "source_report": conflicting_source_report.to_dict(),
        },
    )
    with pytest.raises(HarnessIOError, match="provenance.source_report"):
        derive_episode_trace(
            run_id="run-a",
            policy_events=_policy_events(),
            source_artifacts=(conflicting_source_report_scoring, policy_artifact),
            score_report=ScoreReport(
                source=str(scoring_path.resolve()),
                parsed_at_utc="2026-04-24T00:00:03Z",
                total=7.5,
                trials={
                    "trial_1": TrialScore(total=7.5, tier_1=1.0, tier_2=2.5, tier_3=4.0)
                },
            ),
            generated_at_utc="2026-04-24T00:00:05Z",
        )

    missing_scoring_without_report = ArtifactRef(
        kind="scoring_yaml",
        path=str(tmp_path / "missing_scoring.yaml"),
        sha256="c" * 64,
        provenance={"producer": "pytest", "run_id": "run-a"},
    )
    with pytest.raises(HarnessIOError, match="scoring_yaml source artifact path must exist"):
        derive_episode_trace(
            run_id="run-a",
            policy_events=_policy_events(),
            source_artifacts=(missing_scoring_without_report, policy_artifact),
            generated_at_utc="2026-04-24T00:00:05Z",
        )

    missing_scoring_file_uri_without_report = ArtifactRef(
        kind="scoring_yaml",
        uri=(tmp_path / "missing_scoring_uri.yaml").as_uri(),
        sha256="c" * 64,
        provenance={"producer": "pytest", "run_id": "run-a"},
    )
    with pytest.raises(HarnessIOError, match="scoring_yaml source artifact path must exist"):
        derive_episode_trace(
            run_id="run-a",
            policy_events=_policy_events(),
            source_artifacts=(missing_scoring_file_uri_without_report, policy_artifact),
            generated_at_utc="2026-04-24T00:00:05Z",
        )

    scoring_a_path = tmp_path / "scoring_a.yaml"
    scoring_b_path = tmp_path / "scoring_b.yaml"
    scoring_a_path.write_text("total: 7.5\n", encoding="utf-8")
    scoring_b_path.write_text("total: 7.5\n", encoding="utf-8")
    with pytest.raises(SchemaValidationError, match="path and file URI"):
        ArtifactRef(
            kind="scoring_yaml",
            path=str(scoring_a_path),
            uri=scoring_b_path.as_uri(),
            sha256=sha256_file(scoring_a_path),
            provenance={"producer": "pytest", "run_id": "run-a"},
        )

    policy_path = tmp_path / "policy_trace.jsonl"
    policy_path.write_text(
        json.dumps(
            {
                **_policy_events()[0].to_dict(),
                "payload": {"task_id": "different_task"},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    bound_policy = ArtifactRef(
        kind="policy_trace_jsonl",
        path=str(policy_path),
        sha256=sha256_file(policy_path),
        provenance={"producer": "pytest", "run_id": "run-a"},
    )
    with pytest.raises(HarnessIOError, match="must match policy_events"):
        derive_episode_trace(
            run_id="run-a",
            policy_events=_policy_events(),
            source_artifacts=(scoring_artifact, bound_policy),
            score_report=_score_report(),
            generated_at_utc="2026-04-24T00:00:05Z",
        )


def test_episode_trace_rejects_inconsistent_summaries_and_indices() -> None:
    trace = _episode_trace()
    trial = trace.trials[0]
    with pytest.raises(HarnessIOError, match="action_event_count"):
        TrialTrace(
            trial_id=trial.trial_id,
            start_elapsed_sec=trial.start_elapsed_sec,
            end_elapsed_sec=trial.end_elapsed_sec,
            event_count=trial.event_count,
            action_event_count=0,
            nonzero_action_event_count=trial.nonzero_action_event_count,
            safety_guard_event_count=trial.safety_guard_event_count,
            error_event_count=trial.error_event_count,
            events=trial.events,
        )

    bad_run_event = TimelineEvent.from_dict(
        {
            **trace.run_events[0].to_dict(),
            "event_index": 99,
        }
    )
    with pytest.raises(HarnessIOError, match="contiguous"):
        EpisodeTrace(
            run_id=trace.run_id,
            generated_at_utc=trace.generated_at_utc,
            source_artifacts=trace.source_artifacts,
            trials=trace.trials,
            run_events=(bad_run_event,),
        )


def test_training_signal_report_rejects_weak_source_trace_and_non_string_keys() -> None:
    trace = _episode_trace()
    with pytest.raises(HarnessIOError, match="source_trace.sha256"):
        derive_training_signal_report(
            episode_trace=trace,
            source_trace=ArtifactRef(
                kind="episode_trace",
                path="/tmp/episode_trace.json",
                provenance={
                    "producer": "pytest",
                    "run_id": "run-a",
                    "derivation": "derive_episode_trace",
                },
            ),
        )
    with pytest.raises(HarnessIOError, match="provenance run_id"):
        TrainingSignalReport(
            run_id="run-a",
            generated_at_utc="2026-04-24T00:00:06Z",
            source_trace=ArtifactRef(
                kind="episode_trace",
                uri="memory://pytest/episode_trace.json",
                sha256="b" * 64,
                provenance={
                    "producer": "pytest",
                    "run_id": "other-run",
                    "derivation": "derive_episode_trace",
                },
            ),
        )
    with pytest.raises(HarnessIOError, match="keys must be nonempty strings"):
        TimelineEvent(
            event_index=0,
            event_kind=TimelineEventKind.policy_event,
            elapsed_sec=0.0,
            source="pytest",
            leakage_class=LeakageClass.legal_policy_input,
            payload=cast(Any, {1: "not-json-object-contract"}),
        )
    with pytest.raises(HarnessIOError, match="keys must be nonempty strings"):
        TrainingSignal(
            signal_id="tsig_bad_evidence",
            kind=TrainingSignalKind.failure_label,
            target="failure",
            weight=1.0,
            source="pytest",
            source_event_indices=(1,),
            extraction_method="episode_trace.v1.failure_label",
            leakage_class=LeakageClass.post_hoc_label,
            evidence=cast(Any, {1: "not-json-object-contract"}),
        )


def test_training_signal_report_binds_source_trace_digest_and_content(tmp_path: Path) -> None:
    trace = _episode_trace()
    bogus_trace = derive_episode_trace(
        run_id="run-a",
        policy_events=_policy_events(),
        source_artifacts=_source_artifacts(include_reports=True),
        score_report=_score_report(),
        generated_at_utc="2026-04-24T00:00:05Z",
    )
    source_path = tmp_path / "episode_trace.json"
    write_json(source_path, bogus_trace.to_dict())

    with pytest.raises(HarnessIOError, match="does not match supplied episode trace"):
        derive_training_signal_report(
            episode_trace=trace,
            source_trace=ArtifactRef(
                kind="episode_trace",
                path=str(source_path),
                sha256=sha256_file(source_path),
                provenance={
                    "producer": "pytest",
                    "run_id": "run-a",
                    "derivation": "derive_episode_trace",
                },
            ),
        )

    with pytest.raises(HarnessIOError, match="source_trace.path must exist"):
        derive_training_signal_report(
            episode_trace=trace,
            source_trace=ArtifactRef(
                kind="episode_trace",
                path=str(tmp_path / "missing_episode_trace.json"),
                sha256=_episode_trace_sha256(trace),
                provenance={
                    "producer": "pytest",
                    "run_id": "run-a",
                    "derivation": "derive_episode_trace",
                },
            ),
        )

    with pytest.raises(HarnessIOError, match="source_trace.path must exist"):
        derive_training_signal_report(
            episode_trace=trace,
            source_trace=ArtifactRef(
                kind="episode_trace",
                uri=(tmp_path / "missing_episode_trace_uri.json").as_uri(),
                sha256=_episode_trace_sha256(trace),
                provenance={
                    "producer": "pytest",
                    "run_id": "run-a",
                    "derivation": "derive_episode_trace",
                },
            ),
        )

    copied_source_path = tmp_path / "episode_trace_copy.json"
    copied_uri_path = tmp_path / "episode_trace_uri_copy.json"
    write_json(copied_source_path, trace.to_dict())
    write_json(copied_uri_path, trace.to_dict())
    with pytest.raises(SchemaValidationError, match="path and file URI"):
        ArtifactRef(
            kind="episode_trace",
            path=str(copied_source_path),
            uri=copied_uri_path.as_uri(),
            sha256=sha256_file(copied_source_path),
            provenance={
                "producer": "pytest",
                "run_id": "run-a",
                "derivation": "derive_episode_trace",
            },
        )

    with pytest.raises(HarnessIOError, match="sha256 must match supplied"):
        derive_training_signal_report(
            episode_trace=trace,
            source_trace=ArtifactRef(
                kind="episode_trace",
                uri="memory://pytest/episode_trace.json",
                sha256="b" * 64,
                provenance={
                    "producer": "pytest",
                    "run_id": "run-a",
                    "derivation": "derive_episode_trace",
                },
            ),
        )


def test_training_signal_report_binds_reward_failure_source_reports() -> None:
    trace = _episode_trace()
    reward_report, failure_report, reward_artifact, failure_artifact = _training_signal_source_reports()

    with pytest.raises(HarnessIOError, match="source_reward_report is required"):
        derive_training_signal_report(
            episode_trace=trace,
            source_trace=_source_trace_ref(trace),
            reward_report=reward_report,
            failure_report=failure_report,
            source_failure_report=failure_artifact,
        )

    forged_reward_artifact = ArtifactRef(
        kind="reward_report",
        uri="memory://pytest/reward_report.json",
        sha256="0" * 64,
        provenance={
            "producer": "pytest",
            "run_id": "run-a",
            "derivation": "derive_reward_failure_reports",
        },
    )
    with pytest.raises(HarnessIOError, match="source_reward_report.sha256"):
        derive_training_signal_report(
            episode_trace=trace,
            source_trace=_source_trace_ref(trace),
            reward_report=reward_report,
            failure_report=failure_report,
            source_reward_report=forged_reward_artifact,
            source_failure_report=failure_artifact,
        )

    uri_only_reward_artifact = ArtifactRef(
        kind="reward_report",
        uri="memory://pytest/reward_report.json",
        sha256=_report_sha256(reward_report.to_dict()),
        provenance={
            "producer": "pytest",
            "run_id": "run-a",
            "derivation": "derive_reward_failure_reports",
        },
    )
    with pytest.raises(HarnessIOError, match="local byte-verifiable"):
        derive_training_signal_report(
            episode_trace=trace,
            source_trace=_source_trace_ref(trace),
            reward_report=reward_report,
            failure_report=failure_report,
            source_reward_report=uri_only_reward_artifact,
            source_failure_report=failure_artifact,
        )

    contradictory_reward = RewardReport(
        run_id="run-a",
        generated_at_utc="2026-04-24T00:00:04Z",
        terms=(
            reward_report.terms[0],
            RewardTerm(
                name="diagnostic.margin",
                value=999.0,
                signal_kind=RewardSignalKind.diagnostic,
                leakage_class=LeakageClass.privileged_eval_signal,
                source=_SCORE_SOURCE,
                trial_id="task_1__policy_call_0001",
            ),
        ),
    )
    with pytest.raises(HarnessIOError, match="reward_report terms must match"):
        derive_training_signal_report(
            episode_trace=trace,
            source_trace=_source_trace_ref(trace),
            reward_report=contradictory_reward,
            failure_report=failure_report,
            source_reward_report=_typed_report_artifact("reward_report", contradictory_reward),
            source_failure_report=failure_artifact,
        )

    payload = derive_training_signal_report(
        episode_trace=trace,
        source_trace=_source_trace_ref(trace),
        reward_report=reward_report,
        failure_report=failure_report,
        source_reward_report=reward_artifact,
        source_failure_report=failure_artifact,
    ).to_dict()
    payload["signals"].append(dict(payload["signals"][0]))
    with pytest.raises(HarnessIOError, match="duplicate signal_id"):
        TrainingSignalReport.from_dict(payload)

    uri_only_payload = derive_training_signal_report(
        episode_trace=trace,
        source_trace=_source_trace_ref(trace),
        reward_report=reward_report,
        failure_report=failure_report,
        source_reward_report=reward_artifact,
        source_failure_report=failure_artifact,
    ).to_dict()
    uri_only_payload["source_reward_report"] = ArtifactRef(
        kind="reward_report",
        uri="memory://pytest/reward_report.json",
        sha256=_report_sha256(reward_report.to_dict()),
        provenance={
            "producer": "pytest",
            "run_id": "run-a",
            "derivation": "derive_reward_failure_reports",
        },
    ).to_dict()
    with pytest.raises(HarnessIOError, match="local byte-verifiable"):
        TrainingSignalReport.from_dict(uri_only_payload)

    legacy_payload = derive_training_signal_report(
        episode_trace=trace,
        source_trace=_source_trace_ref(trace),
        reward_report=reward_report,
        failure_report=failure_report,
        source_reward_report=reward_artifact,
        source_failure_report=failure_artifact,
    ).to_dict()
    legacy_payload["schema_version"] = 1
    with pytest.raises(HarnessIOError, match="schema_version must be 2"):
        TrainingSignalReport.from_dict(legacy_payload)


def test_training_signals_preserve_trace_trial_ids_for_mapped_labels() -> None:
    reward_report = RewardReport(
        run_id="run-a",
        generated_at_utc="2026-04-24T00:00:04Z",
        terms=(
            RewardTerm(
                name="diagnostic.margin",
                value=123.0,
                signal_kind=RewardSignalKind.diagnostic,
                leakage_class=LeakageClass.privileged_eval_signal,
                source=_SCORE_SOURCE,
                trial_id="trial_1",
            ),
        ),
    )
    failure_report = FailureReport(
        run_id="run-a",
        generated_at_utc="2026-04-24T00:00:04Z",
        labels=(
            FailureLabel(
                kind=FailureKind.no_partial_or_full_insertion,
                severity=FailureSeverity.blocker,
                summary="No insertion credit.",
                source=_SCORE_SOURCE,
                leakage_class=LeakageClass.privileged_eval_signal,
                trial_id="trial_1",
            ),
        ),
    )
    reward_artifact = _typed_report_artifact("reward_report", reward_report)
    failure_artifact = _typed_report_artifact("failure_report", failure_report)
    trace = derive_episode_trace(
        run_id="run-a",
        policy_events=_policy_events(),
        source_artifacts=_source_artifacts(include_reports=True),
        score_report=_score_report(),
        reward_report=reward_report,
        failure_report=failure_report,
        generated_at_utc="2026-04-24T00:00:05Z",
    )

    report = derive_training_signal_report(
        episode_trace=trace,
        source_trace=_source_trace_ref(trace),
        reward_report=reward_report,
        failure_report=failure_report,
        source_reward_report=reward_artifact,
        source_failure_report=failure_artifact,
        generated_at_utc="2026-04-24T00:00:06Z",
    )

    label_signals = tuple(
        signal
        for signal in report.signals
        if signal.kind in {TrainingSignalKind.reward_term, TrainingSignalKind.failure_label}
    )
    assert {signal.trial_id for signal in label_signals} == {"task_1__policy_call_0001"}
    assert {signal.evidence["trial_id"] for signal in label_signals} == {"trial_1"}
    assert {signal.evidence["source_trial_id"] for signal in label_signals} == {"trial_1"}


def test_training_signal_from_dict_requires_explicit_runtime_boundary_flags() -> None:
    payload = TrainingSignal(
        signal_id="tsig_test_failure",
        kind=TrainingSignalKind.failure_label,
        target="failure",
        weight=1.0,
        source="pytest",
        source_event_indices=(1,),
        extraction_method="episode_trace.v1.failure_label",
        leakage_class=LeakageClass.post_hoc_label,
        evidence={"kind": "failure"},
    ).to_dict()

    for field_name in (
        "offline_only",
        "runtime_allowed",
        "consumable_by_policy_runtime",
        "signal_id",
        "source_event_indices",
        "extraction_method",
    ):
        missing = dict(payload)
        del missing[field_name]
        with pytest.raises(HarnessIOError, match=field_name):
            TrainingSignal.from_dict(missing)

    for bad_indices in ((2, 1), (1, 1)):
        malformed = {**payload, "source_event_indices": list(bad_indices)}
        with pytest.raises(HarnessIOError, match="source_event_indices"):
            TrainingSignal.from_dict(malformed)


def test_training_signal_report_rejects_privileged_action_signal_extraction() -> None:
    privileged_action = TimelineEvent(
        event_index=1,
        event_kind=TimelineEventKind.action_event,
        elapsed_sec=0.25,
        source="pytest",
        leakage_class=LeakageClass.privileged_eval_signal,
        trial_id="task_1__policy_call_0001",
        payload={
            "policy_event_type": "action_published",
            "policy_payload": {
                "linear": [0.1, 0.0, 0.0],
                "angular": [0.0, 0.0, 0.0],
            },
            "official_trial_id": "trial_1",
        },
    )
    trace = _episode_trace()
    trial = trace.trials[0]
    mutated_trial = TrialTrace(
        trial_id=trial.trial_id,
        start_elapsed_sec=trial.start_elapsed_sec,
        end_elapsed_sec=trial.end_elapsed_sec,
        event_count=trial.event_count,
        action_event_count=trial.action_event_count,
        nonzero_action_event_count=trial.nonzero_action_event_count,
        safety_guard_event_count=trial.safety_guard_event_count,
        error_event_count=trial.error_event_count,
        score=trial.score,
        events=(trial.events[0], privileged_action, trial.events[2]),
    )
    privileged_trace = EpisodeTrace(
        run_id=trace.run_id,
        generated_at_utc=trace.generated_at_utc,
        source_artifacts=trace.source_artifacts,
        trials=(mutated_trial,),
        run_events=trace.run_events,
    )
    reward_report, failure_report, reward_artifact, failure_artifact = _training_signal_source_reports()
    with pytest.raises(HarnessIOError, match="legal_policy_action_output"):
        derive_training_signal_report(
            episode_trace=privileged_trace,
            source_trace=_source_trace_ref(privileged_trace),
            reward_report=reward_report,
            failure_report=failure_report,
            source_reward_report=reward_artifact,
            source_failure_report=failure_artifact,
        )
