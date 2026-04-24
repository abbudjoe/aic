import json
from pathlib import Path

import pytest

import aic_signal_harness.reducers.policy_trace as policy_trace_reducer
from aic_signal_harness import (
    ArtifactRef,
    HarnessIOError,
    LeakageClass,
    PolicyTraceEvent,
    PolicyTraceEventType,
    PolicyTraceReduction,
    PolicyTraceReport,
    PolicyTraceTrialReport,
    analyze_policy_trace_jsonl,
    reduce_policy_trace_jsonl,
    sha256_file,
)


def _event(
    *,
    index: int,
    trial_id: str = "trial_1",
    event_type: str = "planner_event",
    elapsed_sec: float = 0.0,
    run_id: str = "gate2-candidate",
    payload: dict | None = None,
    leakage_class: str | None = None,
    official_trial_id: str | None = None,
) -> dict:
    event_leakage_class = (
        leakage_class
        if leakage_class is not None
        else "legal_policy_action_output"
        if event_type in {"action_selected", "action_published"}
        else "legal_policy_input"
    )
    event = {
        "schema_version": 1,
        "run_id": run_id,
        "trial_id": trial_id,
        "event_index": index,
        "event_type": event_type,
        "elapsed_sec": elapsed_sec,
        "emitted_at_utc": f"2026-04-23T00:00:{index:02d}Z",
        "source": "aic_lewm_policy",
        "leakage_class": event_leakage_class,
        "payload": {} if payload is None else payload,
    }
    if official_trial_id is not None:
        event["official_trial_id"] = official_trial_id
    return event


def _write_jsonl(path: Path, rows: tuple[dict, ...]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_policy_trace_event_round_trips_strictly() -> None:
    event = PolicyTraceEvent(
        run_id="gate2-candidate",
        trial_id="trial_1",
        event_index=2,
        event_type=PolicyTraceEventType.action_selected,
        elapsed_sec=0.2,
        emitted_at_utc="2026-04-23T00:00:02Z",
        source="aic_lewm_policy",
        leakage_class=LeakageClass.legal_policy_action_output,
        official_trial_id="trial_1",
        payload={
            "linear": [0.01, 0.0, 0.0],
            "angular": [0.0, 0.0, 0.0],
            "frame_id": "base_link",
        },
    )

    assert PolicyTraceEvent.from_dict(event.to_dict()) == event
    assert event.to_dict()["official_trial_id"] == "trial_1"


def test_policy_trace_event_rejects_unknown_fields() -> None:
    payload = _event(index=0)
    payload["unexpected"] = True

    with pytest.raises(HarnessIOError, match="unknown fields"):
        PolicyTraceEvent.from_dict(payload)


def test_policy_trace_event_requires_action_vectors() -> None:
    with pytest.raises(HarnessIOError, match="linear is required"):
        PolicyTraceEvent.from_dict(
            _event(
                index=0,
                event_type="action_selected",
                payload={"angular": [0.0, 0.0, 0.0]},
            )
        )


def test_policy_trace_event_rejects_bad_official_trial_id() -> None:
    with pytest.raises(HarnessIOError, match="official_trial_id"):
        PolicyTraceEvent.from_dict(
            _event(index=0, event_type="task_started", official_trial_id="task_1")
        )


def test_policy_trace_event_rejects_privileged_action_outputs() -> None:
    with pytest.raises(HarnessIOError, match="legal_policy_action_output"):
        PolicyTraceEvent.from_dict(
            _event(
                index=0,
                event_type="action_published",
                leakage_class="privileged_eval_signal",
                payload={
                    "linear": [0.01, 0.0, 0.0],
                    "angular": [0.0, 0.0, 0.0],
                },
            )
        )


def test_policy_trace_report_round_trips_strictly() -> None:
    report = PolicyTraceReport(
        source="/tmp/policy_trace.jsonl",
        run_id="gate2-candidate",
        reduced_at_utc="2026-04-23T00:00:03Z",
        event_count=2,
        start_elapsed_sec=0.0,
        end_elapsed_sec=0.1,
        event_type_counts={"task_started": 1, "task_finished": 1},
        leakage_classes=(LeakageClass.legal_policy_input,),
        trials=(
            PolicyTraceTrialReport(
                trial_id="trial_1",
                event_count=2,
                first_event_index=0,
                last_event_index=1,
                start_elapsed_sec=0.0,
                end_elapsed_sec=0.1,
                event_type_counts={"task_started": 1, "task_finished": 1},
            ),
        ),
    )

    assert PolicyTraceReport.from_dict(report.to_dict()) == report


def test_analyze_policy_trace_jsonl_summarizes_trials_and_actions(tmp_path: Path) -> None:
    trace_path = tmp_path / "policy_trace.jsonl"
    _write_jsonl(
        trace_path,
        (
            _event(index=0, event_type="task_started", elapsed_sec=0.0),
            _event(
                index=1,
                event_type="action_selected",
                elapsed_sec=0.1,
                payload={
                    "linear": [0.01, 0.0, 0.0],
                    "angular": [0.0, 0.0, 0.0],
                    "frame_id": "base_link",
                },
            ),
            _event(
                index=2,
                event_type="action_published",
                elapsed_sec=0.2,
                payload={
                    "linear": [0.0, 0.0, 0.0],
                    "angular": [0.0, 0.0, 0.0],
                    "frame_id": "base_link",
                },
            ),
            _event(index=3, event_type="safety_guard", elapsed_sec=0.3),
            _event(index=4, trial_id="trial_2", event_type="task_started", elapsed_sec=0.0),
            _event(index=5, trial_id="trial_2", event_type="error", elapsed_sec=0.1),
        ),
    )

    report = analyze_policy_trace_jsonl(
        trace_path,
        reduced_at_utc="2026-04-23T00:00:06Z",
    )

    assert report.source == str(trace_path.resolve())
    assert report.run_id == "gate2-candidate"
    assert report.reduced_at_utc == "2026-04-23T00:00:06Z"
    assert report.event_count == 6
    assert report.event_type_counts == {
        "task_started": 2,
        "action_selected": 1,
        "action_published": 1,
        "safety_guard": 1,
        "error": 1,
    }
    assert tuple(trial.trial_id for trial in report.trials) == ("trial_1", "trial_2")
    assert report.trials[0].action_event_count == 2
    assert report.trials[0].nonzero_action_event_count == 1
    assert report.trials[0].safety_guard_event_count == 1
    assert report.trials[1].error_event_count == 1


def test_policy_trace_reduction_defaults_to_source_timestamp(tmp_path: Path) -> None:
    trace_path = tmp_path / "policy_trace.jsonl"
    _write_jsonl(
        trace_path,
        (
            _event(index=0, event_type="task_started", elapsed_sec=0.0),
            _event(index=1, event_type="task_finished", elapsed_sec=0.1),
        ),
    )

    first_report = analyze_policy_trace_jsonl(trace_path)
    second_report = analyze_policy_trace_jsonl(trace_path)
    first_reduction = reduce_policy_trace_jsonl(trace_path)
    second_reduction = reduce_policy_trace_jsonl(trace_path)

    assert first_report == second_report
    assert first_reduction.report == second_reduction.report
    assert first_report.reduced_at_utc == "2026-04-23T00:00:01Z"
    assert first_reduction.report.reduced_at_utc == "2026-04-23T00:00:01Z"


def test_reduce_policy_trace_jsonl_binds_artifact_identity(tmp_path: Path) -> None:
    trace_path = tmp_path / "policy_trace.jsonl"
    _write_jsonl(trace_path, (_event(index=0, event_type="task_started"),))

    reduction = reduce_policy_trace_jsonl(
        trace_path,
        uri="gs://bucket/run/policy_trace.jsonl",
        provenance={"producer": "pytest"},
        reduced_at_utc="2026-04-23T00:00:00Z",
    )

    assert isinstance(reduction, PolicyTraceReduction)
    assert reduction.artifact == ArtifactRef(
        kind="policy_trace_jsonl",
        path=str(trace_path.resolve()),
        uri="gs://bucket/run/policy_trace.jsonl",
        sha256=sha256_file(trace_path),
        provenance={"producer": "pytest"},
    )
    assert reduction.report.reduced_at_utc == "2026-04-23T00:00:00Z"
    assert len(reduction.events) == 1
    assert reduction.events[0].run_id == reduction.report.run_id


def test_reduce_policy_trace_jsonl_rejects_split_path_file_uri_identity(tmp_path: Path) -> None:
    trace_path = tmp_path / "policy_trace.jsonl"
    other_trace_path = tmp_path / "other_policy_trace.jsonl"
    _write_jsonl(trace_path, (_event(index=0, event_type="task_started"),))
    _write_jsonl(other_trace_path, (_event(index=0, event_type="task_started"),))

    with pytest.raises(HarnessIOError, match="path and file URI"):
        reduce_policy_trace_jsonl(trace_path, uri=other_trace_path.as_uri())


def test_reduce_policy_trace_jsonl_rejects_malformed_uri_as_harness_error(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "policy_trace.jsonl"
    _write_jsonl(trace_path, (_event(index=0, event_type="task_started"),))

    for uri, message in (
        ("not-a-uri", "URI must include a scheme"),
        ("s3:path", "URI must include a network location"),
        ("x:", "URI must include a network location"),
    ):
        with pytest.raises(HarnessIOError, match=message):
            reduce_policy_trace_jsonl(trace_path, uri=uri)


def test_reduce_policy_trace_jsonl_rejects_source_mutation_during_reduction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace_path = tmp_path / "policy_trace.jsonl"
    _write_jsonl(trace_path, (_event(index=0, event_type="task_started"),))
    original_sha256_file = policy_trace_reducer.sha256_file

    def mutating_sha256_file(path: str | Path) -> str:
        digest = original_sha256_file(path)
        with Path(path).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_event(index=1, event_type="task_finished", elapsed_sec=0.1)) + "\n")
        return digest

    monkeypatch.setattr(policy_trace_reducer, "sha256_file", mutating_sha256_file)

    with pytest.raises(HarnessIOError, match="changed during reduction"):
        reduce_policy_trace_jsonl(trace_path)


def test_policy_trace_reduction_rejects_digest_mismatch(tmp_path: Path) -> None:
    trace_path = tmp_path / "policy_trace.jsonl"
    _write_jsonl(trace_path, (_event(index=0, event_type="task_started"),))
    report = analyze_policy_trace_jsonl(trace_path, reduced_at_utc="2026-04-23T00:00:00Z")

    with pytest.raises(HarnessIOError, match="sha256 must match"):
        PolicyTraceReduction(
            artifact=ArtifactRef(
                kind="policy_trace_jsonl",
                path=str(trace_path.resolve()),
                sha256="0" * 64,
            ),
            report=report,
        )


def test_policy_trace_reduction_rejects_empty_or_mismatched_events(tmp_path: Path) -> None:
    trace_path = tmp_path / "policy_trace.jsonl"
    _write_jsonl(trace_path, (_event(index=0, event_type="task_started"),))
    report = analyze_policy_trace_jsonl(trace_path, reduced_at_utc="2026-04-23T00:00:00Z")
    artifact = ArtifactRef(
        kind="policy_trace_jsonl",
        path=str(trace_path.resolve()),
        sha256=sha256_file(trace_path),
    )

    with pytest.raises(HarnessIOError, match="events must not be empty"):
        PolicyTraceReduction(artifact=artifact, report=report)

    with pytest.raises(HarnessIOError, match="all match report.run_id"):
        PolicyTraceReduction(
            artifact=artifact,
            report=report,
            events=(
                PolicyTraceEvent.from_dict(
                    _event(index=0, event_type="task_started", run_id="other-run")
                ),
            ),
        )


def test_policy_trace_reduction_revalidates_events_against_report(tmp_path: Path) -> None:
    trace_path = tmp_path / "policy_trace.jsonl"
    _write_jsonl(trace_path, (_event(index=0, event_type="task_started"),))
    report = analyze_policy_trace_jsonl(trace_path, reduced_at_utc="2026-04-23T00:00:00Z")
    mismatched_report = PolicyTraceReport(
        source=report.source,
        run_id=report.run_id,
        reduced_at_utc=report.reduced_at_utc,
        event_count=report.event_count,
        start_elapsed_sec=report.start_elapsed_sec,
        end_elapsed_sec=0.5,
        event_type_counts=report.event_type_counts,
        leakage_classes=report.leakage_classes,
        trials=(
            PolicyTraceTrialReport(
                trial_id=report.trials[0].trial_id,
                event_count=report.trials[0].event_count,
                first_event_index=report.trials[0].first_event_index,
                last_event_index=report.trials[0].last_event_index,
                start_elapsed_sec=report.trials[0].start_elapsed_sec,
                end_elapsed_sec=0.5,
                event_type_counts=report.trials[0].event_type_counts,
            ),
        ),
    )

    with pytest.raises(HarnessIOError, match="reconstruct report exactly"):
        PolicyTraceReduction(
            artifact=ArtifactRef(
                kind="policy_trace_jsonl",
                path=str(trace_path.resolve()),
                sha256=sha256_file(trace_path),
            ),
            report=mismatched_report,
            events=(PolicyTraceEvent.from_dict(_event(index=0, event_type="task_started")),),
        )


def test_analyze_policy_trace_jsonl_rejects_blank_lines(tmp_path: Path) -> None:
    trace_path = tmp_path / "policy_trace.jsonl"
    trace_path.write_text(json.dumps(_event(index=0)) + "\n\n", encoding="utf-8")

    with pytest.raises(HarnessIOError, match="must not be blank"):
        analyze_policy_trace_jsonl(trace_path)


def test_analyze_policy_trace_jsonl_rejects_noncontiguous_indices(tmp_path: Path) -> None:
    trace_path = tmp_path / "policy_trace.jsonl"
    _write_jsonl(trace_path, (_event(index=0), _event(index=2, elapsed_sec=0.1)))

    with pytest.raises(HarnessIOError, match="contiguous"):
        analyze_policy_trace_jsonl(trace_path)


def test_analyze_policy_trace_jsonl_rejects_mixed_run_ids(tmp_path: Path) -> None:
    trace_path = tmp_path / "policy_trace.jsonl"
    _write_jsonl(
        trace_path,
        (
            _event(index=0),
            _event(index=1, elapsed_sec=0.1, run_id="other-run"),
        ),
    )

    with pytest.raises(HarnessIOError, match="exactly one run_id"):
        analyze_policy_trace_jsonl(trace_path)


def test_analyze_policy_trace_jsonl_rejects_nonmonotonic_trial_time(tmp_path: Path) -> None:
    trace_path = tmp_path / "policy_trace.jsonl"
    _write_jsonl(trace_path, (_event(index=0, elapsed_sec=0.2), _event(index=1, elapsed_sec=0.1)))

    with pytest.raises(HarnessIOError, match="nondecreasing"):
        analyze_policy_trace_jsonl(trace_path)
