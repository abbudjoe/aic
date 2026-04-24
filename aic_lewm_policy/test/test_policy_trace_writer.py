from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from aic_lewm_policy.LewmMpcPolicy import LewmMpcPolicy
from aic_lewm_policy.actions import CartesianVelocityAction
from aic_lewm_policy.policy_trace_writer import (
    PolicyTraceWriter,
    action_payload,
)
from aic_lewm_policy.schemas import FinalInsertionServoConfig, PolicyRuntimeConfig
from aic_signal_harness import analyze_policy_trace_jsonl


class _Logger:
    def info(self, message):
        _ = message

    def warn(self, message):
        _ = message

    def error(self, message):
        raise AssertionError(message)


def _config() -> PolicyRuntimeConfig:
    return PolicyRuntimeConfig(
        planner_mode="lewm_mpc",
        control_hz=10.0,
        max_runtime_sec=0.55,
        frame_id="gripper/tcp",
        linear_velocity_limit=0.025,
        angular_velocity_limit=0.2,
        feedback_period_sec=1.0,
        require_checkpoint=False,
        goal_dataset_path=None,
        image_size=16,
        history_size=3,
        frameskip=2,
        planning_horizon=1,
        num_action_candidates=4,
        replay_dataset_path=None,
        replay_hz=20.0,
        replay_time_scale=1.0,
        replay_action_gain=1.0,
        replay_sc_stop_sec=None,
        final_servo=FinalInsertionServoConfig.disabled(),
    )


def _task(*, time_limit: float = 180.0) -> SimpleNamespace:
    return SimpleNamespace(
        id="task_1",
        cable_type="sfp_sc",
        cable_name="cable_0",
        plug_type="sfp",
        plug_name="sfp_tip",
        port_type="sfp",
        port_name="sfp_port_0",
        target_module_name="nic_card_mount_0",
        time_limit=time_limit,
    )


def test_policy_trace_writer_from_env_requires_run_id(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AIC_LEWM_POLICY_TRACE_PATH", str(tmp_path / "policy_trace.jsonl"))
    monkeypatch.delenv("AIC_LEWM_POLICY_TRACE_RUN_ID", raising=False)
    monkeypatch.delenv("AIC_EVAL_RUN_ID", raising=False)
    monkeypatch.delenv("AIC_EXPERIMENT_RUN_ID", raising=False)

    with pytest.raises(ValueError, match="RUN_ID"):
        PolicyTraceWriter.from_env()


def test_policy_trace_writer_emits_reducible_action_trace(tmp_path) -> None:
    trace_path = tmp_path / "policy_trace.jsonl"
    writer = PolicyTraceWriter(path=trace_path, run_id="run-a")

    writer.emit(
        trial_id="trial_1",
        official_trial_id="trial_1",
        event_type="action_published",
        elapsed_sec=0.1,
        payload=action_payload(
            CartesianVelocityAction(
                linear=(0.01, 0.0, 0.0),
                angular=(0.0, 0.0, 0.0),
                frame_id="base_link",
            )
        ),
    )

    report = analyze_policy_trace_jsonl(trace_path)
    assert report.run_id == "run-a"
    assert report.event_count == 1
    assert report.trials[0].nonzero_action_event_count == 1
    assert report.trials[0].trial_id == "trial_1"


def test_policy_trace_writer_emits_explicit_official_trial_contract(tmp_path) -> None:
    trace_path = tmp_path / "policy_trace.jsonl"
    writer = PolicyTraceWriter(path=trace_path, run_id="run-a")

    writer.emit(
        trial_id="task_1__policy_call_0001",
        official_trial_id="trial_2",
        event_type="task_started",
        elapsed_sec=0.0,
        payload={},
    )

    event = analyze_policy_trace_jsonl(trace_path)
    raw = trace_path.read_text(encoding="utf-8")
    assert '"official_trial_id":"trial_2"' in raw
    assert event.trials[0].trial_id == "task_1__policy_call_0001"


def test_policy_trace_writer_resolves_only_explicit_official_trial_mapping(tmp_path) -> None:
    writer = PolicyTraceWriter(
        path=tmp_path / "policy_trace.jsonl",
        run_id="run-a",
        official_trial_ids_by_call=("trial_1", "trial_2"),
        official_trial_ids_by_trace_trial={"task_1__policy_call_0003": "trial_3"},
    )

    assert writer.official_trial_id_for(
        trace_trial_id="task_1__policy_call_0001",
        policy_call_index=1,
    ) == "trial_1"
    assert writer.official_trial_id_for(
        trace_trial_id="task_1__policy_call_0002",
        policy_call_index=2,
    ) == "trial_2"
    assert writer.official_trial_id_for(
        trace_trial_id="task_1__policy_call_0003",
        policy_call_index=3,
    ) == "trial_3"
    assert writer.official_trial_id_for(
        trace_trial_id="task_1__policy_call_0004",
        policy_call_index=4,
    ) is None


def test_policy_trace_writer_rejects_conflicting_official_trial_mapping(tmp_path) -> None:
    writer = PolicyTraceWriter(
        path=tmp_path / "policy_trace.jsonl",
        run_id="run-a",
        official_trial_ids_by_call=("trial_1",),
        official_trial_ids_by_trace_trial={"task_1__policy_call_0001": "trial_2"},
    )

    with pytest.raises(ValueError, match="conflicting"):
        writer.official_trial_id_for(
            trace_trial_id="task_1__policy_call_0001",
            policy_call_index=1,
        )


def test_policy_trace_writer_ignores_task_official_trial_id_unless_trusted(tmp_path) -> None:
    writer = PolicyTraceWriter(
        path=tmp_path / "policy_trace.jsonl",
        run_id="run-a",
        official_trial_ids_by_call=("trial_1",),
    )

    assert writer.official_trial_id_for(
        trace_trial_id="task_1__policy_call_0001",
        policy_call_index=1,
        task_official_trial_id="trial_2",
    ) == "trial_1"
    assert writer.official_trial_id_for(
        trace_trial_id="task_1__policy_call_0002",
        policy_call_index=2,
        task_official_trial_id="not-a-trial-id",
    ) is None


def test_policy_trace_writer_trusts_task_official_trial_id_when_enabled(tmp_path) -> None:
    writer = PolicyTraceWriter(
        path=tmp_path / "policy_trace.jsonl",
        run_id="run-a",
        trust_task_official_trial_id=True,
    )

    assert writer.official_trial_id_for(
        trace_trial_id="task_1__policy_call_0001",
        policy_call_index=1,
        task_official_trial_id="trial_2",
    ) == "trial_2"

    with pytest.raises(ValueError, match="official_trial_id"):
        writer.official_trial_id_for(
            trace_trial_id="task_1__policy_call_0002",
            policy_call_index=2,
            task_official_trial_id="task_2",
        )


def test_policy_trace_writer_from_env_loads_explicit_official_trial_mapping(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("AIC_LEWM_POLICY_TRACE_PATH", str(tmp_path / "policy_trace.jsonl"))
    monkeypatch.setenv("AIC_LEWM_POLICY_TRACE_RUN_ID", "run-a")
    monkeypatch.setenv("AIC_LEWM_POLICY_TRACE_OFFICIAL_TRIAL_IDS", '["trial_1","trial_2"]')

    writer = PolicyTraceWriter.from_env()

    assert writer.official_trial_id_for(
        trace_trial_id="task_1__policy_call_0002",
        policy_call_index=2,
    ) == "trial_2"


def test_policy_trace_writer_from_env_loads_task_official_trial_trust(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("AIC_LEWM_POLICY_TRACE_PATH", str(tmp_path / "policy_trace.jsonl"))
    monkeypatch.setenv("AIC_LEWM_POLICY_TRACE_RUN_ID", "run-a")
    monkeypatch.setenv("AIC_LEWM_POLICY_TRACE_TRUST_TASK_OFFICIAL_TRIAL_ID", "true")

    writer = PolicyTraceWriter.from_env()

    assert writer.official_trial_id_for(
        trace_trial_id="task_1__policy_call_0001",
        policy_call_index=1,
        task_official_trial_id="trial_3",
    ) == "trial_3"


def test_policy_trace_writer_from_env_rejects_bad_task_official_trial_trust(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("AIC_LEWM_POLICY_TRACE_PATH", str(tmp_path / "policy_trace.jsonl"))
    monkeypatch.setenv("AIC_LEWM_POLICY_TRACE_RUN_ID", "run-a")
    monkeypatch.setenv("AIC_LEWM_POLICY_TRACE_TRUST_TASK_OFFICIAL_TRIAL_ID", "yes")

    with pytest.raises(ValueError, match="TRUST_TASK_OFFICIAL_TRIAL_ID"):
        PolicyTraceWriter.from_env()


def test_policy_trace_writer_rejects_duplicate_env_map_keys(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AIC_LEWM_POLICY_TRACE_PATH", str(tmp_path / "policy_trace.jsonl"))
    monkeypatch.setenv("AIC_LEWM_POLICY_TRACE_RUN_ID", "run-a")
    monkeypatch.setenv(
        "AIC_LEWM_POLICY_TRACE_OFFICIAL_TRIAL_ID_MAP",
        '{"task_1__policy_call_0001":"trial_1","task_1__policy_call_0001":"trial_2"}',
    )

    with pytest.raises(ValueError, match="duplicate trace trial ids"):
        PolicyTraceWriter.from_env()


def test_policy_trace_writer_rejects_duplicate_normalized_map_keys(
    tmp_path,
    monkeypatch,
) -> None:
    with pytest.raises(ValueError, match="duplicate trace trial ids"):
        PolicyTraceWriter(
            path=tmp_path / "policy_trace.jsonl",
            run_id="run-a",
            official_trial_ids_by_trace_trial={
                "task_1__policy_call_0001": "trial_1",
                " task_1__policy_call_0001 ": "trial_2",
            },
        )

    monkeypatch.setenv("AIC_LEWM_POLICY_TRACE_PATH", str(tmp_path / "env_policy_trace.jsonl"))
    monkeypatch.setenv("AIC_LEWM_POLICY_TRACE_RUN_ID", "run-a")
    monkeypatch.setenv(
        "AIC_LEWM_POLICY_TRACE_OFFICIAL_TRIAL_ID_MAP",
        '{"task_1__policy_call_0001":"trial_1"," task_1__policy_call_0001 ":"trial_2"}',
    )

    with pytest.raises(ValueError, match="duplicate trace trial ids"):
        PolicyTraceWriter.from_env()


def test_policy_trace_writer_rejects_empty_csv_trial_id_entries(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AIC_LEWM_POLICY_TRACE_PATH", str(tmp_path / "policy_trace.jsonl"))
    monkeypatch.setenv("AIC_LEWM_POLICY_TRACE_RUN_ID", "run-a")
    monkeypatch.setenv("AIC_LEWM_POLICY_TRACE_OFFICIAL_TRIAL_IDS", "trial_1,,trial_2")

    with pytest.raises(ValueError, match="empty entries"):
        PolicyTraceWriter.from_env()


def test_policy_trace_writer_rejects_bad_official_trial_id(tmp_path) -> None:
    writer = PolicyTraceWriter(path=tmp_path / "policy_trace.jsonl", run_id="run-a")

    with pytest.raises(ValueError, match="official_trial_id"):
        writer.emit(
            trial_id="trial_1",
            official_trial_id="task_1",
            event_type="task_started",
            elapsed_sec=0.0,
            payload={},
        )


def test_policy_trace_writer_rejects_boolean_elapsed(tmp_path) -> None:
    writer = PolicyTraceWriter(path=tmp_path / "policy_trace.jsonl", run_id="run-a")

    with pytest.raises(ValueError, match="elapsed_sec"):
        writer.emit(
            trial_id="trial_1",
            event_type="task_started",
            elapsed_sec=True,
            payload={},
        )


def test_policy_trace_writer_rejects_blank_trial_id(tmp_path) -> None:
    writer = PolicyTraceWriter(path=tmp_path / "policy_trace.jsonl", run_id="run-a")

    with pytest.raises(ValueError, match="trial_id"):
        writer.emit(
            trial_id=" ",
            event_type="task_started",
            elapsed_sec=0.0,
            payload={},
        )


def test_policy_trace_writer_rejects_unknown_payload_objects(tmp_path) -> None:
    writer = PolicyTraceWriter(path=tmp_path / "policy_trace.jsonl", run_id="run-a")

    with pytest.raises(ValueError, match="unsupported type"):
        writer.emit(
            trial_id="trial_1",
            event_type="task_started",
            elapsed_sec=0.0,
            payload={"raw_object": object()},
        )


def test_policy_trace_writer_rejects_non_string_payload_keys(tmp_path) -> None:
    writer = PolicyTraceWriter(path=tmp_path / "policy_trace.jsonl", run_id="run-a")

    with pytest.raises(ValueError, match="payload keys"):
        writer.emit(
            trial_id="trial_1",
            event_type="task_started",
            elapsed_sec=0.0,
            payload={1: "not-json-object-contract"},
        )


def test_policy_rejects_malformed_task_official_trial_id(tmp_path) -> None:
    trace_path = tmp_path / "policy_trace.jsonl"

    class Planner:
        def reset(self, task):
            _ = task

        def select_action(self, task, observation, elapsed_sec):  # pragma: no cover
            raise AssertionError("planner should not run before official id validation")

    policy = object.__new__(LewmMpcPolicy)
    policy._config = _config()
    policy._planner = Planner()
    policy._trace = PolicyTraceWriter(
        path=trace_path,
        run_id="run-a",
        trust_task_official_trial_id=True,
    )
    policy.get_logger = lambda: _Logger()
    task = _task(time_limit=0.0)
    task.official_trial_id = " "

    with pytest.raises(ValueError, match="task.official_trial_id"):
        policy.insert_cable(task, lambda: object(), lambda **_: None, lambda _: None)


def test_policy_ignores_malformed_task_official_trial_id_by_default(tmp_path) -> None:
    trace_path = tmp_path / "policy_trace.jsonl"
    published: list[CartesianVelocityAction] = []

    class Planner:
        def reset(self, task):
            _ = task

        def select_action(self, task, observation, elapsed_sec):  # pragma: no cover - nonpositive task exits first
            raise AssertionError("planner should not run for nonpositive task time")

    policy = object.__new__(LewmMpcPolicy)
    policy._config = _config()
    policy._planner = Planner()
    policy._trace = PolicyTraceWriter(path=trace_path, run_id="run-a")
    policy.get_logger = lambda: _Logger()
    policy._publish_action = lambda action, move_robot: published.append(action)
    policy.time_now = lambda: 0.0
    task = _task(time_limit=0.0)
    task.official_trial_id = " "

    assert policy.insert_cable(task, lambda: object(), lambda **_: None, lambda _: None)
    raw = trace_path.read_text(encoding="utf-8")
    assert "official_trial_id" not in raw
    assert published[-1].linear == (0.0, 0.0, 0.0)


def test_insert_cable_writes_policy_trace_when_configured(tmp_path) -> None:
    trace_path = tmp_path / "policy_trace.jsonl"
    published: list[CartesianVelocityAction] = []

    class Planner:
        def reset(self, task):
            _ = task

        def select_action(self, task, observation, elapsed_sec):  # pragma: no cover - nonpositive task exits first
            raise AssertionError("planner should not run for nonpositive task time")

    policy = object.__new__(LewmMpcPolicy)
    policy._config = _config()
    policy._planner = Planner()
    policy._trace = PolicyTraceWriter(
        path=trace_path,
        run_id="run-a",
        official_trial_ids_by_call=("trial_1",),
    )
    policy.get_logger = lambda: _Logger()
    policy._publish_action = lambda action, move_robot: published.append(action)
    policy.time_now = lambda: 0.0

    assert policy.insert_cable(_task(time_limit=0.0), lambda: object(), lambda **_: None, lambda _: None)

    report = analyze_policy_trace_jsonl(trace_path)
    assert report.run_id == "run-a"
    assert report.event_type_counts == {
        "task_started": 1,
        "action_published": 1,
        "task_finished": 1,
    }
    raw = trace_path.read_text(encoding="utf-8")
    assert raw.count('"official_trial_id":"trial_1"') == 3
    assert published[-1].linear == (0.0, 0.0, 0.0)


def test_repeated_task_ids_are_emitted_as_distinct_trace_trials(tmp_path) -> None:
    trace_path = tmp_path / "policy_trace.jsonl"
    published: list[CartesianVelocityAction] = []
    now_values = iter((100.0, 100.2, 200.0, 200.2))

    class Planner:
        def reset(self, task):
            _ = task

        def select_action(self, task, observation, elapsed_sec):
            _ = task, observation, elapsed_sec
            return CartesianVelocityAction.zero("gripper/tcp")

    policy = object.__new__(LewmMpcPolicy)
    policy._config = _config()
    policy._planner = Planner()
    policy._trace = PolicyTraceWriter(
        path=trace_path,
        run_id="run-a",
        official_trial_ids_by_call=("trial_1", "trial_2"),
    )
    policy.get_logger = lambda: _Logger()
    policy._publish_action = lambda action, move_robot: published.append(action)
    policy.time_now = lambda: next(now_values)

    assert policy.insert_cable(_task(time_limit=0.1), lambda: None, lambda **_: None, lambda _: None)
    assert policy.insert_cable(_task(time_limit=0.1), lambda: None, lambda **_: None, lambda _: None)

    report = analyze_policy_trace_jsonl(trace_path)
    assert [trial.trial_id for trial in report.trials] == [
        "task_1__policy_call_0001",
        "task_1__policy_call_0002",
    ]
    assert report.event_count == 12
    assert report.event_type_counts["task_finished"] == 2
    assert len(published) == 4
    raw_events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert {event["official_trial_id"] for event in raw_events[:6]} == {"trial_1"}
    assert {event["official_trial_id"] for event in raw_events[6:]} == {"trial_2"}
