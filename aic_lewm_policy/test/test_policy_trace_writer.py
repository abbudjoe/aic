from __future__ import annotations

from types import SimpleNamespace

import pytest

from aic_lewm_policy.LewmMpcPolicy import LewmMpcPolicy
from aic_lewm_policy.actions import CartesianVelocityAction
from aic_lewm_policy.policy_trace_writer import PolicyTraceWriter, action_payload
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
    policy._trace = PolicyTraceWriter(path=trace_path, run_id="run-a")
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
    policy._trace = PolicyTraceWriter(path=trace_path, run_id="run-a")
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
