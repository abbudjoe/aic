import pytest

from aic_signal_harness import (
    HarnessIOError,
    McapEvalBundleReport,
    McapEvalFirstContact,
    McapEvalTaskHints,
    McapEvalTrialReport,
)


def _trial_source(index: int) -> str:
    return f"/tmp/eval/bag_trial_{index}_20260423_120000_000/bag_trial_{index}_0.mcap"


def _uri_trial_source(index: int) -> str:
    return f"gs://bucket/eval/bag_trial_{index}_20260423_120000_000/bag_trial_{index}_0.mcap"


def _empty_trial(index: int) -> dict[str, object]:
    return {
        "trial_id": f"trial_{index}",
        "source": _trial_source(index),
        "size_bytes": 100 + index,
        "sha256": (hex(index)[2:] * 64)[:64],
        "controller_state_count": 0,
        "pose_command_count": 0,
        "off_limit_contact_count": 0,
    }


def _empty_uri_trial(index: int) -> dict[str, object]:
    return {
        **_empty_trial(index),
        "source": _uri_trial_source(index),
    }


def _sc_trial() -> McapEvalTrialReport:
    return McapEvalTrialReport(
        trial_id="trial_1",
        source=_trial_source(1),
        size_bytes=123,
        sha256="a" * 64,
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
    )


def _uri_bundle_report() -> McapEvalBundleReport:
    return McapEvalBundleReport(
        source="gs://bucket/eval",
        analyzed_at_utc="2026-04-23T12:00:00Z",
        contact_margin_sec=0.25,
        stop_step_sec=0.05,
        recommended_env={},
        trials=tuple(
            McapEvalTrialReport.from_dict(_empty_uri_trial(index))
            for index in (1, 2, 3)
        ),
    )


def test_mcap_eval_bundle_report_round_trips_strictly() -> None:
    report = McapEvalBundleReport(
        source="/tmp/eval",
        analyzed_at_utc="2026-04-23T12:00:00Z",
        contact_margin_sec=0.25,
        stop_step_sec=0.05,
        recommended_env={"AIC_LEWM_SC_REPLAY_STOP_SEC": "0.15"},
        trials=(
            _sc_trial(),
            McapEvalTrialReport.from_dict(_empty_trial(2)),
            McapEvalTrialReport.from_dict(_empty_trial(3)),
        ),
    )

    assert McapEvalBundleReport.from_dict(report.to_dict()) == report


def test_mcap_eval_bundle_report_round_trips_strictly_with_uri_sources() -> None:
    report = _uri_bundle_report()

    assert McapEvalBundleReport.from_dict(report.to_dict()) == report


def test_mcap_eval_bundle_report_rejects_unknown_nested_fields() -> None:
    with pytest.raises(HarnessIOError, match="unknown fields"):
        McapEvalBundleReport.from_dict(
            {
                "schema_version": 1,
                "source": "/tmp/eval",
                "analyzed_at_utc": "2026-04-23T12:00:00Z",
                "contact_margin_sec": 0.25,
                "stop_step_sec": 0.05,
                "recommended_env": {},
                "trials": [
                    {
                        **_empty_trial(1),
                        "task_hints": {"port_type": "sfp", "unexpected": "boom"},
                    },
                    _empty_trial(2),
                    _empty_trial(3),
                ],
            }
        )


def test_mcap_eval_bundle_report_from_dict_rejects_relative_trial_source() -> None:
    with pytest.raises(HarnessIOError, match="absolute local path or uri"):
        McapEvalBundleReport.from_dict(
            {
                "schema_version": 1,
                "source": "/tmp/eval",
                "analyzed_at_utc": "2026-04-23T12:00:00Z",
                "contact_margin_sec": 0.25,
                "stop_step_sec": 0.05,
                "recommended_env": {},
                "trials": [
                    {
                        **_empty_trial(1),
                        "source": "bag_trial_1_20260423_120000_000/bag_trial_1_0.mcap",
                    },
                    _empty_trial(2),
                    _empty_trial(3),
                ],
            }
        )


def test_mcap_eval_bundle_report_rejects_dot_segment_local_trial_source() -> None:
    with pytest.raises(HarnessIOError, match="must not contain '.' or '..' path segments"):
        McapEvalBundleReport.from_dict(
            {
                "schema_version": 1,
                "source": "/tmp/eval",
                "analyzed_at_utc": "2026-04-23T12:00:00Z",
                "contact_margin_sec": 0.25,
                "stop_step_sec": 0.05,
                "recommended_env": {},
                "trials": [
                    {
                        **_empty_trial(1),
                        "source": "/tmp/eval/../other/bag_trial_1_0.mcap",
                    },
                    _empty_trial(2),
                    _empty_trial(3),
                ],
            }
        )


def test_mcap_eval_bundle_report_requires_the_official_three_trial_sequence() -> None:
    with pytest.raises(HarnessIOError, match="official three-trial sequence"):
        McapEvalBundleReport.from_dict(
            {
                "schema_version": 1,
                "source": "/tmp/eval",
                "analyzed_at_utc": "2026-04-23T12:00:00Z",
                "contact_margin_sec": 0.25,
                "stop_step_sec": 0.05,
                "recommended_env": {},
                "trials": [_empty_trial(1), _empty_trial(2)],
            }
        )


def test_mcap_eval_bundle_report_rejects_local_trial_under_uri_bundle() -> None:
    with pytest.raises(HarnessIOError, match="uri source must not contain local trial sources"):
        McapEvalBundleReport.from_dict(
            {
                "schema_version": 1,
                "source": "gs://bucket/eval",
                "analyzed_at_utc": "2026-04-23T12:00:00Z",
                "contact_margin_sec": 0.25,
                "stop_step_sec": 0.05,
                "recommended_env": {},
                "trials": [
                    {
                        **_empty_trial(1),
                        "source": "gs://bucket/eval/bag_trial_1/bag_trial_1_0.mcap",
                    },
                    _empty_trial(2),
                    {
                        **_empty_trial(3),
                        "source": "gs://bucket/eval/bag_trial_3/bag_trial_3_0.mcap",
                    },
                ],
            }
        )


def test_mcap_eval_bundle_report_rejects_non_child_uri_trial_source() -> None:
    with pytest.raises(HarnessIOError, match="rooted under mcap bundle.source when bundle.source is a uri"):
        McapEvalBundleReport.from_dict(
            {
                "schema_version": 1,
                "source": "gs://bucket/eval",
                "analyzed_at_utc": "2026-04-23T12:00:00Z",
                "contact_margin_sec": 0.25,
                "stop_step_sec": 0.05,
                "recommended_env": {},
                "trials": [
                    {
                        **_empty_trial(1),
                        "source": "gs://bucket/eval/bag_trial_1/bag_trial_1_0.mcap",
                    },
                    {
                        **_empty_trial(2),
                        "source": "gs://bucket/other/bag_trial_2/bag_trial_2_0.mcap",
                    },
                    {
                        **_empty_trial(3),
                        "source": "gs://bucket/eval/bag_trial_3/bag_trial_3_0.mcap",
                    },
                ],
            }
        )


def test_mcap_eval_bundle_report_rejects_dot_segment_uri_trial_source() -> None:
    with pytest.raises(HarnessIOError, match="must not contain '.' or '..' path segments"):
        McapEvalBundleReport.from_dict(
            {
                "schema_version": 1,
                "source": "gs://bucket/eval",
                "analyzed_at_utc": "2026-04-23T12:00:00Z",
                "contact_margin_sec": 0.25,
                "stop_step_sec": 0.05,
                "recommended_env": {},
                "trials": [
                    {
                        **_empty_uri_trial(1),
                        "source": "gs://bucket/eval/../other/bag_trial_1_0.mcap",
                    },
                    _empty_uri_trial(2),
                    _empty_uri_trial(3),
                ],
            }
        )


def test_mcap_eval_trial_report_rejects_impossible_controller_context() -> None:
    with pytest.raises(
        HarnessIOError,
        match="controller context requires controller_state_count > 0",
    ):
        McapEvalTrialReport.from_dict(
            {
                **_empty_trial(1),
                "off_limit_contact_count": 1,
                "first_off_limit_contact": {
                    "log_time_ns": 1_300_000_000,
                    "collision1": "plug",
                    "collision2": "enclosure",
                    "log_elapsed_sec": 0.3,
                    "nearest_controller_elapsed_sec": 0.4,
                    "nearest_tcp_position": [1.0, 2.0, 3.0],
                    "nearest_tcp_error": [0.1, 0.2, 0.3],
                },
            }
        )


def test_mcap_eval_trial_report_rejects_first_contact_when_contact_count_is_zero() -> None:
    with pytest.raises(
        HarnessIOError,
        match="must be null when off_limit_contact_count is 0",
    ):
        McapEvalTrialReport.from_dict(
            {
                **_empty_trial(1),
                "first_off_limit_contact": {
                    "log_time_ns": 1_300_000_000,
                    "collision1": "plug",
                    "collision2": "enclosure",
                },
            }
        )


def test_mcap_eval_trial_report_rejects_impossible_command_context() -> None:
    with pytest.raises(
        HarnessIOError,
        match="command context requires pose_command_count > 0",
    ):
        McapEvalTrialReport.from_dict(
            {
                **_empty_trial(1),
                "controller_state_count": 1,
                "controller_stamp_start_sec": 10.0,
                "controller_stamp_end_sec": 10.4,
                "controller_duration_sec": 0.4,
                "final_tcp_position": [1.0, 2.0, 3.0],
                "final_tcp_error": [0.1, 0.2, 0.3],
                "off_limit_contact_count": 1,
                "first_off_limit_contact": {
                    "log_time_ns": 1_300_000_000,
                    "collision1": "plug",
                    "collision2": "enclosure",
                    "log_elapsed_sec": 0.3,
                    "nearest_controller_elapsed_sec": 0.4,
                    "nearest_tcp_position": [1.0, 2.0, 3.0],
                    "nearest_tcp_error": [0.1, 0.2, 0.3],
                    "nearest_command_elapsed_sec": 0.15,
                    "nearest_command_linear": [0.01, 0.02, 0.03],
                    "nearest_command_angular": [0.04, 0.05, 0.06],
                },
            }
        )


def test_mcap_eval_bundle_report_rejects_recommended_stop_formula_mismatch() -> None:
    with pytest.raises(
        HarnessIOError,
        match="must match the bundle-derived sc stop recommendation",
    ):
        McapEvalBundleReport.from_dict(
            {
                "schema_version": 1,
                "source": "/tmp/eval",
                "analyzed_at_utc": "2026-04-23T12:00:00Z",
                "contact_margin_sec": 0.25,
                "stop_step_sec": 0.05,
                "recommended_env": {"AIC_LEWM_SC_REPLAY_STOP_SEC": "0.10"},
                "trials": [
                    {
                        **_sc_trial().to_dict(),
                        "first_off_limit_contact": {
                            **_sc_trial().first_off_limit_contact.to_dict(),
                            "recommended_stop_sec": 0.10,
                        },
                    },
                    _empty_trial(2),
                    _empty_trial(3),
                ],
            }
        )
