import h5py
import numpy as np

from aic_lewm_policy.replay_policy_check import build_replay_report


def _write_dataset(path):
    with h5py.File(path, "w") as handle:
        handle.create_dataset("ep_len", data=np.asarray([2, 3], dtype=np.int32))
        handle.create_dataset("ep_offset", data=np.asarray([0, 2], dtype=np.int64))
        handle.create_dataset(
            "action",
            data=np.asarray(
                [
                    [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.3, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.4, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            ),
        )
        dtype = h5py.string_dtype(encoding="utf-8")
        handle.create_dataset("task_id", data=np.asarray(["task_1"] * 5, dtype=dtype))
        handle.create_dataset("plug_type", data=np.asarray(["sfp"] * 5, dtype=dtype))
        handle.create_dataset("port_type", data=np.asarray(["sfp"] * 5, dtype=dtype))
        handle.create_dataset(
            "target_module_name",
            data=np.asarray(
                [
                    "nic_card_mount_0",
                    "nic_card_mount_0",
                    "nic_card_mount_1",
                    "nic_card_mount_1",
                    "nic_card_mount_1",
                ],
                dtype=dtype,
            ),
        )


def test_replay_policy_check_reports_exact_selection(tmp_path):
    dataset = tmp_path / "demo.h5"
    _write_dataset(dataset)

    report = build_replay_report(dataset)

    assert report["episode_count"] == 2
    assert report["replay_hz"] == 10.0
    assert report["score"]["exact_selection_fraction"] == 1.0
    assert {check["selected_episode"] for check in report["checks"]} == {0, 1}
    assert report["checks"][0]["duration_sec_at_replay_hz"] == 0.2
    assert report["checks"][0]["replay_stop_sec"] is None
    assert report["checks"][0]["effective_duration_sec"] == 0.2
    assert report["checks"][0]["final_servo"]["enabled"] is False
    assert report["checks"][0]["final_servo"]["window_sec"] is None
    assert report["checks"][0]["final_servo"]["force_guard_mode"] is None
