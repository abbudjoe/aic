from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_aic_model_does_not_start_hidden_tf_executor_before_discovery() -> None:
    source = (REPO_ROOT / "aic_model/aic_model/aic_model.py").read_text(encoding="utf-8")
    module = ast.parse(source)

    transform_listener_calls = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "TransformListener"
    ]

    assert transform_listener_calls
    for call in transform_listener_calls:
        spin_thread = next(
            (keyword.value for keyword in call.keywords if keyword.arg == "spin_thread"),
            None,
        )
        assert isinstance(spin_thread, ast.Constant)
        assert spin_thread.value is False


def test_gcp_eval_wrapper_passes_replay_dataset_contract_to_container() -> None:
    vm_script = (
        REPO_ROOT / "aic_lewm_policy/cloud/gcp/vm_eval_learned_policy.sh"
    ).read_text(encoding="utf-8")
    launch_script = (
        REPO_ROOT / "aic_lewm_policy/cloud/gcp/run_learned_eval.sh"
    ).read_text(encoding="utf-8")

    assert "-e AIC_LEWM_GOAL_DATASET=" in vm_script
    assert "-e AIC_LEWM_REPLAY_DATASET=" in vm_script
    assert "export AIC_LEWM_GOAL_DATASET=" in launch_script
    assert "export AIC_LEWM_REPLAY_DATASET=" in launch_script
    assert 'AIC_LEWM_REPLAY_HZ:-10' in vm_script
    assert 'AIC_LEWM_REPLAY_HZ:-10' in launch_script
    assert "AIC_LEWM_SC_REPLAY_STOP_SEC=" in vm_script
    assert "AIC_LEWM_SC_REPLAY_STOP_SEC=" in launch_script
    assert "AIC_LEWM_FINAL_SERVO_ENABLED=" in vm_script
    assert "AIC_LEWM_FINAL_SERVO_ENABLED=" in launch_script
    assert "AIC_LEWM_FINAL_SERVO_FORCE_GUARD_MODE=" in vm_script
    assert "AIC_LEWM_FINAL_SERVO_FORCE_GUARD_MODE=" in launch_script
    assert "AIC_LEWM_SFP_FINAL_SERVO_LINEAR=" in vm_script
    assert "AIC_LEWM_SFP_FINAL_SERVO_LINEAR=" in launch_script


def test_gcp_eval_wrapper_uses_short_docker_dns_aliases() -> None:
    vm_script = (
        REPO_ROOT / "aic_lewm_policy/cloud/gcp/vm_eval_learned_policy.sh"
    ).read_text(encoding="utf-8")

    assert "cut -c1-36" in vm_script
    assert "sha256sum" in vm_script
    assert 'SAFE_RUN_ID="${SAFE_PREFIX}-${RUN_HASH}"' in vm_script
