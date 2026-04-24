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
    assert "AIC_LEWM_POLICY_TRACE_PATH=" in vm_script
    assert "AIC_LEWM_POLICY_TRACE_RUN_ID=" in vm_script
    assert "AIC_LEWM_POLICY_TRACE_REQUIRED=" in launch_script
    assert "AIC_MODEL_IMAGE_ID=" in launch_script
    assert "AIC_HARNESS_GATE_ID=" in launch_script
    assert "AIC_HARNESS_MIN_IMPROVEMENT=" in launch_script
    assert "AIC_HARNESS_BASELINE_PATH=" in launch_script


def test_eval_wrapper_finalizes_live_eval_with_neutral_harness() -> None:
    vm_script = (
        REPO_ROOT / "aic_lewm_policy/cloud/gcp/vm_eval_learned_policy.sh"
    ).read_text(encoding="utf-8")

    assert 'HARNESS_ROOT="$RESULT_ROOT/harness"' in vm_script
    assert 'POLICY_TRACE_CONTAINER_PREFIX="/aic_results/harness/"' in vm_script
    assert 'POLICY_TRACE_HOST_PATH="$HARNESS_ROOT/$POLICY_TRACE_RELATIVE_PATH"' in vm_script
    assert "AIC_LEWM_POLICY_TRACE_CONTAINER_PATH must live under" in vm_script
    assert '-v "$HARNESS_ROOT:/aic_results/harness"' in vm_script
    assert "-m aic_signal_harness.live_eval" in vm_script
    assert "--scoring-yaml" in vm_script
    assert "--model-image-id" in vm_script
    assert "--policy-trace" in vm_script
    assert "--ledger" in vm_script
    assert "policy_trace_report.json" in vm_script
    assert "policy_trace_artifact.json" in vm_script
    assert "AIC_POLICY_TRACE_ARTIFACT_PATH=" in vm_script
    assert "AIC_HARNESS_MANIFEST_PATH=" in vm_script
    assert "AIC_HARNESS_LEDGER_ENTRY_PATH=" in vm_script
    assert "AIC_HARNESS_NEXT_EXPERIMENT_PATH=" in vm_script
    assert "Missing required policy trace JSONL" in vm_script


def test_gcp_eval_wrapper_uses_short_docker_dns_aliases() -> None:
    vm_script = (
        REPO_ROOT / "aic_lewm_policy/cloud/gcp/vm_eval_learned_policy.sh"
    ).read_text(encoding="utf-8")

    assert "cut -c1-36" in vm_script
    assert "sha256sum" in vm_script
    assert 'SAFE_RUN_ID="${SAFE_PREFIX}-${RUN_HASH}"' in vm_script
