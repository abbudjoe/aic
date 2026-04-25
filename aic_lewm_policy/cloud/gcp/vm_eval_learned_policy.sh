#!/usr/bin/env bash
set -euo pipefail

AIC_EVAL_RUN_ID="${AIC_EVAL_RUN_ID:?Set AIC_EVAL_RUN_ID}"
AIC_MODEL_IMAGE="${AIC_MODEL_IMAGE:-aic-lewm-learned:latest}"
AIC_MODEL_IMAGE_ID="${AIC_MODEL_IMAGE_ID:-$(sudo docker image inspect --format '{{.Id}}' "$AIC_MODEL_IMAGE")}"
AIC_EVAL_IMAGE="${AIC_EVAL_IMAGE:-ghcr.io/intrinsic-dev/aic/aic_eval:latest}"
AIC_EVAL_TIMEOUT_SEC="${AIC_EVAL_TIMEOUT_SEC:-1800}"

trim_env_value() {
  local value="${1:-}"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

RESULT_ROOT="${AIC_EVAL_RESULT_ROOT:-$HOME/aic_results/$AIC_EVAL_RUN_ID}"
HARNESS_ROOT="$RESULT_ROOT/harness"
POLICY_TRACE_CONTAINER_PATH="${AIC_LEWM_POLICY_TRACE_CONTAINER_PATH:-/aic_results/harness/policy_trace.jsonl}"
POLICY_TRACE_CONTAINER_PREFIX="/aic_results/harness/"
if [[ "$POLICY_TRACE_CONTAINER_PATH" != "$POLICY_TRACE_CONTAINER_PREFIX"* ]]; then
  echo "AIC_LEWM_POLICY_TRACE_CONTAINER_PATH must live under $POLICY_TRACE_CONTAINER_PREFIX" >&2
  exit 2
fi
POLICY_TRACE_RELATIVE_PATH="${POLICY_TRACE_CONTAINER_PATH#"$POLICY_TRACE_CONTAINER_PREFIX"}"
case "/$POLICY_TRACE_RELATIVE_PATH/" in
  */../* | */./* | *//*)
    echo "AIC_LEWM_POLICY_TRACE_CONTAINER_PATH must not contain relative path segments" >&2
    exit 2
    ;;
esac
if [[ -z "$POLICY_TRACE_RELATIVE_PATH" ]]; then
  echo "AIC_LEWM_POLICY_TRACE_CONTAINER_PATH must name a JSONL file" >&2
  exit 2
fi
if [[ "${AIC_LEWM_POLICY_TRACE_REQUIRED:-1}" != "0" && "${AIC_LEWM_POLICY_TRACE_REQUIRED:-1}" != "false" ]]; then
  POLICY_TRACE_OFFICIAL_TRIAL_IDS_TRIMMED="$(trim_env_value "${AIC_LEWM_POLICY_TRACE_OFFICIAL_TRIAL_IDS:-}")"
  POLICY_TRACE_OFFICIAL_TRIAL_ID_MAP_TRIMMED="$(trim_env_value "${AIC_LEWM_POLICY_TRACE_OFFICIAL_TRIAL_ID_MAP:-}")"
  POLICY_TRACE_TRUST_TASK_OFFICIAL_TRIAL_ID_TRIMMED="$(trim_env_value "${AIC_LEWM_POLICY_TRACE_TRUST_TASK_OFFICIAL_TRIAL_ID:-0}")"
  if [[ -z "$POLICY_TRACE_OFFICIAL_TRIAL_IDS_TRIMMED" \
    && -z "$POLICY_TRACE_OFFICIAL_TRIAL_ID_MAP_TRIMMED" \
    && "$POLICY_TRACE_TRUST_TASK_OFFICIAL_TRIAL_ID_TRIMMED" != "1" \
    && "$POLICY_TRACE_TRUST_TASK_OFFICIAL_TRIAL_ID_TRIMMED" != "true" ]]; then
    {
      echo "AIC_LEWM_POLICY_TRACE_OFFICIAL_TRIAL_IDS or AIC_LEWM_POLICY_TRACE_OFFICIAL_TRIAL_ID_MAP"
      echo "must be set before live eval when policy trace is required."
      echo "Set AIC_LEWM_POLICY_TRACE_TRUST_TASK_OFFICIAL_TRIAL_ID=1 only when the official Task"
      echo "message supplies official_trial_id."
    } >&2
    exit 2
  fi
fi
POLICY_TRACE_HOST_PATH="$HARNESS_ROOT/$POLICY_TRACE_RELATIVE_PATH"
HARNESS_LEDGER_PATH="${AIC_HARNESS_LEDGER_PATH:-$HARNESS_ROOT/ledger.jsonl}"
POLICY_TRAINING_REPORT_PATH="${AIC_POLICY_TRAINING_REPORT_PATH:-${AIC_HARNESS_POLICY_TRAINING_REPORT_PATH:-}}"
POLICY_TRAINING_REPORT_REQUIRED="${AIC_POLICY_TRAINING_REPORT_REQUIRED:-${AIC_HARNESS_POLICY_TRAINING_REPORT_REQUIRED:-0}}"
RUNTIME_POLICY_CHECKPOINT_PATH="${AIC_RUNTIME_POLICY_CHECKPOINT_PATH:-$PWD/aic_lewm_policy/runtime_artifacts/aic_lewm_epoch_100_object.ckpt}"
RUNTIME_POLICY_CHECKPOINT_CONTAINER_PATH="${AIC_RUNTIME_POLICY_CHECKPOINT_CONTAINER_PATH:-${AIC_LEWM_CHECKPOINT:-/opt/aic_lewm/aic_lewm_epoch_100_object.ckpt}}"
case "$RUNTIME_POLICY_CHECKPOINT_CONTAINER_PATH" in
  /*) ;;
  *)
    echo "AIC_RUNTIME_POLICY_CHECKPOINT_CONTAINER_PATH must be absolute" >&2
    exit 2
    ;;
esac
case "/${RUNTIME_POLICY_CHECKPOINT_CONTAINER_PATH#/}/" in
  */../* | */./* | *//*)
    echo "AIC_RUNTIME_POLICY_CHECKPOINT_CONTAINER_PATH must not contain relative path segments" >&2
    exit 2
    ;;
esac
if [[ -z "$POLICY_TRAINING_REPORT_PATH" ]]; then
  if [[ "$POLICY_TRAINING_REPORT_REQUIRED" != "0" \
    && "$POLICY_TRAINING_REPORT_REQUIRED" != "false" ]]; then
    echo "AIC_POLICY_TRAINING_REPORT_PATH is required for trained policy eval finalization" >&2
    exit 2
  fi
  if [[ "${AIC_HARNESS_GATE_ID:-}" == "trained_policy_live_eval" ]]; then
    echo "AIC_HARNESS_GATE_ID=trained_policy_live_eval requires AIC_POLICY_TRAINING_REPORT_PATH" >&2
    exit 2
  fi
else
  if [[ ! -f "$POLICY_TRAINING_REPORT_PATH" ]]; then
    echo "AIC_POLICY_TRAINING_REPORT_PATH must point to a file: $POLICY_TRAINING_REPORT_PATH" >&2
    exit 2
  fi
  if [[ ! -f "$RUNTIME_POLICY_CHECKPOINT_PATH" ]]; then
    echo "AIC_RUNTIME_POLICY_CHECKPOINT_PATH must point to a file: $RUNTIME_POLICY_CHECKPOINT_PATH" >&2
    exit 2
  fi
fi
# Docker network DNS labels have practical length limits; keep the full run id
# for result paths, but use a short deterministic alias for container hostnames.
SAFE_PREFIX="$(printf '%s' "$AIC_EVAL_RUN_ID" | tr -c 'A-Za-z0-9_.-' '-' | cut -c1-36)"
RUN_HASH="$(printf '%s' "$AIC_EVAL_RUN_ID" | sha256sum | cut -c1-12)"
SAFE_RUN_ID="${SAFE_PREFIX}-${RUN_HASH}"
NETWORK_NAME="aic_eval_${SAFE_RUN_ID}"
EVAL_CONTAINER="aic_eval_${SAFE_RUN_ID}"
MODEL_CONTAINER="aic_model_${SAFE_RUN_ID}"
EVAL_DOCKER_ARGS=()
MODEL_DOCKER_ARGS=()
DOCKER_GPU_ARGS=()

if [[ "${AIC_DOCKER_GPUS:-0}" != "0" && "${AIC_DOCKER_GPUS:-0}" != "false" ]]; then
  DOCKER_GPU_ARGS=(--gpus "${AIC_DOCKER_GPUS:-all}")
fi

if [[ -n "$POLICY_TRAINING_REPORT_PATH" ]]; then
  MODEL_DOCKER_ARGS+=(
    -v "$RUNTIME_POLICY_CHECKPOINT_PATH:$RUNTIME_POLICY_CHECKPOINT_CONTAINER_PATH:ro"
  )
fi

if [[ "${AIC_EVAL_USE_LOCAL_LAUNCH:-1}" != "0" ]]; then
  AIC_EVAL_LAUNCH_OVERRIDE_PATH="${AIC_EVAL_LAUNCH_OVERRIDE_PATH:-$PWD/aic_bringup/launch/aic_gz_bringup.launch.py}"
  if [[ -f "$AIC_EVAL_LAUNCH_OVERRIDE_PATH" ]]; then
    EVAL_DOCKER_ARGS+=(
      -v "$AIC_EVAL_LAUNCH_OVERRIDE_PATH:/ws_aic/install/share/aic_bringup/launch/aic_gz_bringup.launch.py:ro"
    )
  else
    echo "AIC eval launch override missing: $AIC_EVAL_LAUNCH_OVERRIDE_PATH" >&2
    exit 2
  fi
fi

mkdir -p "$RESULT_ROOT/eval" "$HARNESS_ROOT"

cleanup() {
  set +e
  if sudo docker ps -a --format '{{.Names}}' | grep -qx "$EVAL_CONTAINER"; then
    sudo docker logs "$EVAL_CONTAINER" >"$RESULT_ROOT/eval.log" 2>&1
  fi
  if sudo docker ps -a --format '{{.Names}}' | grep -qx "$MODEL_CONTAINER"; then
    sudo docker logs "$MODEL_CONTAINER" >"$RESULT_ROOT/model.log" 2>&1
  fi
  sudo docker rm -f "$MODEL_CONTAINER" "$EVAL_CONTAINER" >/dev/null 2>&1
  sudo docker network rm "$NETWORK_NAME" >/dev/null 2>&1
  sudo chown -R "$USER:$USER" "$RESULT_ROOT" >/dev/null 2>&1
}
trap cleanup EXIT

sudo docker rm -f "$MODEL_CONTAINER" "$EVAL_CONTAINER" >/dev/null 2>&1 || true
sudo docker network rm "$NETWORK_NAME" >/dev/null 2>&1 || true
sudo docker network create "$NETWORK_NAME" >/dev/null
sudo docker pull "$AIC_EVAL_IMAGE"

sudo docker run -d \
  --name "$EVAL_CONTAINER" \
  "${DOCKER_GPU_ARGS[@]}" \
  --network "$NETWORK_NAME" \
  -e AIC_EVAL_PASSWD=CHANGE_IN_PROD \
  -e AIC_MODEL_PASSWD=CHANGE_IN_PROD \
  -e AIC_RESULTS_DIR=/aic_results \
  -v "$RESULT_ROOT/eval:/aic_results" \
  "${EVAL_DOCKER_ARGS[@]}" \
  "$AIC_EVAL_IMAGE" \
  gazebo_gui:=false \
  launch_rviz:=false \
  ground_truth:=false \
  start_aic_engine:=true \
  shutdown_on_aic_engine_exit:=true \
  gz_verbosity_level:="${AIC_GZ_VERBOSITY_LEVEL:-1}" \
  model_discovery_timeout_seconds:=600

sudo docker run -d \
  --name "$MODEL_CONTAINER" \
  "${DOCKER_GPU_ARGS[@]}" \
  --network "$NETWORK_NAME" \
  -v "$HARNESS_ROOT:/aic_results/harness" \
  "${MODEL_DOCKER_ARGS[@]}" \
  -e RMW_IMPLEMENTATION=rmw_zenoh_cpp \
  -e ZENOH_ROUTER_CHECK_ATTEMPTS=-1 \
  -e AIC_ROUTER_ADDR="$EVAL_CONTAINER:7447" \
  -e AIC_MODEL_PASSWD=CHANGE_IN_PROD \
  -e AIC_EVAL_RUN_ID="$AIC_EVAL_RUN_ID" \
  -e AIC_LEWM_POLICY_TRACE_PATH="$POLICY_TRACE_CONTAINER_PATH" \
  -e AIC_LEWM_POLICY_TRACE_RUN_ID="$AIC_EVAL_RUN_ID" \
  -e AIC_LEWM_POLICY_TRACE_OFFICIAL_TRIAL_IDS="${AIC_LEWM_POLICY_TRACE_OFFICIAL_TRIAL_IDS:-}" \
  -e AIC_LEWM_POLICY_TRACE_OFFICIAL_TRIAL_ID_MAP="${AIC_LEWM_POLICY_TRACE_OFFICIAL_TRIAL_ID_MAP:-}" \
  -e AIC_LEWM_POLICY_TRACE_TRUST_TASK_OFFICIAL_TRIAL_ID="${AIC_LEWM_POLICY_TRACE_TRUST_TASK_OFFICIAL_TRIAL_ID:-0}" \
  -e AIC_LEWM_PLANNER_MODE="${AIC_LEWM_PLANNER_MODE:-lewm_mpc}" \
  -e AIC_LEWM_CHECKPOINT="$RUNTIME_POLICY_CHECKPOINT_CONTAINER_PATH" \
  -e AIC_LEWM_DEVICE="${AIC_LEWM_DEVICE:-cpu}" \
  -e AIC_LEWM_REQUIRE_CHECKPOINT="${AIC_LEWM_REQUIRE_CHECKPOINT:-1}" \
  -e AIC_LEWM_GOAL_DATASET="${AIC_LEWM_GOAL_DATASET:-/opt/aic_lewm/aic_qualification_train.h5}" \
  -e AIC_LEWM_REPLAY_DATASET="${AIC_LEWM_REPLAY_DATASET:-${AIC_LEWM_GOAL_DATASET:-/opt/aic_lewm/aic_qualification_train.h5}}" \
  -e AIC_LEWM_MAX_RUNTIME_SEC="${AIC_LEWM_MAX_RUNTIME_SEC:-30}" \
  -e AIC_LEWM_CONTROL_HZ="${AIC_LEWM_CONTROL_HZ:-4}" \
  -e AIC_LEWM_COMMAND_FRAME="${AIC_LEWM_COMMAND_FRAME:-}" \
  -e AIC_LEWM_LINEAR_VEL_LIMIT="${AIC_LEWM_LINEAR_VEL_LIMIT:-}" \
  -e AIC_LEWM_ANGULAR_VEL_LIMIT="${AIC_LEWM_ANGULAR_VEL_LIMIT:-}" \
  -e AIC_LEWM_NUM_ACTION_CANDIDATES="${AIC_LEWM_NUM_ACTION_CANDIDATES:-4}" \
  -e AIC_LEWM_PLANNING_HORIZON="${AIC_LEWM_PLANNING_HORIZON:-1}" \
  -e AIC_LEWM_REPLAY_HZ="${AIC_LEWM_REPLAY_HZ:-10}" \
  -e AIC_LEWM_REPLAY_TIME_SCALE="${AIC_LEWM_REPLAY_TIME_SCALE:-1}" \
  -e AIC_LEWM_REPLAY_ACTION_GAIN="${AIC_LEWM_REPLAY_ACTION_GAIN:-1}" \
  -e AIC_LEWM_SC_REPLAY_STOP_SEC="${AIC_LEWM_SC_REPLAY_STOP_SEC:-}" \
  -e AIC_LEWM_FINAL_SERVO_ENABLED="${AIC_LEWM_FINAL_SERVO_ENABLED:-0}" \
  -e AIC_LEWM_FINAL_SERVO_DURATION_SEC="${AIC_LEWM_FINAL_SERVO_DURATION_SEC:-1.25}" \
  -e AIC_LEWM_FINAL_SERVO_FORCE_GUARD_N="${AIC_LEWM_FINAL_SERVO_FORCE_GUARD_N:-18}" \
  -e AIC_LEWM_FINAL_SERVO_FORCE_GUARD_MODE="${AIC_LEWM_FINAL_SERVO_FORCE_GUARD_MODE:-absolute}" \
  -e AIC_LEWM_SFP_FINAL_SERVO_LINEAR="${AIC_LEWM_SFP_FINAL_SERVO_LINEAR:-0,0,-0.02}" \
  -e AIC_LEWM_SFP_FINAL_SERVO_ANGULAR="${AIC_LEWM_SFP_FINAL_SERVO_ANGULAR:-0,0,0}" \
  -e AIC_LEWM_SC_FINAL_SERVO_LINEAR="${AIC_LEWM_SC_FINAL_SERVO_LINEAR:-0,0,0}" \
  -e AIC_LEWM_SC_FINAL_SERVO_ANGULAR="${AIC_LEWM_SC_FINAL_SERVO_ANGULAR:-0,0,0}" \
  "$AIC_MODEL_IMAGE" \
  --ros-args \
  -p use_sim_time:=true \
  -p policy:=aic_lewm_policy.LewmMpcPolicy

set +e
timeout "$AIC_EVAL_TIMEOUT_SEC" sudo docker wait "$EVAL_CONTAINER" >"$RESULT_ROOT/eval_exit_code.txt"
wait_status=$?
set -e

if [[ "$wait_status" -ne 0 ]]; then
  echo "Eval container did not complete cleanly; timeout/status=$wait_status" >&2
  exit "$wait_status"
fi

eval_exit_code="$(cat "$RESULT_ROOT/eval_exit_code.txt")"
if [[ "$eval_exit_code" != "0" ]]; then
  echo "Eval container exited with code $eval_exit_code" >&2
  exit "$eval_exit_code"
fi

POLICY_TRACE_REQUIRED="${AIC_LEWM_POLICY_TRACE_REQUIRED:-1}"
if [[ "$POLICY_TRACE_REQUIRED" != "0" && "$POLICY_TRACE_REQUIRED" != "false" ]]; then
  if [[ ! -s "$POLICY_TRACE_HOST_PATH" ]]; then
    echo "Missing required policy trace JSONL: $POLICY_TRACE_HOST_PATH" >&2
    exit 2
  fi
fi

if [[ -n "$POLICY_TRAINING_REPORT_PATH" ]]; then
  FINALIZE_ARGS=(
    -m aic_signal_harness.train_eval_promote
    finalize
    --run-id "$AIC_EVAL_RUN_ID"
    --result-root "$RESULT_ROOT"
    --harness-root "$HARNESS_ROOT"
    --scoring-yaml "$RESULT_ROOT/eval/scoring.yaml"
    --ledger "$HARNESS_LEDGER_PATH"
    --policy-training-report "$POLICY_TRAINING_REPORT_PATH"
    --runtime-policy-checkpoint "$RUNTIME_POLICY_CHECKPOINT_PATH"
    --runtime-checkpoint-container-path "$RUNTIME_POLICY_CHECKPOINT_CONTAINER_PATH"
    --model-image "$AIC_MODEL_IMAGE"
    --model-image-id "$AIC_MODEL_IMAGE_ID"
    --planner-mode "${AIC_LEWM_PLANNER_MODE:-lewm_mpc}"
    --gate-id "${AIC_HARNESS_GATE_ID:-trained_policy_live_eval}"
    --min-improvement "${AIC_HARNESS_MIN_IMPROVEMENT:-1.0}"
  )
else
  FINALIZE_ARGS=(
    -m aic_signal_harness.live_eval
    finalize
    --run-id "$AIC_EVAL_RUN_ID"
    --result-root "$RESULT_ROOT"
    --harness-root "$HARNESS_ROOT"
    --scoring-yaml "$RESULT_ROOT/eval/scoring.yaml"
    --ledger "$HARNESS_LEDGER_PATH"
    --model-image "$AIC_MODEL_IMAGE"
    --model-image-id "$AIC_MODEL_IMAGE_ID"
    --planner-mode "${AIC_LEWM_PLANNER_MODE:-lewm_mpc}"
    --gate-id "${AIC_HARNESS_GATE_ID:-live_eval}"
    --min-improvement "${AIC_HARNESS_MIN_IMPROVEMENT:-1.0}"
  )
fi
if [[ -s "$POLICY_TRACE_HOST_PATH" ]]; then
  FINALIZE_ARGS+=(--policy-trace "$POLICY_TRACE_HOST_PATH")
fi
if [[ -n "${AIC_HARNESS_BASELINE_PATH:-}" ]]; then
  FINALIZE_ARGS+=(--baseline "$AIC_HARNESS_BASELINE_PATH")
fi
if [[ -n "${AIC_HARNESS_UPDATE_BASELINE_PATH:-}" ]]; then
  FINALIZE_ARGS+=(--update-baseline "$AIC_HARNESS_UPDATE_BASELINE_PATH")
fi
if [[ -n "${AIC_HARNESS_EXPERIMENT_ID:-}" ]]; then
  FINALIZE_ARGS+=(--experiment-id "$AIC_HARNESS_EXPERIMENT_ID")
fi
if [[ -n "${AIC_HARNESS_HYPOTHESIS:-}" ]]; then
  FINALIZE_ARGS+=(--hypothesis "$AIC_HARNESS_HYPOTHESIS")
fi
if [[ -n "${AIC_HARNESS_BACKEND_KIND:-}" ]]; then
  FINALIZE_ARGS+=(--backend-kind "$AIC_HARNESS_BACKEND_KIND")
fi
if [[ "${AIC_HARNESS_BOOTSTRAP_PROMOTION:-0}" != "0" && "${AIC_HARNESS_BOOTSTRAP_PROMOTION:-0}" != "false" ]]; then
  FINALIZE_ARGS+=(--bootstrap-promotion)
fi
if [[ "${AIC_HARNESS_ELIGIBLE_FOR_SUBMISSION:-0}" != "0" && "${AIC_HARNESS_ELIGIBLE_FOR_SUBMISSION:-0}" != "false" ]]; then
  FINALIZE_ARGS+=(--eligible-for-submission)
fi
if [[ "${AIC_HARNESS_OVERWRITE:-0}" != "0" && "${AIC_HARNESS_OVERWRITE:-0}" != "false" ]]; then
  FINALIZE_ARGS+=(--overwrite)
fi
if [[ "${AIC_HARNESS_NO_APPEND_LEDGER:-0}" != "0" && "${AIC_HARNESS_NO_APPEND_LEDGER:-0}" != "false" ]]; then
  FINALIZE_ARGS+=(--no-append-ledger)
fi
if [[ "${AIC_HARNESS_NO_NEXT_EXPERIMENT:-0}" != "0" && "${AIC_HARNESS_NO_NEXT_EXPERIMENT:-0}" != "false" ]]; then
  FINALIZE_ARGS+=(--no-next-experiment)
fi
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" python3 "${FINALIZE_ARGS[@]}"

echo "AIC_EVAL_RESULT_ROOT=$RESULT_ROOT"
if [[ -f "$RESULT_ROOT/eval/scoring.yaml" ]]; then
  echo "AIC_EVAL_SCORING_PATH=$RESULT_ROOT/eval/scoring.yaml"
fi
if [[ -f "$POLICY_TRACE_HOST_PATH" ]]; then
  echo "AIC_POLICY_TRACE_PATH=$POLICY_TRACE_HOST_PATH"
fi
if [[ -f "$HARNESS_ROOT/policy_trace_report.json" ]]; then
  echo "AIC_POLICY_TRACE_REPORT_PATH=$HARNESS_ROOT/policy_trace_report.json"
fi
if [[ -f "$HARNESS_ROOT/policy_trace_artifact.json" ]]; then
  echo "AIC_POLICY_TRACE_ARTIFACT_PATH=$HARNESS_ROOT/policy_trace_artifact.json"
fi
if [[ -f "$HARNESS_ROOT/run_manifest.json" ]]; then
  echo "AIC_HARNESS_MANIFEST_PATH=$HARNESS_ROOT/run_manifest.json"
fi
if [[ -f "$HARNESS_ROOT/ledger_entry.json" ]]; then
  echo "AIC_HARNESS_LEDGER_ENTRY_PATH=$HARNESS_ROOT/ledger_entry.json"
fi
if [[ -f "$HARNESS_ROOT/next_experiment.json" ]]; then
  echo "AIC_HARNESS_NEXT_EXPERIMENT_PATH=$HARNESS_ROOT/next_experiment.json"
fi
if [[ -f "$HARNESS_ROOT/live_eval_summary.json" ]]; then
  echo "AIC_HARNESS_SUMMARY_PATH=$HARNESS_ROOT/live_eval_summary.json"
fi
if [[ -f "$HARNESS_ROOT/trained_policy_eval_binding.json" ]]; then
  echo "AIC_TRAIN_EVAL_PROMOTE_BINDING_PATH=$HARNESS_ROOT/trained_policy_eval_binding.json"
fi
if [[ -f "$HARNESS_ROOT/train_eval_promote_summary.json" ]]; then
  echo "AIC_TRAIN_EVAL_PROMOTE_SUMMARY_PATH=$HARNESS_ROOT/train_eval_promote_summary.json"
fi
