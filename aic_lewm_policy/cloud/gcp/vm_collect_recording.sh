#!/usr/bin/env bash
set -euo pipefail

AIC_RECORDING_RUN_ID="${AIC_RECORDING_RUN_ID:?Set AIC_RECORDING_RUN_ID}"
AIC_RECORDING_IMAGE="${AIC_RECORDING_IMAGE:-aic-lewm-recording:latest}"
AIC_EVAL_IMAGE="${AIC_EVAL_IMAGE:-ghcr.io/intrinsic-dev/aic/aic_eval:latest}"
AIC_RECORDING_DATASET_NAME="${AIC_RECORDING_DATASET_NAME:-aic_qualification_train}"
AIC_RECORDING_TIMEOUT_SEC="${AIC_RECORDING_TIMEOUT_SEC:-1800}"

RESULT_ROOT="${AIC_RECORDING_RESULT_ROOT:-$HOME/aic_results/$AIC_RECORDING_RUN_ID}"
DATASET_PATH="$RESULT_ROOT/$AIC_RECORDING_DATASET_NAME.h5"
SAFE_RUN_ID="$(printf '%s' "$AIC_RECORDING_RUN_ID" | tr -c 'A-Za-z0-9_.-' '-')"
NETWORK_NAME="aic_record_${SAFE_RUN_ID}"
EVAL_CONTAINER="aic_eval_${SAFE_RUN_ID}"
MODEL_CONTAINER="aic_model_${SAFE_RUN_ID}"
EVAL_DOCKER_ARGS=()
DOCKER_GPU_ARGS=()

if [[ "${AIC_DOCKER_GPUS:-all}" != "0" && "${AIC_DOCKER_GPUS:-all}" != "false" ]]; then
  DOCKER_GPU_ARGS=(--gpus "${AIC_DOCKER_GPUS:-all}")
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

mkdir -p "$RESULT_ROOT/eval"

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
  ground_truth:=true \
  start_aic_engine:=true \
  shutdown_on_aic_engine_exit:=true \
  gz_verbosity_level:="${AIC_GZ_VERBOSITY_LEVEL:-1}" \
  model_discovery_timeout_seconds:=600

sudo docker run -d \
  --name "$MODEL_CONTAINER" \
  "${DOCKER_GPU_ARGS[@]}" \
  --network "$NETWORK_NAME" \
  -e RMW_IMPLEMENTATION=rmw_zenoh_cpp \
  -e ZENOH_ROUTER_CHECK_ATTEMPTS=-1 \
  -e AIC_ROUTER_ADDR="$EVAL_CONTAINER:7447" \
  -e AIC_MODEL_PASSWD=CHANGE_IN_PROD \
  -e AIC_LEWM_DATASET_PATH="/aic_results/$AIC_RECORDING_DATASET_NAME.h5" \
  -e AIC_LEWM_RECORD_HZ="${AIC_LEWM_RECORD_HZ:-20}" \
  -e AIC_LEWM_RECORD_FLUSH_EVERY="${AIC_LEWM_RECORD_FLUSH_EVERY:-10}" \
  -e AIC_LEWM_RECORD_IMAGE_SCALE="${AIC_LEWM_RECORD_IMAGE_SCALE:-0.25}" \
  -e AIC_LEWM_CHEATCODE_APPROACH_STEPS="${AIC_LEWM_CHEATCODE_APPROACH_STEPS:-40}" \
  -e AIC_LEWM_CHEATCODE_APPROACH_SLEEP="${AIC_LEWM_CHEATCODE_APPROACH_SLEEP:-0.02}" \
  -e AIC_LEWM_CHEATCODE_INSERT_Z_STEP="${AIC_LEWM_CHEATCODE_INSERT_Z_STEP:-0.003}" \
  -e AIC_LEWM_CHEATCODE_INSERT_SLEEP="${AIC_LEWM_CHEATCODE_INSERT_SLEEP:-0.02}" \
  -e AIC_LEWM_CHEATCODE_STABILIZE_SLEEP="${AIC_LEWM_CHEATCODE_STABILIZE_SLEEP:-1.0}" \
  -v "$RESULT_ROOT:/aic_results" \
  "$AIC_RECORDING_IMAGE" \
  --ros-args \
  -p use_sim_time:=true \
  -p policy:=aic_lewm_policy.RecordingCheatCode

set +e
timeout "$AIC_RECORDING_TIMEOUT_SEC" sudo docker wait "$EVAL_CONTAINER" >"$RESULT_ROOT/eval_exit_code.txt"
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

if [[ ! -f "$DATASET_PATH" ]]; then
  echo "Recording dataset missing: $DATASET_PATH" >&2
  exit 1
fi

echo "AIC_RECORDING_RESULT_ROOT=$RESULT_ROOT"
echo "AIC_RECORDING_DATASET_PATH=$DATASET_PATH"
if [[ -f "$RESULT_ROOT/eval/scoring.yaml" ]]; then
  echo "AIC_RECORDING_SCORING_PATH=$RESULT_ROOT/eval/scoring.yaml"
fi
