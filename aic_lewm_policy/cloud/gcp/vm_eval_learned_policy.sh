#!/usr/bin/env bash
set -euo pipefail

AIC_EVAL_RUN_ID="${AIC_EVAL_RUN_ID:?Set AIC_EVAL_RUN_ID}"
AIC_MODEL_IMAGE="${AIC_MODEL_IMAGE:-aic-lewm-learned:latest}"
AIC_EVAL_IMAGE="${AIC_EVAL_IMAGE:-ghcr.io/intrinsic-dev/aic/aic_eval:latest}"
AIC_EVAL_TIMEOUT_SEC="${AIC_EVAL_TIMEOUT_SEC:-1800}"

RESULT_ROOT="${AIC_EVAL_RESULT_ROOT:-$HOME/aic_results/$AIC_EVAL_RUN_ID}"
# Docker network DNS labels have practical length limits; keep the full run id
# for result paths, but use a short deterministic alias for container hostnames.
SAFE_PREFIX="$(printf '%s' "$AIC_EVAL_RUN_ID" | tr -c 'A-Za-z0-9_.-' '-' | cut -c1-36)"
RUN_HASH="$(printf '%s' "$AIC_EVAL_RUN_ID" | sha256sum | cut -c1-12)"
SAFE_RUN_ID="${SAFE_PREFIX}-${RUN_HASH}"
NETWORK_NAME="aic_eval_${SAFE_RUN_ID}"
EVAL_CONTAINER="aic_eval_${SAFE_RUN_ID}"
MODEL_CONTAINER="aic_model_${SAFE_RUN_ID}"
EVAL_DOCKER_ARGS=()
DOCKER_GPU_ARGS=()

if [[ "${AIC_DOCKER_GPUS:-0}" != "0" && "${AIC_DOCKER_GPUS:-0}" != "false" ]]; then
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
  ground_truth:=false \
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
	  -e AIC_LEWM_PLANNER_MODE="${AIC_LEWM_PLANNER_MODE:-lewm_mpc}" \
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

echo "AIC_EVAL_RESULT_ROOT=$RESULT_ROOT"
if [[ -f "$RESULT_ROOT/eval/scoring.yaml" ]]; then
  echo "AIC_EVAL_SCORING_PATH=$RESULT_ROOT/eval/scoring.yaml"
fi
