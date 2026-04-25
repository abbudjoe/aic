#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

require_instance

AIC_EVAL_RUN_ID="${AIC_EVAL_RUN_ID:-${AIC_EXPERIMENT_RUN_ID:-learned-$LEWM_RUN_NAME}}"
AIC_MODEL_IMAGE="${AIC_MODEL_IMAGE:-aic-lewm-learned:$AIC_EVAL_RUN_ID}"
REMOTE_AIC_ROOT="${REMOTE_AIC_ROOT:-/home/$USER/src/aic-learned-eval}"
REMOTE_TARBALL="${REMOTE_TARBALL:-/tmp/aic-learned-eval-src.tgz}"
LOCAL_TARBALL="${LOCAL_TARBALL:-$AIC_ROOT/artifacts/aic-learned-eval-src.tgz}"

resolve_existing_file() {
  local path="$1"
  if [[ "$path" != /* ]]; then
    path="$AIC_ROOT/$path"
  fi
  if [[ ! -f "$path" ]]; then
    echo "Required file does not exist: $path" >&2
    exit 2
  fi
  local directory
  directory="$(cd -- "$(dirname -- "$path")" && pwd -P)"
  printf '%s/%s' "$directory" "$(basename -- "$path")"
}

run_local_harness_python() {
  if command -v pixi >/dev/null 2>&1; then
    (cd "$AIC_ROOT" && pixi run python -m "$@")
  else
    (cd "$AIC_ROOT" && PYTHONPATH="$AIC_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 -m "$@")
  fi
}

stage_output_value() {
  local output="$1"
  local key="$2"
  local value
  value="$(printf '%s\n' "$output" | awk -v key="$key" 'index($0, key "=") == 1 { print substr($0, length(key) + 2) }' | tail -n 1)"
  if [[ -z "$value" ]]; then
    echo "stage-training-bundle did not emit $key" >&2
    exit 2
  fi
  printf '%s' "$value"
}

if [[ -z "$AIC_EVAL_RUN_ID" ]]; then
  echo "Set AIC_EVAL_RUN_ID or LEWM_RUN_NAME." >&2
  exit 2
fi

LOCAL_AIC_ROOT="$(cd -- "$AIC_ROOT" && pwd -P)"
REMOTE_POLICY_TRAINING_REPORT_PATH=""
REMOTE_RUNTIME_POLICY_CHECKPOINT_PATH=""
POLICY_TRAINING_REPORT_PATH="${AIC_POLICY_TRAINING_REPORT_PATH:-${AIC_HARNESS_POLICY_TRAINING_REPORT_PATH:-}}"
POLICY_TRAINING_REPORT_REQUIRED="${AIC_POLICY_TRAINING_REPORT_REQUIRED:-${AIC_HARNESS_POLICY_TRAINING_REPORT_REQUIRED:-0}}"
if [[ -n "$POLICY_TRAINING_REPORT_PATH" ]]; then
  if [[ -n "${AIC_RUNTIME_POLICY_CHECKPOINT_PATH:-}" ]]; then
    echo "AIC_RUNTIME_POLICY_CHECKPOINT_PATH must not be set with AIC_POLICY_TRAINING_REPORT_PATH; the staged training bundle supplies the evaluated checkpoint." >&2
    exit 2
  fi
  LOCAL_POLICY_TRAINING_REPORT_PATH="$(resolve_existing_file "$POLICY_TRAINING_REPORT_PATH")"
  TRAINING_BUNDLE_REL="aic_lewm_policy/runtime_artifacts/train_eval_promote/$AIC_EVAL_RUN_ID"
  LOCAL_TRAINING_BUNDLE_ROOT="$AIC_ROOT/$TRAINING_BUNDLE_REL"
  REMOTE_TRAINING_BUNDLE_ROOT="$REMOTE_AIC_ROOT/$TRAINING_BUNDLE_REL"
  stage_output="$(run_local_harness_python \
    aic_signal_harness.train_eval_promote \
    stage-training-bundle \
    --policy-training-report "$LOCAL_POLICY_TRAINING_REPORT_PATH" \
    --bundle-root "$LOCAL_TRAINING_BUNDLE_ROOT" \
    --runtime-bundle-root "$REMOTE_TRAINING_BUNDLE_ROOT" \
    --overwrite)"
  printf '%s\n' "$stage_output"
  REMOTE_POLICY_TRAINING_REPORT_PATH="$(stage_output_value "$stage_output" AIC_STAGED_RUNTIME_POLICY_TRAINING_REPORT_PATH)"
  REMOTE_RUNTIME_POLICY_CHECKPOINT_PATH="$(stage_output_value "$stage_output" AIC_STAGED_RUNTIME_POLICY_CHECKPOINT_PATH)"
elif [[ "$POLICY_TRAINING_REPORT_REQUIRED" != "0" && "$POLICY_TRAINING_REPORT_REQUIRED" != "false" ]]; then
  echo "AIC_POLICY_TRAINING_REPORT_PATH is required for trained policy eval finalization." >&2
  exit 2
else
  LOCAL_RUNTIME_POLICY_CHECKPOINT_PATH="$(resolve_existing_file "${AIC_RUNTIME_POLICY_CHECKPOINT_PATH:-$AIC_ROOT/aic_lewm_policy/runtime_artifacts/aic_lewm_epoch_100_object.ckpt}")"
  if [[ "$LOCAL_RUNTIME_POLICY_CHECKPOINT_PATH" != "$LOCAL_AIC_ROOT/"* ]]; then
    echo "AIC_RUNTIME_POLICY_CHECKPOINT_PATH must be under AIC_ROOT so it is staged to the VM." >&2
    exit 2
  fi
  REMOTE_RUNTIME_POLICY_CHECKPOINT_PATH="$REMOTE_AIC_ROOT/${LOCAL_RUNTIME_POLICY_CHECKPOINT_PATH#"$LOCAL_AIC_ROOT/"}"
fi

if [[ ! -f "$AIC_ROOT/aic_lewm_policy/runtime_artifacts/aic_qualification_train.h5" ]]; then
  echo "Missing staged goal dataset under aic_lewm_policy/runtime_artifacts." >&2
  exit 2
fi

mkdir -p "$(dirname "$LOCAL_TARBALL")"
tar \
  --exclude='.git' \
  --exclude='.pixi' \
  --exclude='artifacts' \
  --exclude='**/__pycache__' \
  -czf "$LOCAL_TARBALL" \
  -C "$AIC_ROOT" .

gcloud compute scp "$LOCAL_TARBALL" "$GCP_INSTANCE:$REMOTE_TARBALL" \
  --project "$GCP_PROJECT" \
  --zone "$GCP_ZONE"

gcloud compute scp "$SCRIPT_DIR/vm_eval_learned_policy.sh" \
  "$GCP_INSTANCE:~/vm_eval_learned_policy.sh" \
  --project "$GCP_PROJECT" \
  --zone "$GCP_ZONE"

remote_command="
set -euo pipefail
rm -rf '$(quote_for_remote "$REMOTE_AIC_ROOT")'
mkdir -p '$(quote_for_remote "$REMOTE_AIC_ROOT")'
tar -xzf '$(quote_for_remote "$REMOTE_TARBALL")' -C '$(quote_for_remote "$REMOTE_AIC_ROOT")'
cd '$(quote_for_remote "$REMOTE_AIC_ROOT")'
if [[ '$(quote_for_remote "${AIC_MODEL_SKIP_BUILD:-0}")' == '1' ]]; then
  sudo docker image inspect '$(quote_for_remote "$AIC_MODEL_IMAGE")' >/dev/null
else
  sudo DOCKER_BUILDKIT=1 docker build -f docker/aic_lewm_policy/Dockerfile -t '$(quote_for_remote "$AIC_MODEL_IMAGE")' .
fi
AIC_MODEL_IMAGE_ID=\"\$(sudo docker image inspect --format '{{.Id}}' '$(quote_for_remote "$AIC_MODEL_IMAGE")')\"
chmod +x ~/vm_eval_learned_policy.sh
export AIC_EVAL_RUN_ID='$(quote_for_remote "$AIC_EVAL_RUN_ID")'
export AIC_MODEL_IMAGE='$(quote_for_remote "$AIC_MODEL_IMAGE")'
export AIC_MODEL_IMAGE_ID=\"\$AIC_MODEL_IMAGE_ID\"
export AIC_EVAL_IMAGE='$(quote_for_remote "${AIC_EVAL_IMAGE:-ghcr.io/intrinsic-dev/aic/aic_eval:latest}")'
export AIC_EVAL_USE_LOCAL_LAUNCH='$(quote_for_remote "${AIC_EVAL_USE_LOCAL_LAUNCH:-1}")'
export AIC_GZ_VERBOSITY_LEVEL='$(quote_for_remote "${AIC_GZ_VERBOSITY_LEVEL:-1}")'
export AIC_DOCKER_GPUS='$(quote_for_remote "${AIC_DOCKER_GPUS:-0}")'
export AIC_LEWM_POLICY_TRACE_REQUIRED='$(quote_for_remote "${AIC_LEWM_POLICY_TRACE_REQUIRED:-1}")'
export AIC_LEWM_POLICY_TRACE_CONTAINER_PATH='$(quote_for_remote "${AIC_LEWM_POLICY_TRACE_CONTAINER_PATH:-/aic_results/harness/policy_trace.jsonl}")'
export AIC_LEWM_POLICY_TRACE_OFFICIAL_TRIAL_IDS='$(quote_for_remote "${AIC_LEWM_POLICY_TRACE_OFFICIAL_TRIAL_IDS:-}")'
export AIC_LEWM_POLICY_TRACE_OFFICIAL_TRIAL_ID_MAP='$(quote_for_remote "${AIC_LEWM_POLICY_TRACE_OFFICIAL_TRIAL_ID_MAP:-}")'
export AIC_LEWM_POLICY_TRACE_TRUST_TASK_OFFICIAL_TRIAL_ID='$(quote_for_remote "${AIC_LEWM_POLICY_TRACE_TRUST_TASK_OFFICIAL_TRIAL_ID:-0}")'
export AIC_HARNESS_GATE_ID='$(quote_for_remote "${AIC_HARNESS_GATE_ID:-}")'
export AIC_HARNESS_MIN_IMPROVEMENT='$(quote_for_remote "${AIC_HARNESS_MIN_IMPROVEMENT:-1.0}")'
export AIC_HARNESS_LEDGER_PATH='$(quote_for_remote "${AIC_HARNESS_LEDGER_PATH:-}")'
export AIC_HARNESS_BASELINE_PATH='$(quote_for_remote "${AIC_HARNESS_BASELINE_PATH:-}")'
export AIC_HARNESS_UPDATE_BASELINE_PATH='$(quote_for_remote "${AIC_HARNESS_UPDATE_BASELINE_PATH:-}")'
export AIC_POLICY_TRAINING_REPORT_PATH='$(quote_for_remote "$REMOTE_POLICY_TRAINING_REPORT_PATH")'
export AIC_POLICY_TRAINING_REPORT_REQUIRED='$(quote_for_remote "$POLICY_TRAINING_REPORT_REQUIRED")'
export AIC_RUNTIME_POLICY_CHECKPOINT_PATH='$(quote_for_remote "$REMOTE_RUNTIME_POLICY_CHECKPOINT_PATH")'
export AIC_RUNTIME_POLICY_CHECKPOINT_CONTAINER_PATH='$(quote_for_remote "${AIC_RUNTIME_POLICY_CHECKPOINT_CONTAINER_PATH:-}")'
export AIC_HARNESS_EXPERIMENT_ID='$(quote_for_remote "${AIC_HARNESS_EXPERIMENT_ID:-}")'
export AIC_HARNESS_HYPOTHESIS='$(quote_for_remote "${AIC_HARNESS_HYPOTHESIS:-}")'
export AIC_HARNESS_BACKEND_KIND='$(quote_for_remote "${AIC_HARNESS_BACKEND_KIND:-}")'
export AIC_HARNESS_BOOTSTRAP_PROMOTION='$(quote_for_remote "${AIC_HARNESS_BOOTSTRAP_PROMOTION:-0}")'
export AIC_HARNESS_ELIGIBLE_FOR_SUBMISSION='$(quote_for_remote "${AIC_HARNESS_ELIGIBLE_FOR_SUBMISSION:-0}")'
export AIC_HARNESS_OVERWRITE='$(quote_for_remote "${AIC_HARNESS_OVERWRITE:-0}")'
export AIC_HARNESS_NO_APPEND_LEDGER='$(quote_for_remote "${AIC_HARNESS_NO_APPEND_LEDGER:-0}")'
export AIC_HARNESS_NO_NEXT_EXPERIMENT='$(quote_for_remote "${AIC_HARNESS_NO_NEXT_EXPERIMENT:-0}")'
export AIC_EVAL_TIMEOUT_SEC='$(quote_for_remote "${AIC_EVAL_TIMEOUT_SEC:-1800}")'
export AIC_LEWM_PLANNER_MODE='$(quote_for_remote "${AIC_LEWM_PLANNER_MODE:-lewm_mpc}")'
export AIC_LEWM_DEVICE='$(quote_for_remote "${AIC_LEWM_DEVICE:-cpu}")'
export AIC_LEWM_REQUIRE_CHECKPOINT='$(quote_for_remote "${AIC_LEWM_REQUIRE_CHECKPOINT:-1}")'
export AIC_LEWM_GOAL_DATASET='$(quote_for_remote "${AIC_LEWM_GOAL_DATASET:-/opt/aic_lewm/aic_qualification_train.h5}")'
export AIC_LEWM_REPLAY_DATASET='$(quote_for_remote "${AIC_LEWM_REPLAY_DATASET:-${AIC_LEWM_GOAL_DATASET:-/opt/aic_lewm/aic_qualification_train.h5}}")'
export AIC_LEWM_MAX_RUNTIME_SEC='$(quote_for_remote "${AIC_LEWM_MAX_RUNTIME_SEC:-30}")'
export AIC_LEWM_CONTROL_HZ='$(quote_for_remote "${AIC_LEWM_CONTROL_HZ:-4}")'
export AIC_LEWM_COMMAND_FRAME='$(quote_for_remote "${AIC_LEWM_COMMAND_FRAME:-}")'
export AIC_LEWM_LINEAR_VEL_LIMIT='$(quote_for_remote "${AIC_LEWM_LINEAR_VEL_LIMIT:-}")'
export AIC_LEWM_ANGULAR_VEL_LIMIT='$(quote_for_remote "${AIC_LEWM_ANGULAR_VEL_LIMIT:-}")'
export AIC_LEWM_NUM_ACTION_CANDIDATES='$(quote_for_remote "${AIC_LEWM_NUM_ACTION_CANDIDATES:-4}")'
export AIC_LEWM_PLANNING_HORIZON='$(quote_for_remote "${AIC_LEWM_PLANNING_HORIZON:-1}")'
export AIC_LEWM_REPLAY_HZ='$(quote_for_remote "${AIC_LEWM_REPLAY_HZ:-10}")'
export AIC_LEWM_REPLAY_TIME_SCALE='$(quote_for_remote "${AIC_LEWM_REPLAY_TIME_SCALE:-1}")'
export AIC_LEWM_REPLAY_ACTION_GAIN='$(quote_for_remote "${AIC_LEWM_REPLAY_ACTION_GAIN:-1}")'
export AIC_LEWM_SC_REPLAY_STOP_SEC='$(quote_for_remote "${AIC_LEWM_SC_REPLAY_STOP_SEC:-}")'
export AIC_LEWM_FINAL_SERVO_ENABLED='$(quote_for_remote "${AIC_LEWM_FINAL_SERVO_ENABLED:-0}")'
export AIC_LEWM_FINAL_SERVO_DURATION_SEC='$(quote_for_remote "${AIC_LEWM_FINAL_SERVO_DURATION_SEC:-1.25}")'
export AIC_LEWM_FINAL_SERVO_FORCE_GUARD_N='$(quote_for_remote "${AIC_LEWM_FINAL_SERVO_FORCE_GUARD_N:-18}")'
export AIC_LEWM_FINAL_SERVO_FORCE_GUARD_MODE='$(quote_for_remote "${AIC_LEWM_FINAL_SERVO_FORCE_GUARD_MODE:-absolute}")'
export AIC_LEWM_SFP_FINAL_SERVO_LINEAR='$(quote_for_remote "${AIC_LEWM_SFP_FINAL_SERVO_LINEAR:-0,0,-0.02}")'
export AIC_LEWM_SFP_FINAL_SERVO_ANGULAR='$(quote_for_remote "${AIC_LEWM_SFP_FINAL_SERVO_ANGULAR:-0,0,0}")'
export AIC_LEWM_SC_FINAL_SERVO_LINEAR='$(quote_for_remote "${AIC_LEWM_SC_FINAL_SERVO_LINEAR:-0,0,0}")'
export AIC_LEWM_SC_FINAL_SERVO_ANGULAR='$(quote_for_remote "${AIC_LEWM_SC_FINAL_SERVO_ANGULAR:-0,0,0}")'
~/vm_eval_learned_policy.sh
"

gcloud compute ssh "$GCP_INSTANCE" \
  --project "$GCP_PROJECT" \
  --zone "$GCP_ZONE" \
  --command "$remote_command"
