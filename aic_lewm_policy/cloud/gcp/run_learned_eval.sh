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

if [[ -z "$AIC_EVAL_RUN_ID" ]]; then
  echo "Set AIC_EVAL_RUN_ID or LEWM_RUN_NAME." >&2
  exit 2
fi

if [[ ! -f "$AIC_ROOT/aic_lewm_policy/runtime_artifacts/aic_lewm_epoch_100_object.ckpt" ]]; then
  echo "Missing staged checkpoint under aic_lewm_policy/runtime_artifacts." >&2
  exit 2
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
export AIC_HARNESS_GATE_ID='$(quote_for_remote "${AIC_HARNESS_GATE_ID:-live_eval}")'
export AIC_HARNESS_MIN_IMPROVEMENT='$(quote_for_remote "${AIC_HARNESS_MIN_IMPROVEMENT:-1.0}")'
export AIC_HARNESS_LEDGER_PATH='$(quote_for_remote "${AIC_HARNESS_LEDGER_PATH:-}")'
export AIC_HARNESS_BASELINE_PATH='$(quote_for_remote "${AIC_HARNESS_BASELINE_PATH:-}")'
export AIC_HARNESS_UPDATE_BASELINE_PATH='$(quote_for_remote "${AIC_HARNESS_UPDATE_BASELINE_PATH:-}")'
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
