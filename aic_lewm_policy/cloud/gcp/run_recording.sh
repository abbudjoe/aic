#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

require_instance

AIC_RECORDING_RUN_ID="${AIC_RECORDING_RUN_ID:-${AIC_EXPERIMENT_RUN_ID:-$LEWM_RUN_NAME}}"
AIC_RECORDING_IMAGE="${AIC_RECORDING_IMAGE:-aic-lewm-recording:$AIC_RECORDING_RUN_ID}"
REMOTE_AIC_ROOT="${REMOTE_AIC_ROOT:-/home/$USER/src/aic-recording}"
REMOTE_TARBALL="${REMOTE_TARBALL:-/tmp/aic-recording-src.tgz}"
LOCAL_TARBALL="${LOCAL_TARBALL:-$AIC_ROOT/artifacts/aic-recording-src.tgz}"

if [[ -z "$AIC_RECORDING_RUN_ID" ]]; then
  echo "Set AIC_RECORDING_RUN_ID or LEWM_RUN_NAME." >&2
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

gcloud compute scp "$SCRIPT_DIR/vm_collect_recording.sh" \
  "$GCP_INSTANCE:~/vm_collect_recording.sh" \
  --project "$GCP_PROJECT" \
  --zone "$GCP_ZONE"

remote_command="
set -euo pipefail
rm -rf '$(quote_for_remote "$REMOTE_AIC_ROOT")'
mkdir -p '$(quote_for_remote "$REMOTE_AIC_ROOT")'
tar -xzf '$(quote_for_remote "$REMOTE_TARBALL")' -C '$(quote_for_remote "$REMOTE_AIC_ROOT")'
cd '$(quote_for_remote "$REMOTE_AIC_ROOT")'
if [[ '$(quote_for_remote "${AIC_RECORDING_SKIP_BUILD:-0}")' == '1' ]]; then
  sudo docker image inspect '$(quote_for_remote "$AIC_RECORDING_IMAGE")' >/dev/null
else
  sudo DOCKER_BUILDKIT=1 docker build -f docker/aic_lewm_policy/Dockerfile -t '$(quote_for_remote "$AIC_RECORDING_IMAGE")' .
fi
chmod +x ~/vm_collect_recording.sh
export AIC_RECORDING_RUN_ID='$(quote_for_remote "$AIC_RECORDING_RUN_ID")'
export AIC_RECORDING_IMAGE='$(quote_for_remote "$AIC_RECORDING_IMAGE")'
export AIC_RECORDING_DATASET_NAME='$(quote_for_remote "$AIC_DATASET_NAME")'
export AIC_EVAL_IMAGE='$(quote_for_remote "${AIC_EVAL_IMAGE:-ghcr.io/intrinsic-dev/aic/aic_eval:latest}")'
export AIC_EVAL_USE_LOCAL_LAUNCH='$(quote_for_remote "${AIC_EVAL_USE_LOCAL_LAUNCH:-1}")'
export AIC_GZ_VERBOSITY_LEVEL='$(quote_for_remote "${AIC_GZ_VERBOSITY_LEVEL:-1}")'
export AIC_DOCKER_GPUS='$(quote_for_remote "${AIC_DOCKER_GPUS:-all}")'
export AIC_RECORDING_TIMEOUT_SEC='$(quote_for_remote "${AIC_RECORDING_TIMEOUT_SEC:-1800}")'
export AIC_LEWM_RECORD_HZ='$(quote_for_remote "${AIC_LEWM_RECORD_HZ:-20}")'
export AIC_LEWM_RECORD_FLUSH_EVERY='$(quote_for_remote "${AIC_LEWM_RECORD_FLUSH_EVERY:-10}")'
export AIC_LEWM_RECORD_IMAGE_SCALE='$(quote_for_remote "${AIC_LEWM_RECORD_IMAGE_SCALE:-0.25}")'
export AIC_LEWM_CHEATCODE_APPROACH_STEPS='$(quote_for_remote "${AIC_LEWM_CHEATCODE_APPROACH_STEPS:-40}")'
export AIC_LEWM_CHEATCODE_APPROACH_SLEEP='$(quote_for_remote "${AIC_LEWM_CHEATCODE_APPROACH_SLEEP:-0.02}")'
export AIC_LEWM_CHEATCODE_INSERT_Z_STEP='$(quote_for_remote "${AIC_LEWM_CHEATCODE_INSERT_Z_STEP:-0.003}")'
export AIC_LEWM_CHEATCODE_INSERT_SLEEP='$(quote_for_remote "${AIC_LEWM_CHEATCODE_INSERT_SLEEP:-0.02}")'
export AIC_LEWM_CHEATCODE_STABILIZE_SLEEP='$(quote_for_remote "${AIC_LEWM_CHEATCODE_STABILIZE_SLEEP:-1.0}")'
~/vm_collect_recording.sh
"

gcloud compute ssh "$GCP_INSTANCE" \
  --project "$GCP_PROJECT" \
  --zone "$GCP_ZONE" \
  --command "$remote_command"
