#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
AIC_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ENV_FILE="${AIC_LEWM_GCP_ENV:-$SCRIPT_DIR/.env}"

if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi

GCP_PROJECT="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null || true)}"
GCP_REGION="${GCP_REGION:-us-west4}"
GCP_ZONE="${GCP_ZONE:-us-west4-a}"
GCP_INSTANCE="${GCP_INSTANCE:-aic-lewm-l4}"
GCP_MACHINE_TYPE="${GCP_MACHINE_TYPE:-g2-standard-16}"
GCP_BOOT_DISK_SIZE="${GCP_BOOT_DISK_SIZE:-300GB}"
GCP_BOOT_DISK_TYPE="${GCP_BOOT_DISK_TYPE:-pd-balanced}"
GCP_IMAGE_PROJECT="${GCP_IMAGE_PROJECT:-deeplearning-platform-release}"
GCP_IMAGE_FAMILY="${GCP_IMAGE_FAMILY:-pytorch-2-9-cu129-ubuntu-2204-nvidia-580}"
GCP_PREFIX="${GCP_PREFIX:-aic-lewm}"

LEWM_LOCAL_ROOT="${LEWM_LOCAL_ROOT:-$HOME/le-wm}"
LEWM_LOCAL_ROOT="${LEWM_LOCAL_ROOT%/}"
AIC_DATASET_NAME="${AIC_DATASET_NAME:-aic_qualification_train}"
AIC_DATASET_LOCAL="${AIC_DATASET_LOCAL:-}"
LEWM_RUN_NAME="${LEWM_RUN_NAME:-aic-lewm-baseline}"
LEWM_MAX_EPOCHS="${LEWM_MAX_EPOCHS:-100}"
LEWM_BATCH_SIZE="${LEWM_BATCH_SIZE:-64}"
REMOTE_STABLEWM_HOME="${REMOTE_STABLEWM_HOME:-}"

if [[ -n "${GCP_BUCKET:-}" ]]; then
  GCP_BUCKET="${GCP_BUCKET%/}"
  AIC_DATASET_GCS_URI="${AIC_DATASET_GCS_URI:-$GCP_BUCKET/$GCP_PREFIX/datasets/$AIC_DATASET_NAME.h5}"
  GCS_OUTPUT_PREFIX="${GCS_OUTPUT_PREFIX:-$GCP_BUCKET/$GCP_PREFIX/runs}"
else
  AIC_DATASET_GCS_URI="${AIC_DATASET_GCS_URI:-}"
  GCS_OUTPUT_PREFIX="${GCS_OUTPUT_PREFIX:-}"
fi

gcloud_base_args() {
  printf '%s\n' --project "$GCP_PROJECT" --zone "$GCP_ZONE"
}

require_command() {
  local command_name="$1"
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Missing required command: $command_name" >&2
    exit 2
  fi
}

require_gcp_project() {
  require_command gcloud
  if [[ -z "$GCP_PROJECT" ]]; then
    echo "Set GCP_PROJECT or configure gcloud's core/project." >&2
    exit 2
  fi
}

require_bucket() {
  if [[ -z "${GCP_BUCKET:-}" ]]; then
    echo "Set GCP_BUCKET in $ENV_FILE, for example gs://your-bucket-name." >&2
    exit 2
  fi
}

require_dataset_uri() {
  if [[ -n "${AIC_DATASET_GCS_URI:-}" ]]; then
    return
  fi
  if [[ -n "${GCP_BUCKET:-}" ]]; then
    AIC_DATASET_GCS_URI="$GCP_BUCKET/$GCP_PREFIX/datasets/$AIC_DATASET_NAME.h5"
    return
  fi
  echo "Set AIC_DATASET_GCS_URI or GCP_BUCKET in $ENV_FILE before training." >&2
  exit 2
}

require_instance() {
  require_gcp_project
  if ! gcloud compute instances describe "$GCP_INSTANCE" \
    --project "$GCP_PROJECT" \
    --zone "$GCP_ZONE" >/dev/null 2>&1; then
    echo "Instance $GCP_INSTANCE does not exist in $GCP_ZONE. Run launch_l4_vm.sh first." >&2
    exit 2
  fi
}

quote_for_remote() {
  printf "%s" "$1" | sed "s/'/'\\\\''/g"
}
