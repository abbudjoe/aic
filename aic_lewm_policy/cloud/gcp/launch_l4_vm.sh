#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

require_gcp_project

if gcloud compute instances describe "$GCP_INSTANCE" \
  --project "$GCP_PROJECT" \
  --zone "$GCP_ZONE" >/dev/null 2>&1; then
  echo "Instance already exists: $GCP_INSTANCE ($GCP_ZONE)"
  exit 0
fi

"$SCRIPT_DIR/check_quota.sh" --quiet

gcloud compute instances create "$GCP_INSTANCE" \
  --project "$GCP_PROJECT" \
  --zone "$GCP_ZONE" \
  --machine-type "$GCP_MACHINE_TYPE" \
  --accelerator type=nvidia-l4,count=1 \
  --maintenance-policy TERMINATE \
  --provisioning-model STANDARD \
  --image-project "$GCP_IMAGE_PROJECT" \
  --image-family "$GCP_IMAGE_FAMILY" \
  --boot-disk-size "$GCP_BOOT_DISK_SIZE" \
  --boot-disk-type "$GCP_BOOT_DISK_TYPE" \
  --boot-disk-auto-delete \
  --scopes cloud-platform \
  --labels purpose=aic-lewm

echo "Created $GCP_INSTANCE in $GCP_ZONE."
echo "Next: $SCRIPT_DIR/bootstrap_vm.sh"
