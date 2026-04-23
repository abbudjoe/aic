#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

quiet=false
if [[ "${1:-}" == "--quiet" ]]; then
  quiet=true
fi

require_gcp_project

global_gpu_limit="$(
  gcloud compute project-info describe \
    --project "$GCP_PROJECT" \
    --flatten='quotas' \
    --format='table(quotas.metric,quotas.limit,quotas.usage)' |
    awk '$1 == "GPUS_ALL_REGIONS" { print $2 }'
)"

regional_l4_limit="$(
  gcloud compute regions describe "$GCP_REGION" \
    --project "$GCP_PROJECT" \
    --flatten='quotas' \
    --format='table(quotas.metric,quotas.limit,quotas.usage)' |
    awk '$1 == "NVIDIA_L4_GPUS" { print $2 }'
)"

global_gpu_limit="${global_gpu_limit:-0}"
regional_l4_limit="${regional_l4_limit:-0}"

if [[ "$quiet" == false ]]; then
  echo "Project: $GCP_PROJECT"
  echo "Region:  $GCP_REGION"
  echo "Global GPUS_ALL_REGIONS quota: $global_gpu_limit"
  echo "Regional NVIDIA_L4_GPUS quota: $regional_l4_limit"
fi

if ! awk -v value="$global_gpu_limit" 'BEGIN { exit(value >= 1 ? 0 : 1) }'; then
  cat >&2 <<EOF
Global GPU quota is blocking L4 VM creation.

Request this quota increase before launching:
  service: compute.googleapis.com
  quota:   GPUS_ALL_REGIONS
  limit:   1
  project: $GCP_PROJECT

Console:
  https://console.cloud.google.com/iam-admin/quotas?project=$GCP_PROJECT
EOF
  exit 2
fi

if ! awk -v value="$regional_l4_limit" 'BEGIN { exit(value >= 1 ? 0 : 1) }'; then
  cat >&2 <<EOF
Regional L4 quota is blocking L4 VM creation.

Request this quota increase before launching:
  service: compute.googleapis.com
  quota:   NVIDIA_L4_GPUS
  region:  $GCP_REGION
  limit:   1
  project: $GCP_PROJECT
EOF
  exit 2
fi

if [[ "$quiet" == false ]]; then
  echo "Quota check passed for one L4 VM."
fi
