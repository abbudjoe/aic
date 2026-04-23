#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

require_instance

gcloud compute scp "$SCRIPT_DIR/vm_bootstrap.sh" "$GCP_INSTANCE:~/vm_bootstrap.sh" \
  --project "$GCP_PROJECT" \
  --zone "$GCP_ZONE"

gcloud compute ssh "$GCP_INSTANCE" \
  --project "$GCP_PROJECT" \
  --zone "$GCP_ZONE" \
  --command "chmod +x ~/vm_bootstrap.sh && ~/vm_bootstrap.sh"

echo "Bootstrapped $GCP_INSTANCE."
