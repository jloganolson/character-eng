#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEFAULT_ENV_FILE="${CHARACTER_ENG_GCP_ENV:-$ROOT/deploy/gcp.env}"

if [[ $# -ge 2 && -f "${1:-}" ]]; then
  ENV_FILE="$1"
  shift
else
  ENV_FILE="$DEFAULT_ENV_FILE"
fi

if [[ ! -f "$ENV_FILE" ]]; then
  echo "missing env file: $ENV_FILE" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$ENV_FILE"

EMAIL="${1:-}"
MEMBER_KIND="${MEMBER_KIND:-user}"
ROLE_ID="${GCP_OPERATOR_ROLE_ID:-characterEngVmOperator}"

if [[ -z "$EMAIL" ]]; then
  echo "usage: $(basename "$0") [env-file] user@gmail.com" >&2
  exit 1
fi

if [[ "$EMAIL" == *:* ]]; then
  MEMBER="$EMAIL"
else
  MEMBER="${MEMBER_KIND}:${EMAIL}"
fi

require_var() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "missing required env var: $name" >&2
    exit 1
  fi
}

require_var GCP_PROJECT_ID
require_var GCP_ZONE
require_var GCP_INSTANCE_NAME

ROLE_NAME="projects/${GCP_PROJECT_ID}/roles/${ROLE_ID}"

service_account_email() {
  gcloud compute instances describe "$GCP_INSTANCE_NAME" \
    --project "$GCP_PROJECT_ID" \
    --zone "$GCP_ZONE" \
    --format='value(serviceAccounts[0].email)'
}

gcloud projects remove-iam-policy-binding "$GCP_PROJECT_ID" \
  --member "$MEMBER" \
  --role "$ROLE_NAME" >/dev/null 2>&1 || true

gcloud projects remove-iam-policy-binding "$GCP_PROJECT_ID" \
  --member "$MEMBER" \
  --role "roles/compute.osLogin" >/dev/null 2>&1 || true

SA_EMAIL="$(service_account_email)"
if [[ -n "$SA_EMAIL" ]]; then
  gcloud iam service-accounts remove-iam-policy-binding "$SA_EMAIL" \
    --project "$GCP_PROJECT_ID" \
    --member "$MEMBER" \
    --role "roles/iam.serviceAccountUser" >/dev/null 2>&1 || true
fi

cat <<EOF
Removed access for ${MEMBER}

Project: ${GCP_PROJECT_ID}
Instance: ${GCP_INSTANCE_NAME}
Zone: ${GCP_ZONE}
EOF
