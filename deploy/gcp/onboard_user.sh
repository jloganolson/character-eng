#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
resolve_default_env_file() {
  if [[ -n "${CHARACTER_ENG_GCP_ENV:-}" ]]; then
    printf '%s\n' "$CHARACTER_ENG_GCP_ENV"
    return 0
  fi
  if [[ -f "$ROOT/deploy/gcp.env" ]]; then
    printf '%s\n' "$ROOT/deploy/gcp.env"
    return 0
  fi
  printf '%s\n' "$ROOT/deploy/gcp.shared-remote.env"
}

DEFAULT_ENV_FILE="$(resolve_default_env_file)"

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
ROLE_TITLE="${GCP_OPERATOR_ROLE_TITLE:-Character Eng VM Operator}"

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
ROLE_FILE="$(mktemp)"
cleanup() {
  rm -f "$ROLE_FILE"
}
trap cleanup EXIT

cat >"$ROLE_FILE" <<EOF
title: ${ROLE_TITLE}
description: Start/stop/status access for Character Engine Compute Engine VMs
stage: GA
includedPermissions:
- compute.instances.get
- compute.instances.list
- compute.instances.start
- compute.instances.stop
- compute.projects.get
- compute.zoneOperations.get
- compute.zoneOperations.list
EOF

ensure_custom_role() {
  if gcloud iam roles describe "$ROLE_NAME" --project "$GCP_PROJECT_ID" >/dev/null 2>&1; then
    gcloud iam roles update "$ROLE_ID" \
      --project "$GCP_PROJECT_ID" \
      --file "$ROLE_FILE" >/dev/null
  else
    gcloud iam roles create "$ROLE_ID" \
      --project "$GCP_PROJECT_ID" \
      --file "$ROLE_FILE" >/dev/null
  fi
}

service_account_email() {
  gcloud compute instances describe "$GCP_INSTANCE_NAME" \
    --project "$GCP_PROJECT_ID" \
    --zone "$GCP_ZONE" \
    --format='value(serviceAccounts[0].email)'
}

ensure_custom_role

gcloud projects add-iam-policy-binding "$GCP_PROJECT_ID" \
  --member "$MEMBER" \
  --role "$ROLE_NAME" >/dev/null

gcloud projects add-iam-policy-binding "$GCP_PROJECT_ID" \
  --member "$MEMBER" \
  --role "roles/compute.osLogin" >/dev/null

SA_EMAIL="$(service_account_email)"
if [[ -n "$SA_EMAIL" ]]; then
  gcloud iam service-accounts add-iam-policy-binding "$SA_EMAIL" \
    --project "$GCP_PROJECT_ID" \
    --member "$MEMBER" \
    --role "roles/iam.serviceAccountUser" >/dev/null
fi

cat <<EOF
Granted access for ${MEMBER}

Project: ${GCP_PROJECT_ID}
Instance: ${GCP_INSTANCE_NAME}
Zone: ${GCP_ZONE}
Operator role: ${ROLE_NAME}
OS Login role: roles/compute.osLogin
Service account user: ${SA_EMAIL:-<none>}

Next steps for the user:
  1. gcloud auth login
  2. gcloud config set project ${GCP_PROJECT_ID}
  3. ./deploy/gcp/doctor.sh
  4. ./scripts/run_hot_remote_webrtc.sh
EOF
