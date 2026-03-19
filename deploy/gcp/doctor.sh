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

CURRENT_ACCOUNT="$(gcloud config get-value account 2>/dev/null || true)"
CURRENT_PROJECT="$(gcloud config get-value project 2>/dev/null || true)"

echo "account: ${CURRENT_ACCOUNT:-<unset>}"
echo "project: ${CURRENT_PROJECT:-<unset>}"
echo "target project: ${GCP_PROJECT_ID}"
echo "target instance: ${GCP_INSTANCE_NAME}"
echo "target zone: ${GCP_ZONE}"

if [[ "$CURRENT_PROJECT" != "$GCP_PROJECT_ID" ]]; then
  echo "warning: gcloud project is not set to ${GCP_PROJECT_ID}" >&2
fi

if ! gcloud compute instances describe "$GCP_INSTANCE_NAME" \
  --project "$GCP_PROJECT_ID" \
  --zone "$GCP_ZONE" \
  --format='get(status)' >/tmp/character_eng_vm_status.$$ 2>/tmp/character_eng_vm_status_err.$$; then
  cat /tmp/character_eng_vm_status_err.$$ >&2
  rm -f /tmp/character_eng_vm_status.$$ /tmp/character_eng_vm_status_err.$$
  exit 1
fi

STATUS="$(cat /tmp/character_eng_vm_status.$$)"
rm -f /tmp/character_eng_vm_status.$$ /tmp/character_eng_vm_status_err.$$

MANAGER_URL="$("$ROOT/deploy/gcp/vm_ctl.sh" "$ENV_FILE" url)"
LIVEKIT_URL="$("$ROOT/deploy/gcp/vm_ctl.sh" "$ENV_FILE" livekit-url)"
OS_LOGIN_ENABLED="$(
  gcloud compute instances describe "$GCP_INSTANCE_NAME" \
    --project "$GCP_PROJECT_ID" \
    --zone "$GCP_ZONE" \
    --format='json(metadata.items)' | python3 -c '
import json,sys
items = json.load(sys.stdin).get("metadata", {}).get("items", [])
value = ""
for item in items:
    if item.get("key") == "enable-oslogin":
        value = item.get("value", "")
        break
print(value)
'
)"

echo "vm status: ${STATUS}"
echo "os login: ${OS_LOGIN_ENABLED:-<unset>}"
echo "manager url: ${MANAGER_URL}"
echo "livekit url: ${LIVEKIT_URL}"

if [[ "$STATUS" == "RUNNING" ]]; then
  if gcloud compute ssh "$GCP_INSTANCE_NAME" \
    --project "$GCP_PROJECT_ID" \
    --zone "$GCP_ZONE" \
    --command 'echo ssh-ok' >/tmp/character_eng_doctor_ssh.$$ 2>/tmp/character_eng_doctor_ssh_err.$$; then
    printf 'ssh: '
    cat /tmp/character_eng_doctor_ssh.$$
  else
    echo "ssh: failed" >&2
    cat /tmp/character_eng_doctor_ssh_err.$$ >&2
    rm -f /tmp/character_eng_doctor_ssh.$$ /tmp/character_eng_doctor_ssh_err.$$
    exit 1
  fi
  rm -f /tmp/character_eng_doctor_ssh.$$ /tmp/character_eng_doctor_ssh_err.$$
else
  echo "ssh: skipped (vm not running)"
fi

echo "next:"
echo "  ./scripts/run_hot_remote_webrtc.sh"
