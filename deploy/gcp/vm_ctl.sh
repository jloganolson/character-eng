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

GCP_PROJECT_ID="${GCP_PROJECT_ID:-}"
GCP_ZONE="${GCP_ZONE:-}"
GCP_INSTANCE_NAME="${GCP_INSTANCE_NAME:-}"
MANAGER_PORT="${MANAGER_PORT:-7870}"
CADDY_DOMAIN="${CADDY_DOMAIN:-}"
PUBLIC_HOST="${PUBLIC_HOST:-}"
CHARACTER_ENG_MANAGER_TOKEN="${CHARACTER_ENG_MANAGER_TOKEN:-}"
WAIT_INTERVAL="${WAIT_INTERVAL:-5}"
READY_TIMEOUT="${READY_TIMEOUT:-600}"

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

usage() {
  cat <<EOF
Usage: $(basename "$0") [env-file] <command>

Commands:
  status         Print instance status
  ip             Print external IP
  start          Start the VM
  stop           Stop the VM
  restart        Restart the VM
  wait           Wait for manager health
  url            Print manager base URL
EOF
}

gcloud_instance() {
  gcloud compute instances "$@" \
    --project "$GCP_PROJECT_ID" \
    --zone "$GCP_ZONE"
}

instance_status() {
  gcloud_instance describe "$GCP_INSTANCE_NAME" --format='get(status)'
}

instance_ip() {
  gcloud_instance describe "$GCP_INSTANCE_NAME" --format='get(networkInterfaces[0].accessConfigs[0].natIP)'
}

metadata_value() {
  local key="$1"
  gcloud_instance describe "$GCP_INSTANCE_NAME" --format='json' | python3 -c '
import json
import sys

target = sys.argv[1]
items = json.load(sys.stdin).get("metadata", {}).get("items", [])
for item in items:
    if item.get("key") == target:
        print(item.get("value", ""))
        break
' "$key"
}

manager_base_url() {
  local caddy_domain="$CADDY_DOMAIN"
  local public_host="$PUBLIC_HOST"
  if [[ -z "$caddy_domain" ]]; then
    caddy_domain="$(metadata_value 'character-eng-caddy-domain')"
  fi
  if [[ -z "$public_host" ]]; then
    public_host="$(metadata_value 'character-eng-public-host')"
  fi
  if [[ -n "$caddy_domain" ]]; then
    printf 'https://%s' "$caddy_domain"
    return 0
  fi
  if [[ -n "$public_host" ]]; then
    printf 'http://%s:%s' "$public_host" "$MANAGER_PORT"
    return 0
  fi
  local ip
  ip="$(instance_ip)"
  printf 'http://%s:%s' "$ip" "$MANAGER_PORT"
}

curl_health() {
  local url
  url="$(manager_base_url)/health"
  if [[ -n "$CHARACTER_ENG_MANAGER_TOKEN" ]]; then
    curl -fsS -H "Authorization: Bearer ${CHARACTER_ENG_MANAGER_TOKEN}" "$url"
  else
    curl -fsS "$url"
  fi
}

wait_for_manager() {
  local elapsed=0
  until curl_health >/dev/null 2>&1; do
    sleep "$WAIT_INTERVAL"
    elapsed=$((elapsed + WAIT_INTERVAL))
    if (( elapsed >= READY_TIMEOUT )); then
      echo "manager did not become healthy within ${READY_TIMEOUT}s" >&2
      return 1
    fi
  done
}

cmd="${1:-}"
if [[ -z "$cmd" ]]; then
  usage
  exit 1
fi

case "$cmd" in
  status)
    instance_status
    ;;
  ip)
    instance_ip
    ;;
  start)
    if [[ "$(instance_status)" != "RUNNING" ]]; then
      gcloud_instance start "$GCP_INSTANCE_NAME"
    fi
    ;;
  stop)
    if [[ "$(instance_status)" != "TERMINATED" ]]; then
      gcloud_instance stop "$GCP_INSTANCE_NAME"
    fi
    ;;
  restart)
    gcloud_instance reset "$GCP_INSTANCE_NAME"
    ;;
  wait)
    wait_for_manager
    ;;
  url)
    manager_base_url
    ;;
  *)
    usage
    exit 1
    ;;
esac
