#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${1:-$ROOT/deploy/gcp.env}"
STARTUP_SCRIPT="$ROOT/deploy/gcp/startup.sh"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "missing env file: $ENV_FILE" >&2
  echo "copy deploy/gcp.env.example to deploy/gcp.env and fill it in" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "$ENV_FILE"

require_var() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "missing required env var: $name" >&2
    exit 1
  fi
}

require_var GCP_PROJECT_ID
require_var GCP_REGION
require_var GCP_ZONE
require_var GCP_INSTANCE_NAME
require_var GCP_MACHINE_TYPE
require_var GCP_ACCELERATOR_TYPE
require_var GCP_ACCELERATOR_COUNT
require_var GCP_BOOT_DISK_SIZE_GB
require_var GCP_RUNTIME_DISK_NAME
require_var GCP_RUNTIME_DISK_SIZE_GB
require_var GCP_RUNTIME_MOUNT_PATH
require_var GCP_SOURCE_IMAGE_PROJECT
require_var GCP_SOURCE_IMAGE_FAMILY
require_var GCP_NETWORK
require_var GCP_TAGS
require_var CONTAINER_IMAGE
require_var CONTAINER_NAME
require_var CONTAINER_MODE
require_var MANAGER_PORT
require_var WORKSPACE_MOUNT_PATH
require_var CHARACTER
require_var START_PAUSED

if [[ ! -f "$STARTUP_SCRIPT" ]]; then
  echo "missing startup script: $STARTUP_SCRIPT" >&2
  exit 1
fi

firewall_ports="tcp:22"
if [[ -n "${CADDY_DOMAIN:-}" ]]; then
  firewall_ports="$firewall_ports,tcp:80,tcp:443"
else
  firewall_ports="$firewall_ports,tcp:${MANAGER_PORT}"
fi

firewall_rule_name="${GCP_INSTANCE_NAME}-allow"

tmp_dir="$(mktemp -d)"
cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

app_env_file="$tmp_dir/character-eng.env"
cat >"$app_env_file" <<EOF
MODE=${CONTAINER_MODE}
MANAGER_PORT=${MANAGER_PORT}
CHARACTER=${CHARACTER}
START_PAUSED=${START_PAUSED}
HF_TOKEN=${HF_TOKEN:-}
CHARACTER_ENG_MANAGER_TOKEN=${CHARACTER_ENG_MANAGER_TOKEN:-}
EOF

metadata_pairs=(
  "character-eng-container-image=${CONTAINER_IMAGE}"
  "character-eng-container-name=${CONTAINER_NAME}"
  "character-eng-manager-port=${MANAGER_PORT}"
  "character-eng-runtime-disk=${GCP_RUNTIME_DISK_NAME}"
  "character-eng-runtime-mount=${GCP_RUNTIME_MOUNT_PATH}"
  "character-eng-workspace-mount=${WORKSPACE_MOUNT_PATH}"
)

if [[ -n "${PUBLIC_HOST:-}" ]]; then
  metadata_pairs+=("character-eng-public-host=${PUBLIC_HOST}")
fi
if [[ -n "${CADDY_DOMAIN:-}" ]]; then
  metadata_pairs+=("character-eng-caddy-domain=${CADDY_DOMAIN}")
fi
if [[ -n "${CADDY_EMAIL:-}" ]]; then
  metadata_pairs+=("character-eng-caddy-email=${CADDY_EMAIL}")
fi

metadata_file_pairs=("startup-script=${STARTUP_SCRIPT}" "character-eng-env=${app_env_file}")
if [[ -n "${REGISTRY_SERVER:-}" ]]; then
  printf '%s' "$REGISTRY_SERVER" >"$tmp_dir/registry-server"
  metadata_file_pairs+=("character-eng-registry-server=$tmp_dir/registry-server")
fi
if [[ -n "${REGISTRY_USERNAME:-}" ]]; then
  printf '%s' "$REGISTRY_USERNAME" >"$tmp_dir/registry-username"
  metadata_file_pairs+=("character-eng-registry-username=$tmp_dir/registry-username")
fi
if [[ -n "${REGISTRY_PASSWORD:-}" ]]; then
  printf '%s' "$REGISTRY_PASSWORD" >"$tmp_dir/registry-password"
  metadata_file_pairs+=("character-eng-registry-password=$tmp_dir/registry-password")
fi

join_by_comma() {
  local IFS=,
  printf '%s' "$*"
}

run() {
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '+'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

disk_exists=0
if gcloud compute disks describe "$GCP_RUNTIME_DISK_NAME" --project "$GCP_PROJECT_ID" --zone "$GCP_ZONE" >/dev/null 2>&1; then
  disk_exists=1
fi
if [[ "$disk_exists" == "0" ]]; then
  run gcloud compute disks create "$GCP_RUNTIME_DISK_NAME" \
    --project "$GCP_PROJECT_ID" \
    --zone "$GCP_ZONE" \
    --type pd-balanced \
    --size "${GCP_RUNTIME_DISK_SIZE_GB}GB"
fi

if [[ -n "${GCP_ADDRESS_NAME:-}" ]]; then
  if ! gcloud compute addresses describe "$GCP_ADDRESS_NAME" --project "$GCP_PROJECT_ID" --region "$GCP_REGION" >/dev/null 2>&1; then
    run gcloud compute addresses create "$GCP_ADDRESS_NAME" \
      --project "$GCP_PROJECT_ID" \
      --region "$GCP_REGION"
  fi
fi

if ! gcloud compute firewall-rules describe "$firewall_rule_name" --project "$GCP_PROJECT_ID" >/dev/null 2>&1; then
  run gcloud compute firewall-rules create "$firewall_rule_name" \
    --project "$GCP_PROJECT_ID" \
    --network "$GCP_NETWORK" \
    --allow "$firewall_ports" \
    --target-tags "$GCP_TAGS"
fi

instance_args=(
  gcloud compute instances create "$GCP_INSTANCE_NAME"
  --project "$GCP_PROJECT_ID"
  --zone "$GCP_ZONE"
  --machine-type "$GCP_MACHINE_TYPE"
  --boot-disk-size "${GCP_BOOT_DISK_SIZE_GB}GB"
  --image-project "$GCP_SOURCE_IMAGE_PROJECT"
  --image-family "$GCP_SOURCE_IMAGE_FAMILY"
  --network "$GCP_NETWORK"
  --tags "$GCP_TAGS"
  --maintenance-policy TERMINATE
  --restart-on-failure
  --scopes cloud-platform
  --accelerator "type=${GCP_ACCELERATOR_TYPE},count=${GCP_ACCELERATOR_COUNT}"
  --disk "name=${GCP_RUNTIME_DISK_NAME},device-name=${GCP_RUNTIME_DISK_NAME},mode=rw,boot=no,auto-delete=no"
)

if [[ -n "${GCP_ADDRESS_NAME:-}" ]]; then
  instance_args+=(--address "$GCP_ADDRESS_NAME")
fi

instance_args+=(--metadata "$(join_by_comma "${metadata_pairs[@]}")")
instance_args+=(--metadata-from-file "$(join_by_comma "${metadata_file_pairs[@]}")")

run "${instance_args[@]}"

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  gcloud compute instances describe "$GCP_INSTANCE_NAME" \
    --project "$GCP_PROJECT_ID" \
    --zone "$GCP_ZONE" \
    --format='value(networkInterfaces[0].accessConfigs[0].natIP)'
fi
