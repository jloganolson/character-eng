#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '[character-eng-gcp] %s\n' "$*"
}

metadata() {
  curl -fsS -H 'Metadata-Flavor: Google' \
    "http://metadata.google.internal/computeMetadata/v1/$1"
}

metadata_attr() {
  metadata "instance/attributes/$1"
}

metadata_attr_optional() {
  metadata_attr "$1" 2>/dev/null || true
}

service_account_token() {
  metadata 'instance/service-accounts/default/token' | python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])'
}

install_packages() {
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y ca-certificates curl gnupg jq docker.io
  systemctl enable --now docker
}

configure_nvidia_runtime() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    log "nvidia-smi not found; install NVIDIA drivers or use a GPU-ready base image"
    return 1
  fi
  if command -v nvidia-ctk >/dev/null 2>&1; then
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker
    return 0
  fi
  . /etc/os-release
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL "https://nvidia.github.io/libnvidia-container/${ID}${VERSION_ID}/libnvidia-container.list" \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    >/etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update
  apt-get install -y nvidia-container-toolkit
  nvidia-ctk runtime configure --runtime=docker
  systemctl restart docker
}

ensure_mount() {
  local disk_name="$1"
  local mount_path="$2"
  local device="/dev/disk/by-id/google-${disk_name}"

  mkdir -p "$mount_path"
  if [[ ! -b "$device" ]]; then
    log "runtime disk device missing: $device"
    return 1
  fi
  if ! blkid "$device" >/dev/null 2>&1; then
    mkfs.ext4 -F "$device"
  fi
  if ! grep -q " $mount_path " /etc/fstab; then
    local uuid
    uuid="$(blkid -s UUID -o value "$device")"
    echo "UUID=$uuid $mount_path ext4 defaults,nofail 0 2" >> /etc/fstab
  fi
  mount -a
}

configure_caddy() {
  local domain="$1"
  local email="$2"
  local caddy_root="${RUNTIME_MOUNT:-/var/lib/character-eng-caddy}/caddy"

  mkdir -p "${caddy_root}/data" "${caddy_root}/config"
  cat >"${caddy_root}/Caddyfile" <<EOF
{
  email ${email}
}

${domain} {
  reverse_proxy 127.0.0.1:${MANAGER_PORT}
}
EOF
  docker pull caddy:2
  docker rm -f character-eng-caddy >/dev/null 2>&1 || true
  docker run -d \
    --name character-eng-caddy \
    --restart unless-stopped \
    --network host \
    -v "${caddy_root}/Caddyfile:/etc/caddy/Caddyfile" \
    -v "${caddy_root}/data:/data" \
    -v "${caddy_root}/config:/config" \
    caddy:2 >/dev/null
}

CONTAINER_IMAGE="$(metadata_attr 'character-eng-container-image')"
CONTAINER_NAME="$(metadata_attr_optional 'character-eng-container-name')"
CONTAINER_NAME="${CONTAINER_NAME:-character-eng}"
MANAGER_PORT="$(metadata_attr_optional 'character-eng-manager-port')"
MANAGER_PORT="${MANAGER_PORT:-7870}"
RUNTIME_DISK="$(metadata_attr_optional 'character-eng-runtime-disk')"
RUNTIME_MOUNT="$(metadata_attr_optional 'character-eng-runtime-mount')"
WORKSPACE_MOUNT_PATH="$(metadata_attr_optional 'character-eng-workspace-mount')"
WORKSPACE_MOUNT_PATH="${WORKSPACE_MOUNT_PATH:-/workspace}"
PUBLIC_HOST="$(metadata_attr_optional 'character-eng-public-host')"
CADDY_DOMAIN="$(metadata_attr_optional 'character-eng-caddy-domain')"
CADDY_EMAIL="$(metadata_attr_optional 'character-eng-caddy-email')"
REGISTRY_SERVER="$(metadata_attr_optional 'character-eng-registry-server')"
REGISTRY_USERNAME="$(metadata_attr_optional 'character-eng-registry-username')"
REGISTRY_PASSWORD="$(metadata_attr_optional 'character-eng-registry-password')"
APP_ENV_FILE="/etc/character-eng.env"
REGISTRY_HOST="${CONTAINER_IMAGE%%/*}"

metadata_attr_optional 'character-eng-env' >"$APP_ENV_FILE"
chmod 600 "$APP_ENV_FILE"

if [[ -n "$RUNTIME_DISK" && -n "$RUNTIME_MOUNT" ]]; then
  ensure_mount "$RUNTIME_DISK" "$RUNTIME_MOUNT"
fi

if [[ -z "$PUBLIC_HOST" && -n "$CADDY_DOMAIN" ]]; then
  PUBLIC_HOST="$CADDY_DOMAIN"
fi
if [[ -z "$PUBLIC_HOST" ]]; then
  PUBLIC_HOST="$(metadata 'instance/network-interfaces/0/access-configs/0/external-ip')"
fi
echo "PUBLIC_HOST=$PUBLIC_HOST" >>"$APP_ENV_FILE"

install_packages
configure_nvidia_runtime

if [[ -n "$CADDY_DOMAIN" ]]; then
  configure_caddy "$CADDY_DOMAIN" "$CADDY_EMAIL"
  publish_args=(-p "127.0.0.1:${MANAGER_PORT}:${MANAGER_PORT}")
else
  publish_args=(-p "${MANAGER_PORT}:${MANAGER_PORT}")
fi

if [[ -n "$REGISTRY_SERVER" && -n "$REGISTRY_USERNAME" && -n "$REGISTRY_PASSWORD" ]]; then
  printf '%s' "$REGISTRY_PASSWORD" | docker login "$REGISTRY_SERVER" --username "$REGISTRY_USERNAME" --password-stdin
elif [[ "$REGISTRY_HOST" == *".pkg.dev" ]]; then
  service_account_token | docker login -u oauth2accesstoken --password-stdin "https://${REGISTRY_HOST}"
fi

docker pull "$CONTAINER_IMAGE"
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

docker run -d \
  --name "$CONTAINER_NAME" \
  --restart unless-stopped \
  --gpus all \
  --env-file "$APP_ENV_FILE" \
  -v "${RUNTIME_MOUNT}:${WORKSPACE_MOUNT_PATH}" \
  "${publish_args[@]}" \
  "$CONTAINER_IMAGE"

log "container started: $CONTAINER_NAME"
if [[ -n "$CADDY_DOMAIN" ]]; then
  log "waiting for TLS at https://${CADDY_DOMAIN}"
else
  log "manager available on http://${PUBLIC_HOST}:${MANAGER_PORT}"
  log "browser mic/camera will require HTTPS; use Caddy + a real hostname for full remote UX"
fi
