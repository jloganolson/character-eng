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

env_file_value() {
  local env_file="$1"
  local key="$2"
  python3 - "$env_file" "$key" <<'PY'
import sys
from pathlib import Path

env_file = Path(sys.argv[1])
target = sys.argv[2]
value = ""

if env_file.exists():
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in raw_line:
            continue
        key, current = raw_line.split("=", 1)
        if key == target:
            value = current

print(value)
PY
}

append_env_line() {
  local env_file="$1"
  local key="$2"
  local value="$3"
  printf '%s=%s\n' "$key" "$value" >>"$env_file"
}

resolve_secret_ref() {
  local ref="$1"
  local access_token="${SECRET_MANAGER_ACCESS_TOKEN:-}"
  local url

  if [[ -z "$access_token" ]]; then
    access_token="$(service_account_token)"
    SECRET_MANAGER_ACCESS_TOKEN="$access_token"
  fi

  if [[ "$ref" == projects/*/secrets/*/versions/* ]]; then
    url="https://secretmanager.googleapis.com/v1/${ref}:access"
  else
    local secret_name="$ref"
    local version="latest"
    if [[ "$ref" == *:* ]]; then
      secret_name="${ref%%:*}"
      version="${ref##*:}"
    fi
    url="https://secretmanager.googleapis.com/v1/projects/${PROJECT_ID}/secrets/${secret_name}/versions/${version}:access"
  fi

  curl -fsS \
    -H "Authorization: Bearer ${access_token}" \
    "$url" \
    | python3 -c 'import base64,json,sys; print(base64.b64decode(json.load(sys.stdin)["payload"]["data"]).decode("utf-8"))'
}

build_app_env_file() {
  local raw_env_file="$1"
  local output_env_file="$2"

  : >"$output_env_file"
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" || "$line" == \#* ]] && continue
    [[ "$line" == *=* ]] || continue
    local key="${line%%=*}"
    local value="${line#*=}"
    if [[ "$key" == GCP_SECRET_* ]]; then
      local target_key="${key#GCP_SECRET_}"
      local resolved_value
      resolved_value="$(resolve_secret_ref "$value")"
      append_env_line "$output_env_file" "$target_key" "$resolved_value"
    else
      append_env_line "$output_env_file" "$key" "$value"
    fi
  done <"$raw_env_file"
  chmod 600 "$output_env_file"
}

write_remote_hot_env_file() {
  local source_env_file="$1"
  local output_env_file="$2"
  shift 2
  local env_keys=("$@")

  : >"$output_env_file"
  for key in "${env_keys[@]}"; do
    local value
    value="$(env_file_value "$source_env_file" "$key")"
    if [[ -n "$value" ]]; then
      append_env_line "$output_env_file" "$key" "$value"
    fi
  done
  chmod 644 "$output_env_file"
}

disable_auto_shutdown() {
  rm -f /usr/local/bin/character-eng-auto-shutdown-check.sh
  rm -f /etc/systemd/system/character-eng-auto-shutdown-check.service
  rm -f /etc/systemd/system/character-eng-auto-shutdown-check.timer
  systemctl disable --now character-eng-auto-shutdown-check.timer >/dev/null 2>&1 || true
  systemctl daemon-reload
}

configure_auto_shutdown() {
  local enabled="$1"
  local schedule_time="$2"
  local schedule_timezone="$3"
  local script_path="/usr/local/bin/character-eng-auto-shutdown-check.sh"
  local service_path="/etc/systemd/system/character-eng-auto-shutdown-check.service"
  local timer_path="/etc/systemd/system/character-eng-auto-shutdown-check.timer"
  local normalized_enabled="${enabled,,}"

  if [[ "$normalized_enabled" != "1" && "$normalized_enabled" != "true" && "$normalized_enabled" != "yes" ]]; then
    disable_auto_shutdown
    log "nightly auto-shutdown disabled"
    return 0
  fi

  if [[ ! "$schedule_time" =~ ^[0-9]{2}:[0-9]{2}$ ]]; then
    log "invalid AUTO_SHUTDOWN_TIME '${schedule_time}'; expected HH:MM"
    exit 1
  fi

  cat >"$script_path" <<EOF
#!/usr/bin/env bash
set -euo pipefail

target_time="${schedule_time}"
target_timezone="${schedule_timezone}"
current_time="\$(TZ="\$target_timezone" date +%H:%M)"

if [[ "\$current_time" != "\$target_time" ]]; then
  exit 0
fi

printf '[character-eng-gcp] nightly auto-shutdown at %s %s\n' "\$target_time" "\$target_timezone" | systemd-cat -t character-eng-auto-shutdown
/usr/bin/systemctl poweroff
EOF
  chmod 755 "$script_path"

  cat >"$service_path" <<EOF
[Unit]
Description=Character Eng auto-shutdown check

[Service]
Type=oneshot
ExecStart=${script_path}
EOF

  cat >"$timer_path" <<EOF
[Unit]
Description=Character Eng auto-shutdown timer

[Timer]
OnBootSec=30s
OnUnitActiveSec=60s
AccuracySec=1s
Persistent=false
Unit=character-eng-auto-shutdown-check.service

[Install]
WantedBy=timers.target
EOF

  chmod 644 "$service_path" "$timer_path"
  systemctl daemon-reload
  systemctl enable --now character-eng-auto-shutdown-check.timer >/dev/null
  log "nightly auto-shutdown scheduled for ${schedule_time} ${schedule_timezone}"
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
  local manager_domain="$1"
  local livekit_domain="$2"
  local email="$3"
  local caddy_root="${RUNTIME_MOUNT:-/var/lib/character-eng-caddy}/caddy"

  mkdir -p "${caddy_root}/data" "${caddy_root}/config"
  cat >"${caddy_root}/Caddyfile" <<EOF
{
  email ${email}
}

EOF
  if [[ -n "$manager_domain" ]]; then
    cat >>"${caddy_root}/Caddyfile" <<EOF
${manager_domain} {
  reverse_proxy 127.0.0.1:${MANAGER_PORT}
}
EOF
  fi
  if [[ -n "$livekit_domain" ]]; then
    cat >>"${caddy_root}/Caddyfile" <<EOF
${livekit_domain} {
  reverse_proxy 127.0.0.1:${LIVEKIT_HTTP_PORT}
}
EOF
  fi
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

start_livekit() {
  local livekit_root="${RUNTIME_MOUNT:-/var/lib/character-eng-livekit}/livekit"
  local public_ip="$1"

  mkdir -p "${livekit_root}/config"
  cat >"${livekit_root}/config/livekit.yaml" <<EOF
port: ${LIVEKIT_HTTP_PORT}
rtc:
  tcp_port: ${LIVEKIT_TCP_PORT}
  port_range_start: ${LIVEKIT_UDP_START}
  port_range_end: ${LIVEKIT_UDP_END}
  use_external_ip: true
keys:
  ${LIVEKIT_API_KEY}: ${LIVEKIT_API_SECRET}
EOF

  docker pull "${LIVEKIT_IMAGE}"
  docker rm -f character-eng-livekit >/dev/null 2>&1 || true
  docker run -d \
    --name character-eng-livekit \
    --restart unless-stopped \
    --network host \
    -v "${livekit_root}/config/livekit.yaml:/etc/livekit.yaml" \
    "${LIVEKIT_IMAGE}" \
    --config /etc/livekit.yaml \
    --bind 0.0.0.0 \
    --node-ip "${public_ip}" >/dev/null
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
CADDY_LIVEKIT_DOMAIN="$(metadata_attr_optional 'character-eng-caddy-livekit-domain')"
CADDY_EMAIL="$(metadata_attr_optional 'character-eng-caddy-email')"
REGISTRY_SERVER="$(metadata_attr_optional 'character-eng-registry-server')"
REGISTRY_USERNAME="$(metadata_attr_optional 'character-eng-registry-username')"
REGISTRY_PASSWORD="$(metadata_attr_optional 'character-eng-registry-password')"
LIVEKIT_ENABLED="$(metadata_attr_optional 'character-eng-livekit-enabled')"
LIVEKIT_IMAGE="$(metadata_attr_optional 'character-eng-livekit-image')"
LIVEKIT_HTTP_PORT="$(metadata_attr_optional 'character-eng-livekit-http-port')"
LIVEKIT_TCP_PORT="$(metadata_attr_optional 'character-eng-livekit-tcp-port')"
LIVEKIT_UDP_START="$(metadata_attr_optional 'character-eng-livekit-udp-start')"
LIVEKIT_UDP_END="$(metadata_attr_optional 'character-eng-livekit-udp-end')"
APP_ENV_FILE="/etc/character-eng.env"
REMOTE_HOT_ENV_FILE="/etc/character-eng.remote-hot.env"
REGISTRY_HOST="${CONTAINER_IMAGE%%/*}"
PROJECT_ID="$(metadata 'project/project-id')"

raw_env_file="$(mktemp)"
metadata_attr_optional 'character-eng-env' >"$raw_env_file"
chmod 600 "$raw_env_file"
build_app_env_file "$raw_env_file" "$APP_ENV_FILE"
rm -f "$raw_env_file"

LIVEKIT_ENABLED="${LIVEKIT_ENABLED,,}"
LIVEKIT_IMAGE="${LIVEKIT_IMAGE:-livekit/livekit-server:latest}"
LIVEKIT_HTTP_PORT="${LIVEKIT_HTTP_PORT:-7880}"
LIVEKIT_TCP_PORT="${LIVEKIT_TCP_PORT:-7881}"
LIVEKIT_UDP_START="${LIVEKIT_UDP_START:-50100}"
LIVEKIT_UDP_END="${LIVEKIT_UDP_END:-50120}"
AUTO_SHUTDOWN_ENABLED="$(env_file_value "$APP_ENV_FILE" 'AUTO_SHUTDOWN_ENABLED')"
AUTO_SHUTDOWN_TIME="$(env_file_value "$APP_ENV_FILE" 'AUTO_SHUTDOWN_TIME')"
AUTO_SHUTDOWN_TIMEZONE="$(env_file_value "$APP_ENV_FILE" 'AUTO_SHUTDOWN_TIMEZONE')"
LIVEKIT_API_KEY="$(env_file_value "$APP_ENV_FILE" 'LIVEKIT_API_KEY')"
LIVEKIT_API_SECRET="$(env_file_value "$APP_ENV_FILE" 'LIVEKIT_API_SECRET')"
if [[ -z "$LIVEKIT_API_KEY" ]]; then
  LIVEKIT_API_KEY="$(metadata_attr_optional 'character-eng-livekit-api-key')"
  if [[ -n "$LIVEKIT_API_KEY" ]]; then
    append_env_line "$APP_ENV_FILE" "LIVEKIT_API_KEY" "$LIVEKIT_API_KEY"
  fi
fi
if [[ -z "$LIVEKIT_API_SECRET" ]]; then
  LIVEKIT_API_SECRET="$(metadata_attr_optional 'character-eng-livekit-api-secret')"
  if [[ -n "$LIVEKIT_API_SECRET" ]]; then
    append_env_line "$APP_ENV_FILE" "LIVEKIT_API_SECRET" "$LIVEKIT_API_SECRET"
  fi
fi

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

if [[ "$LIVEKIT_ENABLED" == "1" || "$LIVEKIT_ENABLED" == "true" || "$LIVEKIT_ENABLED" == "yes" ]]; then
  if [[ -z "$LIVEKIT_API_KEY" || -z "$LIVEKIT_API_SECRET" ]]; then
    log "LiveKit is enabled but LIVEKIT_API_KEY / LIVEKIT_API_SECRET were not provided in the VM env"
    exit 1
  fi
  livekit_public_url="ws://${PUBLIC_HOST}:${LIVEKIT_HTTP_PORT}"
  if [[ -n "$CADDY_LIVEKIT_DOMAIN" ]]; then
    livekit_public_url="wss://${CADDY_LIVEKIT_DOMAIN}"
  fi
  cat >>"$APP_ENV_FILE" <<EOF
CHARACTER_ENG_LIVEKIT_ENABLED=true
CHARACTER_ENG_LIVEKIT_URL=${livekit_public_url}
CHARACTER_ENG_LIVEKIT_API_KEY=${LIVEKIT_API_KEY}
CHARACTER_ENG_LIVEKIT_API_SECRET=${LIVEKIT_API_SECRET}
EOF
fi

write_remote_hot_env_file "$APP_ENV_FILE" "$REMOTE_HOT_ENV_FILE" \
  CEREBRAS_API_KEY \
  GROQ_API_KEY \
  GEMINI_API_KEY \
  DEEPGRAM_API_KEY \
  ELEVENLABS_API_KEY \
  HF_TOKEN \
  LIVEKIT_API_KEY \
  LIVEKIT_API_SECRET \
  CHARACTER_ENG_LIVEKIT_API_KEY \
  CHARACTER_ENG_LIVEKIT_API_SECRET

install_packages
configure_nvidia_runtime
configure_auto_shutdown \
  "${AUTO_SHUTDOWN_ENABLED:-0}" \
  "${AUTO_SHUTDOWN_TIME:-02:00}" \
  "${AUTO_SHUTDOWN_TIMEZONE:-America/Los_Angeles}"

if [[ "$LIVEKIT_ENABLED" == "1" || "$LIVEKIT_ENABLED" == "true" || "$LIVEKIT_ENABLED" == "yes" ]]; then
  start_livekit "$PUBLIC_HOST"
fi

if [[ -n "$CADDY_DOMAIN" ]]; then
  configure_caddy "$CADDY_DOMAIN" "$CADDY_LIVEKIT_DOMAIN" "$CADDY_EMAIL"
  publish_args=(-p "127.0.0.1:${MANAGER_PORT}:${MANAGER_PORT}")
else
  if [[ -n "$CADDY_LIVEKIT_DOMAIN" ]]; then
    configure_caddy "$CADDY_DOMAIN" "$CADDY_LIVEKIT_DOMAIN" "$CADDY_EMAIL"
  fi
  publish_args=(-p "${MANAGER_PORT}:${MANAGER_PORT}")
fi
publish_args+=(
  -p "127.0.0.1:7860:7860"
  -p "127.0.0.1:8003:8003"
)

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
if [[ "$LIVEKIT_ENABLED" == "1" || "$LIVEKIT_ENABLED" == "true" || "$LIVEKIT_ENABLED" == "yes" ]]; then
  if [[ -n "$CADDY_LIVEKIT_DOMAIN" ]]; then
    log "livekit signaling available at wss://${CADDY_LIVEKIT_DOMAIN}"
  else
    log "livekit signaling available at ws://${PUBLIC_HOST}:${LIVEKIT_HTTP_PORT}"
  fi
fi
