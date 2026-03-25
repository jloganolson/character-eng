#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

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

ENV_FILE="$(resolve_default_env_file)"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "missing env file: $ENV_FILE" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$ENV_FILE"

GCP_PROJECT_ID="${GCP_PROJECT_ID:-}"
GCP_ZONE="${GCP_ZONE:-}"
GCP_INSTANCE_NAME="${GCP_INSTANCE_NAME:-}"
LOCAL_VISION_PORT="${LOCAL_VISION_PORT:-17860}"
LOCAL_POCKET_PORT="${LOCAL_POCKET_PORT:-18003}"
LOCAL_TUNNEL_HOST="${LOCAL_TUNNEL_HOST:-127.0.0.1}"
WAIT_INTERVAL="${WAIT_INTERVAL:-2}"
READY_TIMEOUT="${READY_TIMEOUT:-240}"
TUNNEL_DIR="${TUNNEL_DIR:-$ROOT/.cache/gcp-remote-hot-webrtc}"
TUNNEL_PIDFILE="$TUNNEL_DIR/tunnel.pid"
TUNNEL_LOGFILE="$TUNNEL_DIR/tunnel.log"
REMOTE_RUNTIME_ENV_FILE="${REMOTE_RUNTIME_ENV_FILE:-/etc/character-eng.remote-hot.env}"
REMOTE_RUNTIME_ENV_CACHE="$TUNNEL_DIR/remote-runtime.env"
REMOTE_SERVICE_HOST="${REMOTE_SERVICE_HOST:-127.0.0.1}"
LIVEKIT_HTTP_PORT="${LIVEKIT_HTTP_PORT:-7880}"
LIVEKIT_ENABLED="${LIVEKIT_ENABLED:-1}"
LIVEKIT_API_KEY="${LIVEKIT_API_KEY:-devkey}"
LIVEKIT_API_SECRET="${LIVEKIT_API_SECRET:-secret}"
TRANSPORT_VIDEO_METRICS_PATH="$TUNNEL_DIR/video_metrics.json"
TRANSPORT_AUDIO_METRICS_PATH="$TUNNEL_DIR/audio_metrics.json"

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

mkdir -p "$TUNNEL_DIR"

log() {
  printf '==> %s\n' "$*"
}

cleanup_tunnel() {
  if [[ -f "$TUNNEL_PIDFILE" ]]; then
    local pid
    pid="$(cat "$TUNNEL_PIDFILE" 2>/dev/null || true)"
    if [[ -n "$pid" ]]; then
      kill "$pid" >/dev/null 2>&1 || true
      sleep 1
      kill -9 "$pid" >/dev/null 2>&1 || true
    fi
    rm -f "$TUNNEL_PIDFILE"
  fi
  rm -f "$REMOTE_RUNTIME_ENV_CACHE"
}

tunnel_alive() {
  [[ -f "$TUNNEL_PIDFILE" ]] || return 1
  local pid
  pid="$(cat "$TUNNEL_PIDFILE" 2>/dev/null || true)"
  [[ -n "$pid" ]] || return 1
  kill -0 "$pid" 2>/dev/null
}

wait_for_http() {
  local label="$1"
  local url="$2"
  local elapsed=0
  while ! curl -sf "$url" >/dev/null 2>&1; do
    sleep "$WAIT_INTERVAL"
    elapsed=$((elapsed + WAIT_INTERVAL))
    if (( elapsed >= READY_TIMEOUT )); then
      echo "${label} did not become ready within ${READY_TIMEOUT}s" >&2
      return 1
    fi
  done
}

remote_http_ready() {
  local url="$1"
  gcloud compute ssh "$GCP_INSTANCE_NAME" \
    --project "$GCP_PROJECT_ID" \
    --zone "$GCP_ZONE" \
    --command "curl -fsS '$url' >/dev/null" \
    >/dev/null 2>&1
}

wait_for_remote_http() {
  local label="$1"
  local url="$2"
  local elapsed=0
  while ! remote_http_ready "$url"; do
    sleep "$WAIT_INTERVAL"
    elapsed=$((elapsed + WAIT_INTERVAL))
    if (( elapsed >= READY_TIMEOUT )); then
      echo "${label} did not become ready within ${READY_TIMEOUT}s" >&2
      return 1
    fi
  done
}

ensure_vm_running() {
  local status
  status="$(./deploy/gcp/vm_ctl.sh "$ENV_FILE" status)"
  if [[ "$status" != "RUNNING" ]]; then
    log "Starting remote GCP heavy VM"
    ./deploy/gcp/vm_ctl.sh "$ENV_FILE" start
  fi
  log "Waiting for remote manager and LiveKit"
  ./deploy/gcp/vm_ctl.sh "$ENV_FILE" wait >/dev/null
  ./deploy/gcp/vm_ctl.sh "$ENV_FILE" wait-livekit >/dev/null
}

start_tunnel() {
  if tunnel_alive; then
    return 0
  fi
  cleanup_tunnel
  : >"$TUNNEL_LOGFILE"
  log "Opening SSH tunnel for remote vision and Pocket-TTS"
  (
    exec setsid gcloud compute ssh "$GCP_INSTANCE_NAME" \
      --project "$GCP_PROJECT_ID" \
      --zone "$GCP_ZONE" \
      -- -N \
      -L "${LOCAL_TUNNEL_HOST}:${LOCAL_VISION_PORT}:${REMOTE_SERVICE_HOST}:7860" \
      -L "${LOCAL_TUNNEL_HOST}:${LOCAL_POCKET_PORT}:${REMOTE_SERVICE_HOST}:8003" \
      -o ExitOnForwardFailure=yes \
      -o ServerAliveInterval=30 \
      -o ServerAliveCountMax=3
  ) </dev/null >"$TUNNEL_LOGFILE" 2>&1 &
  echo "$!" >"$TUNNEL_PIDFILE"
  sleep 2
  if ! tunnel_alive; then
    echo "failed to establish SSH tunnel; see $TUNNEL_LOGFILE" >&2
    tail -n 40 "$TUNNEL_LOGFILE" >&2 || true
    exit 1
  fi
}

fetch_remote_runtime_env() {
  if gcloud compute ssh "$GCP_INSTANCE_NAME" \
    --project "$GCP_PROJECT_ID" \
    --zone "$GCP_ZONE" \
    --command "cat '$REMOTE_RUNTIME_ENV_FILE'" \
    >"$REMOTE_RUNTIME_ENV_CACHE" 2>/dev/null; then
    chmod 600 "$REMOTE_RUNTIME_ENV_CACHE"
    # shellcheck disable=SC1090
    source "$REMOTE_RUNTIME_ENV_CACHE"
    LIVEKIT_API_KEY="${CHARACTER_ENG_LIVEKIT_API_KEY:-${LIVEKIT_API_KEY:-devkey}}"
    LIVEKIT_API_SECRET="${CHARACTER_ENG_LIVEKIT_API_SECRET:-${LIVEKIT_API_SECRET:-secret}}"
    return 0
  fi
  rm -f "$REMOTE_RUNTIME_ENV_CACHE"
  return 1
}

livekit_url() {
  ./deploy/gcp/vm_ctl.sh "$ENV_FILE" livekit-url
}

livekit_probe_url() {
  local url
  url="$(livekit_url)"
  url="${url/wss:/https:}"
  url="${url/ws:/http:}"
  printf '%s/' "${url%/}"
}

trap 'cleanup_tunnel' EXIT

if [[ "$LIVEKIT_ENABLED" != "1" && "$LIVEKIT_ENABLED" != "true" && "$LIVEKIT_ENABLED" != "yes" ]]; then
  echo "LIVEKIT_ENABLED is not set in $ENV_FILE" >&2
  exit 1
fi

ensure_vm_running

if fetch_remote_runtime_env; then
  log "Synced remote runtime env from GCP VM"
else
  log "Remote runtime env not available; falling back to local env/.env"
fi

if [[ "$LIVEKIT_API_KEY" == "devkey" || "$LIVEKIT_API_SECRET" == "secret" ]]; then
  echo "LiveKit credentials are missing; set LIVEKIT_API_KEY / LIVEKIT_API_SECRET in GCP or deploy/gcp.env" >&2
  exit 1
fi

log "Waiting for remote vision, Pocket-TTS, and LiveKit"
wait_for_remote_http "remote vision" "http://${REMOTE_SERVICE_HOST}:7860/model_status"
wait_for_remote_http "remote Pocket-TTS" "http://${REMOTE_SERVICE_HOST}:8003/"
wait_for_remote_http "remote LiveKit" "http://127.0.0.1:${LIVEKIT_HTTP_PORT}/"
start_tunnel

log "Waiting for tunneled remote vision and Pocket-TTS"
wait_for_http "remote vision" "http://${LOCAL_TUNNEL_HOST}:${LOCAL_VISION_PORT}/model_status"
wait_for_http "remote Pocket-TTS" "http://${LOCAL_TUNNEL_HOST}:${LOCAL_POCKET_PORT}/"
wait_for_http "remote LiveKit" "$(livekit_probe_url)"

REMOTE_LIVEKIT_URL="$(livekit_url)"
log "Running local app against remote heavy services + remote LiveKit"
log "Remote LiveKit: ${REMOTE_LIVEKIT_URL}"
env \
  CLEAN_START="${CLEAN_START:-1}" \
  CHARACTER_ENG_TRANSPORT_MODE=remote_hot_webrtc \
  CHARACTER_ENG_TRANSPORT_VIDEO_METRICS_PATH="$TRANSPORT_VIDEO_METRICS_PATH" \
  CHARACTER_ENG_TRANSPORT_AUDIO_METRICS_PATH="$TRANSPORT_AUDIO_METRICS_PATH" \
  CHARACTER_ENG_VISION_URL="http://${LOCAL_TUNNEL_HOST}:${LOCAL_VISION_PORT}" \
  CHARACTER_ENG_VISION_AUTO_LAUNCH=false \
  CHARACTER_ENG_TTS_BACKEND=pocket \
  CHARACTER_ENG_TTS_SERVER_URL="http://${LOCAL_TUNNEL_HOST}:${LOCAL_POCKET_PORT}" \
  CHARACTER_ENG_LIVEKIT_ENABLED=true \
  CHARACTER_ENG_LIVEKIT_URL="${REMOTE_LIVEKIT_URL}" \
  CHARACTER_ENG_LIVEKIT_API_KEY="${LIVEKIT_API_KEY}" \
  CHARACTER_ENG_LIVEKIT_API_SECRET="${LIVEKIT_API_SECRET}" \
  ./scripts/run_local.sh "$@"
