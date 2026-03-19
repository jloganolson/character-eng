#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VISION_PORT="${VISION_PORT:-7860}"
VLLM_PORT="${VLLM_PORT:-8000}"
POCKET_PORT="${POCKET_PORT:-8003}"
WAIT_INTERVAL="${WAIT_INTERVAL:-2}"
READY_TIMEOUT="${READY_TIMEOUT:-12}"
TUNNEL_DIR="${TUNNEL_DIR:-$ROOT/.cache/livekit-remote-hot}"
TRANSPORT_VIDEO_METRICS_PATH="$TUNNEL_DIR/video_metrics.json"
TRANSPORT_AUDIO_METRICS_PATH="$TUNNEL_DIR/audio_metrics.json"

log() {
  printf '==> %s\n' "$*"
}

kill_matching() {
  local pattern="$1"
  local label="$2"
  local pids
  pids="$(pgrep -f "$pattern" || true)"
  if [[ -z "$pids" ]]; then
    return 0
  fi
  log "Stopping ${label}: ${pids//$'\n'/ }"
  while IFS= read -r pid; do
    [[ -n "$pid" ]] || continue
    kill "$pid" >/dev/null 2>&1 || true
  done <<< "$pids"
  sleep 1
  while IFS= read -r pid; do
    [[ -n "$pid" ]] || continue
    kill -9 "$pid" >/dev/null 2>&1 || true
  done <<< "$(pgrep -f "$pattern" || true)"
}

kill_port() {
  local port="$1"
  local label="$2"
  local pids
  pids="$(lsof -ti:"$port" 2>/dev/null || true)"
  if [[ -z "$pids" ]]; then
    return 0
  fi
  log "Clearing ${label} on :${port}: ${pids//$'\n'/ }"
  while IFS= read -r pid; do
    [[ -n "$pid" ]] || continue
    kill "$pid" >/dev/null 2>&1 || true
  done <<< "$pids"
  sleep 1
  while IFS= read -r pid; do
    [[ -n "$pid" ]] || continue
    kill -9 "$pid" >/dev/null 2>&1 || true
  done <<< "$(lsof -ti:"$port" 2>/dev/null || true)"
}

phase() {
  local index="$1"
  local total="$2"
  local title="$3"
  local eta="$4"
  printf '\n==> [%s/%s] %s' "$index" "$total" "$title"
  if [[ -n "$eta" ]]; then
    printf ' (%s)' "$eta"
  fi
  printf '\n'
}

wait_with_progress() {
  local label="$1"
  local timeout="$2"
  local eta="$3"
  local probe="$4"
  local elapsed=0
  local spinner='|/-\'
  local spin_idx=0
  local percent=0
  local spin_char=''

  printf '    %s ' "$label"
  while ! eval "$probe" >/dev/null 2>&1; do
    spin_char="${spinner:spin_idx:1}"
    spin_idx=$(((spin_idx + 1) % 4))
    elapsed=$((elapsed + WAIT_INTERVAL))
    if (( timeout > 0 )); then
      percent=$(( elapsed * 100 / timeout ))
      if (( percent > 99 )); then
        percent=99
      fi
    else
      percent=0
    fi
    printf '\r    %s %s %2ss elapsed · %2s%% of %ss budget' "$label" "$spin_char" "$elapsed" "$percent" "$timeout"
    if [[ -n "$eta" ]]; then
      printf ' · typical %s' "$eta"
    fi
    sleep "$WAIT_INTERVAL"
    if (( elapsed >= timeout )); then
      printf '\n'
      return 1
    fi
  done
  printf '\r    %s ready in %ss' "$label" "$elapsed"
  if [[ -n "$eta" ]]; then
    printf ' · typical %s' "$eta"
  fi
  printf '\n'
}

models_fully_ready() {
  local status_json
  status_json="$(curl -sf "http://127.0.0.1:${VISION_PORT}/model_status" || true)"
  [[ -n "$status_json" ]] || return 1
  python3 - <<'PY' "$status_json"
import json
import sys

status = json.loads(sys.argv[1])
if status.get("vllm") not in {"ready", "off", "unavailable"}:
    raise SystemExit(1)
for key in ("sam3", "face", "person"):
    item = status.get(key)
    if isinstance(item, dict):
        state = item.get("status", "unknown")
    else:
        state = item or "unknown"
    if state not in {"ready", "off", "unavailable", "disabled"}:
        raise SystemExit(1)
PY
}

wait_for_http() {
  local label="$1"
  local url="$2"
  local timeout="${3:-$READY_TIMEOUT}"
  local eta="${4:-2-8s}"
  wait_with_progress "$label" "$timeout" "$eta" "curl -sf '$url'"
}

wait_for_models_ready() {
  local timeout="${1:-$READY_TIMEOUT}"
  local eta="${2:-3-12s}"
  wait_with_progress "model trackers" "$timeout" "$eta" "models_fully_ready"
}

ensure_heavy_ready() {
  wait_for_http "Pocket-TTS" "http://127.0.0.1:${POCKET_PORT}/" "$READY_TIMEOUT" "2-8s" || return 1
  wait_for_http "vLLM" "http://127.0.0.1:${VLLM_PORT}/v1/models" "$READY_TIMEOUT" "2-10s" || return 1
  wait_for_http "vision app" "http://127.0.0.1:${VISION_PORT}/model_status" "$READY_TIMEOUT" "2-10s" || return 1
  wait_for_models_ready "$READY_TIMEOUT" "3-12s" || return 1
  return 0
}

ensure_livekit_dev() {
  if docker ps --filter name=character-eng-livekit-dev --format '{{.Names}}' | grep -qx 'character-eng-livekit-dev'; then
    log "LiveKit dev server already running at ws://127.0.0.1:7880"
    return 0
  fi
  log "Starting LiveKit dev server"
  ./deploy/livekit/start_local_dev.sh >/dev/null
}

force_external_only_vision() {
  log "Restarting vision service in injected-frame mode for WebRTC validation"
  kill_matching '/services/vision/start\.sh' 'vision launcher'
  kill_port "$VISION_PORT" 'vision service'
}

mkdir -p "$TUNNEL_DIR"

phase 1 4 "Checking local heavy services" "usually 5-15s if already warm"
if ! ensure_heavy_ready; then
  log "Heavy services look stale; repairing with ./scripts/run_heavy.sh"
  phase 2 4 "Repairing heavy stack" "usually 20-90s depending on warm caches"
  ./scripts/run_heavy.sh
fi

phase 3 4 "Re-checking heavy services" "usually 5-15s"
if ! ensure_heavy_ready; then
  printf '==> Heavy services are still not ready after repair.\n' >&2
  printf '==> Try ./scripts/run_local.sh for a full cold boot and inspect logs/heavy/.\n' >&2
  exit 1
fi

phase 4 4 "Starting LiveKit transport + local app" "dashboard should print within ~5-20s"
ensure_livekit_dev
force_external_only_vision
log "Metrics will be written to:"
printf '    video: %s\n' "$TRANSPORT_VIDEO_METRICS_PATH"
printf '    audio: %s\n' "$TRANSPORT_AUDIO_METRICS_PATH"
log "Handing off to ./scripts/run_local.sh"

exec env \
  CLEAN_START=0 \
  KEEP_POCKET=1 \
  CHARACTER_ENG_TRANSPORT_MODE=remote_hot_webrtc \
  CHARACTER_ENG_TRANSPORT_VIDEO_METRICS_PATH="$TRANSPORT_VIDEO_METRICS_PATH" \
  CHARACTER_ENG_TRANSPORT_AUDIO_METRICS_PATH="$TRANSPORT_AUDIO_METRICS_PATH" \
  CHARACTER_ENG_LIVEKIT_ENABLED=true \
  CHARACTER_ENG_LIVEKIT_URL="${CHARACTER_ENG_LIVEKIT_URL:-ws://127.0.0.1:7880}" \
  CHARACTER_ENG_LIVEKIT_API_KEY="${CHARACTER_ENG_LIVEKIT_API_KEY:-devkey}" \
  CHARACTER_ENG_LIVEKIT_API_SECRET="${CHARACTER_ENG_LIVEKIT_API_SECRET:-secret}" \
  ./scripts/run_local.sh "$@"
