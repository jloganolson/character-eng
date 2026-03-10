#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/.uv-cache}"

VISION_PORT="${VISION_PORT:-7860}"
VLLM_PORT="${VLLM_PORT:-8000}"
POCKET_PORT="${POCKET_PORT:-8003}"
POCKET_VOICE="${POCKET_VOICE:-$ROOT/voices/greg.safetensors}"
HEAVY_LOG_DIR="${HEAVY_LOG_DIR:-$ROOT/logs/heavy}"
POCKET_BIN="${POCKET_BIN:-}"
WAIT_INTERVAL="${WAIT_INTERVAL:-2}"
VISION_TIMEOUT="${VISION_TIMEOUT:-240}"
POCKET_TIMEOUT="${POCKET_TIMEOUT:-30}"
VISION_PIDFILE="$HEAVY_LOG_DIR/vision_stack.pid"
POCKET_PIDFILE="$HEAVY_LOG_DIR/pocket_tts.pid"

mkdir -p "$HEAVY_LOG_DIR"

log() {
  printf '==> %s\n' "$*"
}

kill_stale_vllm_gpu_compute() {
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  local pids
  pids="$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader,nounits 2>/dev/null | awk -F',' '/VLLM::EngineCore/ {gsub(/ /, "", $1); print $1}')"
  if [[ -z "$pids" ]]; then
    return 0
  fi
  log "Clearing stale vLLM GPU compute: ${pids//$'\n'/ }"
  while IFS= read -r pid; do
    [[ -n "$pid" ]] || continue
    kill "$pid" >/dev/null 2>&1 || true
  done <<< "$pids"
  sleep 1
  while IFS= read -r pid; do
    [[ -n "$pid" ]] || continue
    kill -9 "$pid" >/dev/null 2>&1 || true
  done <<< "$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader,nounits 2>/dev/null | awk -F',' '/VLLM::EngineCore/ {gsub(/ /, "", $1); print $1}')"
}

start_bg() {
  local label="$1"
  local logfile="$2"
  local pidfile="$3"
  shift 3
  log "Starting ${label} (log: ${logfile})"
  : >"$logfile"
  setsid "$@" >"$logfile" 2>&1 < /dev/null &
  echo "$!" >"$pidfile"
}

pid_alive() {
  local pidfile="$1"
  [[ -f "$pidfile" ]] || return 1
  local pid
  pid="$(cat "$pidfile" 2>/dev/null || true)"
  [[ -n "$pid" ]] || return 1
  kill -0 "$pid" 2>/dev/null
}

wait_for_http() {
  local label="$1"
  local url="$2"
  local timeout="$3"
  local pidfile="$4"
  local logfile="$5"
  local elapsed=0

  log "Waiting for ${label}"
  while ! curl -sf "$url" >/dev/null 2>&1; do
    if [[ -n "$pidfile" ]] && ! pid_alive "$pidfile"; then
      log "${label} process exited before becoming ready."
      [[ -f "$logfile" ]] && tail -n 40 "$logfile"
      return 1
    fi
    sleep "$WAIT_INTERVAL"
    elapsed=$((elapsed + WAIT_INTERVAL))
    printf '.'
    if (( elapsed >= timeout )); then
      printf '\n'
      log "${label} did not become ready within ${timeout}s."
      [[ -f "$logfile" ]] && tail -n 40 "$logfile"
      return 1
    fi
  done
  printf ' ready (%ss)\n' "$elapsed"
}

print_model_status() {
  local status_json
  status_json="$(curl -sf "http://127.0.0.1:${VISION_PORT}/model_status" || true)"
  [[ -n "$status_json" ]] || return 0
  python3 - <<'PY' "$status_json"
import json, sys
status = json.loads(sys.argv[1])
parts = [f"vllm={status.get('vllm', 'unknown')}"]
for key in ("sam3", "face", "person"):
    item = status.get(key)
    if isinstance(item, dict):
        parts.append(f"{key}={item.get('status', 'unknown')}")
    elif item is not None:
        parts.append(f"{key}={item}")
print("==> Model status: " + " · ".join(parts))
PY
}

models_fully_ready() {
  local status_json
  status_json="$(curl -sf "http://127.0.0.1:${VISION_PORT}/model_status" || true)"
  [[ -n "$status_json" ]] || return 1
  python3 - <<'PY' "$status_json"
import json, sys
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

wait_for_models_ready() {
  local elapsed=0
  local timeout="$1"
  log "Waiting for model trackers"
  while ! models_fully_ready; do
    sleep "$WAIT_INTERVAL"
    elapsed=$((elapsed + WAIT_INTERVAL))
    print_model_status
    if (( elapsed >= timeout )); then
      log "Model trackers did not reach ready within ${timeout}s."
      return 1
    fi
  done
  print_model_status
}

if [[ -z "$POCKET_BIN" ]]; then
  if [[ -x "$ROOT/.venv/bin/pocket-tts" ]]; then
    POCKET_BIN="$ROOT/.venv/bin/pocket-tts"
  else
    POCKET_BIN="$(command -v pocket-tts || true)"
  fi
fi

kill_stale_vllm_gpu_compute

if curl -sf "http://127.0.0.1:${VISION_PORT}/" >/dev/null 2>&1; then
  log "Vision service already running on :${VISION_PORT}"
else
  start_bg "vision stack" "$HEAVY_LOG_DIR/vision_stack.log" "$VISION_PIDFILE" "$ROOT/services/vision/start.sh"
fi

if curl -sf "http://127.0.0.1:${POCKET_PORT}/" >/dev/null 2>&1; then
  log "Pocket-TTS already running on :${POCKET_PORT}"
else
  if [[ -z "$POCKET_BIN" ]]; then
    log "Pocket-TTS binary not found; expected .venv/bin/pocket-tts or PATH entry"
    exit 1
  fi
  start_bg "Pocket-TTS" "$HEAVY_LOG_DIR/pocket_tts.log" "$POCKET_PIDFILE" "$POCKET_BIN" serve --port "$POCKET_PORT" --voice "$POCKET_VOICE"
fi

wait_for_http "Pocket-TTS" "http://127.0.0.1:${POCKET_PORT}/" "$POCKET_TIMEOUT" "$POCKET_PIDFILE" "$HEAVY_LOG_DIR/pocket_tts.log"
wait_for_http "vLLM" "http://127.0.0.1:${VLLM_PORT}/v1/models" "$VISION_TIMEOUT" "$VISION_PIDFILE" "$HEAVY_LOG_DIR/vision_stack.log"
wait_for_http "vision app" "http://127.0.0.1:${VISION_PORT}/model_status" "$VISION_TIMEOUT" "$VISION_PIDFILE" "$HEAVY_LOG_DIR/vision_stack.log"
wait_for_models_ready "$VISION_TIMEOUT"

log "Heavy services requested."
log "Vision:  http://127.0.0.1:${VISION_PORT}/"
log "vLLM:    http://127.0.0.1:${VLLM_PORT}/v1/models"
log "Pocket:  http://127.0.0.1:${POCKET_PORT}/"
print_model_status
log "Use ./scripts/run_hot.sh for the fast app/UI layer."
