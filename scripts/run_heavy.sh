#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT/.venv/bin/python"
else
  PYTHON_BIN="$(command -v python3)"
fi

if [[ -z "${CHARACTER_ENG_RUNTIME_ROOT:-}" && "$ROOT" == "/app" && -d /workspace ]]; then
  export CHARACTER_ENG_RUNTIME_ROOT="/workspace/character-eng-runtime"
fi

if [[ -n "${CHARACTER_ENG_RUNTIME_ROOT:-}" ]]; then
  CACHE_ROOT_DEFAULT="${CHARACTER_ENG_CACHE_ROOT:-$CHARACTER_ENG_RUNTIME_ROOT/.cache}"
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$CACHE_ROOT_DEFAULT/uv}"
else
  CACHE_ROOT_DEFAULT="${CHARACTER_ENG_CACHE_ROOT:-$ROOT/.cache}"
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/.uv-cache}"
fi
export HF_HOME="${HF_HOME:-$CACHE_ROOT_DEFAULT/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TORCH_HOME="${TORCH_HOME:-$CACHE_ROOT_DEFAULT/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_ROOT_DEFAULT/xdg}"

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
STALE_VLLM_CLEARED=0
EXIT_AFTER_READY=0
PRINT_STATUS_JSON=0
TEARDOWN_AFTER_READY=0
STARTED_VISION=0
STARTED_POCKET=0
if [[ -z "${REQUIRE_POCKET_TTS:-}" ]]; then
  if [[ "$ROOT" == "/app" ]]; then
    REQUIRE_POCKET_TTS=0
  else
    REQUIRE_POCKET_TTS=1
  fi
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --once)
      EXIT_AFTER_READY=1
      ;;
    --json)
      PRINT_STATUS_JSON=1
      ;;
    --teardown)
      EXIT_AFTER_READY=1
      TEARDOWN_AFTER_READY=1
      ;;
    *)
      echo "unknown arg: $1" >&2
      echo "expected optional args: --once, --json, --teardown" >&2
      exit 1
      ;;
  esac
  shift
done

mkdir -p "$HEAVY_LOG_DIR"
mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME"

log() {
  printf '==> %s\n' "$*"
}

vllm_http_ready() {
  curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1
}

vision_http_ready() {
  curl -sf "http://127.0.0.1:${VISION_PORT}/model_status" >/dev/null 2>&1
}

kill_stale_vllm_gpu_compute() {
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  local pids
  pids="$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader,nounits 2>/dev/null | awk -F',' '/VLLM::EngineCore/ {gsub(/ /, "", $1); print $1}')"
  if [[ -z "$pids" ]]; then
    return 0
  fi
  STALE_VLLM_CLEARED=1
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

stop_pidfile_process() {
  local pidfile="$1"
  [[ -f "$pidfile" ]] || return 0
  local pid
  pid="$(cat "$pidfile" 2>/dev/null || true)"
  [[ -n "$pid" ]] || return 0
  kill "$pid" >/dev/null 2>&1 || true
  sleep 1
  kill -9 "$pid" >/dev/null 2>&1 || true
  rm -f "$pidfile"
}

teardown_started_services() {
  if (( STARTED_POCKET )); then
    log "Stopping Pocket-TTS started by this smoke run"
    stop_pidfile_process "$POCKET_PIDFILE"
    lsof -ti:"$POCKET_PORT" 2>/dev/null | xargs kill -9 2>/dev/null || true
  fi
  if (( STARTED_VISION )); then
    log "Stopping vision stack started by this smoke run"
    stop_pidfile_process "$VISION_PIDFILE"
    lsof -ti:"$VISION_PORT" 2>/dev/null | xargs kill -9 2>/dev/null || true
    lsof -ti:"$VLLM_PORT" 2>/dev/null | xargs kill -9 2>/dev/null || true
  fi
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

model_error_summary() {
  local status_json
  status_json="$(curl -sf "http://127.0.0.1:${VISION_PORT}/model_status" || true)"
  [[ -n "$status_json" ]] || return 1
  python3 - <<'PY' "$status_json"
import json, sys
status = json.loads(sys.argv[1])
errors = []
for key in ("vllm", "sam3", "face", "person"):
    item = status.get(key)
    if isinstance(item, dict):
        state = item.get("status", "unknown")
        detail = str(item.get("error", "")).strip()
    else:
        state = item or "unknown"
        detail = ""
    if state == "error":
        errors.append(f"{key}={detail or 'error'}")
if not errors:
    raise SystemExit(1)
print(" · ".join(errors))
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
    if error_summary="$(model_error_summary)"; then
      print_model_status
      log "Model tracker error: ${error_summary}"
      return 1
    fi
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

restart_vision_stack() {
  local pids
  pids="$(lsof -ti:"$VISION_PORT" 2>/dev/null || true)"
  if [[ -n "$pids" ]]; then
    log "Clearing stale vision app on :${VISION_PORT}: ${pids//$'\n'/ }"
    while IFS= read -r pid; do
      [[ -n "$pid" ]] || continue
      kill "$pid" >/dev/null 2>&1 || true
    done <<< "$pids"
    sleep 1
    while IFS= read -r pid; do
      [[ -n "$pid" ]] || continue
      kill -9 "$pid" >/dev/null 2>&1 || true
    done <<< "$(lsof -ti:"$VISION_PORT" 2>/dev/null || true)"
  fi
  rm -f "$VISION_PIDFILE"
  start_bg "vision stack" "$HEAVY_LOG_DIR/vision_stack.log" "$VISION_PIDFILE" "$ROOT/services/vision/start.sh"
  STARTED_VISION=1
}

if [[ -z "$POCKET_BIN" ]]; then
  if [[ -x "$ROOT/.venv/bin/pocket-tts" ]]; then
    POCKET_BIN="$ROOT/.venv/bin/pocket-tts"
  else
    POCKET_BIN="$("$PYTHON_BIN" -m character_eng.pocket_runtime --ensure --print-bin || true)"
    if [[ -z "$POCKET_BIN" ]]; then
      POCKET_BIN="$(command -v pocket-tts || true)"
    fi
  fi
fi

kill_stale_vllm_gpu_compute

if (( STALE_VLLM_CLEARED )); then
  log "Restarting vision stack after clearing stale vLLM compute"
  restart_vision_stack
elif vision_http_ready && vllm_http_ready; then
  log "Vision service already running on :${VISION_PORT}"
elif vision_http_ready || curl -sf "http://127.0.0.1:${VISION_PORT}/" >/dev/null 2>&1; then
  log "Vision stack is up but unhealthy; restarting it"
  print_model_status || true
  restart_vision_stack
else
  restart_vision_stack
fi

POCKET_REQUIRED="$REQUIRE_POCKET_TTS"
POCKET_READY=0
if curl -sf "http://127.0.0.1:${POCKET_PORT}/" >/dev/null 2>&1; then
  log "Pocket-TTS already running on :${POCKET_PORT}"
  POCKET_READY=1
else
  if [[ -z "$POCKET_BIN" ]]; then
    if [[ "$POCKET_REQUIRED" == "1" ]]; then
      log "Pocket-TTS binary not found and runtime bootstrap failed"
      exit 1
    fi
    log "Pocket-TTS unavailable; continuing without filler/TTS warm service"
  else
    start_bg "Pocket-TTS" "$HEAVY_LOG_DIR/pocket_tts.log" "$POCKET_PIDFILE" "$POCKET_BIN" serve --port "$POCKET_PORT" --voice "$POCKET_VOICE"
    STARTED_POCKET=1
    if wait_for_http "Pocket-TTS" "http://127.0.0.1:${POCKET_PORT}/" "$POCKET_TIMEOUT" "$POCKET_PIDFILE" "$HEAVY_LOG_DIR/pocket_tts.log"; then
      POCKET_READY=1
    elif [[ "$POCKET_REQUIRED" == "1" ]]; then
      exit 1
    else
      log "Pocket-TTS failed to warm; continuing without filler/TTS warm service"
    fi
  fi
fi

wait_for_http "vLLM" "http://127.0.0.1:${VLLM_PORT}/v1/models" "$VISION_TIMEOUT" "$VISION_PIDFILE" "$HEAVY_LOG_DIR/vision_stack.log"
wait_for_http "vision app" "http://127.0.0.1:${VISION_PORT}/model_status" "$VISION_TIMEOUT" "$VISION_PIDFILE" "$HEAVY_LOG_DIR/vision_stack.log"
wait_for_models_ready "$VISION_TIMEOUT"

log "Heavy services requested."
log "Vision:  http://127.0.0.1:${VISION_PORT}/"
log "vLLM:    http://127.0.0.1:${VLLM_PORT}/v1/models"
if (( POCKET_READY )); then
  log "Pocket:  http://127.0.0.1:${POCKET_PORT}/"
else
  log "Pocket:  unavailable"
fi
print_model_status
if (( PRINT_STATUS_JSON )); then
  curl -sf "http://127.0.0.1:${VISION_PORT}/model_status"
  printf '\n'
fi
log "Use ./scripts/run_hot.sh for the fast app/UI layer."
if (( EXIT_AFTER_READY )); then
  if (( TEARDOWN_AFTER_READY )); then
    teardown_started_services
  fi
  exit 0
fi
