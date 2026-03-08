#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CHARACTER="${CHARACTER:-greg}"
CLEAN_START="${CLEAN_START:-1}"
KEEP_POCKET="${KEEP_POCKET:-0}"
DASHBOARD_PORT="${DASHBOARD_PORT:-7862}"
VISION_PORT="${VISION_PORT:-7860}"
VLLM_PORT="${VLLM_PORT:-8000}"
POCKET_PORT="${POCKET_PORT:-8003}"

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

clean_start() {
  log "Cleaning old local stack"
  kill_matching 'uv run -m character_eng' 'character-eng app'
  kill_matching 'python[^ ]* -m character_eng' 'character-eng app'
  kill_matching '/services/vision/start\.sh' 'vision launcher'
  kill_matching '/tools/vllm/bin/vllm serve ' 'vLLM server'

  kill_port "$DASHBOARD_PORT" 'dashboard/bridge'
  kill_port "$VISION_PORT" 'vision service'
  kill_port "$VLLM_PORT" 'vLLM'
  if [[ "$KEEP_POCKET" != "1" ]]; then
    kill_matching 'pocket-tts serve' 'Pocket-TTS'
    kill_port "$POCKET_PORT" 'Pocket-TTS'
  else
    log "Keeping Pocket-TTS on :${POCKET_PORT}"
  fi
}

if [[ "$CLEAN_START" == "1" ]]; then
  clean_start
else
  log "Skipping clean start (CLEAN_START=0)"
fi

log "Starting local stack for ${CHARACTER}"
log "Stack ports: dashboard=${DASHBOARD_PORT} vision=${VISION_PORT} vllm=${VLLM_PORT} pocket=${POCKET_PORT}"
log "Vision bootstrap logs will stream below once the app starts vision"
exec uv run -m character_eng --character "$CHARACTER" --vision "$@"
