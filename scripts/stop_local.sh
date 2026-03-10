#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DASHBOARD_PORT="${DASHBOARD_PORT:-7862}"
VISION_PORT="${VISION_PORT:-7860}"
VLLM_PORT="${VLLM_PORT:-8000}"
POCKET_PORT="${POCKET_PORT:-8003}"

log() {
  printf '==> %s\n' "$*"
}

kill_vllm_gpu_compute() {
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  local pids
  pids="$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader,nounits 2>/dev/null | awk -F',' '/VLLM::EngineCore/ {gsub(/ /, "", $1); print $1}')"
  if [[ -z "$pids" ]]; then
    return 0
  fi
  log "Stopping stray vLLM GPU compute: ${pids//$'\n'/ }"
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

log "Stopping local app and heavy services"
kill_matching 'uv run -m character_eng' 'character-eng app'
kill_matching 'python[^ ]* -m character_eng' 'character-eng app'
kill_matching '/services/vision/start\.sh' 'vision launcher'
kill_matching '/tools/vllm/bin/vllm serve ' 'vLLM server'
kill_matching 'pocket-tts serve' 'Pocket-TTS'
kill_vllm_gpu_compute

kill_port "$DASHBOARD_PORT" 'dashboard/bridge'
kill_port "$VISION_PORT" 'vision service'
kill_port "$VLLM_PORT" 'vLLM'
kill_port "$POCKET_PORT" 'Pocket-TTS'
kill_vllm_gpu_compute

rm -f "$ROOT/logs/heavy/vision_stack.pid" "$ROOT/logs/heavy/pocket_tts.pid"

log "Local stack stopped."
