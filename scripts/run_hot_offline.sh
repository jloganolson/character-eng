#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/.uv-cache}"

CHARACTER="${CHARACTER:-greg}"
DASHBOARD_PORT="${DASHBOARD_PORT:-7862}"

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

kill_matching 'uv run -m character_eng' 'character-eng app'
kill_matching 'python[^ ]* -m character_eng' 'character-eng app'
kill_port "$DASHBOARD_PORT" 'dashboard/bridge'

log "Starting archive-only dashboard for ${CHARACTER}"
exec uv run -m character_eng --character "$CHARACTER" --archive-only "$@"
