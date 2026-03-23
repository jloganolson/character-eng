#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/.uv-cache}"

CHARACTER="${CHARACTER:-greg}"
VOICE_ON_START="${VOICE_ON_START:-1}"
VISION_ON_START="${VISION_ON_START:-1}"
START_PAUSED="${START_PAUSED:-1}"

log() {
  printf '==> %s\n' "$*"
}

log "Starting live app for ${CHARACTER}"
log "Mode: voice=${VOICE_ON_START} vision=${VISION_ON_START} start_paused=${START_PAUSED}"
log "Use ./scripts/stop_live.sh to stop only the app/dashboard layer."

cmd=(uv run -m character_eng --character "$CHARACTER")
if [[ "$VISION_ON_START" == "1" ]]; then
  cmd+=(--vision)
fi
if [[ "$VOICE_ON_START" == "1" ]]; then
  cmd+=(--voice)
fi
if [[ "$START_PAUSED" == "1" ]]; then
  cmd+=(--start-paused)
fi

exec "${cmd[@]}" "$@"
