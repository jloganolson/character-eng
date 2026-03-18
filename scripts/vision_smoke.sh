#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TEARDOWN="${TEARDOWN:-0}"

log() {
  printf '==> %s\n' "$*"
}

log "Heavy runtime smoke starting"
log "Ports: vision=${VISION_PORT:-7860} vllm=${VLLM_PORT:-8000} pocket=${POCKET_PORT:-8003}"
log "Runtime root: ${CHARACTER_ENG_RUNTIME_ROOT:-$ROOT/.runtime}"

args=(--once --json)
if [[ "$TEARDOWN" == "1" ]]; then
  args+=(--teardown)
fi

"$ROOT/scripts/run_heavy.sh" "${args[@]}" "$@"

if [[ "$TEARDOWN" == "1" ]]; then
  log "Heavy runtime smoke passed and stopped the services it started."
else
  log "Heavy runtime smoke passed; services remain warm for follow-up debugging."
fi
