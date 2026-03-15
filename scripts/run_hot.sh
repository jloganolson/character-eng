#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VISION_PORT="${VISION_PORT:-7860}"
VLLM_PORT="${VLLM_PORT:-8000}"
POCKET_PORT="${POCKET_PORT:-8003}"
WAIT_INTERVAL="${WAIT_INTERVAL:-2}"
READY_TIMEOUT="${READY_TIMEOUT:-12}"

log() {
  printf '==> %s\n' "$*"
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
  local elapsed=0
  while ! curl -sf "$url" >/dev/null 2>&1; do
    sleep "$WAIT_INTERVAL"
    elapsed=$((elapsed + WAIT_INTERVAL))
    if (( elapsed >= timeout )); then
      return 1
    fi
  done
  return 0
}

wait_for_models_ready() {
  local timeout="${1:-$READY_TIMEOUT}"
  local elapsed=0
  while ! models_fully_ready; do
    sleep "$WAIT_INTERVAL"
    elapsed=$((elapsed + WAIT_INTERVAL))
    if (( elapsed >= timeout )); then
      return 1
    fi
  done
  return 0
}

ensure_heavy_ready() {
  wait_for_http "Pocket-TTS" "http://127.0.0.1:${POCKET_PORT}/" || return 1
  wait_for_http "vLLM" "http://127.0.0.1:${VLLM_PORT}/v1/models" || return 1
  wait_for_http "vision app" "http://127.0.0.1:${VISION_PORT}/model_status" || return 1
  wait_for_models_ready || return 1
  return 0
}

if ! ensure_heavy_ready; then
  log "Heavy services look stale; repairing with ./scripts/run_heavy.sh"
  ./scripts/run_heavy.sh
fi

if ! ensure_heavy_ready; then
  printf '==> Heavy services are still not ready after repair.\n' >&2
  printf '==> Try ./scripts/run_local.sh for a full cold boot and inspect logs/heavy/.\n' >&2
  exit 1
fi

exec env CLEAN_START=0 KEEP_POCKET=1 ./scripts/run_local.sh "$@"
