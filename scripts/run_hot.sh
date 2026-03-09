#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VISION_PORT="${VISION_PORT:-7860}"
VLLM_PORT="${VLLM_PORT:-8000}"
POCKET_PORT="${POCKET_PORT:-8003}"

require_http() {
  local label="$1"
  local url="$2"
  if ! curl -sf "$url" >/dev/null 2>&1; then
    printf '==> %s not ready at %s\n' "$label" "$url" >&2
    printf '==> Run ./scripts/run_heavy.sh first, or use ./scripts/run_local.sh for a full cold boot.\n' >&2
    exit 1
  fi
}

require_http "Pocket-TTS" "http://127.0.0.1:${POCKET_PORT}/"
require_http "vLLM" "http://127.0.0.1:${VLLM_PORT}/v1/models"
require_http "vision app" "http://127.0.0.1:${VISION_PORT}/model_status"

exec env CLEAN_START=0 KEEP_POCKET=1 ./scripts/run_local.sh "$@"
