#!/usr/bin/env bash
set -euo pipefail

cd /app

MODE="${MODE:-session-manager}"
PUBLIC_HOST="${PUBLIC_HOST:-127.0.0.1}"
MANAGER_PORT="${MANAGER_PORT:-7870}"
CHARACTER="${CHARACTER:-greg}"
START_PAUSED="${START_PAUSED:-1}"
VISION="${VISION:-0}"

case "$MODE" in
  session-manager)
    args=(uv run python -m character_eng.session_manager --port "$MANAGER_PORT" --public-host "$PUBLIC_HOST")
    if [[ -n "${CHARACTER_ENG_MANAGER_TOKEN:-}" ]]; then
      args+=(--admin-token "$CHARACTER_ENG_MANAGER_TOKEN")
    fi
    exec "${args[@]}"
    ;;
  app)
    args=(uv run -m character_eng --browser --character "$CHARACTER")
    if [[ "$VISION" == "1" ]]; then
      args+=(--vision)
    fi
    if [[ "$START_PAUSED" == "1" ]]; then
      args+=(--start-paused)
    fi
    exec "${args[@]}"
    ;;
  heavy)
    exec ./scripts/run_heavy.sh
    ;;
  full)
    ./scripts/run_heavy.sh
    args=(uv run python -m character_eng.session_manager --port "$MANAGER_PORT" --public-host "$PUBLIC_HOST")
    if [[ -n "${CHARACTER_ENG_MANAGER_TOKEN:-}" ]]; then
      args+=(--admin-token "$CHARACTER_ENG_MANAGER_TOKEN")
    fi
    exec "${args[@]}"
    ;;
  *)
    echo "unknown MODE: $MODE" >&2
    echo "expected one of: session-manager, app, heavy, full" >&2
    exit 1
    ;;
esac
