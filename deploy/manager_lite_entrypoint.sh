#!/usr/bin/env bash
set -euo pipefail

cd /app

export PYTHONUNBUFFERED=1
export CHARACTER_ENG_SESSION_RUNTIME="${CHARACTER_ENG_SESSION_RUNTIME:-stub}"

PUBLIC_HOST="${PUBLIC_HOST:-127.0.0.1}"
MANAGER_PORT="${MANAGER_PORT:-7870}"

args=(python3 -m character_eng.session_manager --port "$MANAGER_PORT" --public-host "$PUBLIC_HOST")
if [[ -n "${CHARACTER_ENG_MANAGER_TOKEN:-}" ]]; then
  args+=(--admin-token "$CHARACTER_ENG_MANAGER_TOKEN")
fi

exec "${args[@]}"
