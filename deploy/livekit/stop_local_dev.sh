#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${LIVEKIT_CONTAINER_NAME:-character-eng-livekit-dev}"
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
echo "LiveKit dev server stopped: ${CONTAINER_NAME}"
