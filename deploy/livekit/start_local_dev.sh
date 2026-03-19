#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${LIVEKIT_CONTAINER_NAME:-character-eng-livekit-dev}"
IMAGE="${LIVEKIT_IMAGE:-livekit/livekit-server:latest}"
HTTP_PORT="${LIVEKIT_HTTP_PORT:-7880}"
TCP_PORT="${LIVEKIT_TCP_PORT:-7881}"
UDP_START="${LIVEKIT_UDP_START:-50100}"
UDP_END="${LIVEKIT_UDP_END:-50120}"

if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  docker rm -f "$CONTAINER_NAME" >/dev/null
fi

docker run -d \
  --name "$CONTAINER_NAME" \
  -p "${HTTP_PORT}:7880" \
  -p "${TCP_PORT}:7881/tcp" \
  -p "${UDP_START}-${UDP_END}:${UDP_START}-${UDP_END}/udp" \
  "$IMAGE" \
  --dev \
  --bind 0.0.0.0 >/dev/null

echo "LiveKit dev server started:"
echo "  ws://127.0.0.1:${HTTP_PORT}"
echo "  api key: devkey"
echo "  api secret: secret"
