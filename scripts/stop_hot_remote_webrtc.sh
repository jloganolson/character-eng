#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ENV_FILE="${CHARACTER_ENG_GCP_ENV:-$ROOT/deploy/gcp.env}"
TUNNEL_DIR="${TUNNEL_DIR:-$ROOT/.cache/gcp-remote-hot-webrtc}"
TUNNEL_PIDFILE="$TUNNEL_DIR/tunnel.pid"
STOP_VM="${STOP_VM:-1}"

log() {
  printf '==> %s\n' "$*"
}

if [[ -f "$TUNNEL_PIDFILE" ]]; then
  pid="$(cat "$TUNNEL_PIDFILE" 2>/dev/null || true)"
  if [[ -n "${pid:-}" ]]; then
    log "Stopping remote WebRTC tunnel ($pid)"
    kill "$pid" >/dev/null 2>&1 || true
    sleep 1
    kill -9 "$pid" >/dev/null 2>&1 || true
  fi
  rm -f "$TUNNEL_PIDFILE"
fi

if [[ "$STOP_VM" == "1" ]]; then
  log "Stopping GCP VM"
  ./deploy/gcp/vm_ctl.sh "$ENV_FILE" stop
fi
