#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

resolve_default_env_file() {
  if [[ -n "${CHARACTER_ENG_GCP_ENV:-}" ]]; then
    printf '%s\n' "$CHARACTER_ENG_GCP_ENV"
    return 0
  fi
  if [[ -f "$ROOT/deploy/gcp.env" ]]; then
    printf '%s\n' "$ROOT/deploy/gcp.env"
    return 0
  fi
  printf '%s\n' "$ROOT/deploy/gcp.shared-remote.env"
}

ENV_FILE="$(resolve_default_env_file)"
TUNNEL_DIR="${TUNNEL_DIR:-$ROOT/.cache/gcp-remote-hot-webrtc}"
TUNNEL_PIDFILE="$TUNNEL_DIR/tunnel.pid"
STOP_VM="${STOP_VM:-1}"
VM_CTL_BIN="${VM_CTL_BIN:-./deploy/gcp/vm_ctl.sh}"

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
  "$VM_CTL_BIN" "$ENV_FILE" stop
fi
