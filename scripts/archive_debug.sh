#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <archive-id|path> [issue summary]" >&2
  exit 1
fi

ARCHIVE_REF="$1"
ISSUE_SUMMARY="${2:-}"

cd "$ROOT"
python3 scripts/archive_debug.py "$ARCHIVE_REF" "$ISSUE_SUMMARY"
