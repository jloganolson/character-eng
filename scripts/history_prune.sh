#!/usr/bin/env bash
set -euo pipefail

uv run -m character_eng.history prune "$@"
