#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_SKILLS_DIR="$ROOT/.codex/skills"
CODEX_HOME_DIR="${CODEX_HOME:-$HOME/.codex}"
TARGET_DIR="$CODEX_HOME_DIR/skills"
FORCE=0

if [[ "${1:-}" == "--force" ]]; then
  FORCE=1
fi

mkdir -p "$TARGET_DIR"

if [[ ! -d "$REPO_SKILLS_DIR" ]]; then
  echo "No repo skills found at $REPO_SKILLS_DIR" >&2
  exit 1
fi

installed_any=0
for skill_dir in "$REPO_SKILLS_DIR"/*; do
  [[ -d "$skill_dir" ]] || continue
  skill_name="$(basename "$skill_dir")"
  target="$TARGET_DIR/$skill_name"
  if [[ -e "$target" && ! -L "$target" ]]; then
    if [[ "$FORCE" -ne 1 ]]; then
      echo "Refusing to replace existing non-symlink: $target" >&2
      echo "Re-run with --force to back it up and install the repo-tracked skill." >&2
      exit 1
    fi
    backup="${target}.bak.$(date +%Y%m%d%H%M%S)"
    mv "$target" "$backup"
    echo "Backed up existing skill: $target -> $backup"
  fi
  ln -sfn "$skill_dir" "$target"
  echo "Installed skill: $skill_name -> $skill_dir"
  installed_any=1
done

if [[ "$installed_any" -eq 0 ]]; then
  echo "No repo skills found at $REPO_SKILLS_DIR" >&2
  exit 1
fi
