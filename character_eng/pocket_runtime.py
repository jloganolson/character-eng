"""Resolve or install a dedicated Pocket-TTS runtime outside the core app env."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
POCKET_TTS_SPEC = os.environ.get("POCKET_TTS_SPEC", "pocket-tts==1.1.1")


def runtime_root() -> Path:
    configured = os.environ.get("CHARACTER_ENG_RUNTIME_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    if PROJECT_ROOT == Path("/app") and Path("/workspace").exists():
        return Path("/workspace/character-eng-runtime")
    return PROJECT_ROOT / ".runtime"


def pocket_env_dir() -> Path:
    return runtime_root() / "pocket-tts"


def pocket_cache_dir() -> Path:
    return runtime_root() / ".cache" / "uv"


def resolve_uv_binary() -> str | None:
    explicit = os.environ.get("UV_BIN", "").strip()
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.append(PROJECT_ROOT / ".venv" / "bin" / "uv")
    candidates.append(Path.home() / ".local" / "bin" / "uv")
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())
    return shutil.which("uv")


def resolve_pocket_tts_binary() -> str | None:
    explicit = os.environ.get("POCKET_BIN", "").strip()
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.append(PROJECT_ROOT / ".venv" / "bin" / "pocket-tts")
    candidates.append(pocket_env_dir() / "bin" / "pocket-tts")
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())
    return shutil.which("pocket-tts")


def ensure_pocket_tts_binary() -> str | None:
    existing = resolve_pocket_tts_binary()
    if existing:
        return existing

    env_dir = pocket_env_dir()
    env_python = env_dir / "bin" / "python"
    cache_dir = pocket_cache_dir()
    env_dir.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", str(cache_dir))
    uv_bin = resolve_uv_binary()
    if not uv_bin:
        raise RuntimeError(f"uv binary not found for Pocket-TTS bootstrap (python={sys.executable})")
    subprocess.run(
        [uv_bin, "venv", str(env_dir)],
        check=True,
        env=env,
    )
    subprocess.run(
        [uv_bin, "pip", "install", "--python", str(env_python), POCKET_TTS_SPEC],
        check=True,
        env=env,
    )
    return resolve_pocket_tts_binary()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensure", action="store_true", help="Install Pocket-TTS if no binary is available.")
    parser.add_argument("--print-bin", action="store_true", help="Print the resolved Pocket-TTS binary path.")
    args = parser.parse_args()

    binary = ensure_pocket_tts_binary() if args.ensure else resolve_pocket_tts_binary()
    if args.print_bin and binary:
        print(binary)
    return 0 if binary else 1


if __name__ == "__main__":
    raise SystemExit(main())
