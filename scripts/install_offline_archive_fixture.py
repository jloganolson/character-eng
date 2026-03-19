#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


FIXTURE_ID = "greg-offline-canonical-v1"
FIXTURE_TITLE = "Greg Offline Canonical v1"
MEDIA_PATH_KEYS = {
    "assistant_audio_path",
    "conversation_audio_path",
    "playback_path",
    "user_audio_path",
    "video_frames_path",
    "video_path",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def fixture_root() -> Path:
    return repo_root() / "fixtures" / "offline_archives" / FIXTURE_ID


def history_root() -> Path:
    return repo_root() / "history"


def target_session_dir() -> Path:
    return history_root() / "sessions" / FIXTURE_ID


def detect_source_paths(manifest: dict) -> tuple[str, str]:
    media = manifest.get("media", {}) or {}
    for key in MEDIA_PATH_KEYS:
        value = str(media.get(key, "")).strip()
        if not value:
            continue
        path = Path(value)
        parts = path.parts
        if "sessions" not in parts:
            continue
        sessions_index = parts.index("sessions")
        if sessions_index + 1 >= len(parts):
            continue
        session_dir = Path(*parts[:sessions_index + 2])
        history_dir = Path(*parts[:sessions_index])
        return str(session_dir), str(history_dir)
    raise RuntimeError("fixture manifest does not include enough media paths to detect its source session")


def rewrite_archive_metadata(session_dir: Path) -> None:
    manifest_path = session_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_session_id = str(manifest.get("session_id") or "").strip()
    source_title = str(manifest.get("title") or "").strip()
    if not source_session_id:
        raise RuntimeError("fixture manifest is missing session_id")
    source_session_dir, source_history_dir = detect_source_paths(manifest)
    target_history_dir = history_root()
    target_dir = target_session_dir()

    replacements = [
        (source_session_dir, str(target_dir)),
        (source_history_dir, str(target_history_dir)),
        (source_session_id, FIXTURE_ID),
    ]
    if source_title:
        replacements.append((source_title, FIXTURE_TITLE))

    for path in session_dir.rglob("*"):
        if not path.is_file() or path.suffix not in {".json", ".jsonl"}:
            continue
        original = path.read_text(encoding="utf-8")
        updated = original
        for old, new in replacements:
            updated = updated.replace(old, new)
        if updated != original:
            path.write_text(updated, encoding="utf-8")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["session_id"] = FIXTURE_ID
    manifest["title"] = FIXTURE_TITLE
    manifest["promoted"] = True
    media = dict(manifest.get("media", {}) or {})
    media["assistant_audio_path"] = str(target_dir / "media" / "audio" / "assistant_output.wav.gz")
    media["conversation_audio_path"] = str(target_dir / "media" / "audio" / "conversation_mix.wav.gz")
    media["playback_path"] = str(target_dir / "media" / "playback.mp4")
    media["user_audio_path"] = str(target_dir / "media" / "audio" / "user_input.wav.gz")
    media["video_frames_path"] = str(target_dir / "media" / "video" / "frames.jsonl")
    media["video_path"] = str(target_dir / "media" / "video" / "session_capture.mp4")
    manifest["media"] = media
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def install_fixture(*, force: bool = False, quiet: bool = False) -> Path:
    source_dir = fixture_root()
    target_dir = target_session_dir()
    if not source_dir.exists():
        raise FileNotFoundError(f"missing offline archive fixture: {source_dir}")
    if target_dir.exists() and not force:
        if not quiet:
            print(f"offline archive already installed at {target_dir}")
        return target_dir
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir, target_dir)
    rewrite_archive_metadata(target_dir)
    if not quiet:
        print(f"installed offline archive fixture to {target_dir}")
    return target_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Install the canonical offline dashboard archive fixture into local history.")
    parser.add_argument("--force", action="store_true", help="replace an existing local install")
    parser.add_argument("--quiet", action="store_true", help="suppress success output")
    args = parser.parse_args()
    install_fixture(force=args.force, quiet=args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
