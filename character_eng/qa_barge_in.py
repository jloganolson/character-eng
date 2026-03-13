from __future__ import annotations

import argparse
import json
import threading
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from unittest.mock import MagicMock

from character_eng.history import HISTORY_ROOT
from character_eng.voice import VoiceIO


@dataclass
class BargeInSnippet:
    manifest_path: Path
    title: str
    session_id: str
    tags: list[str]
    transcript_text: str
    reason: str
    speech_started_s: float | None
    transcript_final_s: float | None
    assistant_first_audio_s: float | None
    audio_started_before_speech: bool | None
    assistant_text: str
    interrupted_text: str
    user_audio_path: Path | None
    assistant_audio_path: Path | None

    @property
    def expected_cancel(self) -> bool | None:
        lowered = {item.strip().lower() for item in self.tags}
        if "false-barge-in" in lowered:
            return False
        if "real-barge-in" in lowered or "expected-barge-in" in lowered:
            return True
        return None


@dataclass
class BargeInEvalResult:
    snippet: BargeInSnippet
    mode: str
    cancelled: bool
    barged_in: bool
    queued_text: str

    @property
    def matches_expectation(self) -> bool | None:
        if self.snippet.expected_cancel is None:
            return None
        return self.cancelled == self.snippet.expected_cancel


def _wav_duration_s(path: Path | None) -> float | None:
    if path is None or not path.exists():
        return None
    with wave.open(str(path), "rb") as wav:
        frames = wav.getnframes()
        rate = wav.getframerate() or 1
    return frames / rate


def iter_barge_in_snippets(root: Path = HISTORY_ROOT, *, required_tags: set[str] | None = None) -> Iterable[BargeInSnippet]:
    for bucket in ("moments", "pinned", "sessions"):
        bucket_dir = root / bucket
        if not bucket_dir.exists():
            continue
        for candidate in sorted(bucket_dir.iterdir()):
            manifest_path = candidate / "manifest.json"
            if not manifest_path.exists():
                continue
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if str(payload.get("capture_mode") or payload.get("bundle", {}).get("capture_mode") or "") != "barge_in_turn":
                continue
            tags = [str(item).strip() for item in payload.get("tags", []) if str(item).strip()]
            lowered = {item.lower() for item in tags}
            if required_tags and not required_tags.issubset(lowered):
                continue
            bundle = dict(payload.get("bundle") or {})
            barge = dict(bundle.get("barge_in") or {})
            media_dir = candidate / "media" / "audio"
            user_audio_path = next(
                (path for path in (media_dir / "user_input_clip.wav", media_dir / "user_input_clip.wav.gz") if path.exists()),
                None,
            )
            assistant_audio_path = next(
                (path for path in (media_dir / "assistant_output_clip.wav", media_dir / "assistant_output_clip.wav.gz") if path.exists()),
                None,
            )
            yield BargeInSnippet(
                manifest_path=manifest_path,
                title=str(payload.get("title") or candidate.name),
                session_id=str(payload.get("session_id") or candidate.name),
                tags=tags,
                transcript_text=str(barge.get("transcript_text") or ""),
                reason=str(barge.get("reason") or ""),
                speech_started_s=_float_or_none(barge.get("speech_started_s")),
                transcript_final_s=_float_or_none(barge.get("transcript_final_s")),
                assistant_first_audio_s=_float_or_none(barge.get("assistant_first_audio_s")),
                audio_started_before_speech=_bool_or_none(barge.get("audio_started_before_speech")),
                assistant_text=str(barge.get("assistant_text") or ""),
                interrupted_text=str(barge.get("interrupted_text") or ""),
                user_audio_path=user_audio_path,
                assistant_audio_path=assistant_audio_path,
            )


def _float_or_none(value) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _bool_or_none(value) -> bool | None:
    if value is None:
        return None
    return bool(value)


class _SpeakerStub:
    def __init__(self) -> None:
        self._audio_started = threading.Event()

    def flush(self) -> None:
        pass


def evaluate_barge_in_snippet(snippet: BargeInSnippet, *, aec: bool) -> BargeInEvalResult:
    voice = VoiceIO(aec=aec)
    voice._speaker = _SpeakerStub()
    voice._tts = MagicMock()
    voice._started = True
    voice._is_speaking = True
    voice._cancelled.clear()
    voice._barged_in = False
    if aec:
        voice._aec = MagicMock()
    else:
        voice._aec = None

    events: list[tuple[float, str]] = []
    if snippet.speech_started_s is not None:
        events.append((snippet.speech_started_s, "speech_started"))
    if snippet.transcript_final_s is not None and snippet.transcript_text:
        events.append((snippet.transcript_final_s, "transcript_final"))
    events.sort(key=lambda item: item[0])

    queued_text = ""
    for when_s, kind in events:
        if snippet.assistant_first_audio_s is not None and snippet.assistant_first_audio_s <= when_s:
            voice._speaker._audio_started.set()
        if kind == "speech_started":
            voice._on_turn_start()
        elif kind == "transcript_final":
            voice._on_transcript(snippet.transcript_text)
            try:
                queued_text = voice._event_queue.get_nowait()
            except Exception:
                queued_text = ""

    return BargeInEvalResult(
        snippet=snippet,
        mode="aec" if aec else "legacy",
        cancelled=voice._cancelled.is_set(),
        barged_in=voice._barged_in,
        queued_text=queued_text,
    )


def _render_result(result: BargeInEvalResult) -> str:
    expected = result.snippet.expected_cancel
    expected_text = "?" if expected is None else ("cancel" if expected else "continue")
    actual_text = "cancel" if result.cancelled else "continue"
    status = "n/a" if result.matches_expectation is None else ("ok" if result.matches_expectation else "mismatch")
    user_dur = _wav_duration_s(result.snippet.user_audio_path)
    assistant_dur = _wav_duration_s(result.snippet.assistant_audio_path)
    return " | ".join([
        result.mode,
        status,
        result.snippet.title,
        f"expected {expected_text}",
        f"actual {actual_text}",
        f"reason {result.snippet.reason or '-'}",
        f"transcript {result.snippet.transcript_text or '-'}",
        f"user_audio {user_dur:.2f}s" if user_dur is not None else "user_audio n/a",
        f"assistant_audio {assistant_dur:.2f}s" if assistant_dur is not None else "assistant_audio n/a",
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate barge-in snippets against current VoiceIO policy.")
    parser.add_argument("--root", type=Path, default=HISTORY_ROOT)
    parser.add_argument("--tag", action="append", default=[], help="Require snippet tag (can be repeated)")
    parser.add_argument("--mode", choices=["aec", "legacy", "both"], default="both")
    args = parser.parse_args()

    required_tags = {item.strip().lower() for item in args.tag if item.strip()}
    snippets = list(iter_barge_in_snippets(args.root, required_tags=required_tags or None))
    if not snippets:
        print("No barge-in snippets found.")
        return 1

    had_mismatch = False
    for snippet in snippets:
        modes = [args.mode] if args.mode in {"aec", "legacy"} else ["aec", "legacy"]
        for mode in modes:
            result = evaluate_barge_in_snippet(snippet, aec=(mode == "aec"))
            print(_render_result(result))
            if result.matches_expectation is False:
                had_mismatch = True
    return 1 if had_mismatch else 0


if __name__ == "__main__":
    raise SystemExit(main())
