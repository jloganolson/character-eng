from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
import wave
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from character_eng.person import PeopleState, Person
from character_eng.scenario import (
    ScenarioGuardrails,
    ScenarioScript,
    Stage,
    StageExit,
    VisualRequirements,
)
from character_eng.vision.client import VisionClient
from character_eng.world import Goals, Script, Beat, WorldState

HISTORY_ROOT = Path(__file__).resolve().parent.parent / "history"
SESSIONS_DIR = HISTORY_ROOT / "sessions"
MOMENTS_DIR = HISTORY_ROOT / "moments"
PINNED_DIR = HISTORY_ROOT / "pinned"
CATALOG_DIR = HISTORY_ROOT / "catalog"
DEFAULT_FREE_WARNING_GIB = 50.0
DEFAULT_VISION_CAPTURE_FPS = 2.0
DEFAULT_PLAYBACK_VIDEO_FPS = 30.0
DEFAULT_EVENT_PLAYBACK_WINDOW_S = 0.85
VISION_RECORDER_STOP_TIMEOUT_S = 5.0
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _now_ts() -> float:
    return time.time()


def _iso(ts: float | None = None) -> str:
    return datetime.fromtimestamp(ts or _now_ts(), tz=timezone.utc).isoformat()


def _display_stamp(ts: float | None = None) -> str:
    return datetime.fromtimestamp(ts or _now_ts()).strftime("%Y-%m-%d %H:%M")


def _default_session_title(character: str, ts: float | None = None) -> str:
    label = (character or "session").replace("_", " ").strip()
    return f"{_display_stamp(ts)} {label}".strip()


def _slug(text: str, fallback: str = "item") -> str:
    cleaned = _SLUG_RE.sub("-", (text or "").strip().lower()).strip("-")
    return cleaned or fallback


def _json_dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def _json_load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True) + "\n")


def _ffmpeg_available() -> bool:
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=2,
        )
        return True
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return False


def _run_ffmpeg(args: list[str]) -> bool:
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", *args],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=60,
        )
        return True
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return False


def _ffprobe_json(args: list[str]) -> dict:
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=20,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return {}
    try:
        return json.loads(proc.stdout.decode("utf-8", errors="ignore") or "{}")
    except json.JSONDecodeError:
        return {}


def _video_is_valid(path: Path | None) -> bool:
    if path is None or not path.exists():
        return False
    payload = _ffprobe_json([
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,avg_frame_rate,duration",
        "-of",
        "json",
        str(path),
    ])
    streams = payload.get("streams") or []
    if not streams:
        return False
    stream = streams[0] or {}
    codec_name = str(stream.get("codec_name") or "").strip()
    avg_rate = str(stream.get("avg_frame_rate") or "").strip()
    try:
        duration = float(stream.get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    return bool(codec_name and avg_rate and avg_rate != "0/0" and duration > 0.0)


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                pass
    return total


def _disk_free_gib(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024 ** 3)


def _ensure_roots(root: Path) -> None:
    for directory in (
        root,
        root / "sessions",
        root / "moments",
        root / "pinned",
        root / "catalog",
        root / ".discarding",
    ):
        directory.mkdir(parents=True, exist_ok=True)


def _remove_tree_async(path: Path) -> None:
    def _worker() -> None:
        shutil.rmtree(path, ignore_errors=True)
    threading.Thread(target=_worker, daemon=True, name="history-discard-cleanup").start()


def serialize_world(world: WorldState | None) -> dict:
    if world is None:
        return {}
    return {
        "static": list(world.static),
        "dynamic": dict(world.dynamic),
        "events": list(world.events),
        "pending": list(world.pending),
        "next_id": int(getattr(world, "_next_id", 1)),
    }


def deserialize_world(payload: dict | None) -> WorldState | None:
    if not payload:
        return None
    world = WorldState(
        static=list(payload.get("static", [])),
        dynamic=dict(payload.get("dynamic", {})),
        events=list(payload.get("events", [])),
        pending=list(payload.get("pending", [])),
    )
    world._next_id = int(payload.get("next_id", len(world.dynamic) + 1))
    return world


def serialize_people(people: PeopleState | None) -> dict:
    if people is None:
        return {}
    return {
        "next_id": int(getattr(people, "_next_id", 1)),
        "people": [
            {
                "person_id": person.person_id,
                "name": person.name,
                "presence": person.presence,
                "facts": dict(person.facts),
                "history": list(person.history),
                "next_fact_id": int(getattr(person, "_next_fact_id", 1)),
            }
            for person in people.people.values()
        ],
    }


def deserialize_people(payload: dict | None) -> PeopleState | None:
    if not payload:
        return None
    people = PeopleState()
    people._next_id = int(payload.get("next_id", 1))
    for entry in payload.get("people", []):
        person = Person(
            person_id=str(entry.get("person_id", "")),
            name=entry.get("name"),
            presence=str(entry.get("presence", "approaching")),
            facts=dict(entry.get("facts", {})),
            history=list(entry.get("history", [])),
        )
        person._next_fact_id = int(entry.get("next_fact_id", len(person.facts) + 1))
        people.people[person.person_id] = person
    return people


def serialize_scenario(scenario: ScenarioScript | None) -> dict:
    if scenario is None:
        return {}
    return {
        "name": scenario.name,
        "start": scenario.start,
        "current_stage": scenario.current_stage,
        "gaze_targets": list(scenario.gaze_targets),
        "guardrails": asdict(scenario.guardrails),
        "visual_requirements": asdict(scenario.visual_requirements),
        "stages": {
            name: {
                "name": stage.name,
                "goal": stage.goal,
                "exits": [asdict(exit_obj) for exit_obj in stage.exits],
                "visual_requirements": asdict(stage.visual_requirements),
            }
            for name, stage in scenario.stages.items()
        },
    }


def deserialize_scenario(payload: dict | None) -> ScenarioScript | None:
    if not payload:
        return None
    stages: dict[str, Stage] = {}
    for name, stage_payload in payload.get("stages", {}).items():
        stages[name] = Stage(
            name=str(stage_payload.get("name", name)),
            goal=str(stage_payload.get("goal", "")),
            exits=[
                StageExit(
                    condition=str(exit_payload.get("condition", "")),
                    goto=str(exit_payload.get("goto", "")),
                    label=str(exit_payload.get("label", "")),
                    visual_signals=list(exit_payload.get("visual_signals", [])),
                )
                for exit_payload in stage_payload.get("exits", [])
            ],
            visual_requirements=VisualRequirements(
                constant_questions=list(stage_payload.get("visual_requirements", {}).get("constant_questions", [])),
                constant_sam_targets=list(stage_payload.get("visual_requirements", {}).get("constant_sam_targets", [])),
            ),
        )
    return ScenarioScript(
        name=str(payload.get("name", "")),
        stages=stages,
        start=str(payload.get("start", "")),
        current_stage=str(payload.get("current_stage", payload.get("start", ""))),
        gaze_targets=list(payload.get("gaze_targets", [])),
        guardrails=ScenarioGuardrails(
            always=list(payload.get("guardrails", {}).get("always", [])),
            on_first_visible_person=list(payload.get("guardrails", {}).get("on_first_visible_person", [])),
            on_user_leaving=list(payload.get("guardrails", {}).get("on_user_leaving", [])),
        ),
        visual_requirements=VisualRequirements(
            constant_questions=list(payload.get("visual_requirements", {}).get("constant_questions", [])),
            constant_sam_targets=list(payload.get("visual_requirements", {}).get("constant_sam_targets", [])),
        ),
    )


def serialize_script(script: Script | None) -> dict:
    if script is None:
        return {}
    return {
        "beats": [
            {
                "line": beat.line,
                "intent": beat.intent,
                "condition": beat.condition,
            }
            for beat in script.beats
        ],
        "index": int(getattr(script, "_index", 0)),
    }


def deserialize_script(payload: dict | None) -> Script | None:
    if not payload:
        return None
    script = Script(
        beats=[
            Beat(
                line=str(beat.get("line", "")),
                intent=str(beat.get("intent", "")),
                condition=str(beat.get("condition", "")),
            )
            for beat in payload.get("beats", [])
        ]
    )
    script._index = int(payload.get("index", 0))
    return script


def _write_pcm_wav(path: Path, pcm: bytes, *, sample_rate: int, channels: int = 1) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)


def _read_wav_bytes(path: Path) -> tuple[dict, bytes]:
    if path.suffix == ".gz":
        raw = gzip.open(path, "rb").read()
        handle: io.BytesIO | Path = io.BytesIO(raw)
    else:
        handle = path
    with wave.open(handle, "rb") as wav:
        params = {
            "channels": wav.getnchannels(),
            "sample_width": wav.getsampwidth(),
            "sample_rate": wav.getframerate(),
        }
        frames = wav.readframes(wav.getnframes())
    return params, frames


def _read_audio_file_bytes(path: Path) -> bytes:
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as fh:
            return fh.read()
    return path.read_bytes()


def clip_audio_track(source: Path, target: Path, *, start_s: float, end_s: float) -> bool:
    if not source.exists() or end_s <= start_s:
        return False
    params, frames = _read_wav_bytes(source)
    bytes_per_frame = params["channels"] * params["sample_width"]
    sample_rate = params["sample_rate"]
    start_index = max(0, int(start_s * sample_rate)) * bytes_per_frame
    end_index = max(start_index, int(end_s * sample_rate)) * bytes_per_frame
    clipped = frames[start_index:end_index]
    if not clipped:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    _write_pcm_wav(target, clipped, sample_rate=sample_rate, channels=params["channels"])
    return True


def resolve_session_path(root: Path, ref: str) -> Path | None:
    candidate = Path(ref)
    if candidate.exists():
        if candidate.is_dir():
            return candidate
        if candidate.name == "manifest.json":
            return candidate.parent
    for base in (root / "sessions", root / "pinned", root / "moments"):
        possible = base / ref
        if possible.exists():
            return possible
        if not base.exists():
            continue
        for child in base.iterdir():
            manifest_path = child / "manifest.json"
            if not manifest_path.exists():
                continue
            try:
                manifest = _json_load(manifest_path)
            except json.JSONDecodeError:
                continue
            if manifest.get("session_id") == ref or child.name == ref:
                return child
    return None


def list_checkpoints(session_path: Path) -> list[Path]:
    checkpoints_dir = session_path / "checkpoints"
    if not checkpoints_dir.exists():
        return []
    return sorted(checkpoints_dir.glob("*.json"))


def load_checkpoint(session_path: Path, index: int | None = None) -> dict:
    direct_checkpoint = session_path / "checkpoint.json"
    if direct_checkpoint.exists():
        return _json_load(direct_checkpoint)
    checkpoints = list_checkpoints(session_path)
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoints in {session_path}")
    selected = checkpoints[-1] if index is None else checkpoints[index]
    return _json_load(selected)


def resolve_checkpoint_for_event_time(session_path: Path, event_time_s: float | None = None) -> dict:
    direct_checkpoint = session_path / "checkpoint.json"
    if direct_checkpoint.exists():
        return _json_load(direct_checkpoint)
    checkpoints = list_checkpoints(session_path)
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoints in {session_path}")
    if event_time_s is None:
        return _json_load(checkpoints[-1])
    manifest = _json_load(session_path / "manifest.json")
    started_at = float(manifest.get("started_at", 0.0) or 0.0)
    anchor = started_at + float(event_time_s or 0.0)
    selected = checkpoints[0]
    for candidate in checkpoints:
        payload = _json_load(candidate)
        if float(payload.get("timestamp", 0.0) or 0.0) <= anchor:
            selected = candidate
        else:
            break
    return _json_load(selected)


def _parse_frames_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    entries: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict) and isinstance(payload.get("frames"), list):
            entries.extend(dict(entry) for entry in payload["frames"])
        elif isinstance(payload, dict):
            entries.append(payload)
    return entries


def _parse_events_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    events: list[dict] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            continue
        payload.setdefault("seq", index)
        events.append(payload)
    return events


@dataclass
class PlaybackPlan:
    source_path: Path
    source_kind: str
    session_id: str
    checkpoint_payload: dict
    checkpoint_index: int
    checkpoint_label: str
    audio_path: Path | None
    audio_start_s: float
    video_frames: list[dict]

    def summary(self) -> dict:
        return {
            "source_path": str(self.source_path),
            "source_kind": self.source_kind,
            "session_id": self.session_id,
            "checkpoint_index": self.checkpoint_index,
            "checkpoint_label": self.checkpoint_label,
            "audio_path": str(self.audio_path) if self.audio_path is not None else "",
            "audio_start_s": self.audio_start_s,
            "video_frame_count": len(self.video_frames),
        }


def _nearest_frame_entry(session_path: Path, *, event_time_s: float) -> dict | None:
    manifest_path = session_path / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = _json_load(manifest_path)
    started_at = float(manifest.get("started_at", 0.0) or 0.0)
    anchor = started_at + float(event_time_s or 0.0)
    media_dir = session_path / "media"
    best_entry = None
    best_delta = None
    for entry in _parse_frames_jsonl(media_dir / "video" / "frames.jsonl"):
        frame_ts = float(entry.get("timestamp", 0.0) or 0.0)
        delta = abs(frame_ts - anchor)
        if best_delta is None or delta < best_delta:
            frame_path = media_dir / "video" / str(entry.get("path", ""))
            if not frame_path.exists():
                continue
            best_entry = {
                **entry,
                "abs_path": str(frame_path),
                "relative_to_event_s": round(frame_ts - anchor, 3),
            }
            best_delta = delta
    return best_entry


def _nearest_frame_entry_by_timestamp(session_path: Path, *, frame_timestamp: float) -> dict | None:
    media_dir = session_path / "media"
    best_entry = None
    best_delta = None
    for entry in _parse_frames_jsonl(media_dir / "video" / "frames.jsonl"):
        frame_ts = float(entry.get("timestamp", 0.0) or 0.0)
        delta = abs(frame_ts - float(frame_timestamp))
        if best_delta is None or delta < best_delta:
            frame_path = media_dir / "video" / str(entry.get("path", ""))
            if not frame_path.exists():
                continue
            best_entry = {
                **entry,
                "abs_path": str(frame_path),
                "relative_to_target_s": round(frame_ts - float(frame_timestamp), 3),
            }
            best_delta = delta
    return best_entry


class PlaybackRunner:
    def __init__(
        self,
        plan: PlaybackPlan,
        *,
        audio_callback: Callable[[bytes], None] | None = None,
        video_callback: Callable[[bytes, dict], None] | None = None,
        speed: float = 1.0,
        audio_chunk_ms: int = 100,
    ):
        self.plan = plan
        self.audio_callback = audio_callback
        self.video_callback = video_callback
        self.speed = max(float(speed), 0.1)
        self.audio_chunk_ms = max(int(audio_chunk_ms), 10)
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()
        self._started_at = 0.0

    def start(self) -> None:
        if self._threads:
            return
        self._stop.clear()
        self._started_at = time.monotonic()
        if self.audio_callback is not None and self.plan.audio_path is not None:
            self._threads.append(threading.Thread(target=self._run_audio, daemon=True))
        if self.video_callback is not None and self.plan.video_frames:
            self._threads.append(threading.Thread(target=self._run_video, daemon=True))
        for thread in self._threads:
            thread.start()

    def join(self, timeout: float | None = None) -> None:
        deadline = None if timeout is None else time.monotonic() + timeout
        for thread in list(self._threads):
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            thread.join(timeout=remaining)

    def stop(self) -> None:
        self._stop.set()
        self.join(timeout=2.0)
        self._threads.clear()

    @property
    def is_alive(self) -> bool:
        return any(thread.is_alive() for thread in self._threads)

    def _sleep_until(self, relative_s: float) -> bool:
        target = self._started_at + (max(relative_s, 0.0) / self.speed)
        while not self._stop.is_set():
            remaining = target - time.monotonic()
            if remaining <= 0:
                return True
            time.sleep(min(remaining, 0.02))
        return False

    def _run_audio(self) -> None:
        assert self.plan.audio_path is not None
        params, frames = _read_wav_bytes(self.plan.audio_path)
        sample_rate = int(params["sample_rate"])
        bytes_per_frame = int(params["channels"]) * int(params["sample_width"])
        start_frame = max(0, int(self.plan.audio_start_s * sample_rate))
        chunk_frames = max(1, int(sample_rate * (self.audio_chunk_ms / 1000.0)))
        offset = start_frame * bytes_per_frame
        while offset < len(frames) and not self._stop.is_set():
            frame_index = offset // bytes_per_frame
            relative_s = max(0.0, (frame_index - start_frame) / sample_rate)
            if not self._sleep_until(relative_s):
                break
            end_offset = min(len(frames), offset + chunk_frames * bytes_per_frame)
            chunk = frames[offset:end_offset]
            if chunk:
                self.audio_callback(chunk)
            offset = end_offset

    def _run_video(self) -> None:
        for entry in self.plan.video_frames:
            if self._stop.is_set():
                break
            relative_s = float(entry.get("relative_s", 0.0) or 0.0)
            if not self._sleep_until(relative_s):
                break
            frame_path = Path(entry["abs_path"])
            if not frame_path.exists():
                continue
            self.video_callback(
                frame_path.read_bytes(),
                {
                    "relative_s": relative_s,
                    "timestamp": float(entry.get("timestamp", 0.0) or 0.0),
                    "path": str(frame_path),
                    "source": str(entry.get("source", "history_playback")),
                },
            )


@dataclass
class AudioTrackWriter:
    path: Path
    sample_rate: int
    channels: int = 1
    origin_ts: float = 0.0
    _wav: wave.Wave_write | None = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    bytes_written: int = 0
    first_chunk_start_ts: float | None = None
    last_chunk_end_ts: float | None = None

    @property
    def bytes_per_frame(self) -> int:
        return self.channels * 2

    @property
    def total_frames(self) -> int:
        return self.bytes_written // self.bytes_per_frame

    def _write_silence_frames(self, frame_count: int) -> None:
        if frame_count <= 0:
            return
        silence = b"\x00" * (frame_count * self.bytes_per_frame)
        self._wav.writeframes(silence)
        self.bytes_written += len(silence)

    def append(self, pcm: bytes, *, timestamp: float | None = None) -> None:
        if not pcm:
            return
        chunk_ts = float(timestamp or _now_ts())
        chunk_frames = len(pcm) // self.bytes_per_frame
        if chunk_frames <= 0:
            return
        chunk_duration_s = chunk_frames / float(self.sample_rate)
        chunk_start_ts = max(self.origin_ts, chunk_ts - chunk_duration_s)
        with self._lock:
            if self._wav is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._wav = wave.open(str(self.path), "wb")
                self._wav.setnchannels(self.channels)
                self._wav.setsampwidth(2)
                self._wav.setframerate(self.sample_rate)
            expected_start_frame = max(0, int(round((chunk_start_ts - self.origin_ts) * self.sample_rate)))
            gap_frames = expected_start_frame - self.total_frames
            self._write_silence_frames(gap_frames)
            self._wav.writeframes(pcm)
            self.bytes_written += len(pcm)
            if self.first_chunk_start_ts is None:
                self.first_chunk_start_ts = chunk_start_ts
            self.last_chunk_end_ts = chunk_start_ts + chunk_duration_s

    def close(self) -> None:
        with self._lock:
            if self._wav is not None:
                self._wav.close()
                self._wav = None

    def compress(self) -> Path | None:
        self.close()
        if not self.path.exists():
            return None
        gz_path = self.path.with_suffix(self.path.suffix + ".gz")
        with self.path.open("rb") as src, gzip.open(gz_path, "wb", compresslevel=9) as dst:
            shutil.copyfileobj(src, dst)
        self.path.unlink()
        self.path = gz_path
        return gz_path

    def start_offset_s(self) -> float | None:
        if self.first_chunk_start_ts is None:
            return None
        return max(0.0, self.first_chunk_start_ts - self.origin_ts)


class VisionFramePoller:
    def __init__(self, service_url: str, fps: float, callback):
        self._service_url = service_url
        self._fps = max(fps, 0.1)
        self._callback = callback
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        client = VisionClient(self._service_url)
        interval = 1.0 / self._fps
        while not self._stop.wait(0.0):
            started = _now_ts()
            try:
                jpeg = client.capture_frame_jpeg(annotated=False, max_width=1280)
            except Exception:
                jpeg = b""
            if jpeg:
                self._callback(jpeg, timestamp=started, source="vision_poll")
            elapsed = _now_ts() - started
            remaining = max(0.0, interval - elapsed)
            if self._stop.wait(remaining):
                break

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None


class VisionVideoRecorder:
    def __init__(self, service_url: str, output_path: Path, *, fps: float = DEFAULT_PLAYBACK_VIDEO_FPS):
        self._service_url = service_url.rstrip("/")
        self._output_path = output_path
        self._fps = max(float(fps or DEFAULT_PLAYBACK_VIDEO_FPS), 1.0)
        self._first_frame_at: float | None = None
        self._proc: subprocess.Popen[bytes] | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._response = None
        self._response_lock = threading.Lock()

    @property
    def output_path(self) -> Path:
        return self._output_path

    @property
    def first_frame_at(self) -> float | None:
        return self._first_frame_at

    def start(self) -> None:
        if self._proc is not None or self._thread is not None or not _ffmpeg_available():
            return
        self._stop.clear()
        self._first_frame_at = None
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._output_path.unlink()
        except OSError:
            pass
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-fflags",
            "+genpts",
            "-f",
            "mjpeg",
            "-r",
            f"{self._fps:.2f}",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            str(self._output_path),
        ]
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (FileNotFoundError, OSError):
            self._proc = None
            return
        self._thread = threading.Thread(target=self._run, daemon=True, name="vision-video-recorder")
        self._thread.start()

    def _run(self) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            return
        request = urllib.request.Request(f"{self._service_url}/video_feed")
        response = None
        try:
            response = urllib.request.urlopen(request, timeout=10)
            with self._response_lock:
                self._response = response
            buffer = bytearray()
            while not self._stop.is_set():
                try:
                    chunk = response.read(65536)
                except (AttributeError, OSError, urllib.error.URLError):
                    break
                if not chunk:
                    break
                buffer.extend(chunk)
                while True:
                    start = buffer.find(b"\xff\xd8")
                    if start < 0:
                        if len(buffer) > 1_000_000:
                            del buffer[:-4]
                        break
                    end = buffer.find(b"\xff\xd9", start + 2)
                    if end < 0:
                        if start > 0:
                            del buffer[:start]
                        break
                    jpeg = bytes(buffer[start:end + 2])
                    del buffer[:end + 2]
                    if not jpeg:
                        continue
                    if self._first_frame_at is None:
                        self._first_frame_at = _now_ts()
                    try:
                        proc.stdin.write(jpeg)
                        proc.stdin.flush()
                    except OSError:
                        self._stop.set()
                        break
        except (OSError, urllib.error.URLError, ValueError):
            pass
        finally:
            with self._response_lock:
                self._response = None
            if response is not None:
                try:
                    response.close()
                except OSError:
                    pass
            if proc.stdin is not None:
                try:
                    proc.stdin.close()
                except OSError:
                    pass

    def stop(self) -> Path | None:
        if self._proc is None:
            return self._output_path if self._output_path.exists() else None
        self._stop.set()
        proc = self._proc
        self._proc = None
        with self._response_lock:
            response = self._response
            self._response = None
        if response is not None:
            try:
                response.close()
            except OSError:
                pass
        if proc.stdin is not None:
            try:
                proc.stdin.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        try:
            proc.wait(timeout=VISION_RECORDER_STOP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    pass
        return self._output_path if _video_is_valid(self._output_path) else None

    def abort(self) -> None:
        if self._proc is None and self._thread is None:
            return
        self._stop.set()
        with self._response_lock:
            response = self._response
            self._response = None
        if response is not None:
            try:
                response.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=1)
            self._thread = None
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass


def _mix_conversation_audio(
    *,
    user_audio_path: Path | None,
    assistant_audio_path: Path | None,
    output_path: Path,
) -> Path | None:
    if not _ffmpeg_available():
        return None
    inputs = [path for path in (user_audio_path, assistant_audio_path) if path is not None and path.exists()]
    if not inputs:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if len(inputs) == 1:
        if _run_ffmpeg(["-i", str(inputs[0]), "-c:a", "pcm_s16le", str(output_path)]):
            return output_path if output_path.exists() else None
        return None
    filter_complex = (
        "[0:a]aresample=48000,volume=1.0[a0];"
        "[1:a]aresample=48000,volume=1.0[a1];"
        "[a0][a1]amix=inputs=2:duration=longest:normalize=0[aout]"
    )
    ok = _run_ffmpeg([
        "-i", str(inputs[0]),
        "-i", str(inputs[1]),
        "-filter_complex", filter_complex,
        "-map", "[aout]",
        "-c:a", "pcm_s16le",
        str(output_path),
    ])
    return output_path if ok and output_path.exists() else None


def _materialize_wav_for_ffmpeg(source: Path | None, *, temp_path: Path) -> Path | None:
    if source is None or not source.exists():
        return None
    if source.suffix != ".gz":
        return source
    try:
        raw = _read_audio_file_bytes(source)
    except OSError:
        return None
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path.write_bytes(raw)
    return temp_path


def _mux_playback_media(
    *,
    video_path: Path | None,
    audio_path: Path | None,
    output_path: Path,
    video_offset_s: float = 0.0,
) -> Path | None:
    if video_path is None or not video_path.exists():
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if audio_path is None or not audio_path.exists():
        shutil.copy2(video_path, output_path)
        return output_path
    args = []
    if video_offset_s > 0.001:
        args.extend(["-itsoffset", f"{video_offset_s:.3f}"])
    args.extend([
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        str(output_path),
    ])
    ok = _run_ffmpeg(args)
    return output_path if ok and output_path.exists() else None


def _build_video_from_frames(
    *,
    frames_jsonl: Path | None,
    output_path: Path,
    fps: float = DEFAULT_PLAYBACK_VIDEO_FPS,
) -> Path | None:
    if frames_jsonl is None or not frames_jsonl.exists() or not _ffmpeg_available():
        return None
    entries: list[dict] = []
    for line in frames_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "frames" in payload:
            nested = payload.get("frames") or []
            if isinstance(nested, list):
                entries.extend(item for item in nested if isinstance(item, dict))
            continue
        if isinstance(payload, dict):
            entries.append(payload)
    entries = [entry for entry in entries if entry.get("path")]
    if not entries:
        return None
    ordered = sorted(entries, key=lambda item: float(item.get("timestamp", 0.0) or 0.0))
    concat_path = output_path.with_suffix(".frames.txt")
    concat_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for index, entry in enumerate(ordered):
        frame_path = (frames_jsonl.parent / str(entry.get("path"))).resolve()
        if not frame_path.exists():
            continue
        escaped_frame = str(frame_path).replace("'", "'\\''")
        lines.append(f"file '{escaped_frame}'")
        if index + 1 < len(ordered):
            current_ts = float(entry.get("timestamp", 0.0) or 0.0)
            next_ts = float(ordered[index + 1].get("timestamp", current_ts) or current_ts)
            duration = max(1.0 / max(fps, 1.0), next_ts - current_ts)
            lines.append(f"duration {duration:.6f}")
    last_entry = next((entry for entry in reversed(ordered) if (frames_jsonl.parent / str(entry.get('path'))).exists()), None)
    if last_entry is None:
        return None
    last_path = (frames_jsonl.parent / str(last_entry.get("path"))).resolve()
    escaped_last = str(last_path).replace("'", "'\\''")
    lines.append(f"file '{escaped_last}'")
    concat_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        ok = _run_ffmpeg([
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_path),
            "-vf",
            f"fps={max(float(fps or DEFAULT_PLAYBACK_VIDEO_FPS), 1.0):.2f}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ])
    finally:
        try:
            concat_path.unlink()
        except OSError:
            pass
    return output_path if ok and _video_is_valid(output_path) else None


class SessionArchive:
    def __init__(
        self,
        *,
        root: Path,
        session_id: str,
        character: str,
        model: str,
        model_name: str = "",
        free_warning_gib: float = DEFAULT_FREE_WARNING_GIB,
    ):
        self.root = root
        self.session_id = session_id
        self.character = character
        self.model = model
        self.model_name = model_name
        self.free_warning_gib = free_warning_gib
        self.path = root / "sessions" / session_id
        self.checkpoints_dir = self.path / "checkpoints"
        self.media_dir = self.path / "media"
        self.video_dir = self.media_dir / "video"
        self.annotations_path = self.path / "annotations.jsonl"
        self.events_path = self.path / "events.jsonl"
        self.manifest_path = self.path / "manifest.json"
        self._event_lock = threading.Lock()
        self._manifest_lock = threading.RLock()
        self._checkpoint_count = 0
        self._event_count = 0
        self._video_index = 0
        self._started_at = _now_ts()
        self._vision_poller: VisionFramePoller | None = None
        self._video_recorder: VisionVideoRecorder | None = None
        self._user_audio = AudioTrackWriter(self.media_dir / "audio" / "user_input.wav", sample_rate=16000, origin_ts=self._started_at)
        self._assistant_audio = AudioTrackWriter(self.media_dir / "audio" / "assistant_output.wav", sample_rate=24000, origin_ts=self._started_at)
        self._video_started_at: float | None = None
        self._video_first_frame_at: float | None = None
        self.path.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self._write_manifest(
            {
                "version": 1,
                "session_id": session_id,
                "character": character,
                "title": _default_session_title(character, self._started_at),
                "model": model,
                "model_name": model_name,
                "started_at": self._started_at,
                "started_at_iso": _iso(self._started_at),
                "ended_at": None,
                "ended_at_iso": "",
                "event_count": 0,
                "checkpoint_count": 0,
                "annotation_count": 0,
                "moment_count": 0,
                "promoted": False,
                "pinned_path": "",
                "record_mode": "full",
                "replay_capture_enabled": True,
                "debug_only_reason": "",
                "rewind_count": 0,
                "last_rewind": {},
                "disk_warning": self.disk_warning(),
                "free_gib": round(_disk_free_gib(self.root), 2),
                "media": {
                    "user_audio_path": "",
                    "assistant_audio_path": "",
                    "conversation_audio_path": "",
                    "video_path": "",
                    "playback_path": "",
                    "video_frames_path": str(self.video_dir / "frames.jsonl"),
                    "video_frame_count": 0,
                },
            }
        )

    def disk_warning(self) -> str:
        free_gib = _disk_free_gib(self.root)
        if free_gib < self.free_warning_gib:
            return (
                f"Low disk space: {free_gib:.1f} GiB free "
                f"(warning threshold {self.free_warning_gib:.1f} GiB)"
            )
        return ""

    def _manifest(self) -> dict:
        with self._manifest_lock:
            return _json_load(self.manifest_path)

    def _write_manifest(self, payload: dict) -> None:
        with self._manifest_lock:
            _json_dump(self.manifest_path, payload)

    def update_manifest(self, **updates: Any) -> dict:
        with self._manifest_lock:
            manifest = _json_load(self.manifest_path)
            manifest.update(updates)
            manifest["free_gib"] = round(_disk_free_gib(self.root), 2)
            manifest["disk_warning"] = self.disk_warning()
            _json_dump(self.manifest_path, manifest)
            return manifest

    def status(self) -> dict:
        manifest = self._manifest()
        return {
            "root": str(self.root),
            "session_id": self.session_id,
            "session_path": str(self.path),
            "title": str(manifest.get("title") or ""),
            "free_gib": manifest.get("free_gib", round(_disk_free_gib(self.root), 2)),
            "warning": manifest.get("disk_warning", ""),
            "event_count": manifest.get("event_count", 0),
            "checkpoint_count": manifest.get("checkpoint_count", 0),
            "annotation_count": manifest.get("annotation_count", 0),
            "moment_count": manifest.get("moment_count", 0),
            "promoted": bool(manifest.get("promoted", False)),
            "record_mode": str(manifest.get("record_mode", "full")),
            "replay_capture_enabled": bool(manifest.get("replay_capture_enabled", True)),
            "debug_only_reason": str(manifest.get("debug_only_reason", "")),
            "rewind_count": int(manifest.get("rewind_count", 0)),
            "last_rewind": dict(manifest.get("last_rewind", {}) or {}),
            "media": manifest.get("media", {}),
        }

    def replay_capture_enabled(self) -> bool:
        manifest = self._manifest()
        return bool(manifest.get("replay_capture_enabled", True))

    def mark_debug_after_rewind(
        self,
        *,
        checkpoint_payload: dict,
        source_session_id: str,
        reason: str,
        source_event_time_s: float | None = None,
    ) -> dict:
        manifest = self._manifest()
        rewind_count = int(manifest.get("rewind_count", 0)) + 1
        rewind_meta = {
            "at_iso": _iso(),
            "source_session_id": source_session_id,
            "checkpoint_index": int(checkpoint_payload.get("checkpoint_index", 0)),
            "checkpoint_label": str(checkpoint_payload.get("label", "")),
            "checkpoint_timestamp": float(checkpoint_payload.get("timestamp", 0.0) or 0.0),
            "source_event_time_s": None if source_event_time_s is None else float(source_event_time_s),
            "reason": reason,
        }
        return self.update_manifest(
            record_mode="debug_only",
            replay_capture_enabled=False,
            debug_only_reason=reason,
            rewind_count=rewind_count,
            last_rewind=rewind_meta,
        )

    def rename(self, title: str) -> dict:
        cleaned = str(title or "").strip()
        if not cleaned:
            raise ValueError("title is required")
        return self.update_manifest(title=cleaned)

    def record_event(self, event_type: str, data: dict, *, timestamp: float | None = None) -> None:
        manifest = self._manifest()
        if manifest.get("ended_at") and event_type != "session_end":
            return
        payload = {
            "timestamp": float(timestamp or _now_ts()),
            "timestamp_iso": _iso(timestamp),
            "type": event_type,
            "data": data,
        }
        with self._event_lock:
            _append_jsonl(self.events_path, payload)
            self._event_count += 1
        self.update_manifest(event_count=self._event_count)

    def capture_checkpoint(
        self,
        *,
        label: str,
        character: str,
        session_snapshot: dict,
        world: WorldState | None,
        people: PeopleState | None,
        scenario: ScenarioScript | None,
        script: Script | None,
        goals: Goals | None,
        log_entries: list[dict],
        context_version: int,
        had_user_input: bool,
    ) -> Path:
        if not self.replay_capture_enabled():
            return self.checkpoints_dir / "disabled.debug-only"
        with self._manifest_lock:
            checkpoint_index = self._checkpoint_count
            path = self.checkpoints_dir / f"{checkpoint_index:04d}_{_slug(label, 'checkpoint')}.json"
        payload = {
            "version": 1,
            "session_id": self.session_id,
            "character": character,
            "label": label,
            "checkpoint_index": checkpoint_index,
            "timestamp": _now_ts(),
            "timestamp_iso": _iso(),
            "context_version": context_version,
            "had_user_input": had_user_input,
            "session": session_snapshot,
            "world": serialize_world(world),
            "people": serialize_people(people),
            "scenario": serialize_scenario(scenario),
            "script": serialize_script(script),
            "goals": {"long_term": goals.long_term if goals is not None else ""},
            "log_entries": log_entries,
        }
        _json_dump(path, payload)
        with self._manifest_lock:
            self._checkpoint_count += 1
            next_count = self._checkpoint_count
        self.update_manifest(checkpoint_count=next_count)
        return path

    def record_user_audio(self, pcm: bytes) -> None:
        if not self.replay_capture_enabled():
            return
        self._user_audio.append(pcm, timestamp=_now_ts())

    def record_assistant_audio(self, pcm: bytes) -> None:
        if not self.replay_capture_enabled():
            return
        self._assistant_audio.append(pcm, timestamp=_now_ts())

    def record_video_frame(self, jpeg_bytes: bytes, *, timestamp: float | None = None, source: str = "vision") -> Path | None:
        if not self.replay_capture_enabled():
            return None
        if not jpeg_bytes:
            return None
        with self._manifest_lock:
            frame_index = self._video_index
            frame_name = f"frame_{frame_index:06d}.jpg"
            frame_path = self.video_dir / frame_name
            frame_path.write_bytes(jpeg_bytes)
            _append_jsonl(
                self.video_dir / "frames.jsonl",
                {
                    "frame_index": frame_index,
                    "timestamp": float(timestamp or _now_ts()),
                    "timestamp_iso": _iso(timestamp),
                    "source": source,
                    "path": frame_name,
                },
            )
            self._video_index += 1
            manifest = _json_load(self.manifest_path)
            media = dict(manifest.get("media", {}))
            media["video_frame_count"] = self._video_index
            media["video_frames_path"] = str(self.video_dir / "frames.jsonl")
        self.update_manifest(media=media)
        return frame_path

    def start_vision_capture(
        self,
        service_url: str,
        *,
        fps: float = DEFAULT_VISION_CAPTURE_FPS,
        playback_video_fps: float = DEFAULT_PLAYBACK_VIDEO_FPS,
    ) -> None:
        if self._vision_poller is not None:
            if self._video_recorder is not None:
                return
        if self._vision_poller is None:
            self._vision_poller = VisionFramePoller(service_url, fps, self.record_video_frame)
            self._vision_poller.start()
        if self._video_recorder is None:
            self._video_started_at = _now_ts()
            self._video_first_frame_at = None
            self._video_recorder = VisionVideoRecorder(
                service_url,
                self.media_dir / "video" / "session_capture.mp4",
                fps=playback_video_fps,
            )
            self._video_recorder.start()

    def stop_vision_capture(self, *, force: bool = False) -> None:
        if self._vision_poller is not None:
            self._vision_poller.stop()
            self._vision_poller = None
        if self._video_recorder is not None:
            recorder = self._video_recorder
            if force:
                recorder.abort()
            else:
                recorder.stop()
                if recorder.first_frame_at is not None:
                    self._video_first_frame_at = recorder.first_frame_at
            self._video_recorder = None

    def promote(self, kind: str = "pinned") -> Path:
        destination_root = self.root / ("pinned" if kind == "pinned" else "moments")
        destination_root.mkdir(parents=True, exist_ok=True)
        destination = destination_root / self.path.name
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(self.path, destination)
        self.update_manifest(promoted=True, pinned_path=str(destination))
        return destination

    def save_annotation(self, payload: dict, *, catalog_path: Path, auto_promote: bool = True) -> Path:
        entry = {
            "saved_at": _iso(),
            "session_id": self.session_id,
            "character": self.character,
            **payload,
        }
        _append_jsonl(self.annotations_path, entry)
        _append_jsonl(catalog_path, entry)
        manifest = self._manifest()
        annotation_count = int(manifest.get("annotation_count", 0)) + 1
        self.update_manifest(annotation_count=annotation_count)
        if auto_promote:
            self.promote("pinned")
        return self.annotations_path

    def list_annotations(self) -> list[dict]:
        if not self.annotations_path.exists():
            return []
        entries: list[dict] = []
        for line in self.annotations_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return entries

    def capture_moment(
        self,
        *,
        title: str,
        event_time_s: float | None,
        window_start_s: float | None = None,
        window_end_s: float | None = None,
        bundle: dict | None,
        tags: list[str] | None = None,
        moments_root: Path,
        window_before_s: float = 10.0,
        window_after_s: float = 10.0,
    ) -> Path:
        started_at = float(self._manifest().get("started_at", self._started_at))
        has_explicit_window = window_start_s is not None and window_end_s is not None
        if has_explicit_window:
            clip_start_s = max(0.0, float(window_start_s or 0.0))
            clip_end_s = max(clip_start_s, float(window_end_s or clip_start_s))
            capture_start_ts = started_at + clip_start_s
            capture_end_ts = started_at + clip_end_s
            anchor = capture_start_ts
        else:
            anchor = started_at + float(event_time_s or 0.0)
            clip_start_s = max(0.0, anchor - started_at - window_before_s)
            clip_end_s = max(clip_start_s, anchor - started_at + window_after_s)
            capture_start_ts = started_at + clip_start_s
            capture_end_ts = started_at + clip_end_s
        moment_dir = moments_root / f"{self.session_id}_{_slug(title, 'moment')}_{int(_now_ts())}"
        moment_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = None
        checkpoints = list_checkpoints(self.path)
        if checkpoints:
            checkpoint = checkpoints[-1]
            for candidate in checkpoints:
                candidate_payload = _json_load(candidate)
                if float(candidate_payload.get("timestamp", 0.0)) <= capture_start_ts:
                    checkpoint = candidate
        if checkpoint is not None:
            shutil.copy2(checkpoint, moment_dir / "checkpoint.json")
        if self.annotations_path.exists():
            shutil.copy2(self.annotations_path, moment_dir / "annotations.jsonl")

        for source_name, target_name in (
            ("user_input.wav.gz", "user_input_clip.wav"),
            ("user_input.wav", "user_input_clip.wav"),
            ("assistant_output.wav.gz", "assistant_output_clip.wav"),
            ("assistant_output.wav", "assistant_output_clip.wav"),
        ):
            source = self.media_dir / "audio" / source_name
            if source.exists():
                clip_audio_track(
                    source,
                    moment_dir / "media" / "audio" / target_name,
                    start_s=clip_start_s,
                    end_s=clip_end_s,
                )
        conversation_mix_candidates = (
            self.media_dir / "audio" / "conversation_mix.wav.gz",
            self.media_dir / "audio" / "conversation_mix.wav",
        )
        conversation_mix = next((path for path in conversation_mix_candidates if path.exists()), None)
        if conversation_mix is not None:
            clip_audio_track(
                conversation_mix,
                moment_dir / "media" / "audio" / "conversation_mix.wav",
                start_s=clip_start_s,
                end_s=clip_end_s,
            )

        frame_entries: list[dict] = []
        frames_jsonl = self.video_dir / "frames.jsonl"
        if frames_jsonl.exists():
            entries = [
                json.loads(line)
                for line in frames_jsonl.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            selected = [
                entry
                for entry in entries
                if capture_start_ts <= float(entry.get("timestamp", 0.0)) <= capture_end_ts
            ]
            for entry in selected:
                src = self.video_dir / str(entry.get("path", ""))
                if not src.exists():
                    continue
                dst = moment_dir / "media" / "video" / src.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                copied = dict(entry)
                copied["path"] = str(Path("media") / "video" / src.name)
                frame_entries.append(copied)
        if frame_entries:
            _append_jsonl(moment_dir / "media" / "video" / "frames.jsonl", {"frames": frame_entries})

        metadata = {
            "version": 1,
            "type": "snippet",
            "title": title,
            "session_id": self.session_id,
            "source_session_path": str(self.path),
            "captured_at": _iso(),
            "event_time_s": event_time_s,
            "window_start_s": clip_start_s if has_explicit_window else None,
            "window_end_s": clip_end_s if has_explicit_window else None,
            "window_before_s": window_before_s,
            "window_after_s": window_after_s,
            "bundle": bundle or {},
            "tags": [str(item).strip() for item in (tags or []) if str(item).strip()],
            "capture_mode": str((bundle or {}).get("capture_mode", "turn_context")),
            "vision_inputs": dict((bundle or {}).get("vision_inputs", {}) or {}),
        }
        _json_dump(moment_dir / "manifest.json", metadata)
        manifest = self._manifest()
        self.update_manifest(moment_count=int(manifest.get("moment_count", 0)) + 1)
        return moment_dir

    def finalize(self) -> dict:
        self.stop_vision_capture()
        raw_user_audio_path = self._user_audio.path if self._user_audio.path.exists() else None
        raw_assistant_audio_path = self._assistant_audio.path if self._assistant_audio.path.exists() else None
        raw_video_path = self.media_dir / "video" / "session_capture.mp4"
        frames_jsonl = self.video_dir / "frames.jsonl"
        user_audio_start_s = self._user_audio.start_offset_s()
        assistant_audio_start_s = self._assistant_audio.start_offset_s()
        video_anchor_ts = self._video_first_frame_at or self._video_started_at
        video_start_s = max(0.0, (video_anchor_ts or self._started_at) - self._started_at) if (
            self._video_started_at is not None or raw_video_path.exists()
        ) else None
        mixed_audio_path = _mix_conversation_audio(
            user_audio_path=raw_user_audio_path,
            assistant_audio_path=raw_assistant_audio_path,
            output_path=self.media_dir / "audio" / "conversation_mix.wav",
        )
        video_for_mux = raw_video_path if _video_is_valid(raw_video_path) else None
        if video_for_mux is None:
            video_for_mux = _build_video_from_frames(
                frames_jsonl=frames_jsonl if frames_jsonl.exists() else None,
                output_path=self.media_dir / "video" / "session_capture_fallback.mp4",
                fps=DEFAULT_PLAYBACK_VIDEO_FPS,
            )
        playback_path = _mux_playback_media(
            video_path=video_for_mux,
            audio_path=mixed_audio_path,
            output_path=self.media_dir / "playback.mp4",
            video_offset_s=float(video_start_s or 0.0),
        )
        user_audio_path = self._user_audio.compress()
        assistant_audio_path = self._assistant_audio.compress()
        mixed_audio_gz_path = None
        if mixed_audio_path is not None and mixed_audio_path.exists():
            mixed_audio_gz_path = mixed_audio_path.with_suffix(mixed_audio_path.suffix + ".gz")
            with mixed_audio_path.open("rb") as src, gzip.open(mixed_audio_gz_path, "wb", compresslevel=9) as dst:
                shutil.copyfileobj(src, dst)
            mixed_audio_path.unlink()
        manifest = self._manifest()
        media = dict(manifest.get("media", {}))
        media["user_audio_path"] = str(user_audio_path) if user_audio_path else ""
        media["assistant_audio_path"] = str(assistant_audio_path) if assistant_audio_path else ""
        media["conversation_audio_path"] = str(mixed_audio_gz_path) if mixed_audio_gz_path else ""
        media["video_path"] = str(video_for_mux) if video_for_mux and video_for_mux.exists() else ""
        media["playback_path"] = str(playback_path) if playback_path else ""
        media["user_audio_start_s"] = float(user_audio_start_s) if user_audio_start_s is not None else None
        media["assistant_audio_start_s"] = float(assistant_audio_start_s) if assistant_audio_start_s is not None else None
        media["video_start_s"] = float(video_start_s) if video_start_s is not None else None
        media["playback_origin_s"] = 0.0
        media["video_frames_path"] = str(self.video_dir / "frames.jsonl")
        media["video_frame_count"] = self._video_index
        return self.update_manifest(
            ended_at=_now_ts(),
            ended_at_iso=_iso(),
            media=media,
        )


class HistoryService:
    def __init__(self, *, root: Path = HISTORY_ROOT, free_warning_gib: float = DEFAULT_FREE_WARNING_GIB):
        self.root = root
        self.free_warning_gib = free_warning_gib
        self._lock = threading.Lock()
        self.current: SessionArchive | None = None
        _ensure_roots(root)

    def status(self) -> dict:
        _ensure_roots(self.root)
        free_gib = round(_disk_free_gib(self.root), 2)
        sessions_dir = self.root / "sessions"
        pinned_dir = self.root / "pinned"
        moments_dir = self.root / "moments"
        session = self.current.status() if self.current is not None else {}
        return {
            "root": str(self.root),
            "free_gib": free_gib,
            "warning_threshold_gib": self.free_warning_gib,
            "warning": (
                f"Low disk space: {free_gib:.1f} GiB free "
                f"(warning threshold {self.free_warning_gib:.1f} GiB)"
                if free_gib < self.free_warning_gib
                else ""
            ),
            "sessions_count": len(list(sessions_dir.iterdir())) if sessions_dir.exists() else 0,
            "pinned_count": len(list(pinned_dir.iterdir())) if pinned_dir.exists() else 0,
            "moments_count": len(list(moments_dir.iterdir())) if moments_dir.exists() else 0,
            "sessions_bytes": _dir_size_bytes(sessions_dir),
            "pinned_bytes": _dir_size_bytes(pinned_dir),
            "moments_bytes": _dir_size_bytes(moments_dir),
            "current_session": session,
        }

    def list_archives(self, limit: int = 50) -> list[dict]:
        items: list[dict] = []
        for bucket in ("sessions", "pinned", "moments"):
            base = self.root / bucket
            if not base.exists():
                continue
            for child in base.iterdir():
                manifest_path = child / "manifest.json"
                if not manifest_path.exists():
                    continue
                try:
                    manifest = _json_load(manifest_path)
                except json.JSONDecodeError:
                    continue
                items.append(
                    {
                        "bucket": bucket,
                        "path": str(child),
                        "name": child.name,
                        "ref": child.name,
                        "session_id": manifest.get("session_id", ""),
                        "character": manifest.get("character", ""),
                        "title": manifest.get("title", ""),
                        "label": manifest.get("title") or manifest.get("character") or child.name,
                        "tags": list(manifest.get("tags", []) or []),
                        "started_at": manifest.get("started_at", 0.0),
                        "started_at_iso": manifest.get("started_at_iso") or manifest.get("captured_at", ""),
                        "checkpoint_count": manifest.get("checkpoint_count", 0),
                        "annotation_count": manifest.get("annotation_count", 0),
                        "moment_count": manifest.get("moment_count", 0),
                    }
                )
        items.sort(key=lambda item: str(item.get("started_at_iso") or item.get("started_at") or ""), reverse=True)
        return items[:limit]

    def list_annotations(self, session_id: str | None = None) -> list[dict]:
        session = self._resolve_active_or_ref(session_id)
        return session.list_annotations()

    def start_session(self, *, session_id: str, character: str, model: str, model_name: str = "") -> SessionArchive:
        with self._lock:
            self.current = SessionArchive(
                root=self.root,
                session_id=session_id,
                character=character,
                model=model,
                model_name=model_name,
                free_warning_gib=self.free_warning_gib,
            )
            return self.current

    def finalize_current(self) -> dict | None:
        with self._lock:
            if self.current is None:
                return None
            return self.current.finalize()

    def discard_current(self) -> dict | None:
        with self._lock:
            if self.current is None:
                return None
            archive = self.current
            archive.stop_vision_capture(force=True)
            archive._user_audio.close()
            archive._assistant_audio.close()
            path = archive.path
            session_id = archive.session_id
            self.current = None
            cleanup_path = path
            try:
                cleanup_path = self.root / ".discarding" / f"{session_id}_{int(time.time() * 1000)}"
                cleanup_path.parent.mkdir(parents=True, exist_ok=True)
                path.rename(cleanup_path)
            except OSError:
                cleanup_path = path
            _remove_tree_async(cleanup_path)
            return {
                "session_id": session_id,
                "path": str(path),
            }

    def record_event(self, event_type: str, data: dict, *, timestamp: float | None = None) -> None:
        with self._lock:
            if self.current is None:
                return
            self.current.record_event(event_type, data, timestamp=timestamp)

    def capture_checkpoint(self, **kwargs: Any) -> Path | None:
        with self._lock:
            if self.current is None:
                return None
            return self.current.capture_checkpoint(**kwargs)

    def record_user_audio(self, pcm: bytes) -> None:
        with self._lock:
            if self.current is not None:
                self.current.record_user_audio(pcm)

    def record_assistant_audio(self, pcm: bytes) -> None:
        with self._lock:
            if self.current is not None:
                self.current.record_assistant_audio(pcm)

    def record_video_frame(self, jpeg_bytes: bytes, *, timestamp: float | None = None, source: str = "vision") -> Path | None:
        with self._lock:
            if self.current is None:
                return None
            return self.current.record_video_frame(jpeg_bytes, timestamp=timestamp, source=source)

    def start_vision_capture(
        self,
        service_url: str,
        *,
        fps: float = DEFAULT_VISION_CAPTURE_FPS,
        playback_video_fps: float = DEFAULT_PLAYBACK_VIDEO_FPS,
    ) -> None:
        with self._lock:
            if self.current is not None:
                self.current.start_vision_capture(
                    service_url,
                    fps=fps,
                    playback_video_fps=playback_video_fps,
                )

    def stop_vision_capture(self) -> None:
        with self._lock:
            if self.current is not None:
                self.current.stop_vision_capture()

    def save_annotation(self, payload: dict, *, session_id: str | None = None, auto_promote: bool = True) -> Path:
        session = self._resolve_active_or_ref(session_id)
        return session.save_annotation(
            payload,
            catalog_path=self.root / "catalog" / "annotations.jsonl",
            auto_promote=auto_promote,
        )

    def capture_moment(self, payload: dict) -> Path:
        session = self._resolve_active_or_ref(payload.get("session_id"))
        return session.capture_moment(
            title=str(payload.get("title") or payload.get("label") or "moment"),
            event_time_s=payload.get("event_time_s"),
            window_start_s=payload.get("window_start_s"),
            window_end_s=payload.get("window_end_s"),
            bundle=payload.get("bundle"),
            tags=list(payload.get("tags") or []),
            moments_root=self.root / "moments",
            window_before_s=float(payload.get("window_before_s", 10.0)),
            window_after_s=float(payload.get("window_after_s", 10.0)),
        )

    def capture_snippet(self, payload: dict) -> Path:
        bundle = dict(payload.get("bundle") or {})
        bundle["capture_mode"] = str(payload.get("capture_mode") or bundle.get("capture_mode") or "turn_context")
        payload = {**payload, "bundle": bundle}
        return self.capture_moment(payload)

    def promote(self, *, session_id: str | None = None, kind: str = "pinned") -> Path:
        session = self._resolve_active_or_ref(session_id)
        return session.promote(kind)

    def rename(self, *, session_id: str | None = None, title: str) -> dict:
        session = self._resolve_active_or_ref(session_id)
        return session.rename(title)

    def load_checkpoint_payload(self, session_ref: str, checkpoint_index: int | None = None) -> dict:
        session_path = resolve_session_path(self.root, session_ref)
        if session_path is None:
            raise FileNotFoundError(f"unknown session: {session_ref}")
        return load_checkpoint(session_path, checkpoint_index)

    def resolve_checkpoint_for_event(self, session_ref: str, event_time_s: float | None = None) -> dict:
        session_path = resolve_session_path(self.root, session_ref)
        if session_path is None:
            raise FileNotFoundError(f"unknown session: {session_ref}")
        return resolve_checkpoint_for_event_time(session_path, event_time_s)

    def nearest_frame_for_event(self, session_ref: str, event_time_s: float | None = None) -> dict:
        session_path = resolve_session_path(self.root, session_ref)
        if session_path is None:
            raise FileNotFoundError(f"unknown session: {session_ref}")
        if event_time_s is None:
            return {}
        return _nearest_frame_entry(session_path, event_time_s=float(event_time_s)) or {}

    def nearest_frame_for_timestamp(self, session_ref: str, frame_timestamp: float | None = None) -> dict:
        session_path = resolve_session_path(self.root, session_ref)
        if session_path is None:
            raise FileNotFoundError(f"unknown session: {session_ref}")
        if frame_timestamp is None:
            return {}
        return _nearest_frame_entry_by_timestamp(session_path, frame_timestamp=float(frame_timestamp)) or {}

    def audio_path_for_session(self, session_ref: str) -> Path | None:
        session_path = resolve_session_path(self.root, session_ref)
        if session_path is None:
            raise FileNotFoundError(f"unknown session: {session_ref}")
        manifest = _json_load(session_path / "manifest.json")
        media = manifest.get("media", {}) or {}
        candidates = []
        mixed = str(media.get("conversation_audio_path", "")).strip()
        if mixed:
            candidates.append(Path(mixed))
        stored = str(media.get("user_audio_path", "")).strip()
        if stored:
            candidates.append(Path(stored))
        candidates.extend([
            session_path / "media" / "audio" / "conversation_mix.wav.gz",
            session_path / "media" / "audio" / "conversation_mix.wav",
            session_path / "media" / "audio" / "user_input.wav.gz",
            session_path / "media" / "audio" / "user_input.wav",
            session_path / "media" / "audio" / "user_input_clip.wav.gz",
            session_path / "media" / "audio" / "user_input_clip.wav",
        ])
        return next((path for path in candidates if path.exists()), None)

    def audio_bytes_for_session(self, session_ref: str) -> bytes:
        path = self.audio_path_for_session(session_ref)
        if path is None:
            raise FileNotFoundError(f"no user audio for session: {session_ref}")
        return _read_audio_file_bytes(path)

    def playback_media_path_for_session(self, session_ref: str) -> Path | None:
        session_path = resolve_session_path(self.root, session_ref)
        if session_path is None:
            raise FileNotFoundError(f"unknown session: {session_ref}")
        manifest = _json_load(session_path / "manifest.json")
        media = manifest.get("media", {}) or {}
        candidates = []
        stored = str(media.get("playback_path", "")).strip()
        if stored:
            candidates.append(Path(stored))
        candidates.append(session_path / "media" / "playback.mp4")
        existing = next((path for path in candidates if _video_is_valid(path)), None)
        if existing is not None:
            return existing
        repaired = self._ensure_playback_media(session_path)
        return repaired if repaired and _video_is_valid(repaired) else None

    def list_events(self, session_ref: str) -> dict:
        session_path = resolve_session_path(self.root, session_ref)
        if session_path is None:
            raise FileNotFoundError(f"unknown session: {session_ref}")
        manifest = _json_load(session_path / "manifest.json")
        media = dict(manifest.get("media", {}) or {})
        playback_path = self.playback_media_path_for_session(session_ref)
        return {
            "session_id": str(manifest.get("session_id") or session_path.name),
            "title": str(manifest.get("title") or session_path.name),
            "character": str(manifest.get("character") or ""),
            "source_kind": str(manifest.get("type") or session_path.parent.name.rstrip("s") or "session"),
            "media": {
                **media,
                "playback_available": bool(playback_path is not None),
                "playback_path": str(playback_path) if playback_path is not None else str(media.get("playback_path") or ""),
            },
            "events": _parse_events_jsonl(session_path / "events.jsonl"),
        }

    def _ensure_playback_media(self, session_path: Path) -> Path | None:
        session_path = session_path.resolve()
        manifest_path = session_path / "manifest.json"
        if not manifest_path.exists():
            return None
        manifest = _json_load(manifest_path)
        media = dict(manifest.get("media", {}) or {})
        output_path = session_path / "media" / "playback.mp4"
        if _video_is_valid(output_path):
            return output_path

        mixed_audio_candidates = []
        mixed = str(media.get("conversation_audio_path", "")).strip()
        if mixed:
            mixed_audio_candidates.append(Path(mixed))
        mixed_audio_candidates.extend([
            session_path / "media" / "audio" / "conversation_mix.wav.gz",
            session_path / "media" / "audio" / "conversation_mix.wav",
        ])
        audio_source = next((path for path in mixed_audio_candidates if path.exists()), None)
        temp_audio_path = session_path / "media" / "audio" / ".conversation_mix_for_mux.wav"
        audio_path = _materialize_wav_for_ffmpeg(audio_source, temp_path=temp_audio_path)

        raw_video_candidates = []
        stored_video = str(media.get("video_path", "")).strip()
        if stored_video:
            raw_video_candidates.append(Path(stored_video))
        raw_video_candidates.extend([
            session_path / "media" / "video" / "session_capture.mp4",
            session_path / "media" / "video" / "session_capture_fallback.mp4",
        ])
        raw_video = next((path for path in raw_video_candidates if _video_is_valid(path)), None)
        if raw_video is None:
            raw_video = _build_video_from_frames(
                frames_jsonl=session_path / "media" / "video" / "frames.jsonl",
                output_path=session_path / "media" / "video" / "session_capture_fallback.mp4",
                fps=DEFAULT_PLAYBACK_VIDEO_FPS,
            )
        try:
            playback = _mux_playback_media(
                video_path=raw_video,
                audio_path=audio_path,
                output_path=output_path,
                video_offset_s=float(media.get("video_start_s") or 0.0),
            )
        finally:
            if temp_audio_path.exists():
                try:
                    temp_audio_path.unlink()
                except OSError:
                    pass
        if playback is None or not _video_is_valid(playback):
            return None
        media["playback_path"] = str(playback)
        if raw_video is not None and raw_video.exists():
            media["video_path"] = str(raw_video)
        _json_dump(manifest_path, {**manifest, "media": media})
        return playback

    def prepare_playback(self, session_ref: str, checkpoint_index: int | None = None) -> PlaybackPlan:
        session_path = resolve_session_path(self.root, session_ref)
        if session_path is None:
            raise FileNotFoundError(f"unknown session: {session_ref}")

        manifest = _json_load(session_path / "manifest.json")
        checkpoint_payload = load_checkpoint(session_path, checkpoint_index)
        source_kind = str(manifest.get("type") or session_path.parent.name.rstrip("s") or "session")
        session_id = str(manifest.get("session_id") or session_path.name)
        checkpoint_idx = int(checkpoint_payload.get("checkpoint_index", 0))
        checkpoint_label = str(checkpoint_payload.get("label", "checkpoint"))
        media_dir = session_path / "media"

        audio_path = None
        audio_start_s = 0.0
        video_frames: list[dict] = []

        if source_kind in {"moment", "snippet"}:
            candidate_audio_paths = (
                media_dir / "audio" / "user_input_clip.wav",
                media_dir / "audio" / "user_input_clip.wav.gz",
            )
            audio_path = next((path for path in candidate_audio_paths if path.exists()), None)
            source_session_path = Path(str(manifest.get("source_session_path", ""))) if manifest.get("source_session_path") else None
            media_start_ts = None
            if source_session_path is not None and (source_session_path / "manifest.json").exists():
                source_manifest = _json_load(source_session_path / "manifest.json")
                if manifest.get("window_start_s") is not None:
                    media_start_ts = float(source_manifest.get("started_at", 0.0)) + float(manifest.get("window_start_s", 0.0) or 0.0)
                else:
                    media_start_ts = (
                        float(source_manifest.get("started_at", 0.0))
                        + float(manifest.get("event_time_s", 0.0) or 0.0)
                        - float(manifest.get("window_before_s", 0.0) or 0.0)
                    )
            frame_entries = _parse_frames_jsonl(media_dir / "video" / "frames.jsonl")
            for index, entry in enumerate(frame_entries):
                frame_path = session_path / str(entry.get("path", ""))
                if not frame_path.exists():
                    continue
                if media_start_ts is not None:
                    relative_s = max(0.0, float(entry.get("timestamp", media_start_ts)) - media_start_ts)
                else:
                    relative_s = float(index) * 0.1
                video_frames.append({
                    **entry,
                    "relative_s": relative_s,
                    "abs_path": str(frame_path),
                })
        else:
            media = manifest.get("media", {}) or {}
            candidate_audio_paths = []
            stored_audio = str(media.get("user_audio_path", "")).strip()
            if stored_audio:
                candidate_audio_paths.append(Path(stored_audio))
            candidate_audio_paths.extend([
                media_dir / "audio" / "user_input.wav.gz",
                media_dir / "audio" / "user_input.wav",
            ])
            audio_path = next((path for path in candidate_audio_paths if path.exists()), None)

            started_at = float(manifest.get("started_at", 0.0) or 0.0)
            checkpoint_ts = float(checkpoint_payload.get("timestamp", started_at) or started_at)
            audio_start_s = max(0.0, checkpoint_ts - started_at)

            frame_entries = _parse_frames_jsonl(media_dir / "video" / "frames.jsonl")
            for entry in frame_entries:
                frame_ts = float(entry.get("timestamp", 0.0) or 0.0)
                if frame_ts < checkpoint_ts:
                    continue
                frame_path = media_dir / "video" / str(entry.get("path", ""))
                if not frame_path.exists():
                    continue
                video_frames.append({
                    **entry,
                    "relative_s": max(0.0, frame_ts - checkpoint_ts),
                    "abs_path": str(frame_path),
                })

        return PlaybackPlan(
            source_path=session_path,
            source_kind=source_kind,
            session_id=session_id,
            checkpoint_payload=checkpoint_payload,
            checkpoint_index=checkpoint_idx,
            checkpoint_label=checkpoint_label,
            audio_path=audio_path,
            audio_start_s=audio_start_s,
            video_frames=video_frames,
        )

    def prepare_event_playback(
        self,
        session_ref: str,
        *,
        event_time_s: float,
        window_before_s: float = DEFAULT_EVENT_PLAYBACK_WINDOW_S,
        window_after_s: float = DEFAULT_EVENT_PLAYBACK_WINDOW_S,
        include_audio: bool = False,
    ) -> PlaybackPlan:
        session_path = resolve_session_path(self.root, session_ref)
        if session_path is None:
            raise FileNotFoundError(f"unknown session: {session_ref}")
        manifest = _json_load(session_path / "manifest.json")
        checkpoint_payload = resolve_checkpoint_for_event_time(session_path, event_time_s)
        checkpoint_idx = int(checkpoint_payload.get("checkpoint_index", 0))
        checkpoint_label = str(checkpoint_payload.get("label", "checkpoint"))
        started_at = float(manifest.get("started_at", 0.0) or 0.0)
        anchor = started_at + float(event_time_s or 0.0)
        media_dir = session_path / "media"

        audio_path = None
        if include_audio:
            candidate_audio_paths = [
                Path(str((manifest.get("media", {}) or {}).get("user_audio_path", ""))),
                media_dir / "audio" / "user_input.wav.gz",
                media_dir / "audio" / "user_input.wav",
            ]
            audio_path = next((path for path in candidate_audio_paths if path and path.exists()), None)

        video_frames: list[dict] = []
        for entry in _parse_frames_jsonl(media_dir / "video" / "frames.jsonl"):
            frame_ts = float(entry.get("timestamp", 0.0) or 0.0)
            if frame_ts < anchor - window_before_s or frame_ts > anchor + window_after_s:
                continue
            frame_path = media_dir / "video" / str(entry.get("path", ""))
            if not frame_path.exists():
                continue
            video_frames.append({
                **entry,
                "relative_s": max(0.0, frame_ts - anchor + window_before_s),
                "abs_path": str(frame_path),
            })

        return PlaybackPlan(
            source_path=session_path,
            source_kind="event",
            session_id=str(manifest.get("session_id") or session_path.name),
            checkpoint_payload=checkpoint_payload,
            checkpoint_index=checkpoint_idx,
            checkpoint_label=checkpoint_label,
            audio_path=audio_path,
            audio_start_s=max(0.0, float(event_time_s or 0.0) - window_before_s) if audio_path is not None else 0.0,
            video_frames=video_frames,
        )

    def prune_unpromoted(
        self,
        *,
        older_than_days: float | None = None,
        until_free_gib: float | None = None,
    ) -> dict:
        removed: list[str] = []
        sessions_dir = self.root / "sessions"
        candidates: list[tuple[float, Path, dict]] = []
        active_session_id = None
        if self.current is not None:
            manifest = self.current._manifest()
            if not manifest.get("ended_at"):
                active_session_id = self.current.session_id
        for session_dir in sessions_dir.iterdir():
            manifest_path = session_dir / "manifest.json"
            if not manifest_path.exists():
                continue
            manifest = _json_load(manifest_path)
            if active_session_id and manifest.get("session_id") == active_session_id:
                continue
            if manifest.get("promoted"):
                continue
            if int(manifest.get("annotation_count", 0)) > 0:
                continue
            candidates.append((float(manifest.get("started_at", 0.0)), session_dir, manifest))
        candidates.sort(key=lambda item: item[0])
        cutoff = None if older_than_days is None else (_now_ts() - older_than_days * 86400)
        for started_at, session_dir, manifest in candidates:
            if cutoff is not None and started_at > cutoff:
                continue
            if until_free_gib is not None and _disk_free_gib(self.root) >= until_free_gib:
                break
            shutil.rmtree(session_dir, ignore_errors=True)
            removed.append(str(session_dir))
        return {
            "removed": removed,
            "free_gib": round(_disk_free_gib(self.root), 2),
            "remaining_sessions": len(list((self.root / "sessions").iterdir())),
        }

    def _resolve_active_or_ref(self, session_id: str | None) -> SessionArchive:
        with self._lock:
            if session_id and self.current is not None and self.current.session_id == session_id:
                return self.current
            if session_id is None and self.current is not None:
                return self.current
        if not session_id:
            raise FileNotFoundError("no active session")
        session_path = resolve_session_path(self.root, session_id)
        if session_path is None:
            raise FileNotFoundError(f"unknown session: {session_id}")
        return self._archive_from_path(session_path)

    def _archive_from_path(self, session_path: Path) -> SessionArchive:
        manifest = _json_load(session_path / "manifest.json")
        archive = object.__new__(SessionArchive)
        archive.root = self.root
        archive.session_id = str(manifest.get("session_id"))
        archive.character = str(manifest.get("character", ""))
        archive.model = str(manifest.get("model", ""))
        archive.model_name = str(manifest.get("model_name", ""))
        archive.free_warning_gib = self.free_warning_gib
        archive.path = session_path
        archive.checkpoints_dir = session_path / "checkpoints"
        archive.media_dir = session_path / "media"
        archive.video_dir = archive.media_dir / "video"
        archive.annotations_path = session_path / "annotations.jsonl"
        archive.events_path = session_path / "events.jsonl"
        archive.manifest_path = session_path / "manifest.json"
        archive._event_lock = threading.Lock()
        archive._manifest_lock = threading.RLock()
        archive._checkpoint_count = int(manifest.get("checkpoint_count", 0))
        archive._event_count = int(manifest.get("event_count", 0))
        archive._video_index = int(manifest.get("media", {}).get("video_frame_count", 0))
        archive._started_at = float(manifest.get("started_at", _now_ts()))
        archive._vision_poller = None
        archive._user_audio = AudioTrackWriter(archive.media_dir / "audio" / "user_input.wav", sample_rate=16000, origin_ts=archive._started_at)
        archive._assistant_audio = AudioTrackWriter(archive.media_dir / "audio" / "assistant_output.wav", sample_rate=24000, origin_ts=archive._started_at)
        return archive


def restore_runtime_state(payload: dict) -> dict:
    return {
        "session_snapshot": payload.get("session", {}),
        "world": deserialize_world(payload.get("world")),
        "people": deserialize_people(payload.get("people")),
        "scenario": deserialize_scenario(payload.get("scenario")),
        "script": deserialize_script(payload.get("script")),
        "goals": Goals(long_term=str(payload.get("goals", {}).get("long_term", ""))),
        "log_entries": list(payload.get("log_entries", [])),
        "had_user_input": bool(payload.get("had_user_input", False)),
        "checkpoint_index": int(payload.get("checkpoint_index", 0)),
        "label": str(payload.get("label", "")),
    }


def catalog_report_annotation(payload: dict, *, root: Path = HISTORY_ROOT) -> Path:
    _ensure_roots(root)
    path = root / "catalog" / "report_annotations.jsonl"
    _append_jsonl(
        path,
        {
            "saved_at": _iso(),
            "type": "report_annotation",
            **payload,
        },
    )
    return path


def _main_usage(root: Path) -> int:
    service = HistoryService(root=root)
    print(json.dumps(service.status(), indent=2, sort_keys=True))
    return 0


def _main_prune(root: Path, older_than_days: float | None, until_free_gib: float | None) -> int:
    service = HistoryService(root=root)
    print(json.dumps(
        service.prune_unpromoted(older_than_days=older_than_days, until_free_gib=until_free_gib),
        indent=2,
        sort_keys=True,
    ))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="History/session archive helpers.")
    parser.add_argument("--root", default=str(HISTORY_ROOT))
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("usage")

    prune = sub.add_parser("prune")
    prune.add_argument("--older-than-days", type=float, default=None)
    prune.add_argument("--until-free-gib", type=float, default=None)

    args = parser.parse_args(argv)
    root = Path(args.root)
    if args.command == "usage":
        return _main_usage(root)
    if args.command == "prune":
        return _main_prune(root, args.older_than_days, args.until_free_gib)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
