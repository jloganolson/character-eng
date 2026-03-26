import json
import subprocess
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from character_eng.history import (
    AudioTrackWriter,
    HistoryService,
    PlaybackRunner,
    VisionVideoRecorder,
    _video_is_valid,
    deserialize_people,
    load_checkpoint,
    resolve_checkpoint_for_event_time,
    resolve_session_path,
    restore_runtime_state,
    serialize_people,
)
from character_eng.person import PeopleState
from character_eng.world import Goals, Script, WorldState
from services.vision.mock_server import MockVisionHandler


def _session_snapshot():
    return {
        "system_prompt": "system",
        "messages": [{"role": "system", "content": "system"}],
        "tagged_system_indices": {},
    }


def _write_test_video(path):
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=320x240:d=1:r=30",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _write_test_jpeg(path):
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=320x240:d=0.04:r=30",
            "-frames:v",
            "1",
            str(path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


class _MjpegHandler(BaseHTTPRequestHandler):
    frame_bytes = b""
    stream_seconds = 1.0
    initial_video_delay_s = 0.0

    def do_GET(self):
        if self.path.startswith("/frame.jpg"):
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(self.frame_bytes)))
            self.end_headers()
            self.wfile.write(self.frame_bytes)
            self.wfile.flush()
            return
        if self.path != "/video_feed":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            if self.initial_video_delay_s > 0:
                time.sleep(self.initial_video_delay_s)
            deadline = time.time() + float(self.stream_seconds)
            while time.time() < deadline:
                payload = (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(self.frame_bytes)}\r\n\r\n".encode()
                    + self.frame_bytes
                    + b"\r\n"
                )
                self.wfile.write(payload)
                self.wfile.flush()
                time.sleep(1.0 / 30.0)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def log_message(self, format, *args):
        pass


def test_start_vision_capture_records_frames_from_mock_server(tmp_path):
    replay_path = Path("services/vision/replays/walkup.json")
    replay_data = json.loads(replay_path.read_text())
    MockVisionHandler.replay_data = replay_data
    MockVisionHandler.start_time = time.time()
    MockVisionHandler.input_mode = "camera"

    server = ThreadingHTTPServer(("127.0.0.1", 0), MockVisionHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_port}"
        service = HistoryService(root=tmp_path, free_warning_gib=0.0)
        archive = service.start_session(session_id="sess-mock-vision", character="greg", model="test")
        archive.start_vision_capture(base_url, fps=4.0, playback_video_fps=8.0)
        time.sleep(0.8)
        archive.stop_vision_capture()

        manifest = json.loads(archive.manifest_path.read_text())
        media = manifest.get("media", {})
        assert int(media.get("video_frame_count", 0)) > 0
        frames_jsonl = archive.video_dir / "frames.jsonl"
        assert frames_jsonl.exists()
        entries = [json.loads(line) for line in frames_jsonl.read_text().splitlines() if line.strip()]
        assert entries
        first_frame = archive.video_dir / str(entries[0]["path"])
        assert first_frame.exists()
        assert first_frame.stat().st_size > 0
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_history_service_records_session_checkpoint_and_restore(tmp_path):
    service = HistoryService(root=tmp_path, free_warning_gib=0.0)
    archive = service.start_session(session_id="sess1", character="greg", model="test")
    manifest = json.loads(archive.manifest_path.read_text())
    assert manifest["title"].endswith("greg")

    world = WorldState(static=["robot"], dynamic={"f1": "cup on table"}, events=["boot"])
    people = PeopleState()
    person = people.add_person(name="Visitor", presence="present")
    person.add_fact("holding phone")
    script = Script()
    goals = Goals(long_term="be helpful")

    archive.record_event("session_start", {"session_id": "sess1"})
    checkpoint_path = archive.capture_checkpoint(
        label="start",
        character="greg",
        session_snapshot=_session_snapshot(),
        world=world,
        people=people,
        scenario=None,
        script=script,
        goals=goals,
        log_entries=[{"type": "send", "input": "hi", "response": "hello"}],
        context_version=2,
        had_user_input=True,
    )
    assert checkpoint_path.exists()

    payload = load_checkpoint(archive.path)
    restored = restore_runtime_state(payload)
    assert restored["world"].dynamic["f1"] == "cup on table"
    assert restored["people"].people["p1"].facts["p1f1"] == "holding phone"
    assert restored["goals"].long_term == "be helpful"
    assert restored["log_entries"][0]["response"] == "hello"


def test_people_serialization_preserves_aliases(tmp_path):
    people = PeopleState()
    person = people.add_person(name="Person 1", presence="present")
    person.add_fact("wearing glasses")
    person.name = "Alex"
    person.remember_alias("Alex")

    payload = serialize_people(people)
    restored = deserialize_people(payload)

    assert restored is not None
    assert restored.people["p1"].name == "Alex"
    assert restored.people["p1"].display_name == "Alex"
    assert restored.people["p1"].aliases == ["Person 1", "Alex"]
    assert restored.get_or_create("Person 1") == "p1"


def test_history_service_can_rename_archive(tmp_path):
    service = HistoryService(root=tmp_path, free_warning_gib=0.0)
    service.start_session(session_id="sess-rename", character="greg", model="test")
    updated = service.rename(session_id="sess-rename", title="greg scarf retry")
    assert updated["title"] == "greg scarf retry"


def test_finalized_archive_ignores_followup_events_except_session_end(tmp_path):
    service = HistoryService(root=tmp_path, free_warning_gib=0.0)
    archive = service.start_session(session_id="sess-finalized", character="greg", model="test")
    archive.record_event("session_start", {"session_id": "sess-finalized"})

    service.finalize_current()
    service.record_event("vision_snapshot_read", {"event_id": "late:snapshot"})
    service.record_event("history_status", {"current_session": {"session_id": "sess-finalized"}})
    service.record_event("session_end", {"session_id": "sess-finalized"})

    events = [
        json.loads(line)
        for line in archive.events_path.read_text().splitlines()
        if line.strip()
    ]
    assert [event["type"] for event in events] == ["session_start", "session_end"]


def test_history_service_can_discard_current_archive(tmp_path):
    service = HistoryService(root=tmp_path, free_warning_gib=0.0)
    archive = service.start_session(session_id="sess-discard", character="greg", model="test")
    archive.record_event("session_start", {"session_id": "sess-discard"})

    result = service.discard_current()

    assert result is not None
    assert result["session_id"] == "sess-discard"
    assert not archive.path.exists()
    assert service.status()["current_session"] == {}


def test_annotations_auto_promote_and_catalog(tmp_path):
    service = HistoryService(root=tmp_path, free_warning_gib=0.0)
    service.start_session(session_id="sess2", character="greg", model="test")
    service.finalize_current()

    path = service.save_annotation(
        {
            "session_id": "sess2",
            "note": "Greg interrupted the user.",
            "tags": ["interruption"],
        },
        session_id="sess2",
    )
    assert path.exists()
    pinned = resolve_session_path(tmp_path, "sess2")
    assert pinned == tmp_path / "sessions" / "sess2"
    assert (tmp_path / "pinned" / "sess2").exists()
    catalog = tmp_path / "catalog" / "annotations.jsonl"
    assert catalog.exists()
    assert "Greg interrupted the user." in catalog.read_text()


def test_capture_moment_writes_audio_and_video_context(tmp_path):
    service = HistoryService(root=tmp_path, free_warning_gib=0.0)
    archive = service.start_session(session_id="sess3", character="greg", model="test")
    archive.record_event("session_start", {"session_id": "sess3"})
    archive.capture_checkpoint(
        label="start",
        character="greg",
        session_snapshot=_session_snapshot(),
        world=WorldState(),
        people=PeopleState(),
        scenario=None,
        script=Script(),
        goals=Goals(),
        log_entries=[],
        context_version=0,
        had_user_input=False,
    )
    archive.record_user_audio(b"\x00\x00" * 16000)
    archive.record_assistant_audio(b"\x00\x00" * 24000)
    archive.record_video_frame(b"\xff\xd8\xff\xd9", timestamp=archive._started_at + 0.5, source="vision")
    service.finalize_current()

    moment_dir = service.capture_moment(
        {
            "session_id": "sess3",
            "title": "phone moment",
            "tags": ["barge-in", "false-barge-in"],
            "event_time_s": 0.5,
            "bundle": {"event": {"type": "vision_pass"}},
            "window_before_s": 0.25,
            "window_after_s": 0.25,
        }
    )
    assert (moment_dir / "manifest.json").exists()
    assert (moment_dir / "checkpoint.json").exists()
    assert (moment_dir / "media" / "audio" / "user_input_clip.wav").exists()
    assert (moment_dir / "media" / "audio" / "assistant_output_clip.wav").exists()
    assert (moment_dir / "media" / "video" / "frame_000000.jpg").exists()
    manifest = json.loads((moment_dir / "manifest.json").read_text())
    assert manifest["type"] == "snippet"
    assert manifest["tags"] == ["barge-in", "false-barge-in"]


def test_capture_moment_reads_plain_wav_from_active_session(tmp_path):
    service = HistoryService(root=tmp_path, free_warning_gib=0.0)
    archive = service.start_session(session_id="sess-active", character="greg", model="test")
    archive.record_event("session_start", {"session_id": "sess-active"})
    archive.capture_checkpoint(
        label="start",
        character="greg",
        session_snapshot=_session_snapshot(),
        world=WorldState(),
        people=PeopleState(),
        scenario=None,
        script=Script(),
        goals=Goals(),
        log_entries=[],
        context_version=0,
        had_user_input=False,
    )
    archive.record_user_audio(b"\x00\x00" * 16000)
    archive.record_assistant_audio(b"\x00\x00" * 24000)
    archive.record_video_frame(b"\xff\xd8\xff\xd9", timestamp=archive._started_at + 0.5, source="vision")

    moment_dir = service.capture_moment(
        {
            "session_id": "sess-active",
            "title": "active wav moment",
            "event_time_s": 0.5,
            "bundle": {"event": {"type": "vision_pass"}},
            "window_before_s": 0.25,
            "window_after_s": 0.25,
        }
    )

    assert (moment_dir / "media" / "audio" / "user_input_clip.wav").exists()
    assert (moment_dir / "media" / "audio" / "assistant_output_clip.wav").exists()


def test_finalize_creates_playback_media_and_mixed_audio(tmp_path):
    service = HistoryService(root=tmp_path, free_warning_gib=0.0)
    archive = service.start_session(session_id="sess-media", character="greg", model="test")
    archive.record_user_audio(b"\x00\x00" * 16000)
    archive.record_assistant_audio(b"\x00\x00" * 24000)
    _write_test_video(archive.media_dir / "video" / "session_capture.mp4")

    manifest = service.finalize_current()
    media = manifest["media"]
    assert Path(media["conversation_audio_path"]).exists()
    assert Path(media["playback_path"]).exists()
    assert service.playback_media_path_for_session("sess-media") == Path(media["playback_path"])


def test_vision_video_recorder_stops_cleanly_from_mjpeg_stream(tmp_path):
    jpeg_path = tmp_path / "recorder.jpg"
    _write_test_jpeg(jpeg_path)
    _MjpegHandler.frame_bytes = jpeg_path.read_bytes()
    _MjpegHandler.stream_seconds = 1.0
    _MjpegHandler.initial_video_delay_s = 0.0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _MjpegHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    output_path = tmp_path / "vision_capture.mp4"
    try:
        recorder = VisionVideoRecorder(
            f"http://127.0.0.1:{server.server_address[1]}",
            output_path,
            fps=30,
        )
        recorder.start()
        time.sleep(0.75)
        result = recorder.stop()
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert result is None or result == output_path


def test_vision_video_recorder_finalizes_valid_mp4_for_longer_capture(tmp_path):
    jpeg_path = tmp_path / "recorder-long.jpg"
    _write_test_jpeg(jpeg_path)
    _MjpegHandler.frame_bytes = jpeg_path.read_bytes()
    _MjpegHandler.stream_seconds = 4.0
    _MjpegHandler.initial_video_delay_s = 0.0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _MjpegHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    output_path = tmp_path / "vision_capture_long.mp4"
    try:
        recorder = VisionVideoRecorder(
            f"http://127.0.0.1:{server.server_address[1]}",
            output_path,
            fps=30,
        )
        recorder.start()
        time.sleep(2.0)
        result = recorder.stop()
    finally:
        _MjpegHandler.stream_seconds = 1.0
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert result == output_path
    assert _video_is_valid(output_path)


def test_finalize_uses_first_video_frame_time_for_mux_offset(tmp_path):
    jpeg_path = tmp_path / "offset-frame.jpg"
    _write_test_jpeg(jpeg_path)
    _MjpegHandler.frame_bytes = jpeg_path.read_bytes()
    _MjpegHandler.stream_seconds = 1.2
    _MjpegHandler.initial_video_delay_s = 0.45
    server = ThreadingHTTPServer(("127.0.0.1", 0), _MjpegHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    service = HistoryService(root=tmp_path / "history", free_warning_gib=0.0)
    try:
        service.start_session(session_id="sess-video-offset", character="greg", model="test")
        service.start_vision_capture(
            f"http://127.0.0.1:{server.server_address[1]}",
            fps=5,
            playback_video_fps=30,
        )
        time.sleep(1.0)
        manifest = service.finalize_current()
    finally:
        _MjpegHandler.stream_seconds = 1.0
        _MjpegHandler.initial_video_delay_s = 0.0
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()
    video_start_s = float(manifest["media"]["video_start_s"] or 0.0)
    assert video_start_s >= 0.35


def test_finalize_current_with_active_vision_capture_stays_reasonably_fast(tmp_path):
    jpeg_path = tmp_path / "finalize-frame.jpg"
    _write_test_jpeg(jpeg_path)
    _MjpegHandler.frame_bytes = jpeg_path.read_bytes()
    _MjpegHandler.initial_video_delay_s = 0.0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _MjpegHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    service = HistoryService(root=tmp_path / "history", free_warning_gib=0.0)
    try:
        archive = service.start_session(session_id="sess-fast-finalize", character="greg", model="test")
        archive.record_user_audio(b"\x00\x00" * 16000)
        archive.record_assistant_audio(b"\x00\x00" * 24000)
        service.start_vision_capture(
            f"http://127.0.0.1:{server.server_address[1]}",
            fps=5,
            playback_video_fps=30,
        )
        time.sleep(0.5)
        started = time.perf_counter()
        manifest = service.finalize_current()
        elapsed = time.perf_counter() - started
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()
    assert elapsed < 5.0
    assert Path(manifest["media"]["playback_path"]).exists()
    assert Path(manifest["media"]["video_path"]).exists()


def test_discard_current_with_active_vision_capture_returns_quickly(tmp_path):
    jpeg_path = tmp_path / "discard-frame.jpg"
    _write_test_jpeg(jpeg_path)
    _MjpegHandler.frame_bytes = jpeg_path.read_bytes()
    _MjpegHandler.initial_video_delay_s = 0.0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _MjpegHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    service = HistoryService(root=tmp_path / "history", free_warning_gib=0.0)
    try:
        service.start_session(session_id="sess-fast-discard", character="greg", model="test")
        service.start_vision_capture(
            f"http://127.0.0.1:{server.server_address[1]}",
            fps=5,
            playback_video_fps=30,
        )
        time.sleep(0.35)
        started = time.perf_counter()
        result = service.discard_current()
        elapsed = time.perf_counter() - started
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()
    assert elapsed < 2.0
    assert result is not None
    assert not (tmp_path / "history" / "sessions" / "sess-fast-discard").exists()


def test_audio_track_writer_preserves_leading_gap(tmp_path):
    writer = AudioTrackWriter(tmp_path / "gap.wav", sample_rate=16000, origin_ts=100.0)
    writer.append(b"\x01\x02" * 1600, timestamp=100.3)
    writer.close()
    with wave.open(str(writer.path), "rb") as wav:
        assert wav.getframerate() == 16000
        frames = wav.readframes(wav.getnframes())
    # 0.2s of silence + 0.1s of audio
    assert len(frames) == 9600
    assert frames[:6400] == b"\x00" * 6400
    assert frames[6400:] == (b"\x01\x02" * 1600)


def test_finalize_rebuilds_playback_from_frames_when_raw_video_invalid(tmp_path):
    service = HistoryService(root=tmp_path, free_warning_gib=0.0)
    archive = service.start_session(session_id="sess-fallback", character="greg", model="test")
    archive.record_user_audio(b"\x00\x00" * 16000)
    archive.record_assistant_audio(b"\x00\x00" * 16000)

    jpg_a = tmp_path / "frame_a.jpg"
    jpg_b = tmp_path / "frame_b.jpg"
    _write_test_jpeg(jpg_a)
    _write_test_jpeg(jpg_b)
    archive.record_video_frame(jpg_a.read_bytes(), timestamp=archive._started_at + 0.0, source="vision")
    archive.record_video_frame(jpg_b.read_bytes(), timestamp=archive._started_at + 0.5, source="vision")
    (archive.media_dir / "video" / "session_capture.mp4").write_bytes(b"broken")

    manifest = service.finalize_current()
    media = manifest["media"]
    assert Path(media["video_path"]).exists()
    assert Path(media["playback_path"]).exists()
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=avg_frame_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            media["playback_path"],
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    assert "30/1" in probe.stdout


def test_playback_media_path_repairs_archive_when_playback_missing(tmp_path):
    service = HistoryService(root=tmp_path, free_warning_gib=0.0)
    archive = service.start_session(session_id="sess-repair", character="greg", model="test")
    archive.record_user_audio(b"\x00\x00" * 16000)
    archive.record_assistant_audio(b"\x00\x00" * 16000)
    jpg = tmp_path / "repair.jpg"
    _write_test_jpeg(jpg)
    archive.record_video_frame(jpg.read_bytes(), timestamp=archive._started_at + 0.0, source="vision")
    (archive.media_dir / "video" / "session_capture.mp4").write_bytes(b"broken")

    manifest = service.finalize_current()
    playback_path = Path(manifest["media"]["playback_path"])
    assert playback_path.exists()
    playback_path.unlink()
    manifest_path = resolve_session_path(tmp_path, "sess-repair") / "manifest.json"
    payload = json.loads(manifest_path.read_text())
    payload["media"]["playback_path"] = ""
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    repaired = service.playback_media_path_for_session("sess-repair")
    assert repaired is not None
    assert repaired.exists()


def test_prune_unpromoted_sessions_leaves_pinned(tmp_path):
    service = HistoryService(root=tmp_path, free_warning_gib=0.0)
    service.start_session(session_id="keep", character="greg", model="test")
    service.finalize_current()
    service.promote(session_id="keep", kind="pinned")

    service.start_session(session_id="drop", character="greg", model="test")
    service.finalize_current()

    result = service.prune_unpromoted(older_than_days=0)
    assert str(tmp_path / "sessions" / "drop") in result["removed"]
    assert (tmp_path / "sessions" / "keep").exists()
    assert not (tmp_path / "sessions" / "drop").exists()


def test_prepare_playback_for_session_uses_checkpoint_offset(tmp_path):
    service = HistoryService(root=tmp_path, free_warning_gib=0.0)
    archive = service.start_session(session_id="sess4", character="greg", model="test")
    archive.capture_checkpoint(
        label="start",
        character="greg",
        session_snapshot=_session_snapshot(),
        world=WorldState(),
        people=PeopleState(),
        scenario=None,
        script=Script(),
        goals=Goals(),
        log_entries=[],
        context_version=0,
        had_user_input=False,
    )
    archive.record_user_audio(b"\x00\x00" * 16000)
    archive.record_video_frame(b"\xff\xd8\xff\xd9", timestamp=archive._started_at + 0.25, source="vision")

    later_checkpoint = archive.capture_checkpoint(
        label="later",
        character="greg",
        session_snapshot=_session_snapshot(),
        world=WorldState(dynamic={"f1": "phone out"}),
        people=PeopleState(),
        scenario=None,
        script=Script(),
        goals=Goals(),
        log_entries=[],
        context_version=1,
        had_user_input=True,
    )
    payload = json.loads(later_checkpoint.read_text())
    frame_ts = payload["timestamp"] + 0.5
    archive.record_video_frame(b"\xff\xd8\xff\xd9", timestamp=frame_ts, source="vision")
    service.finalize_current()

    plan = service.prepare_playback("sess4", 1)
    assert plan.session_id == "sess4"
    assert plan.checkpoint_index == 1
    assert plan.audio_path is not None
    assert plan.audio_start_s >= 0.0
    assert len(plan.video_frames) >= 1
    assert any(round(frame["relative_s"], 3) == 0.5 for frame in plan.video_frames)


def test_resolve_checkpoint_for_event_time_uses_latest_preceding_checkpoint(tmp_path):
    service = HistoryService(root=tmp_path, free_warning_gib=0.0)
    archive = service.start_session(session_id="sess-resolve", character="greg", model="test")
    archive.capture_checkpoint(
        label="start",
        character="greg",
        session_snapshot=_session_snapshot(),
        world=WorldState(),
        people=PeopleState(),
        scenario=None,
        script=Script(),
        goals=Goals(),
        log_entries=[],
        context_version=0,
        had_user_input=False,
    )
    time.sleep(0.02)
    archive.capture_checkpoint(
        label="after-intro",
        character="greg",
        session_snapshot=_session_snapshot(),
        world=WorldState(dynamic={"f1": "intro complete"}),
        people=PeopleState(),
        scenario=None,
        script=Script(),
        goals=Goals(),
        log_entries=[],
        context_version=1,
        had_user_input=True,
    )

    second = load_checkpoint(archive.path, 1)
    manifest = json.loads((archive.path / "manifest.json").read_text())
    event_time_s = float(second["timestamp"]) - float(manifest["started_at"]) + 0.01
    resolved = resolve_checkpoint_for_event_time(archive.path, event_time_s)
    assert resolved["checkpoint_index"] == 1
    assert resolved["label"] == "after-intro"


def test_prepare_playback_for_moment_uses_clipped_media(tmp_path):
    service = HistoryService(root=tmp_path, free_warning_gib=0.0)
    archive = service.start_session(session_id="sess5", character="greg", model="test")
    archive.capture_checkpoint(
        label="start",
        character="greg",
        session_snapshot=_session_snapshot(),
        world=WorldState(),
        people=PeopleState(),
        scenario=None,
        script=Script(),
        goals=Goals(),
        log_entries=[],
        context_version=0,
        had_user_input=False,
    )
    archive.record_event("vision_snapshot_read", {"frame": 1}, timestamp=archive._started_at + 0.25)
    archive.record_event("vision_state_update", {"summary": "person visible"}, timestamp=archive._started_at + 1.0)
    archive.record_event("assistant_reply", {"text": "hello"}, timestamp=archive._started_at + 1.8)
    archive.record_user_audio(b"\x00\x00" * 32000)
    archive.record_video_frame(b"\xff\xd8\xff\xd9", timestamp=archive._started_at + 1.0, source="vision")
    service.finalize_current()

    moment_dir = service.capture_moment(
        {
            "session_id": "sess5",
            "title": "phone moment",
            "event_time_s": 1.0,
            "window_before_s": 0.5,
            "window_after_s": 0.5,
        }
    )

    plan = service.prepare_playback(str(moment_dir))
    assert plan.source_kind == "snippet"
    assert plan.session_id == moment_dir.name
    assert plan.audio_path == moment_dir / "media" / "audio" / "user_input_clip.wav"
    assert plan.audio_start_s == 0.0
    assert len(plan.video_frames) == 1
    assert round(plan.video_frames[0]["relative_s"], 3) == 0.5

    payload = service.list_events(str(moment_dir))
    assert payload["session_id"] == moment_dir.name
    assert payload["started_at"] == pytest.approx(archive._started_at + 0.5)
    assert [event["type"] for event in payload["events"]] == ["vision_state_update"]


def test_debug_only_mode_stops_new_replay_capture_after_rewind(tmp_path):
    service = HistoryService(root=tmp_path, free_warning_gib=0.0)
    archive = service.start_session(session_id="sess-debug", character="greg", model="test")
    checkpoint = archive.capture_checkpoint(
        label="start",
        character="greg",
        session_snapshot=_session_snapshot(),
        world=WorldState(),
        people=PeopleState(),
        scenario=None,
        script=Script(),
        goals=Goals(),
        log_entries=[],
        context_version=0,
        had_user_input=False,
    )
    payload = json.loads(checkpoint.read_text())
    archive.mark_debug_after_rewind(
        checkpoint_payload=payload,
        source_session_id="sess-debug",
        reason="debug-only",
        source_event_time_s=0.2,
    )

    archive.record_user_audio(b"\x00\x00" * 160)
    archive.record_assistant_audio(b"\x00\x00" * 160)
    archive.record_video_frame(b"\xff\xd8\xff\xd9", timestamp=archive._started_at + 0.4)
    archive.capture_checkpoint(
        label="after-rewind",
        character="greg",
        session_snapshot=_session_snapshot(),
        world=WorldState(dynamic={"f1": "ignored"}),
        people=PeopleState(),
        scenario=None,
        script=Script(),
        goals=Goals(),
        log_entries=[],
        context_version=1,
        had_user_input=True,
    )

    assert not (archive.media_dir / "audio" / "user_input.wav").exists()
    assert not (archive.media_dir / "audio" / "assistant_output.wav").exists()
    assert not (archive.video_dir / "frame_000000.jpg").exists()
    checkpoints = list((archive.checkpoints_dir).glob("*.json"))
    assert len(checkpoints) == 1


def test_prepare_event_playback_uses_nearby_frames(tmp_path):
    service = HistoryService(root=tmp_path, free_warning_gib=0.0)
    archive = service.start_session(session_id="sess-event", character="greg", model="test")
    archive.capture_checkpoint(
        label="start",
        character="greg",
        session_snapshot=_session_snapshot(),
        world=WorldState(),
        people=PeopleState(),
        scenario=None,
        script=Script(),
        goals=Goals(),
        log_entries=[],
        context_version=0,
        had_user_input=False,
    )
    archive.record_video_frame(b"\xff\xd8\xff\xd9", timestamp=archive._started_at + 0.5, source="vision")
    archive.record_video_frame(b"\xff\xd8\xff\xd9", timestamp=archive._started_at + 1.5, source="vision")
    service.finalize_current()

    plan = service.prepare_event_playback("sess-event", event_time_s=0.55, include_audio=False)
    assert plan.source_kind == "event"
    assert plan.audio_path is None
    assert len(plan.video_frames) == 1


def test_playback_runner_streams_audio_and_video(tmp_path):
    service = HistoryService(root=tmp_path, free_warning_gib=0.0)
    archive = service.start_session(session_id="sess6", character="greg", model="test")
    archive.capture_checkpoint(
        label="start",
        character="greg",
        session_snapshot=_session_snapshot(),
        world=WorldState(),
        people=PeopleState(),
        scenario=None,
        script=Script(),
        goals=Goals(),
        log_entries=[],
        context_version=0,
        had_user_input=False,
    )
    archive.record_user_audio(b"\x01\x02" * 3200)
    archive.record_video_frame(b"\xff\xd8\xff\xd9", timestamp=archive._started_at + 0.1, source="vision")
    service.finalize_current()

    plan = service.prepare_playback("sess6")
    audio_chunks: list[bytes] = []
    video_chunks: list[dict] = []

    runner = PlaybackRunner(
        plan,
        audio_callback=audio_chunks.append,
        video_callback=lambda jpeg, meta: video_chunks.append({"jpeg": jpeg, **meta}),
        speed=50.0,
        audio_chunk_ms=50,
    )
    runner.start()
    runner.join(timeout=2.0)

    assert audio_chunks
    assert audio_chunks[0]
    assert video_chunks
    assert video_chunks[0]["jpeg"] == b"\xff\xd8\xff\xd9"


def test_session_archive_manifest_survives_concurrent_event_and_video_updates(tmp_path):
    service = HistoryService(root=tmp_path, free_warning_gib=0.0)
    archive = service.start_session(session_id="sess-race", character="greg", model="test")

    def record_events():
        for index in range(20):
            archive.record_event("perception", {"index": index})

    def record_frames():
        for index in range(20):
            archive.record_video_frame(b"\xff\xd8\xff\xd9", timestamp=archive._started_at + index * 0.1)

    event_thread = threading.Thread(target=record_events)
    frame_thread = threading.Thread(target=record_frames)
    event_thread.start()
    frame_thread.start()
    event_thread.join()
    frame_thread.join()

    manifest = json.loads(archive.manifest_path.read_text())
    assert manifest["event_count"] == 20
    assert manifest["media"]["video_frame_count"] == 20
