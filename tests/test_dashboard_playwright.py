from __future__ import annotations

import json
import queue
import re
import subprocess
import threading
import time
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from playwright.sync_api import Error, Page, expect, sync_playwright

from character_eng.dashboard.events import DashboardEvent, DashboardEventCollector
from character_eng.dashboard.server import start_dashboard
from character_eng.history import HistoryService
from character_eng.person import PeopleState
from character_eng.world import Goals, Script, WorldState


def _write_test_video(path: Path, duration_s: float = 2.0):
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s=320x240:d={duration_s}:r=30",
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


def _write_test_jpeg(path: Path):
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=#9ec5e5:s=320x240:d=0.1",
            "-frames:v",
            "1",
            str(path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _collector_push_at(collector: DashboardEventCollector, event_type: str, data: dict, timestamp: float) -> None:
    with collector._lock:
        collector._seq += 1
        event = DashboardEvent(type=event_type, data=data, timestamp=timestamp, seq=collector._seq)
        collector._events.append(event)
        for q in collector._subscribers:
            try:
                q.put_nowait(event)
            except queue.Full:
                pass


def _seed_archive_session(
    history: HistoryService,
    *,
    session_id: str,
    title: str | None = None,
    response_text: str = "Hey, what time is it to you?",
    extra_events: list[tuple[str, dict, float]] | None = None,
    video_duration_s: float = 2.0,
):
    archive = history.start_session(session_id=session_id, character="greg", model="groq-llama-8b")
    archive_start = archive._started_at
    archive.record_event("session_start", {
        "character": "greg",
        "model": "groq-llama-8b",
        "session_id": session_id,
        "stage": "engaged",
    }, timestamp=archive_start + 0.0)
    archive.record_event("user_transcript_final", {"text": "oh hey man"}, timestamp=archive_start + 0.4)
    archive.record_event("turn_start", {"input_text": "oh hey man", "input_type": "send"}, timestamp=archive_start + 0.45)
    archive.record_event("response_done", {"full_text": response_text}, timestamp=archive_start + 1.1)
    for event_type, payload, offset_s in extra_events or []:
        archive.record_event(event_type, payload, timestamp=archive_start + offset_s)
    archive.capture_checkpoint(
        label="start",
        character="greg",
        session_snapshot={"system_prompt": "system", "messages": [], "tagged_system_indices": {}},
        world=WorldState(),
        people=PeopleState(),
        scenario=None,
        script=Script(),
        goals=Goals(),
        log_entries=[],
        context_version=0,
        had_user_input=False,
    )
    history.finalize_current()
    _write_test_video(archive.media_dir / "playback.mp4", duration_s=video_duration_s)
    frame_path = archive.media_dir / "video" / "000001.jpg"
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    _write_test_jpeg(frame_path)
    (archive.media_dir / "video" / "frames.jsonl").write_text(
        json.dumps({
            "timestamp": archive_start + 0.65,
            "path": frame_path.name,
            "source": "vision",
        }) + "\n",
        encoding="utf-8",
    )
    archive_manifest = json.loads(archive.manifest_path.read_text())
    archive_media = dict(archive_manifest.get("media", {}))
    archive_media["playback_path"] = str(archive.media_dir / "playback.mp4")
    archive_media["video_path"] = str(archive.media_dir / "playback.mp4")
    archive_media["video_frames_path"] = str(archive.media_dir / "video" / "frames.jsonl")
    archive_media["video_frame_count"] = 1
    updates = {"media": archive_media}
    if title:
        updates["title"] = title
    archive.update_manifest(**updates)
    return archive


def _seed_turn_summary_archive(history: HistoryService, session_id: str = "sess-turn-summary"):
    archive = history.start_session(session_id=session_id, character="greg", model="groq-llama-8b")
    archive_start = archive._started_at
    archive.record_event("session_start", {
        "character": "greg",
        "model": "groq-llama-8b",
        "session_id": session_id,
        "stage": "watching",
    }, timestamp=archive_start + 0.0)
    archive.record_event("user_transcript_final", {"text": "Do you have water?"}, timestamp=archive_start + 0.35)
    archive.record_event("turn_start", {
        "input_text": "Do you have water?",
        "input_type": "send",
        "trigger_source": "user",
        "turn_kind": "dialogue",
        "input_source": "voice",
    }, timestamp=archive_start + 0.4)
    archive.record_event("prompt_trace", {
        "label": "chat",
        "title": "Chat prompt",
        "provider": "Groq",
        "model": "llama-test",
        "messages": [
            {"role": "system", "content": "Keep it short."},
            {"role": "user", "content": "Do you have water?"},
        ],
        "output": "Free water. Want one?",
        "started_at": archive_start + 0.42,
        "finished_at": archive_start + 0.5,
    }, timestamp=archive_start + 0.5)
    archive.record_event("response_done", {
        "full_text": "Free water. Want one?",
        "turn_kind": "dialogue",
        "turn_outcome": "replied",
        "input_source": "voice",
        "ttft_ms": 190,
    }, timestamp=archive_start + 0.8)
    archive.record_event("eval", {
        "script_status": "advance",
        "thought": "Move to the next beat after answering.",
        "plan_request": "Ask a grounded follow-up.",
    }, timestamp=archive_start + 1.0)
    archive.record_event("director", {
        "status": "advance",
        "thought": "Stay in the same stage.",
        "new_stage": "watching",
    }, timestamp=archive_start + 1.1)
    archive.record_event("plan", {
        "beats": [
            {"intent": "Offer help", "line": "I can point you to the cooler."},
            {"intent": "Ask a follow-up", "line": "You looking for cold water?"},
        ],
    }, timestamp=archive_start + 1.2)
    archive.record_event("beat_advance", {
        "next_intent": "Ask a follow-up",
        "script_complete": False,
    }, timestamp=archive_start + 1.3)
    archive.capture_checkpoint(
        label="turn-summary",
        character="greg",
        session_snapshot={"system_prompt": "system", "messages": [], "tagged_system_indices": {}},
        world=WorldState(),
        people=PeopleState(),
        scenario=None,
        script=Script(),
        goals=Goals(),
        log_entries=[],
        context_version=0,
        had_user_input=True,
    )
    history.finalize_current()
    _write_test_video(archive.media_dir / "playback.mp4", duration_s=4.0)
    frame_path = archive.media_dir / "video" / "000001.jpg"
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    _write_test_jpeg(frame_path)
    (archive.media_dir / "video" / "frames.jsonl").write_text(
        json.dumps({
            "timestamp": archive_start + 0.8,
            "path": frame_path.name,
            "source": "vision",
        }) + "\n",
        encoding="utf-8",
    )
    archive_manifest = json.loads(archive.manifest_path.read_text())
    archive_media = dict(archive_manifest.get("media", {}))
    archive_media["playback_path"] = str(archive.media_dir / "playback.mp4")
    archive_media["video_path"] = str(archive.media_dir / "playback.mp4")
    archive_media["video_frames_path"] = str(archive.media_dir / "video" / "frames.jsonl")
    archive_media["video_frame_count"] = 1
    archive.update_manifest(title="Turn Summary Archive", media=archive_media)
    return archive


def _seed_dense_card_archive(
    history: HistoryService,
    session_id: str = "sess-archive-dense-cards",
    *,
    turn_count: int = 96,
    span_s: float = 576.0,
):
    archive = history.start_session(session_id=session_id, character="greg", model="groq-llama-8b")
    archive_start = archive._started_at
    archive.record_event("session_start", {
        "character": "greg",
        "model": "groq-llama-8b",
        "session_id": session_id,
        "stage": "watching",
    }, timestamp=archive_start + 0.0)

    spacing_s = max(4.5, span_s / max(1, turn_count))
    for idx in range(turn_count):
        turn_offset = 0.4 + (idx * spacing_s)
        archive.record_event("turn_start", {
            "input_text": f"dense user turn {idx}",
            "input_type": "send",
            "trigger_source": "user",
            "turn_kind": "dialogue",
            "input_source": "voice",
        }, timestamp=archive_start + turn_offset)
        archive.record_event("response_done", {
            "full_text": f"Dense reply turn {idx}",
            "turn_kind": "dialogue",
            "turn_outcome": "replied",
            "input_source": "voice",
            "ttft_ms": 160 + idx,
        }, timestamp=archive_start + turn_offset + 0.7)
        archive.record_event("eval", {
            "script_status": "advance" if idx % 2 == 0 else "watching",
            "thought": f"Dense planner thought {idx}",
            "plan_request": f"Dense planner request {idx}",
        }, timestamp=archive_start + turn_offset + 1.3)
        archive.record_event("reconcile", {
            "events": [f"Dense world event {idx}"],
            "add_facts": [f"dense fact {idx}"],
        }, timestamp=archive_start + turn_offset + 1.9)
        archive.record_event("vision_pass", {
            "latest_perception": f"Dense vision turn {idx}",
            "perception_count": 1,
            "vlm_answer_count": 1,
            "object_count": 1,
            "object_labels": [f"marker-{idx}"],
        }, timestamp=archive_start + turn_offset + 2.5)

    archive.capture_checkpoint(
        label="dense-cards",
        character="greg",
        session_snapshot={"system_prompt": "system", "messages": [], "tagged_system_indices": {}},
        world=WorldState(),
        people=PeopleState(),
        scenario=None,
        script=Script(),
        goals=Goals(),
        log_entries=[],
        context_version=0,
        had_user_input=True,
    )
    history.finalize_current()
    _write_test_video(archive.media_dir / "playback.mp4", duration_s=12.0)
    frame_path = archive.media_dir / "video" / "000001.jpg"
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    _write_test_jpeg(frame_path)
    (archive.media_dir / "video" / "frames.jsonl").write_text(
        json.dumps({
            "timestamp": archive_start + 0.7,
            "path": frame_path.name,
            "source": "vision",
        }) + "\n",
        encoding="utf-8",
    )
    archive_manifest = json.loads(archive.manifest_path.read_text())
    archive_media = dict(archive_manifest.get("media", {}))
    archive_media["playback_path"] = str(archive.media_dir / "playback.mp4")
    archive_media["video_path"] = str(archive.media_dir / "playback.mp4")
    archive_media["video_frames_path"] = str(archive.media_dir / "video" / "frames.jsonl")
    archive_media["video_frame_count"] = 1
    archive.update_manifest(title="Dense Card Archive", media=archive_media)
    return archive


class _VisionHandler(BaseHTTPRequestHandler):
    state = {
        "fail": False,
        "status": {
            "vllm": "ready",
            "vllm_model": "gemma-vision-test",
            "sam3": {"status": "ready"},
            "face": {"status": "ready"},
            "person": {"status": "ready"},
        },
        "memory": {
            "gpu": {
                "allocated_mb": 2048,
                "total_mb": 8192,
            }
        },
    }

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            body = b"<html><body><h1>Vision Test UI</h1></body></html>"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/model_status":
            if self.state["fail"]:
                self.send_error(HTTPStatus.SERVICE_UNAVAILABLE)
                return
            body = json.dumps(self.state["status"]).encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/memory_status":
            if self.state["fail"]:
                self.send_error(HTTPStatus.SERVICE_UNAVAILABLE)
                return
            body = json.dumps(self.state["memory"]).encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_error(HTTPStatus.NOT_FOUND)

    def log_message(self, format, *args):
        pass


class VisionServiceController:
    def __init__(self, server: ThreadingHTTPServer):
        self.server = server
        self.url = f"http://127.0.0.1:{server.server_address[1]}"

    def set_fail(self, value: bool) -> None:
        _VisionHandler.state["fail"] = value


@pytest.fixture()
def browser_page() -> Page:
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1600, "height": 1100})
            yield page
            browser.close()
    except Error as exc:
        pytest.skip(f"Playwright browser not available: {exc}")


@pytest.fixture()
def vision_service() -> VisionServiceController:
    _VisionHandler.state["fail"] = False
    server = ThreadingHTTPServer(("127.0.0.1", 0), _VisionHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield VisionServiceController(server)
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


@pytest.fixture()
def dashboard_server(tmp_path):
    collector = DashboardEventCollector()
    input_queue: queue.Queue[str] = queue.Queue()
    report_dir = tmp_path / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    history_root = tmp_path / "history"
    history = HistoryService(root=history_root, free_warning_gib=0.0)
    _seed_archive_session(history, session_id="sess-archive-playback")
    _seed_archive_session(
        history,
        session_id="sess-archive-long-playback",
        title="Long Playback Archive",
        response_text="Long playback archive is ready for scrub testing.",
        video_duration_s=12.0,
        extra_events=[
            (
                "vision_focus",
                {
                    "constant_question_count": 5,
                    "ephemeral_question_count": 2,
                    "constant_sam_target_count": 5,
                    "ephemeral_sam_target_count": 2,
                    "summary": "Greg is scanning the room for a visible clock.",
                },
                0.8,
            ),
            (
                "eval",
                {
                    "script_status": "watching",
                    "thought": "Planner holds the current beat while Greg inspects the room.",
                    "plan_request": "Keep observing the room.",
                },
                3.8,
            ),
            (
                "reconcile",
                {
                    "events": ["A wall clock is clearly visible near the doorway."],
                    "add_facts": ["clock visible near doorway"],
                },
                8.6,
            ),
        ],
    )
    _seed_archive_session(
        history,
        session_id="sess-archive-media-gap",
        title="Media Gap Archive",
        response_text="Media gap archive is ready for scrub testing.",
        video_duration_s=4.2,
        extra_events=[
            (
                "eval",
                {
                    "script_status": "watching",
                    "thought": "Planner holds the current beat while Greg inspects the room.",
                    "plan_request": "Keep observing the room.",
                },
                3.8,
            ),
            (
                "reconcile",
                {
                    "events": ["A wall clock is clearly visible near the doorway."],
                    "add_facts": ["clock visible near doorway"],
                },
                8.6,
            ),
        ],
    )
    _seed_turn_summary_archive(history)
    _seed_archive_session(
        history,
        session_id="greg-offline-canonical-v1",
        title="Greg Offline Canonical v1",
        response_text="Canonical archive is ready for offline UX work.",
    )
    _seed_archive_session(
        history,
        session_id="sess-archive-legacy-vision",
        title="Legacy Vision Archive",
        response_text="Legacy vision archive is ready.",
        extra_events=[
            (
                "vision_snapshot",
                {
                    "faces": 1,
                    "persons": 1,
                    "object_labels": ["clock"],
                    "vlm_answers": [{"question": "What changed?", "answer": "A clock is visible."}],
                },
                0.7,
            ),
            (
                "vision_pass",
                {
                    "latest_perception": "A clock is visible above the table.",
                    "perception_count": 1,
                    "vlm_answer_count": 1,
                    "object_count": 1,
                    "object_labels": ["clock"],
                },
                0.9,
            ),
        ],
    )
    _seed_archive_session(
        history,
        session_id="sess-archive-vision-raw-ui",
        title="Vision Raw UI Archive",
        response_text="Vision raw ui archive is ready.",
        extra_events=[
            (
                "vision_snapshot_read",
                {
                    "event_id": "vision-cycle-test:snapshot",
                    "faces": 0,
                    "persons": 0,
                    "objects": 2,
                    "object_labels": ["clock", "table"],
                    "vlm_answer_count": 1,
                },
                0.65,
            ),
            (
                "vlm_answer",
                {
                    "task_id": "person_description_static_visitor",
                    "label": "Person Description Static [Visitor]",
                    "question": "Describe this visible person's appearance.",
                    "answer": "The visitor is wearing glasses.",
                    "target": "person_identity",
                    "target_identity": "Visitor",
                    "target_bbox": [40, 50, 120, 140],
                    "interpret_as": "person_description_static",
                },
                0.66,
            ),
            (
                "sam3_detection",
                {
                    "objects": [
                        {"label": "clock", "confidence": 0.92, "bbox": [0.1, 0.1, 0.4, 0.4]},
                        {"label": "table", "confidence": 0.88, "bbox": [0.2, 0.45, 0.9, 0.9]},
                    ],
                    "persons": [],
                    "timing": {"sam_ms": 0.018},
                },
                0.72,
            ),
        ],
    )
    _seed_dense_card_archive(history)
    archive = history.start_session(session_id="sess-playwright", character="greg", model="groq-llama-8b")
    archive.record_event("session_start", {
        "character": "greg",
        "model": "groq-llama-8b",
        "session_id": "sess-playwright",
        "stage": "greeting",
    })
    archive.capture_checkpoint(
        label="start",
        character="greg",
        session_snapshot={"system_prompt": "system", "messages": [], "tagged_system_indices": {}},
        world=WorldState(),
        people=PeopleState(),
        scenario=None,
        script=Script(),
        goals=Goals(),
        log_entries=[],
        context_version=0,
        had_user_input=False,
    )
    thread, port = start_dashboard(
        collector,
        input_queue,
        port=0,
        report_dir=report_dir,
        history_api=history,
        default_character="greg",
    )
    for _ in range(20):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/state")
            break
        except Exception:
            time.sleep(0.05)
    try:
        yield collector, input_queue, f"http://127.0.0.1:{port}/", report_dir
    finally:
        collector.shutdown()
        thread.server.shutdown()  # type: ignore[attr-defined]
        thread.server.server_close()  # type: ignore[attr-defined]
        thread.join(timeout=2)


def _seed_rendered_card_inventory_fixture(collector: DashboardEventCollector) -> dict[str, dict[str, object]]:
    base = time.time()
    step_s = 0.25
    index = 0

    def push(event_type: str, data: dict) -> None:
        nonlocal index
        _collector_push_at(collector, event_type, data, base + (index * step_s))
        index += 1

    push("session_start", {
        "character": "greg",
        "model": "groq-llama-8b",
        "session_id": "sess-card-sweep",
        "stage": "watching",
    })
    push("runtime_controls", {
        "controls": {"reconcile": True, "vision": True, "auto_beat": True, "director": True},
        "conversation_paused": False,
        "session_stopped": False,
        "startup_pause_pending": False,
        "session_state": "live",
        "voice_active": True,
        "voice_status": {
            "active": True,
            "output_only": False,
            "filler_enabled": False,
            "tts_backend": "pocket",
            "mic_ready": True,
            "speaker_ready": True,
            "stt": {"state": "ready", "detail": "Mic hot."},
            "tts": {"state": "ready", "detail": "Speaker ready.", "backend": "pocket"},
        },
        "vision_active": True,
        "reconcile_thread_alive": True,
        "vision_service_url": "http://127.0.0.1:9",
        "vision_service_health": True,
        "vision_service_managed": False,
        "vision_service_external": True,
        "vision_service_state": "running",
        "vision_service_autostart": False,
        "vision_service_mode": "camera",
        "vision_mock_replay": "",
    })
    push("resume", {})
    push("user_speech_started", {"source": "mic"})
    push("user_transcript_final", {"text": "hello there"})
    push("turn_start", {
        "input_type": "send",
        "input_text": "hello there",
        "trigger_source": "user",
        "turn_kind": "dialogue",
        "input_source": "voice",
    })
    push("response_done", {
        "full_text": "Free water. Want one?",
        "input_source": "voice",
        "stt_text": "hello there",
        "stt_ms": 120,
        "ttft_ms": 190,
        "llm_ms": 280,
        "audio_ms": 640,
        "first_audio_ms": 360,
    })
    push("expression", {"expression": "curious", "gaze": "forward"})
    push("response_guard", {"status": "pass", "issues": [], "rationale": "Stayed in character and avoided control text."})
    push("eval", {"script_status": "advance", "thought": "Keep it short.", "plan_request": "Offer water."})
    push("director", {"status": "open", "thought": "Move to offer.", "new_stage": "offer"})
    push("stage_change", {"new_stage": "offer", "new_goal": "Offer water and ask what they need."})
    push("plan", {"beats": [{"intent": "offer_water"}, {"intent": "ask_need"}]})
    push("beat_advance", {"next_intent": "offer_water", "script_complete": False})
    push("reconcile", {
        "add_facts": ["visitor present"],
        "remove_facts": [],
        "events": ["Greg notices the visitor."],
    })
    push("world_state", {
        "static_facts": [{"id": "w1", "text": "Greg is indoors."}],
        "dynamic_facts": [{"id": "w2", "text": "A visitor is present."}],
        "pending": ["Ask what time it is."],
    })
    push("people_state", {
        "people": [{"id": "p1", "name": "Visitor", "facts": [{"id": "p1f1", "text": "Holding a cup."}]}],
    })
    push("world_update", {
        "facts": ["A visitor is present."],
        "events": ["Greg notices the visitor."],
        "people": ["p1"],
        "fact_count": 2,
        "people_count": 1,
    })
    push("perception", {
        "source": "visual",
        "description": "A person appeared in view.",
        "source_trace": {
            "kind": "presence_tracker",
            "provenance": {
                "label": "Presence tracker",
                "primary": "face_tracker",
                "components": ["face_tracker", "vlm_answers"],
            },
            "object_labels": ["clock", "cup"],
            "supporting_answers": [{"question": "Who is visible?", "answer": "A visitor."}],
        },
    })
    push("vision_focus", {
        "summary": "Watch the visitor and the wall clock.",
        "constant_questions": ["What time is it?"],
        "ephemeral_questions": ["Is the visitor looking at Greg?"],
        "constant_sam_targets": ["person"],
        "ephemeral_sam_targets": ["clock"],
    })
    push("vision_snapshot", {
        "faces": 1,
        "persons": 1,
        "objects": 2,
        "object_labels": ["clock", "cup"],
        "vlm_answers": [{"label": "room_state", "answer": "A visitor stands near a wall clock."}],
    })
    push("vision_pass", {
        "latest_perception": "A person appeared in view.",
        "latest_answer": "A visitor stands near a wall clock.",
        "perception_count": 1,
        "vlm_answer_count": 1,
        "object_count": 2,
    })
    push("vision_snapshot_read", {
        "faces": 1,
        "persons": 1,
        "objects": 2,
        "vlm_answer_count": 1,
        "object_labels": ["clock", "cup"],
    })
    push("face_tracking", {
        "timing": {"total": 0.041},
        "faces": [{"identity": "visitor", "confidence": 0.97, "bbox": [0.1, 0.1, 0.2, 0.2], "looking_at_camera": True}],
    })
    push("sam3_detection", {
        "timing": {"sam3": 0.088, "prompts": 2},
        "persons": [{"identity": "visitor", "confidence": 0.95, "bbox": [0.1, 0.1, 0.2, 0.2]}],
        "objects": [{"label": "clock", "confidence": 0.91, "bbox": [0.4, 0.1, 0.15, 0.15]}],
    })
    push("reid_track", {
        "track_id": 7,
        "identity": "visitor",
        "identity_source": "embedding",
        "timing": {"reid": 0.012},
    })
    push("vision_trigger", {
        "signal": "person_visible",
        "description": "Person visible for three frames.",
        "rule": {"source": "presence", "label": "person_visible", "frames": 3, "task_id": "watch"},
        "streak": 3,
        "evidence": {"tracks": ["visitor"]},
    })
    push("vision_state_update", {
        "task_answers": [{"label": "world_state", "answer": "A visitor stands near a wall clock."}],
        "add_facts": ["A wall clock is visible."],
        "remove_facts": [],
        "events": ["Visitor entered the room."],
        "person_updates": [{"person_id": "p1", "add_facts": ["Standing near the doorway."], "set_presence": "visible"}],
    })
    push("vlm_answer", {
        "label": "room_state",
        "question": "What changed?",
        "answer": "A visitor stands near a wall clock.",
        "target": "person",
        "target_identity": "visitor",
    })
    push("user_transcript_final", {"text": "wait actually never mind"})
    push("turn_start", {
        "input_type": "send",
        "input_text": "wait actually never mind",
        "trigger_source": "user",
        "turn_kind": "dialogue",
        "input_source": "voice",
    })
    push("response_chunk", {"text": "Okay, I can stop."})
    push("response_interrupted", {
        "reason": "barge_in",
        "truncated_text": "Okay",
        "raw_full_text": "Okay, I can stop.",
    })

    return {
        "turn_start": {"card_text": "hello there", "payload_snippets": ['"input_text": "hello there"']},
        "user_speech_started": {"card_text": "speech detected", "payload_snippets": ['"source": "mic"']},
        "user_transcript_final": {"card_text": "hello there", "payload_snippets": ['"text": "hello there"']},
        "assistant_reply": {"card_text": "Free water. Want one?", "payload_snippets": ['"full_text": "Free water. Want one?"']},
        "response_interrupted": {"card_text": "Interrupted after: Okay", "payload_snippets": ['"reason": "barge_in"']},
        "expression": {"card_text": "curious", "payload_snippets": ['"expression": "curious"']},
        "response_guard": {"card_text": "guard pass", "payload_snippets": ['"status": "pass"']},
        "post_response": {"card_text": None, "payload_snippets": ['"new_stage": "offer"', '"next_intent": "offer_water"']},
        "reconcile": {"card_text": None, "payload_snippets": ['"add_facts": [', '"visitor present"']},
        "world_state": {"card_text": None, "payload_snippets": ['"text": "A visitor is present."']},
        "people_state": {"card_text": None, "payload_snippets": ['"name": "Visitor"']},
        "world_update": {"card_text": None, "payload_snippets": ['"Greg notices the visitor."']},
        "perception": {"card_text": None, "payload_snippets": ['"description": "A person appeared in view."']},
        "vision_snapshot": {"card_text": None, "payload_snippets": ['"object_labels": [', '"clock"']},
        "vision_pass": {"card_text": "A person appeared in view.", "payload_snippets": ['"latest_perception": "A person appeared in view."']},
        "vision_focus": {"card_text": None, "payload_snippets": ['"summary": "Watch the visitor and the wall clock."']},
        "vision_snapshot_read": {"card_text": None, "payload_snippets": ['"vlm_answer_count": 1']},
        "face_tracking": {"card_text": None, "payload_snippets": ['"identity": "visitor"']},
        "sam3_detection": {"card_text": None, "payload_snippets": ['"label": "clock"']},
        "reid_track": {"card_text": None, "payload_snippets": ['"identity_source": "embedding"']},
        "vision_trigger": {"card_text": "Person visible for three frames.", "payload_snippets": ['"signal": "person_visible"']},
        "vision_state_update": {"card_text": None, "payload_snippets": ['"A wall clock is visible."']},
        "vlm_answer": {"card_text": "A visitor stands near a wall clock.", "payload_snippets": ['"answer": "A visitor stands near a wall clock."']},
    }


def test_runtime_panel_interactions_in_browser(
    browser_page: Page,
    dashboard_server,
    vision_service: VisionServiceController,
):
    collector, input_queue, dashboard_url, report_dir = dashboard_server
    vision_service_url = vision_service.url
    page = browser_page
    page.goto(dashboard_url)

    expect(page.locator("body")).to_have_class(re.compile(r"\bbooting\b"))
    expect(page.locator("#boot-overlay")).to_be_visible()
    expect(page.locator("#boot-summary")).to_contain_text("Connecting to the local runtime")
    expect(page.locator("#boot-overlay-summary")).to_contain_text("Connecting to the local runtime")
    expect(page.locator("#boot-grid")).to_contain_text("dashboard")
    expect(page.locator("#boot-grid")).to_contain_text("runtime")
    expect(page.locator("#boot-heartbeat")).to_contain_text("page")
    expect(page.locator("#prompt-assets-panel")).to_contain_text("Character prompt template")

    collector.push(
        "runtime_controls",
        {
            "controls": {
                "reconcile": True,
                "vision": True,
                "auto_beat": False,
                "filler": False,
            },
            "conversation_paused": False,
            "session_stopped": True,
            "startup_pause_pending": False,
            "session_state": "ready",
            "voice_active": True,
            "voice_status": {
                "active": True,
                "output_only": False,
                "filler_enabled": False,
                "tts_backend": "pocket",
                "mic_ready": True,
                "speaker_ready": True,
                "stt": {"state": "ready", "detail": "Deepgram socket open."},
                "tts": {"state": "ready", "detail": "Pocket-TTS reachable.", "backend": "pocket"},
            },
            "vision_active": True,
            "reconcile_thread_alive": True,
            "vision_service_url": vision_service_url,
            "vision_service_health": True,
            "vision_service_managed": False,
            "vision_service_external": True,
            "vision_service_state": "external",
            "vision_service_autostart": True,
            "vision_service_mode": "camera",
            "vision_mock_replay": "",
        },
    )

    expect(page.locator("body")).not_to_have_class(re.compile(r"\bbooting\b"))
    expect(page.locator("#boot-overlay")).not_to_be_visible()
    expect(page.locator("#runtime-status")).to_contain_text("state: ready")
    expect(page.locator("#boot-summary")).to_contain_text("Everything is warm. Start when you want to begin a fresh session.")
    expect(page.locator("#boot-overlay-summary")).to_contain_text("Everything is warm. Start when you want to begin a fresh session.")
    expect(page.locator("#archive-load-button")).to_be_visible()
    expect(page.locator("#boot-grid")).to_contain_text("voice")
    expect(page.locator("#boot-grid")).to_contain_text("models")
    expect(page.locator("#runtime-status")).to_contain_text("auto-beat: off")
    expect(page.locator("#runtime-status")).to_contain_text("stt: ready")
    expect(page.locator("#runtime-status")).to_contain_text("tts: ready")
    expect(page.locator("#runtime-start")).to_have_text("Start New Session")
    expect(page.locator("#runtime-stop")).to_be_disabled()
    expect(page.locator("#boot-grid")).to_contain_text("Deepgram socket open.")
    expect(page.locator("#boot-grid")).to_contain_text("Pocket-TTS reachable.")
    expect(page.locator("#vision-service-status")).to_contain_text("source: external service")
    expect(page.locator("#vision-service-status")).to_contain_text(vision_service_url)
    expect(page.locator("#vision-model-status")).to_contain_text("vLLM")
    expect(page.locator("#vision-model-status")).to_contain_text("gemma-vision-test")
    expect(page.locator("#vision-model-status")).to_contain_text("GPU memory")
    expect(page.locator("#vision-service-link")).to_have_attribute("href", vision_service_url)
    expect(page.locator("#time-window-value")).to_have_text("30s")
    expect(page.locator("#runtime-start")).to_have_text("Start New Session")
    expect(page.locator("button[data-lane-toggle='behavior']")).to_contain_text("robot behavior")
    expect(page.locator("button[data-lane-toggle='planner']")).to_have_text("planner")
    expect(page.locator("button[data-lane-toggle='eval']")).to_have_count(0)
    expect(page.locator("button[data-lane-toggle='director']")).to_have_count(0)
    expect(page.locator("button[data-lane-toggle='script']")).to_have_count(0)

    page.locator("#runtime-start").evaluate("(node) => node.click()")
    expect(page.locator("#runtime-start")).to_have_text("Starting...")
    assert input_queue.get(timeout=2) == "/start"

    collector.push(
        "session_start",
        {
            "character": "greg",
            "model": "groq-llama-8b",
            "session_id": "sess-playwright",
            "stage": "greeting",
        },
    )
    collector.push(
        "runtime_controls",
        {
            "controls": {
                "reconcile": True,
                "vision": True,
                "auto_beat": False,
                "filler": False,
            },
            "conversation_paused": True,
            "session_stopped": False,
            "startup_pause_pending": True,
            "session_state": "paused",
            "voice_active": True,
            "voice_status": {
                "active": True,
                "output_only": False,
                "filler_enabled": False,
                "tts_backend": "pocket",
                "mic_ready": True,
                "speaker_ready": True,
                "stt": {"state": "ready", "detail": "Deepgram socket open."},
                "tts": {"state": "ready", "detail": "Pocket-TTS reachable.", "backend": "pocket"},
            },
            "vision_active": True,
            "reconcile_thread_alive": True,
            "vision_service_url": vision_service_url,
            "vision_service_health": True,
            "vision_service_managed": False,
            "vision_service_external": True,
            "vision_service_state": "external",
            "vision_service_autostart": True,
            "vision_service_mode": "camera",
            "vision_mock_replay": "",
        },
    )
    collector.push("pause", {"startup": True, "fresh_session": True})
    expect(page.locator("#h-character")).to_have_text("greg")
    expect(page.locator("#runtime-status")).to_contain_text("state: paused")
    expect(page.locator("#runtime-toggle")).to_have_text("Play")
    expect(page.locator("#boot-summary")).to_contain_text("Fresh session is ready. Press Play when you want to begin.")
    expect(page.locator("#runtime-stop")).to_be_enabled()
    expect(page.locator("#vision-panel")).to_be_visible()

    page.locator("#runtime-toggle").evaluate("(node) => node.click()")
    expect(page.locator("#runtime-toggle")).to_have_text("Playing...")
    assert input_queue.get(timeout=2) == "/play"

    collector.push(
        "runtime_controls",
        {
            "controls": {
                "reconcile": True,
                "vision": True,
                "auto_beat": False,
                "filler": False,
            },
            "conversation_paused": False,
            "session_stopped": False,
            "startup_pause_pending": False,
            "session_state": "live",
            "voice_active": True,
            "voice_status": {
                "active": True,
                "output_only": False,
                "filler_enabled": False,
                "tts_backend": "pocket",
                "mic_ready": True,
                "speaker_ready": True,
                "stt": {"state": "ready", "detail": "Deepgram socket open."},
                "tts": {"state": "ready", "detail": "Pocket-TTS reachable.", "backend": "pocket"},
            },
            "vision_active": True,
            "reconcile_thread_alive": True,
            "vision_service_url": vision_service_url,
            "vision_service_health": True,
            "vision_service_managed": False,
            "vision_service_external": True,
            "vision_service_state": "external",
            "vision_service_autostart": True,
            "vision_service_mode": "camera",
            "vision_mock_replay": "",
        },
    )
    expect(page.locator("#runtime-status")).to_contain_text("state: live")
    expect(page.locator("#runtime-toggle")).to_have_text("Pause")
    expect(page.locator("#runtime-stop")).to_be_enabled()
    expect(page.locator("#vision-panel")).to_be_visible()

    turn_base = time.time()
    collector.push("user_speech_started", {})
    collector.push("user_transcript_final", {"text": "Do you have water?"})
    collector.push("turn_start", {
        "input_type": "send",
        "input_text": "Do you have water?",
        "input_source": "voice",
        "speech_started_ts": turn_base,
        "speech_ended_ts": turn_base + 0.35,
        "transcript_final_ts": turn_base + 0.42,
        "stt_ms": 420,
        "endpointing_ms": 70,
        "stt_text": "Do you have water?",
    })
    collector.push("prompt_trace", {
        "label": "chat",
        "title": "Chat prompt",
        "provider": "Groq",
        "model": "llama-test",
        "messages": [
            {"role": "system", "content": "Keep it short."},
            {"role": "user", "content": "Do you have water?"},
        ],
        "output": "Free water. Want one?",
        "started_at": time.time(),
        "finished_at": time.time(),
    })
    collector.push("response_chunk", {"text": "Free water. "})
    collector.push("response_chunk", {"text": "Want one?"})
    collector.push("response_done", {
        "full_text": "Free water. Want one?",
        "input_source": "voice",
        "speech_started_ts": turn_base,
        "speech_ended_ts": turn_base + 0.35,
        "transcript_final_ts": turn_base + 0.42,
        "stt_ms": 420,
        "endpointing_ms": 70,
        "stt_text": "Do you have water?",
        "llm_start_ts": turn_base + 0.42,
        "response_ttft_ts": turn_base + 0.61,
        "response_done_ts": turn_base + 1.2,
        "tts_request_ts": turn_base + 0.72,
        "tts_synth_done_ts": turn_base + 1.46,
        "first_audio_ts": turn_base + 1.31,
        "spoken_done_ts": turn_base + 2.4,
        "ttft_ms": 190,
        "llm_ms": 780,
        "total_ms": 1980,
        "tts_request_ms": 300,
        "tts_synth_done_ms": 1040,
        "first_audio_ms": 890,
        "audio_ms": 1090,
    })
    collector.push("eval", {"script_status": "advance", "thought": "Keep it tight."})
    collector.push("response_guard", {"status": "pass", "issues": [], "rationale": "Stayed in character."})
    collector.push("expression", {"expression": "curious", "gaze": "forward"})
    collector.push("vision_focus", {
        "stage_goal": "Notice props near the table.",
        "constant_questions": ["Who is here?"],
        "ephemeral_questions": ["Is there a bottle visible?"],
        "constant_sam_targets": ["person"],
        "ephemeral_sam_targets": ["bottle"],
    })
    collector.push("perception", {
        "source": "visual",
        "description": "A person is now holding a flier",
        "source_trace": {
            "kind": "vision_synthesis",
            "provenance": {
                "label": "Vision synthesis LLM",
                "primary": "vision_synthesis_llm",
                "components": ["vlm_answers", "sam3_objects"],
                "inference": "llm synthesis over visual snapshot",
            },
            "object_labels": ["flier", "table"],
            "vlm_answers": [
                {"question": "What is the person holding?", "answer": "A flier.", "slot_type": "activity"},
            ],
            "supporting_answers": [
                {"question": "What is the person holding?", "answer": "A flier.", "slot_type": "activity"},
            ],
            "focus": {
                "ephemeral_questions": ["Is anyone holding a flier?"],
                "ephemeral_sam_targets": ["flier"],
            },
        },
    })

    expect(page.locator(".stream-card[data-event-type='user_speech_started']")).to_have_count(1)
    expect(page.locator(".stream-card[data-event-type='user_transcript_final']")).to_have_count(1)
    expect(page.locator('.stream-lane[data-stream-lane="behavior"] .stream-card').filter(has_text="guard pass")).to_be_visible()
    expect(page.locator('.stream-lane[data-stream-lane="behavior"] .stream-card').filter(has_text="curious")).to_be_visible()
    expect(page.locator('.stream-lane[data-stream-lane="other"] .stream-card').filter(has_text="guard pass")).to_have_count(0)
    reply_card = page.locator(".stream-card[data-event-type='assistant_reply']").filter(has_text="Free water. Want one?").first
    expect(reply_card.locator(".stream-card-uid")).to_have_text(re.compile(r"#\d+"))
    reply_card.click()
    expect(page.locator("#detail-seq")).to_have_text(re.compile(r"#\d+"))
    expect(page.locator("#inspector-summary-title")).to_have_text(re.compile(r"assistant_reply #\d+"))
    inventory = page.evaluate("window.__dashboardTest.eventInventory()")
    assert "assistant_reply" in inventory["visible_cards"]
    assert "post_response" in inventory["compound_cards"]
    assert "history_playback_frame" in inventory["hidden_runtime_events"]
    page.locator("#report-ref-button").click()
    expect(page.locator("#report-ref-status")).to_contain_text("Copied archive ref:")
    expect(page.locator("#report-ref-status")).to_contain_text("sess-playwright")
    expect(page.locator("#detail-type")).to_have_text("assistant_reply")
    expect(page.locator("#detail-label")).to_contain_text("Free water. Want one?")
    expect(page.locator("#detail-detail")).to_contain_text("speech 350ms")
    expect(page.locator("#detail-detail")).to_contain_text("STT 70ms")
    expect(page.locator("#detail-detail")).to_contain_text("TTFT 190ms")
    expect(page.locator("#detail-loop")).to_have_class(re.compile(r"\bactive\b"))
    expect(page.locator("#detail-loop-chips")).to_contain_text("speech 350ms")
    expect(page.locator("#detail-loop-chips")).to_contain_text("STT 70ms")
    expect(page.locator("#detail-loop-chips")).to_contain_text("TTFT 190ms")
    expect(page.locator("#detail-loop-chips")).to_contain_text("first audio")
    expect(page.locator("#detail-loop-chips")).not_to_contain_text("endpointing 0ms")
    page.locator(".stream-card[data-event-type='user_transcript_final']").first.click()
    expect(page.locator("#detail-type")).to_have_text("user_transcript_final")
    expect(page.locator("#detail-detail")).to_contain_text("speech 350ms")
    expect(page.locator("#detail-detail")).to_contain_text("STT 70ms")
    reply_card.click()
    expect(page.locator("#detail-prompts")).to_contain_text("Chat prompt")
    expect(page.locator("#detail-prompts")).to_contain_text("Free water. Want one?")
    expect(page.locator(".prompt-preview-card")).to_have_count(1)
    page.locator("[data-prompt-open='0']").evaluate("(node) => node.click()")
    expect(page.locator("#prompt-modal")).to_have_class(re.compile(r"\bopen\b"))
    expect(page.locator("#prompt-modal")).to_contain_text("Keep it short.")
    expect(page.locator("#prompt-modal")).to_contain_text("Do you have water?")
    expect(page.locator(".prompt-message.role-system")).to_have_count(2)
    expect(page.locator(".prompt-message.role-user")).to_have_count(1)
    expect(page.locator(".prompt-message.role-assistant")).to_have_count(1)
    page.locator("#prompt-modal-close").click()
    expect(page.locator("#inspector-annotations-panel")).to_be_hidden()
    page.locator("#archive-load-button").click()
    expect(page.locator("#archive-picker")).to_have_class(re.compile(r"\bopen\b"))
    page.locator(".archive-row").filter(has=page.locator(".archive-kind", has_text="session")).locator(".archive-load-action").first.click()
    expect(page.locator("#archive-load-button")).to_have_text("Return To Live")
    expect(page.locator("#archive-load-status")).to_contain_text("Browsing archive:")
    expect(page.locator("#runtime-toggle")).to_have_text("Play")
    expect(page.locator("#runtime-toggle")).to_be_disabled()
    expect(page.locator("#runtime-stop")).to_be_disabled()
    expect(page.locator(".stream-card[data-event-type='session_start']")).to_have_count(0)
    expect(page.locator("#archive-load-button")).to_have_text("Return To Live")
    page.locator("#archive-load-button").click()
    expect(page.locator("#archive-load-button")).to_have_text("Load Archive...")
    page.locator("#archive-load-button").click()
    page.locator(".archive-row").filter(has=page.locator(".archive-kind", has_text="session")).locator(".archive-replay-action").first.click()
    assert input_queue.get(timeout=2) == "/replay sess-playwright"
    replay_transcript_card = page.locator(".stream-card[data-event-type='user_transcript_final']").first
    replay_transcript_card.click()
    page.locator("#detail-open-json").click()
    expect(page.locator("#detail-payload")).to_contain_text("\"text\": \"Do you have water?\"")
    expect(page.locator("#inspector-related-panel")).to_have_count(0)
    expect(page.locator("#tab-chronology")).to_have_count(0)
    expect(page.locator("#show-chronology")).to_have_count(0)

    page.locator("#stream-density").select_option("compact")
    reply_card.click()
    expect(page.locator("#detail-type")).to_have_text("assistant_reply")

    perception_dot = page.locator(".stream-dot[data-event-type='perception']").first
    expect(perception_dot).to_be_visible()
    perception_dot.click()
    expect(page.locator("#detail-type")).to_have_text("perception")
    expect(page.locator("#inspector-summary-title")).to_have_text(re.compile(r"perception #\d+"))
    expect(page.locator("#detail-trace")).to_have_count(0)
    expect(page.locator("#detail-structure")).to_have_count(0)
    expect(page.locator("#detail-payload")).to_contain_text('"description": "A person is now holding a flier"')
    expect(page.locator("#detail-payload")).to_contain_text('"primary": "vision_synthesis_llm"')
    expect(page.locator("#detail-payload")).to_contain_text('"label": "Vision synthesis LLM"')
    expect(page.locator("#detail-payload")).to_contain_text('"ephemeral_sam_targets": [')
    expect(page.locator("#detail-payload")).to_contain_text('"flier"')
    expect(page.locator("#detail-payload")).to_contain_text('"supporting_answers": [')

    page.locator("#detail-continue-from-event").click()
    replay_cmd = input_queue.get(timeout=2)
    assert replay_cmd.startswith("/replay_event sess-playwright ")
    assert replay_cmd.endswith(" vision_synthesis_inputs")

    inspector_resizer = page.locator("#inspector-resizer")
    inspector_box = inspector_resizer.bounding_box()
    assert inspector_box is not None
    page.mouse.move(inspector_box["x"] + inspector_box["width"] / 2, inspector_box["y"] + inspector_box["height"] / 2)
    page.mouse.down()
    page.mouse.move(inspector_box["x"] - 500, inspector_box["y"] + inspector_box["height"] / 2)
    page.mouse.up()
    assert int(page.evaluate("parseInt(getComputedStyle(document.documentElement).getPropertyValue('--inspector-width'))")) > 760

    sidebar_resizer = page.locator("#sidebar-resizer")
    sidebar_box = sidebar_resizer.bounding_box()
    assert sidebar_box is not None
    page.mouse.move(sidebar_box["x"] + sidebar_box["width"] / 2, sidebar_box["y"] + sidebar_box["height"] / 2)
    page.mouse.down()
    page.mouse.move(sidebar_box["x"] + 90, sidebar_box["y"] + sidebar_box["height"] / 2)
    page.mouse.up()
    assert page.evaluate("getComputedStyle(document.documentElement).getPropertyValue('--right-panel-width').trim()") != "360px"

    page.locator("button[data-lane-toggle='vision']").click()
    expect(page.locator("[data-stream-lane='vision']")).to_have_class(re.compile(r"\bhidden\b"))
    page.locator("button[data-lane-toggle='vision']").click()
    expect(page.locator("[data-stream-lane='vision']")).not_to_have_class(re.compile(r"\bhidden\b"))

    stream_viewport = page.locator("#stream-viewport")
    viewport_box = stream_viewport.bounding_box()
    assert viewport_box is not None
    page.mouse.move(viewport_box["x"] + 120, viewport_box["y"] + 40)
    page.mouse.down()
    page.mouse.move(viewport_box["x"] + 320, viewport_box["y"] + 40)
    page.mouse.up()
    expect(page.locator("#follow-live")).not_to_be_checked()

    page.locator("#runtime-advanced-fold summary").evaluate("(node) => node.click()")
    with page.expect_popup() as vision_popup_info:
        page.locator("#vision-service-link").click()
    vision_popup = vision_popup_info.value
    vision_popup.wait_for_load_state("domcontentloaded")
    assert "Vision Test UI" in vision_popup.text_content("body")
    vision_popup.close()

    page.locator("#runtime-stop").click()
    expect(page.locator("#save-complete-modal")).to_have_class(re.compile(r"\bopen\b"))
    expect(page.locator("#save-complete-heading")).to_contain_text("Opening Stop Review")
    expect(page.locator("#save-complete-status")).to_contain_text("Pausing before review")
    assert input_queue.get(timeout=2) == "/review"
    collector.push(
        "runtime_controls",
        {
            "controls": {
                "reconcile": True,
                "vision": True,
                "auto_beat": False,
                "filler": False,
            },
            "conversation_paused": True,
            "session_stopped": False,
            "startup_pause_pending": False,
            "session_state": "paused",
            "runtime_phase": "review",
            "voice_active": True,
            "voice_status": {
                "active": True,
                "output_only": False,
                "filler_enabled": False,
                "tts_backend": "pocket",
                "mic_ready": True,
                "speaker_ready": True,
                "stt": {"state": "ready", "detail": "Deepgram socket open."},
                "tts": {"state": "ready", "detail": "Pocket-TTS reachable.", "backend": "pocket"},
            },
            "vision_active": True,
            "reconcile_thread_alive": True,
            "vision_service_url": vision_service_url,
            "vision_service_health": True,
            "vision_service_managed": False,
            "vision_service_external": True,
            "vision_service_state": "external",
            "vision_service_autostart": True,
            "vision_service_mode": "camera",
            "vision_mock_replay": "",
        },
    )
    expect(page.locator("#save-complete-heading")).to_contain_text("Stop Session")
    expect(page.locator("#save-complete-status")).to_contain_text("Paused. Save or discard this session.")
    expect(page.locator("#save-complete-title")).not_to_have_value("")
    expect(page.locator("#runtime-stop")).to_have_text("Stop Session")
    page.locator("#save-complete-title").fill("Greg stop flow")
    page.locator("#save-complete-save").click()
    expect(page.locator("#save-complete-status")).to_contain_text("Saving...")
    expect(page.locator("#runtime-stop")).to_have_text("Saving...")
    assert input_queue.get(timeout=2) == "/save"

    collector.push("session_end", {"total_turns": 2})
    expect(page.locator("body")).not_to_have_class(re.compile(r"\bbooting\b"))
    expect(page.locator("#save-complete-modal")).to_have_class(re.compile(r"\bopen\b"))
    expect(page.locator("#timeline")).to_contain_text("Free water. Want one?")
    collector.push(
        "runtime_controls",
        {
            "controls": {
                "reconcile": True,
                "vision": True,
                "auto_beat": False,
                "filler": False,
            },
            "conversation_paused": False,
            "session_stopped": True,
            "startup_pause_pending": False,
            "session_state": "ready",
            "voice_active": True,
            "voice_status": {
                "active": True,
                "output_only": False,
                "filler_enabled": False,
                "tts_backend": "pocket",
                "mic_ready": True,
                "speaker_ready": True,
                "stt": {"state": "ready", "detail": "Deepgram socket open."},
                "tts": {"state": "ready", "detail": "Pocket-TTS reachable.", "backend": "pocket"},
            },
            "vision_active": True,
            "reconcile_thread_alive": True,
            "vision_service_url": vision_service_url,
            "vision_service_health": True,
            "vision_service_managed": False,
            "vision_service_external": True,
            "vision_service_state": "external",
            "vision_service_autostart": True,
            "vision_service_mode": "camera",
            "vision_mock_replay": "",
        },
    )
    expect(page.locator("#runtime-status")).to_contain_text("state: ready")
    expect(page.locator("#runtime-start")).to_have_text("Start New Session")
    expect(page.locator("#runtime-stop")).to_be_disabled()
    expect(page.locator("#boot-summary")).to_contain_text("Everything is warm. Start when you want to begin a fresh session.")
    expect(page.locator("#report-ref-status")).to_contain_text("Last stopped session: intermediate session log:")
    assert list(report_dir.iterdir())
    expect(page.locator("#save-complete-session-id")).to_contain_text("sess-playwright")
    expect(page.locator("#save-complete-session-path")).to_contain_text("sess-playwright")
    expect(page.locator("#save-complete-report-path")).to_contain_text("sess-playwright")
    expect(page.locator("#save-complete-heading")).to_contain_text("Session Saved")
    expect(page.locator("#save-complete-status")).to_contain_text("Hiding in")
    page.locator("#save-complete-copy-ref").click()
    expect(page.locator("#save-complete-status")).to_contain_text("Copied archive ref:")
    page.locator("#save-complete-load").click()
    expect(page.locator("#save-complete-modal")).not_to_have_class(re.compile(r"\bopen\b"))
    expect(page.locator("#archive-load-button")).to_have_text("Return To Live")
    expect(page.locator("#archive-load-status")).to_contain_text("Greg stop flow")
    expect(page.locator("#runtime-toggle")).to_have_text("Play")
    expect(page.locator("#runtime-toggle")).to_be_disabled()
    page.locator("#archive-load-button").click()
    expect(page.locator("#archive-load-button")).to_have_text("Load Archive...")

    page.locator("#report-ref-button").click()
    expect(page.locator("#report-ref-status")).to_contain_text("Copied archive ref:")
    expect(page.locator("#report-ref-status")).to_contain_text("sess-playwright")

    expect(page.locator("#runtime-start")).to_be_enabled()
    page.locator("#runtime-start").click()
    expect(page.locator("#runtime-start")).to_have_text("Starting...")
    assert input_queue.get(timeout=2) == "/start"

    collector.push(
        "session_start",
        {
            "character": "greg",
            "model": "groq-llama-8b",
            "session_id": "sess-stopped-fresh",
            "stage": "watching",
        },
    )
    collector.push(
        "runtime_controls",
        {
            "controls": {
                "reconcile": True,
                "vision": True,
                "auto_beat": False,
                "filler": False,
            },
            "conversation_paused": True,
            "session_stopped": False,
            "startup_pause_pending": True,
            "session_state": "paused",
            "voice_active": True,
            "voice_status": {
                "active": True,
                "output_only": False,
                "filler_enabled": False,
                "tts_backend": "pocket",
                "mic_ready": True,
                "speaker_ready": True,
                "stt": {"state": "ready", "detail": "Deepgram socket open."},
                "tts": {"state": "ready", "detail": "Pocket-TTS reachable.", "backend": "pocket"},
            },
            "vision_active": True,
            "reconcile_thread_alive": True,
            "vision_service_url": vision_service_url,
            "vision_service_health": True,
            "vision_service_managed": False,
            "vision_service_external": True,
            "vision_service_state": "external",
            "vision_service_autostart": True,
            "vision_service_mode": "camera",
            "vision_mock_replay": "",
        },
    )
    collector.push("pause", {"startup": True, "fresh_session": True})
    expect(page.locator("#h-session")).to_have_text("sess-stopped-fresh")
    expect(page.locator("#runtime-toggle")).to_have_text("Play")
    expect(page.locator("#report-ref-status")).to_have_text("")

    page.locator("#runtime-controls-fold summary").evaluate("(node) => node.click()")
    page.locator("button[data-runtime-control='thinker']").click()
    assert input_queue.get(timeout=2) == "/threads thinker toggle"

    page.locator("button[data-runtime-control='guardrails']").click()
    assert input_queue.get(timeout=2) == "/threads guardrails toggle"

    page.keyboard.press("KeyB")
    assert input_queue.get(timeout=2) == "/beat"

    page.keyboard.press("Digit1")
    assert input_queue.get(timeout=2) == "1"

    expect(page.locator("#vision-model-status")).to_contain_text(re.compile("ready|loading"))

    collector.push(
        "runtime_controls",
        {
            "controls": {
                "reconcile": False,
                "vision": False,
                "auto_beat": True,
                "filler": True,
            },
            "conversation_paused": False,
            "session_stopped": False,
            "startup_pause_pending": False,
            "session_state": "live",
            "voice_active": True,
            "voice_status": {
                "active": True,
                "output_only": False,
                "filler_enabled": True,
                "tts_backend": "pocket",
                "mic_ready": True,
                "speaker_ready": True,
                "stt": {"state": "error", "detail": "Deepgram socket lost."},
                "tts": {"state": "ready", "detail": "Pocket-TTS reachable.", "backend": "pocket"},
            },
            "vision_active": False,
            "reconcile_thread_alive": False,
            "vision_service_url": vision_service_url,
            "vision_service_health": True,
            "vision_service_managed": True,
            "vision_service_external": False,
            "vision_service_state": "managed",
            "vision_service_autostart": False,
            "vision_service_mode": "camera",
            "vision_mock_replay": "walkup.json",
        },
    )

    expect(page.locator("#runtime-status")).to_contain_text("state: live")
    expect(page.locator("#runtime-toggle")).to_have_text("Pause")
    expect(page.locator("#vision-service-status")).to_contain_text("source: managed by app")
    expect(page.locator("#vision-service-status")).to_contain_text("mock: walkup.json")
    expect(page.locator("#runtime-status")).to_contain_text("stt: error")
    expect(page.locator("#boot-summary")).to_contain_text("Voice needs attention before the runtime is ready.")
    expect(page.locator("body")).to_have_class(re.compile(r"\bbooting\b"))

    expect(page.locator("#boot-overlay")).not_to_contain_text("Restart Session")

    collector.push(
        "session_start",
        {
            "character": "greg",
            "model": "groq-llama-8b",
            "session_id": "sess-restarted",
            "stage": "watching",
        },
    )
    collector.push(
        "runtime_controls",
        {
            "controls": {
                "reconcile": True,
                "vision": True,
                "auto_beat": True,
                "filler": True,
            },
            "conversation_paused": True,
            "session_stopped": False,
            "startup_pause_pending": True,
            "session_state": "paused",
            "voice_active": True,
            "voice_status": {
                "active": True,
                "output_only": False,
                "filler_enabled": True,
                "tts_backend": "pocket",
                "mic_ready": True,
                "speaker_ready": True,
                "stt": {"state": "ready", "detail": "Deepgram socket open."},
                "tts": {"state": "ready", "detail": "Pocket-TTS reachable.", "backend": "pocket"},
            },
            "vision_active": True,
            "reconcile_thread_alive": True,
            "vision_service_url": vision_service_url,
            "vision_service_health": True,
            "vision_service_managed": False,
            "vision_service_external": True,
            "vision_service_state": "external",
            "vision_service_autostart": True,
            "vision_service_mode": "camera",
            "vision_mock_replay": "",
        },
    )
    expect(page.locator("#h-session")).to_have_text("sess-restarted")
    expect(page.locator("#runtime-toggle")).to_have_text("Play")

    vision_service.set_fail(True)
    expect(page.locator("#vision-model-status")).to_contain_text("Vision status unavailable", timeout=7000)
    expect(page.locator("#vision-service-status")).to_contain_text("health: unreachable")
    expect(page.locator("#vision-service-status")).to_contain_text("service: error")
    expect(page.locator("body")).to_have_class(re.compile(r"\bbooting\b"))


def test_rendered_card_inventory_inspector_sweep(
    browser_page: Page,
    dashboard_server,
):
    collector, _, dashboard_url, _ = dashboard_server
    page = browser_page
    page.goto(dashboard_url)

    specs = _seed_rendered_card_inventory_fixture(collector)
    raw_types = {"vision_snapshot_read", "face_tracking", "sam3_detection", "reid_track", "vlm_answer"}
    expected_rendered = set(specs) - raw_types

    page.wait_for_function(
        """(expectedCount) => {
            const items = Array.from(document.querySelectorAll('.stream-card[data-event-type], .stream-dot[data-event-type]'));
            return items.length >= expectedCount;
        }""",
        arg=len(expected_rendered),
    )

    inventory = page.evaluate("window.__dashboardTest.eventInventory()")
    assert set(inventory["rendered_cards"]) == expected_rendered
    assert set(inventory["visible_cards"]) == expected_rendered
    assert set(inventory["raw_lane_events"]) == raw_types
    assert set(inventory["folded_runtime_events"]) == {
        "response_chunk",
        "response_done",
        "eval",
        "director",
        "plan",
        "beat_advance",
        "stage_change",
    }

    rendered_dom_types = set(
        page.evaluate(
            """() => Array.from(
                new Set(
                    Array.from(document.querySelectorAll('.stream-card[data-event-type], .stream-dot[data-event-type]'))
                        .map((node) => node.getAttribute('data-event-type'))
                        .filter(Boolean)
                )
            )"""
        )
    )
    assert rendered_dom_types == expected_rendered
    expect(page.locator(".stream-card[data-event-type='plan'], .stream-dot[data-event-type='plan']")).to_have_count(0)
    expect(page.locator(".stream-card[data-event-type='beat_advance'], .stream-dot[data-event-type='beat_advance']")).to_have_count(0)
    expect(page.locator(".stream-card[data-event-type='stage_change'], .stream-dot[data-event-type='stage_change']")).to_have_count(0)

    for event_type in inventory["rendered_cards"]:
        spec = specs[event_type]
        has_card_text = bool(spec["card_text"])
        locator = page.locator(f".stream-card[data-event-type='{event_type}'], .stream-dot[data-event-type='{event_type}']")
        if has_card_text:
            locator = locator.filter(has_text=str(spec["card_text"]))
        expect(locator.first).to_be_visible()
        locator.first.click()
        expect(page.locator("#inspector-summary-title")).to_have_text(re.compile(rf"{re.escape(event_type)} #\d+"))
        expect(page.locator("#detail-type")).to_have_text(event_type)
        expect(page.locator("#detail-trace")).to_have_count(0)
        expect(page.locator("#detail-structure")).to_have_count(0)
        for snippet in spec["payload_snippets"]:
            expect(page.locator("#detail-payload")).to_contain_text(str(snippet))

    for event_type in inventory["raw_lane_events"]:
        spec = specs[event_type]
        page.wait_for_function(f"() => !!window.__dashboardTest.visionRawEventPoint('{event_type}', 0)")
        raw_point = page.evaluate(f"window.__dashboardTest.visionRawEventPoint('{event_type}', 0)")
        assert raw_point is not None
        page.mouse.click(raw_point["clientX"], raw_point["clientY"])
        expect(page.locator("#inspector-summary-title")).to_have_text(re.compile(rf"{re.escape(event_type)} #\d+"))
        expect(page.locator("#detail-type")).to_have_text(event_type)
        expect(page.locator("#detail-trace")).to_have_count(0)
        expect(page.locator("#detail-structure")).to_have_count(0)
        for snippet in spec["payload_snippets"]:
            expect(page.locator("#detail-payload")).to_contain_text(str(snippet))


def test_boot_overlay_ignores_optional_vision_when_runtime_not_active(
    browser_page: Page,
    dashboard_server,
    vision_service: VisionServiceController,
):
    collector, _, dashboard_url, _ = dashboard_server
    page = browser_page
    page.goto(dashboard_url)

    collector.push(
        "runtime_controls",
        {
            "controls": {
                "reconcile": True,
                "vision": True,
                "auto_beat": True,
                "filler": True,
            },
            "conversation_paused": False,
            "session_stopped": True,
            "startup_pause_pending": False,
            "session_state": "ready",
            "voice_active": True,
            "voice_status": {
                "active": True,
                "output_only": False,
                "filler_enabled": True,
                "tts_backend": "pocket",
                "mic_ready": True,
                "speaker_ready": True,
                "stt": {"state": "ready", "detail": "Deepgram socket open."},
                "tts": {"state": "ready", "detail": "Pocket-TTS reachable.", "backend": "pocket"},
            },
            "vision_active": False,
            "reconcile_thread_alive": True,
            "vision_service_url": vision_service.url,
            "vision_service_health": True,
            "vision_service_managed": False,
            "vision_service_external": True,
            "vision_service_state": "external",
            "vision_service_autostart": True,
            "vision_service_mode": "camera",
            "vision_mock_replay": "",
        },
    )

    expect(page.locator("body")).not_to_have_class(re.compile(r"\bbooting\b"))
    expect(page.locator("#boot-overlay")).not_to_be_visible()
    expect(page.locator("#boot-summary")).to_contain_text("Everything is warm. Start when you want to begin a fresh session.")
    expect(page.locator("#boot-grid")).to_contain_text("Vision runtime is inactive for this session.")
    expect(page.locator("#boot-grid")).to_contain_text("Vision models are inactive for this session.")
    expect(page.locator("#runtime-start")).to_have_text("Start New Session")


def test_bridge_vision_feed_src_keeps_session_token(
    browser_page: Page,
    dashboard_server,
    vision_service: VisionServiceController,
):
    collector, _, dashboard_url, _ = dashboard_server
    page = browser_page
    page.add_init_script(
        """
        window.__bridgeConfig = {
          enabled: true,
          accessToken: "bridge-secret",
          visionFeedPath: "/bridge/video_feed"
        };
        """
    )
    page.goto(dashboard_url)

    collector.push(
        "runtime_controls",
        {
            "controls": {
                "reconcile": True,
                "vision": True,
                "auto_beat": True,
                "filler": True,
            },
            "conversation_paused": False,
            "session_stopped": True,
            "startup_pause_pending": False,
            "session_state": "ready",
            "voice_active": True,
            "voice_status": {
                "active": True,
                "output_only": False,
                "filler_enabled": True,
                "tts_backend": "pocket",
                "mic_ready": True,
                "speaker_ready": True,
                "stt": {"state": "ready", "detail": "Deepgram socket open."},
                "tts": {"state": "ready", "detail": "Pocket-TTS reachable.", "backend": "pocket"},
            },
            "vision_active": True,
            "reconcile_thread_alive": True,
            "vision_service_url": vision_service.url,
            "vision_service_health": True,
            "vision_service_managed": False,
            "vision_service_external": True,
            "vision_service_state": "external",
            "vision_service_autostart": True,
            "vision_service_mode": "camera",
            "vision_mock_replay": "",
        },
    )

    expect(page.locator("#vision-feed")).to_have_attribute("src", re.compile(r"/bridge/video_feed\?token=bridge-secret$"))
    expect(page.locator("#vision-service-link")).to_have_attribute("href", re.compile(r"/bridge/video_feed\?token=bridge-secret$"))


def test_boot_overlay_can_open_archive_picker_before_runtime_ready(
    browser_page: Page,
    dashboard_server,
):
    _, _, dashboard_url, _ = dashboard_server
    page = browser_page
    page.goto(dashboard_url)

    expect(page.locator("#boot-overlay")).to_be_visible()
    expect(page.locator("#boot-load-archive")).to_be_visible()
    page.locator("#boot-load-archive").click()
    expect(page.locator("#archive-picker")).to_have_attribute("aria-hidden", "false")
    expect(page.locator("#archive-picker-list")).to_contain_text("sess-archive-playback")


def test_boot_overlay_runtime_toggle_controls_blocked_states(
    browser_page: Page,
    dashboard_server,
):
    collector, input_queue, dashboard_url, _ = dashboard_server
    page = browser_page
    page.goto(dashboard_url)

    collector.push(
        "runtime_controls",
        {
            "controls": {
                "reconcile": True,
                "vision": False,
                "auto_beat": False,
                "filler": False,
            },
            "conversation_paused": False,
            "session_stopped": True,
            "startup_pause_pending": False,
            "session_state": "ready",
            "voice_active": True,
            "voice_status": {
                "active": True,
                "output_only": False,
                "filler_enabled": False,
                "tts_backend": "pocket",
                "mic_ready": True,
                "speaker_ready": True,
                "stt": {"state": "starting", "detail": "Mic path warming."},
                "tts": {"state": "ready", "detail": "Pocket-TTS reachable.", "backend": "pocket"},
            },
            "vision_active": False,
            "reconcile_thread_alive": True,
            "vision_service_url": "",
            "vision_service_health": False,
            "vision_service_managed": False,
            "vision_service_external": False,
            "vision_service_state": "stopped",
            "vision_service_autostart": False,
            "vision_service_mode": "camera",
            "vision_mock_replay": "",
        },
    )

    expect(page.locator("body")).to_have_class(re.compile(r"\bbooting\b"))
    expect(page.locator("body")).not_to_have_class(re.compile(r"\bboot-blocking\b"))
    expect(page.locator("#boot-overlay")).to_be_visible()
    expect(page.locator("#boot-overlay-summary")).to_contain_text("before the runtime can unlock.")
    expect(page.locator("#boot-runtime-toggle")).to_have_text("Start")
    page.locator("#boot-runtime-toggle").click()
    expect(page.locator("#boot-runtime-toggle")).to_have_text("Starting...")
    assert input_queue.get(timeout=2) == "/start"

    collector.push(
        "session_start",
        {
            "character": "greg",
            "model": "groq-llama-8b",
            "session_id": "sess-playwright",
            "stage": "watching",
        },
    )
    collector.push(
        "runtime_controls",
        {
            "controls": {
                "reconcile": True,
                "vision": False,
                "auto_beat": False,
                "filler": False,
            },
            "conversation_paused": True,
            "session_stopped": False,
            "startup_pause_pending": True,
            "session_state": "paused",
            "voice_active": True,
            "voice_status": {
                "active": True,
                "output_only": False,
                "filler_enabled": False,
                "tts_backend": "pocket",
                "mic_ready": True,
                "speaker_ready": True,
                "stt": {"state": "starting", "detail": "Mic path warming."},
                "tts": {"state": "ready", "detail": "Pocket-TTS reachable.", "backend": "pocket"},
            },
            "vision_active": False,
            "reconcile_thread_alive": True,
            "vision_service_url": "",
            "vision_service_health": False,
            "vision_service_managed": False,
            "vision_service_external": False,
            "vision_service_state": "stopped",
            "vision_service_autostart": False,
            "vision_service_mode": "camera",
            "vision_mock_replay": "",
        },
    )
    collector.push("pause", {"startup": True, "fresh_session": True})

    expect(page.locator("body")).to_have_class(re.compile(r"\bbooting\b"))
    expect(page.locator("body")).not_to_have_class(re.compile(r"\bboot-blocking\b"))
    expect(page.locator("#boot-overlay")).to_be_visible()
    expect(page.locator("#boot-overlay-summary")).to_contain_text("before the runtime can unlock.")
    expect(page.locator("#boot-runtime-toggle")).to_have_text("Play")
    page.locator("#boot-runtime-toggle").click()
    expect(page.locator("#boot-runtime-toggle")).to_have_text("Playing...")
    assert input_queue.get(timeout=2) == "/play"


def test_offline_fixture_ready_renders_start_frame(
    browser_page: Page,
    dashboard_server,
):
    _, _, dashboard_url, _ = dashboard_server
    page = browser_page
    page.goto(f"{dashboard_url}?fixture=ready")

    expect(page.locator("body")).not_to_have_class(re.compile(r"\bbooting\b"))
    expect(page.locator("#boot-overlay")).not_to_be_visible()
    expect(page.locator("#boot-summary")).to_contain_text("Everything is warm. Start when you want to begin a fresh session.")
    expect(page.locator("#runtime-start")).to_have_text("Start New Session")
    expect(page.locator("#runtime-status")).to_contain_text("state: ready")
    expect(page.locator("#runtime-status")).to_contain_text("voice: on")


def test_archive_only_runtime_disables_live_controls(
    browser_page: Page,
    dashboard_server,
):
    collector, _, dashboard_url, _ = dashboard_server
    page = browser_page
    page.goto(dashboard_url)

    collector.push(
        "runtime_controls",
        {
            "controls": {
                "reconcile": True,
                "vision": True,
                "auto_beat": False,
                "filler": False,
            },
            "conversation_paused": True,
            "session_stopped": True,
            "startup_pause_pending": False,
            "session_state": "ready",
            "voice_active": False,
            "voice_status": {
                "active": False,
                "output_only": False,
                "filler_enabled": False,
                "tts_backend": "",
                "mic_ready": False,
                "speaker_ready": False,
                "stt": {"state": "off", "detail": "Voice inactive."},
                "tts": {"state": "off", "detail": "Voice inactive.", "backend": ""},
            },
            "vision_active": False,
            "reconcile_thread_alive": False,
            "vision_service_url": "",
            "vision_service_health": False,
            "vision_service_managed": False,
            "vision_service_external": False,
            "vision_service_state": "stopped",
            "vision_service_autostart": False,
            "vision_service_mode": "camera",
            "vision_mock_replay": "",
            "archive_only": True,
            "archive_only_reason": "Archive-only mode is active. Load archives to inspect them; live conversation is disabled.",
        },
    )

    expect(page.locator("body")).not_to_have_class(re.compile(r"\bbooting\b"))
    expect(page.locator("#boot-summary")).to_contain_text("Archive-only mode is ready.")
    expect(page.locator("#runtime-status")).to_contain_text("mode: archive only")
    expect(page.locator("#runtime-start")).to_have_text("Start New Session")
    expect(page.locator("#runtime-start")).to_be_disabled()
    expect(page.locator("#runtime-toggle")).to_have_text("Play")
    expect(page.locator("#runtime-toggle")).to_be_disabled()
    expect(page.locator("#runtime-panel")).not_to_contain_text("Restart Session")
    expect(page.locator("#boot-overlay")).not_to_contain_text("Restart Session")
    expect(page.locator("#vision-service-status")).to_contain_text("source: archive-only")

    page.locator("#archive-load-button").click()
    expect(page.locator("#archive-picker")).to_have_attribute("aria-hidden", "false")
    expect(page.locator(".archive-replay-action").first).to_be_disabled()


def test_archive_query_autoload_hides_boot_overlay(
    browser_page: Page,
    dashboard_server,
):
    collector, _, dashboard_url, _ = dashboard_server
    collector.push(
        "runtime_controls",
        {
            "controls": {
                "reconcile": True,
                "vision": True,
                "auto_beat": False,
                "filler": False,
            },
            "conversation_paused": True,
            "session_stopped": True,
            "startup_pause_pending": False,
            "session_state": "ready",
            "voice_active": False,
            "voice_status": {
                "active": False,
                "output_only": False,
                "filler_enabled": False,
                "tts_backend": "",
                "mic_ready": False,
                "speaker_ready": False,
                "stt": {"state": "off", "detail": "Voice inactive."},
                "tts": {"state": "off", "detail": "Voice inactive.", "backend": ""},
            },
            "vision_active": False,
            "reconcile_thread_alive": False,
            "vision_service_url": "",
            "vision_service_health": False,
            "vision_service_managed": False,
            "vision_service_external": False,
            "vision_service_state": "stopped",
            "vision_service_autostart": False,
            "vision_service_mode": "camera",
            "vision_mock_replay": "",
            "archive_only": True,
            "archive_only_reason": "Archive-only mode is active. Load archives to inspect them; live conversation is disabled.",
        },
    )
    page = browser_page
    page.goto(f"{dashboard_url}?archive=sess-archive-playback")

    expect(page.locator("#archive-load-button")).to_have_text("Return To Live")
    expect(page.locator("#archive-load-status")).to_contain_text("Browsing archive:")
    expect(page.locator(".stream-card").filter(has_text="Hey, what time is it to you?")).to_have_count(1)
    expect(page.locator("#runtime-status")).to_contain_text("mode: archive only")
    expect(page.locator("#runtime-status")).to_contain_text("voice: off")
    expect(page.locator("#vision-model-status")).to_contain_text("Archive browsing does not poll live vision models.")
    expect(page.locator("body")).not_to_have_class(re.compile(r"\bbooting\b"))
    expect(page.locator("#boot-overlay")).not_to_be_visible()


def test_archive_query_loading_boot_override_keeps_real_archive_loaded(
    browser_page: Page,
    dashboard_server,
):
    collector, _, dashboard_url, _ = dashboard_server
    collector.push(
        "runtime_controls",
        {
            "controls": {
                "reconcile": True,
                "vision": True,
                "auto_beat": False,
                "filler": False,
            },
            "conversation_paused": True,
            "session_stopped": True,
            "startup_pause_pending": False,
            "session_state": "ready",
            "voice_active": False,
            "voice_status": {
                "active": False,
                "output_only": False,
                "filler_enabled": False,
                "tts_backend": "",
                "mic_ready": False,
                "speaker_ready": False,
                "stt": {"state": "off", "detail": "Voice inactive."},
                "tts": {"state": "off", "detail": "Voice inactive.", "backend": ""},
            },
            "vision_active": False,
            "reconcile_thread_alive": False,
            "vision_service_url": "",
            "vision_service_health": False,
            "vision_service_managed": False,
            "vision_service_external": False,
            "vision_service_state": "stopped",
            "vision_service_autostart": False,
            "vision_service_mode": "camera",
            "vision_mock_replay": "",
            "archive_only": True,
            "archive_only_reason": "Archive-only mode is active. Load archives to inspect them; live conversation is disabled.",
        },
    )
    page = browser_page
    page.goto(f"{dashboard_url}?archive=greg-offline-canonical-v1&boot=loading")

    expect(page).to_have_url(re.compile(r"archive=greg-offline-canonical-v1"))
    expect(page.locator("#offline-frame-select")).to_have_value("loading")
    expect(page.locator("#ux-spoof-banner")).to_contain_text("Spoofed For UX")
    expect(page.locator("#ux-spoof-banner")).to_contain_text("Greg Offline Canonical v1")
    expect(page.locator(".stream-card").filter(has_text="Canonical archive is ready for offline UX work.")).to_have_count(1)
    expect(page.locator("body")).to_have_class(re.compile(r"\bbooting\b"))
    expect(page.locator("#boot-overlay")).to_be_visible()
    expect(page.locator("#boot-overlay-summary")).to_contain_text("Connecting to the local runtime")
    expect(page.locator("#boot-overlay-sub")).not_to_contain_text("Offline boot override")
    expect(page.locator("#vision-service-status")).not_to_contain_text("Failed to fetch")


def test_archive_query_ready_boot_override_uses_start_frame_with_real_archive(
    browser_page: Page,
    dashboard_server,
):
    collector, _, dashboard_url, _ = dashboard_server
    collector.push(
        "runtime_controls",
        {
            "controls": {
                "reconcile": True,
                "vision": True,
                "auto_beat": False,
                "filler": False,
            },
            "conversation_paused": True,
            "session_stopped": True,
            "startup_pause_pending": False,
            "session_state": "ready",
            "voice_active": False,
            "voice_status": {
                "active": False,
                "output_only": False,
                "filler_enabled": False,
                "tts_backend": "",
                "mic_ready": False,
                "speaker_ready": False,
                "stt": {"state": "off", "detail": "Voice inactive."},
                "tts": {"state": "off", "detail": "Voice inactive.", "backend": ""},
            },
            "vision_active": False,
            "reconcile_thread_alive": False,
            "vision_service_url": "",
            "vision_service_health": False,
            "vision_service_managed": False,
            "vision_service_external": False,
            "vision_service_state": "stopped",
            "vision_service_autostart": False,
            "vision_service_mode": "camera",
            "vision_mock_replay": "",
            "archive_only": True,
            "archive_only_reason": "Archive-only mode is active. Load archives to inspect them; live conversation is disabled.",
        },
    )
    page = browser_page
    page.goto(f"{dashboard_url}?archive=greg-offline-canonical-v1&boot=ready")

    expect(page.locator("#offline-frame-select")).to_have_value("ready")
    expect(page.locator("#ux-spoof-banner")).to_contain_text("Spoofed For UX")
    expect(page.locator(".stream-card").filter(has_text="Canonical archive is ready for offline UX work.")).to_have_count(1)
    expect(page.locator("body")).not_to_have_class(re.compile(r"\bbooting\b"))
    expect(page.locator("#boot-overlay")).not_to_be_visible()
    expect(page.locator("#boot-summary")).to_contain_text("Everything is warm. Start when you want to begin a fresh session.")
    expect(page.locator("#boot-runtime-toggle")).to_have_text("Start")
    expect(page.locator("#vision-service-status")).to_contain_text("service: ready")
    expect(page.locator("#vision-service-status")).not_to_contain_text("Failed to fetch")
    expect(page.locator("#vision-model-status")).to_contain_text("vLLM")


def test_canonical_archive_query_renders_offline_archive_content(
    browser_page: Page,
    dashboard_server,
):
    collector, _, dashboard_url, _ = dashboard_server
    collector.push(
        "runtime_controls",
        {
            "controls": {
                "reconcile": True,
                "vision": True,
                "auto_beat": False,
                "filler": False,
            },
            "conversation_paused": True,
            "session_stopped": True,
            "startup_pause_pending": False,
            "session_state": "ready",
            "voice_active": False,
            "voice_status": {
                "active": False,
                "output_only": False,
                "filler_enabled": False,
                "tts_backend": "",
                "mic_ready": False,
                "speaker_ready": False,
                "stt": {"state": "off", "detail": "Voice inactive."},
                "tts": {"state": "off", "detail": "Voice inactive.", "backend": ""},
            },
            "vision_active": False,
            "reconcile_thread_alive": False,
            "vision_service_url": "",
            "vision_service_health": False,
            "vision_service_managed": False,
            "vision_service_external": False,
            "vision_service_state": "stopped",
            "vision_service_autostart": False,
            "vision_service_mode": "camera",
            "vision_mock_replay": "",
            "archive_only": True,
            "archive_only_reason": "Archive-only mode is active. Load archives to inspect them; live conversation is disabled.",
        },
    )
    page = browser_page
    page.goto(f"{dashboard_url}?archive=greg-offline-canonical-v1")

    expect(page).to_have_url(re.compile(r"archive=greg-offline-canonical-v1"))
    expect(page.locator("#archive-load-button")).to_have_text("Return To Live")
    expect(page.locator("#archive-load-status")).to_contain_text("Browsing archive: Greg Offline Canonical v1")
    expect(page.locator("#runtime-status")).to_contain_text("mode: archive only")
    expect(page.locator(".stream-card").filter(has_text="Canonical archive is ready for offline UX work.")).to_have_count(1)
    expect(page.get_by_role("button", name="vision", exact=True)).to_be_visible()
    expect(page.get_by_role("button", name="vision raw", exact=True)).to_be_visible()
    expect(page.locator("body")).not_to_have_class(re.compile(r"\bbooting\b"))
    expect(page.locator("#boot-overlay")).not_to_be_visible()


def test_legacy_archive_keeps_vision_and_vision_raw_lanes_visible_offline(
    browser_page: Page,
    dashboard_server,
):
    collector, _, dashboard_url, _ = dashboard_server
    collector.push(
        "runtime_controls",
        {
            "controls": {
                "reconcile": True,
                "vision": True,
                "auto_beat": False,
                "filler": False,
            },
            "conversation_paused": True,
            "session_stopped": True,
            "startup_pause_pending": False,
            "session_state": "ready",
            "voice_active": False,
            "voice_status": {
                "active": False,
                "output_only": False,
                "filler_enabled": False,
                "tts_backend": "",
                "mic_ready": False,
                "speaker_ready": False,
                "stt": {"state": "off", "detail": "Voice inactive."},
                "tts": {"state": "off", "detail": "Voice inactive.", "backend": ""},
            },
            "vision_active": False,
            "reconcile_thread_alive": False,
            "vision_service_url": "",
            "vision_service_health": False,
            "vision_service_managed": False,
            "vision_service_external": False,
            "vision_service_state": "stopped",
            "vision_service_autostart": False,
            "vision_service_mode": "camera",
            "vision_mock_replay": "",
            "archive_only": True,
            "archive_only_reason": "Archive-only mode is active. Load archives to inspect them; live conversation is disabled.",
        },
    )
    page = browser_page
    page.goto(f"{dashboard_url}?archive=sess-archive-legacy-vision")

    expect(page.get_by_role("button", name="vision", exact=True)).to_be_visible()
    expect(page.get_by_role("button", name="vision raw", exact=True)).to_be_visible()
    expect(page.locator("[data-stream-lane='vision'] .stream-dot[data-event-type='vision_pass']")).to_have_count(1)
    expect(page.locator("[data-stream-lane='vision_raw']")).to_be_visible()
    expect(page.locator("[data-stream-lane='vision_raw']")).to_contain_text("No raw vision loop events captured yet")


def test_vision_state_event_shows_stacked_prompt_groups_in_browser(
    browser_page: Page,
    dashboard_server,
    vision_service: VisionServiceController,
):
    collector, _, dashboard_url, _ = dashboard_server
    page = browser_page
    page.goto(dashboard_url)

    collector.push(
        "session_start",
        {
            "character": "greg",
            "model": "groq-llama-8b",
            "session_id": "sess-vision-snippet",
            "stage": "watching",
        },
    )
    collector.push(
        "runtime_controls",
        {
            "controls": {
                "reconcile": True,
                "vision": True,
                "auto_beat": False,
                "filler": False,
            },
            "conversation_paused": False,
            "session_stopped": False,
            "startup_pause_pending": False,
            "session_state": "running",
            "voice_active": True,
            "voice_status": {
                "active": True,
                "output_only": False,
                "filler_enabled": False,
                "tts_backend": "pocket",
                "mic_ready": True,
                "speaker_ready": True,
                "stt": {"state": "ready", "detail": "Deepgram socket open."},
                "tts": {"state": "ready", "detail": "Pocket-TTS reachable.", "backend": "pocket"},
            },
            "vision_active": True,
            "reconcile_thread_alive": True,
            "vision_service_url": vision_service.url,
            "vision_service_health": True,
            "vision_service_managed": False,
            "vision_service_external": True,
            "vision_service_state": "external",
            "vision_service_autostart": True,
            "vision_service_mode": "camera",
            "vision_mock_replay": "",
        },
    )
    expect(page.locator("body")).not_to_have_class(re.compile(r"\bbooting\b"))
    collector.push(
        "vision_state_update",
        {
            "summary": "Visitor now has a backpack and a wall clock is visible.",
            "task_answers": [
                {
                    "task_id": "world_state",
                    "label": "World State",
                    "question": "What changed in the room?",
                    "answer": "A wall clock is visible above the table.",
                    "interpret_as": "world_state",
                    "target": "scene",
                },
                {
                    "task_id": "person_description_static",
                    "label": "Person Description Static",
                    "question": "Describe the highlighted person.",
                    "answer": "The highlighted visitor is wearing a black backpack.",
                    "interpret_as": "person_description_static",
                    "target": "person",
                    "target_person_id": "p1",
                },
            ],
            "remove_facts": [],
            "add_facts": ["A wall clock is visible above the table."],
            "events": [],
            "person_updates": [
                {
                    "person_id": "p1",
                    "remove_facts": [],
                    "add_facts": ["The visitor is wearing a black backpack."],
                    "set_name": None,
                    "set_presence": None,
                }
            ],
        },
    )

    dot = page.locator(".stream-dot[data-event-type='vision_state_update']").first
    expect(dot).to_be_visible()
    dot.click()
    expect(page.locator("#inspector-annotations-panel")).to_be_hidden()
    expect(page.locator("#detail-prompts")).to_contain_text("Summary")
    expect(page.locator("#detail-prompts")).to_contain_text("Inputs")
    expect(page.locator("#detail-prompts")).to_contain_text("Effects")
    expect(page.locator("#detail-prompts")).to_contain_text("A wall clock is visible above the table.")
    expect(page.locator("#detail-prompts")).to_contain_text("The visitor is wearing a black backpack.")


def test_vision_state_update_inspector_groups_derived_perception_and_hides_cycle_metadata(
    browser_page: Page,
    dashboard_server,
):
    collector, _, dashboard_url, _ = dashboard_server
    page = browser_page
    page.goto(dashboard_url)

    collector.push(
        "vision_state_update",
        {
            "event_id": "vision-cycle-77:vision_state_update",
            "vision_cycle_id": "vision-cycle-77",
            "summary": "A wall clock is visible.",
            "task_answers": [
                {
                    "task_id": "world_state",
                    "label": "World State",
                    "question": "Describe the room.",
                    "answer": "A wall clock is visible.",
                }
            ],
            "add_facts": ["A wall clock is visible."],
            "remove_facts": [],
            "events": [],
            "person_updates": [],
        },
    )
    collector.push(
        "perception",
        {
            "event_id": "perception-77",
            "input_event_ids": ["vision-cycle-77:vision_state_update"],
            "kind": "vision_state_update",
            "source": "visual",
            "description": "A wall clock is visible.",
        },
    )

    dot = page.locator(".stream-dot[data-event-type='vision_state_update']").first
    expect(dot).to_be_visible()
    dot.evaluate("(node) => node.click()")
    expect(page.locator("#detail-prompts")).to_contain_text("Derived Perception Updates")
    expect(page.locator("#detail-prompts")).to_contain_text("A wall clock is visible.")
    expect(page.locator("#detail-payload")).not_to_contain_text("vision_cycle_id")
    expect(page.locator("#detail-payload")).not_to_contain_text("vision-cycle-77")
    expect(page.locator("#detail-payload")).to_contain_text("vision-snapshot-77:vision_state_update")


def test_vision_panel_stays_visible_without_source(
    browser_page: Page,
    dashboard_server,
):
    collector, _, dashboard_url, _ = dashboard_server
    page = browser_page
    page.goto(dashboard_url)

    collector.push(
        "runtime_controls",
        {
            "controls": {
                "reconcile": True,
                "vision": True,
                "auto_beat": False,
                "filler": False,
            },
            "conversation_paused": False,
            "session_stopped": True,
            "startup_pause_pending": False,
            "session_state": "ready",
            "voice_active": True,
            "voice_status": {
                "active": True,
                "output_only": False,
                "filler_enabled": False,
                "tts_backend": "pocket",
                "mic_ready": True,
                "speaker_ready": True,
                "stt": {"state": "ready", "detail": "Deepgram socket open."},
                "tts": {"state": "ready", "detail": "Pocket-TTS reachable.", "backend": "pocket"},
            },
            "vision_active": True,
            "reconcile_thread_alive": True,
            "vision_service_url": "",
            "vision_service_health": False,
            "vision_service_managed": False,
            "vision_service_external": False,
            "vision_service_state": "stopped",
            "vision_service_autostart": True,
            "vision_service_mode": "camera",
            "vision_mock_replay": "",
        },
    )

    expect(page.locator("#runtime-status")).to_contain_text("state: ready")
    expect(page.locator("#vision-panel")).to_be_visible()
    expect(page.locator("#vision-panel")).to_have_attribute("data-panel-id", "vision")
    expect(page.locator("#vision-service-status")).to_contain_text("source: missing")
    expect(page.locator("#vision-service-status")).to_contain_text("No live vision feed or archive source is available.")
    expect(page.locator("#boot-summary")).to_contain_text("Vision needs attention before the runtime is ready.")


def test_vision_raw_lane_is_explicit_and_shows_placeholder_or_events(
    browser_page: Page,
    dashboard_server,
):
    collector, _, dashboard_url, _ = dashboard_server
    page = browser_page
    page.goto(dashboard_url)

    expect(page.locator("button[data-lane-toggle='vision_raw']")).to_have_text("vision raw")

    collector.push(
        "session_start",
        {
            "character": "greg",
            "model": "groq-llama-8b",
            "session_id": "sess-vision-raw",
            "stage": "watching",
        },
    )
    collector.push(
        "runtime_controls",
        {
            "controls": {
                "reconcile": True,
                "vision": True,
                "auto_beat": False,
                "filler": False,
            },
            "conversation_paused": False,
            "session_stopped": False,
            "startup_pause_pending": False,
            "session_state": "live",
            "voice_active": True,
            "voice_status": {
                "active": True,
                "output_only": False,
                "filler_enabled": False,
                "tts_backend": "pocket",
                "mic_ready": True,
                "speaker_ready": True,
                "stt": {"state": "ready", "detail": "Deepgram socket open."},
                "tts": {"state": "ready", "detail": "Pocket-TTS reachable.", "backend": "pocket"},
            },
            "vision_active": True,
            "reconcile_thread_alive": True,
            "vision_service_url": "",
            "vision_service_health": False,
            "vision_service_managed": False,
            "vision_service_external": False,
            "vision_service_state": "stopped",
            "vision_service_autostart": True,
            "vision_service_mode": "camera",
            "vision_mock_replay": "",
        },
    )
    collector.push(
        "vision_focus",
        {
            "stage_goal": "Check whether a clock is visible.",
            "constant_questions": ["What changed in the room?"],
            "ephemeral_questions": ["Is there a clock visible?"],
            "constant_sam_targets": ["person"],
            "ephemeral_sam_targets": ["clock"],
        },
    )

    expect(page.locator("[data-stream-lane='vision_raw']")).to_be_visible()
    expect(page.locator("[data-stream-lane='vision_raw']")).to_contain_text("No raw vision loop events captured yet")

    collector.push(
        "sam3_detection",
        {
            "persons": [{"identity": "visitor-1", "bbox": [10, 20, 120, 200]}],
            "objects": [{"label": "clock", "bbox": [220, 40, 60, 60]}],
        },
    )
    collector.push(
        "vision_state_update",
        {
            "summary": "A clock is visible above the table.",
            "task_answers": [
                {
                    "task_id": "world_state",
                    "label": "World State",
                    "question": "What changed in the room?",
                    "answer": "A clock is visible above the table.",
                    "interpret_as": "world_state",
                    "target": "scene",
                }
            ],
            "remove_facts": [],
            "add_facts": ["A clock is visible above the table."],
            "events": [],
            "person_updates": [],
        },
    )

    page.wait_for_function("() => !!window.__dashboardTest.visionRawEventPoint('sam3_detection', 0)")
    raw_sam3_point = page.evaluate("window.__dashboardTest.visionRawEventPoint('sam3_detection', 0)")
    assert raw_sam3_point is not None
    page.mouse.move(raw_sam3_point["clientX"], raw_sam3_point["clientY"])
    expect(page.locator("#stream-hover-card")).to_have_class(re.compile(r"\bactive\b"))
    expect(page.locator("#stream-hover-card")).to_contain_text("sam3_detection")
    expect(page.locator("[data-stream-lane='vision'] .stream-dot[data-event-type='vision_state_update']")).to_have_count(1)


def test_live_timeline_caps_dense_raw_vision_events(
    browser_page: Page,
    dashboard_server,
):
    collector, _, dashboard_url, _ = dashboard_server
    page = browser_page
    page.goto(dashboard_url)

    collector.push(
        "session_start",
        {
            "character": "greg",
            "model": "groq-llama-8b",
            "session_id": "sess-live-cap",
            "stage": "watching",
        },
    )
    collector.push(
        "vision_focus",
        {
            "stage_goal": "Track dense scene motion.",
            "constant_questions": ["What changed in the room?"],
            "ephemeral_questions": ["Is the visitor still visible?"],
            "constant_sam_targets": ["person"],
            "ephemeral_sam_targets": ["visitor"],
        },
    )

    for idx in range(1400):
        collector.push(
            "sam3_detection",
            {
                "persons": [],
                "objects": [{"label": f"marker-{idx}", "bbox": [10, 20, 40, 50]}],
            },
        )

    page.wait_for_function(
        "() => { const state = window.__dashboardTest.timelineState(); return state.liveEventCount <= 1202 && state.visibleCardCount <= 1202; }",
        timeout=10_000,
    )
    state = page.evaluate("window.__dashboardTest.timelineState()")
    assert state["liveEventCount"] <= 1202
    assert state["visibleCardCount"] <= 1202


def test_live_prompt_trace_retention_is_bounded(
    browser_page: Page,
    dashboard_server,
):
    collector, _, dashboard_url, _ = dashboard_server
    page = browser_page
    page.goto(dashboard_url)

    collector.push(
        "session_start",
        {
            "character": "greg",
            "model": "groq-llama-8b",
            "session_id": "sess-live-prompt-cap",
            "stage": "watching",
        },
    )

    base_ts = time.time()
    for idx in range(240):
        _collector_push_at(
            collector,
            "prompt_trace",
            {
                "label": "chat",
                "title": f"Chat prompt {idx}",
                "messages": [
                    {"role": "system", "content": "Keep it grounded."},
                    {"role": "user", "content": f"Prompt trace sample {idx}"},
                ],
                "output": f"Output {idx}",
                "started_at": base_ts + (idx * 0.02),
                "finished_at": base_ts + (idx * 0.02) + 0.01,
            },
            base_ts + (idx * 0.02) + 0.01,
        )

    page.wait_for_function(
        "() => { const state = window.__dashboardTest.timelineState(); return state.lastSeq >= 241; }",
        timeout=10_000,
    )
    state = page.evaluate("window.__dashboardTest.timelineState()")
    assert state["promptTraceCount"] <= 160


def test_live_older_events_are_compacted_without_breaking_selection(
    browser_page: Page,
    dashboard_server,
):
    collector, _, dashboard_url, _ = dashboard_server
    page = browser_page
    page.goto(dashboard_url)

    base_ts = time.time() - 240
    _collector_push_at(
        collector,
        "session_start",
        {
            "character": "greg",
            "model": "groq-llama-8b",
            "session_id": "sess-live-compaction",
            "stage": "watching",
        },
        base_ts,
    )

    def push_perception(idx: int, event_ts: float):
        _collector_push_at(
            collector,
            "perception",
            {
                "source": "visual",
                "description": f"Compaction perception {idx}",
                "event_id": f"perception-{idx}",
                "source_trace": {
                    "event_id": f"trace-{idx}",
                    "object_labels": [f"object-{idx}", "chair", "door", "shadow", "ladder"],
                    "vlm_answers": [
                        {
                            "task_id": "world_state",
                            "label": "World State",
                            "question": "Describe the room.",
                            "answer": f"Room state sample {idx}",
                        },
                        {
                            "task_id": "person_description_static",
                            "label": "Person",
                            "question": "Describe the person.",
                            "answer": f"No target person visible {idx}",
                        },
                    ],
                    "focus": {
                        "ephemeral_questions": ["Is there a phone visible?", "Is someone at the door?"],
                        "ephemeral_sam_targets": ["phone", "door"],
                    },
                    "provenance": {
                        "label": "Vision synthesis LLM",
                        "primary": "vision_state_update_llm",
                        "components": ["vlm_answers", "snapshot"],
                    },
                },
            },
            event_ts,
        )

    def push_state_update(idx: int, event_ts: float):
        _collector_push_at(
            collector,
            "vision_state_update",
            {
                "source": "visual",
                "description": f"Compaction state update {idx}",
                "summary": f"Compaction state update {idx}",
                "event_id": f"state-{idx}",
                "task_answers": [
                    {
                        "task_id": "world_state",
                        "label": "World State",
                        "question": "Describe the room.",
                        "answer": f"Updated room summary {idx}",
                    },
                ],
                "add_facts": [f"fact-{idx}", f"secondary fact-{idx}"],
                "events": [f"event-{idx}"],
                "source_trace": {
                    "event_id": f"state-trace-{idx}",
                    "object_labels": [f"object-{idx}", "lamp", "door", "chair"],
                    "vlm_answers": [
                        {
                            "task_id": "world_state",
                            "label": "World State",
                            "question": "Describe the room.",
                            "answer": f"Updated room summary {idx}",
                        },
                    ],
                    "provenance": {
                        "label": "Vision state interpreter",
                        "primary": "vision_state_update_llm",
                        "components": ["vlm_answers"],
                    },
                },
            },
            event_ts,
        )

    def push_raw_detection(idx: int, event_ts: float):
        _collector_push_at(
            collector,
            "sam3_detection",
            {
                "event_id": f"sam3-{idx}",
                "persons": [{"identity": f"visitor-{idx}", "track_id": idx, "bbox": [10, 20, 30, 40]}],
                "objects": [{"label": f"object-{idx}", "bbox": [20, 30, 40, 50]}],
            },
            event_ts,
        )

    for idx in range(240):
        push_perception(idx, base_ts + (idx * 0.18))
    for idx in range(241):
        push_state_update(idx, base_ts + 45 + (idx * 0.18))
        push_raw_detection(idx, base_ts + 90 + (idx * 0.18))

    collector.push(
        "vision_state_update",
        {
            "source": "visual",
            "description": "Current live state update after many historical events.",
            "summary": "Current live state update after many historical events.",
            "task_answers": [
                {
                    "task_id": "world_state",
                    "label": "World State",
                    "question": "Describe the room.",
                    "answer": "Current room state",
                },
            ],
        },
    )

    page.wait_for_function(
        """() => {
            const state = window.__dashboardTest.timelineState();
                return state.compactedLiveEventCount > 0 && state.liveEventCount >= 720;
        }""",
        timeout=10_000,
    )
    state = page.evaluate("window.__dashboardTest.timelineState()")
    assert state["compactedLiveEventCount"] > 0

    page.evaluate("window.__dashboardTest.scrollTimelineToStart()")
    page.wait_for_function(
        """() => document.querySelectorAll(".stream-dot[data-event-type='perception']").length > 0""",
        timeout=10_000,
    )
    page.locator(".stream-dot[data-event-type='perception']").first.evaluate("(node) => node.click()")
    expect(page.locator("#detail-warning")).to_contain_text("compacted")
    expect(page.locator("#detail-payload")).to_contain_text("\"__compacted\": true")


def test_sidebar_registry_layout_and_panel_collapse(
    browser_page: Page,
    dashboard_server,
):
    _, _, dashboard_url, _ = dashboard_server
    page = browser_page
    page.goto(dashboard_url)

    expect(page.locator(".sidebar-group-label").nth(0)).to_have_text("Session Control")
    expect(page.locator(".sidebar-group-label").nth(1)).to_have_text("Live State")
    expect(page.locator(".sidebar-group-label").nth(2)).to_have_text("Tooling")
    expect(page.locator("#plan-panel")).to_be_visible()
    expect(page.locator("#prompt-assets-panel")).to_have_class(re.compile(r"\bcollapsed\b"))
    expect(page.locator("#plan-panel")).to_have_attribute("data-panel-group", "live_state")
    expect(page.locator("#plan-content")).to_contain_text("No active script plan yet")
    assert page.evaluate(
        """() => Array.from(document.querySelectorAll('#sidebar .sidebar-panel'))
        .map((node) => node.dataset.panelId)"""
    ) == ["runtime", "vision", "world", "stage", "plan", "people", "prompt_assets"]

    toggle = page.locator("#plan-panel .sidebar-panel-toggle")
    expect(toggle).to_have_text("Collapse")
    toggle.evaluate("(node) => node.click()")
    expect(page.locator("#plan-panel")).to_have_class(re.compile(r"\bcollapsed\b"))
    expect(toggle).to_have_text("Expand")


def test_archive_boot_live_keeps_vision_panel_above_prompt_assets(
    browser_page: Page,
    dashboard_server,
):
    collector, _, dashboard_url, _ = dashboard_server
    collector.push(
        "runtime_controls",
        {
            "controls": {
                "reconcile": True,
                "vision": True,
                "auto_beat": True,
                "filler": True,
            },
            "conversation_paused": False,
            "session_stopped": False,
            "startup_pause_pending": False,
            "session_state": "live",
            "voice_active": True,
            "voice_status": {
                "active": True,
                "output_only": False,
                "filler_enabled": True,
                "tts_backend": "pocket-tts",
                "mic_ready": True,
                "speaker_ready": True,
                "stt": {"state": "ready", "detail": "Deepgram socket open."},
                "tts": {"state": "ready", "detail": "Pocket-TTS reachable.", "backend": "pocket-tts"},
            },
            "vision_active": True,
            "reconcile_thread_alive": True,
            "vision_service_url": "http://127.0.0.1:7875",
            "vision_service_health": True,
            "vision_service_managed": True,
            "vision_service_external": False,
            "vision_service_state": "managed",
            "vision_service_autostart": True,
            "vision_service_mode": "mock",
            "vision_mock_replay": "walkup.json",
            "archive_only": True,
            "archive_only_reason": "Archive-only mode is active. Load archives to inspect them; live conversation is disabled.",
        },
    )
    page = browser_page
    page.goto(f"{dashboard_url}?archive=sess-archive-playback&boot=live")

    expect(page.locator("#ux-spoof-banner")).to_contain_text("Spoofed For UX")
    expect(page.locator(".stream-card").filter(has_text="Hey, what time is it to you?")).to_have_count(1)
    expect(page.locator("#vision-panel")).to_be_visible()
    assert page.locator(".sidebar-group-label").all_inner_texts() == ["LIVE STATE", "SESSION CONTROL", "TOOLING"]
    assert page.evaluate(
        """() => {
          const panels = Array.from(document.querySelectorAll('#sidebar .sidebar-panel')).map((node) => node.dataset.panelId);
          const vision = document.getElementById('vision-panel');
          const runtime = document.getElementById('runtime-panel');
          const promptAssets = document.getElementById('prompt-assets-panel');
          return (
            panels[0] === 'vision' &&
            !!vision &&
            !!runtime &&
            !!promptAssets &&
            vision.offsetTop < runtime.offsetTop &&
            vision.offsetTop < promptAssets.offsetTop
          );
        }"""
    )


def test_archive_boot_live_runtime_actions_match_spoof_state(
    browser_page: Page,
    dashboard_server,
):
    collector, _, dashboard_url, _ = dashboard_server
    collector.push(
        "runtime_controls",
        {
            "controls": {
                "reconcile": True,
                "vision": True,
                "auto_beat": True,
                "filler": True,
            },
            "conversation_paused": False,
            "session_stopped": False,
            "startup_pause_pending": False,
            "session_state": "live",
            "voice_active": True,
            "voice_status": {
                "active": True,
                "output_only": False,
                "filler_enabled": True,
                "tts_backend": "pocket-tts",
                "mic_ready": True,
                "speaker_ready": True,
                "stt": {"state": "ready", "detail": "Deepgram socket open."},
                "tts": {"state": "ready", "detail": "Pocket-TTS reachable.", "backend": "pocket-tts"},
            },
            "vision_active": True,
            "reconcile_thread_alive": True,
            "vision_service_url": "http://127.0.0.1:7875",
            "vision_service_health": True,
            "vision_service_managed": True,
            "vision_service_external": False,
            "vision_service_state": "managed",
            "vision_service_autostart": True,
            "vision_service_mode": "mock",
            "vision_mock_replay": "walkup.json",
            "archive_only": True,
            "archive_only_reason": "Archive-only mode is active. Load archives to inspect them; live conversation is disabled.",
        },
    )
    page = browser_page
    page.goto(f"{dashboard_url}?archive=sess-archive-playback&boot=live")

    expect(page.locator("#ux-spoof-banner")).to_contain_text("Use this archive URL as the canonical UX test case.")
    expect(page.locator("#ux-spoof-copy-url")).to_be_visible()
    expect(page.locator("#runtime-toggle")).to_have_text("Pause")
    expect(page.locator("#runtime-toggle")).to_be_disabled()
    expect(page.locator("#runtime-stop")).to_have_text("Stop Session")
    expect(page.locator("#runtime-stop")).to_be_disabled()
    expect(page.locator("#runtime-panel")).not_to_contain_text("Restart Session")


def test_loaded_archive_exports_full_window_without_marks(
    browser_page: Page,
    dashboard_server,
):
    collector, _, dashboard_url, _ = dashboard_server
    collector.push(
        "runtime_controls",
        {
            "controls": {
                "reconcile": True,
                "vision": True,
                "auto_beat": False,
                "filler": False,
            },
            "conversation_paused": False,
            "session_stopped": False,
            "startup_pause_pending": False,
            "session_state": "live",
            "voice_active": True,
            "voice_status": {
                "active": True,
                "output_only": False,
                "filler_enabled": False,
                "tts_backend": "pocket",
                "mic_ready": True,
                "speaker_ready": True,
                "stt": {"state": "ready", "detail": "Deepgram socket open."},
                "tts": {"state": "ready", "detail": "Pocket-TTS reachable.", "backend": "pocket"},
            },
            "vision_active": True,
            "reconcile_thread_alive": True,
            "vision_service_url": "http://127.0.0.1:7875",
            "vision_service_health": True,
            "vision_service_managed": True,
            "vision_service_external": False,
            "vision_service_state": "managed",
            "vision_service_autostart": True,
            "vision_service_mode": "mock",
            "vision_mock_replay": "walkup.json",
        },
    )
    page = browser_page
    page.goto(f"{dashboard_url}?archive=sess-archive-playback&boot=live")

    expect(page.locator("#archive-export-status")).to_contain_text("export the full loaded archive")
    page.locator("#archive-export-button").evaluate("(node) => node.click()")
    expect(page.locator("#save-complete-modal")).to_have_class(re.compile(r"\bopen\b"))
    expect(page.locator("#save-complete-heading")).to_contain_text("Export Archive")
    expect(page.locator("#save-complete-title")).not_to_have_value("")
    page.locator("#save-complete-title").fill("Loaded archive export")
    page.locator("#save-complete-save").click()
    expect(page.locator("#save-complete-heading")).to_contain_text("Archive Exported")
    expect(page.locator("#save-complete-session-id")).to_contain_text("sess-archive-playback")
    expect(page.locator("#save-complete-copy-ref")).to_be_visible()
    expect(page.locator("#save-complete-load")).to_be_visible()


def test_archive_raw_event_shows_inspected_frame_in_inspector(
    browser_page: Page,
    dashboard_server,
):
    collector, _, dashboard_url, _ = dashboard_server
    collector.push(
        "runtime_controls",
        {
            "controls": {
                "reconcile": True,
                "vision": True,
                "auto_beat": False,
                "filler": False,
            },
            "conversation_paused": True,
            "session_stopped": True,
            "startup_pause_pending": False,
            "session_state": "ready",
            "voice_active": False,
            "voice_status": {
                "active": False,
                "output_only": False,
                "filler_enabled": False,
                "tts_backend": "",
                "mic_ready": False,
                "speaker_ready": False,
                "stt": {"state": "off", "detail": "Voice inactive."},
                "tts": {"state": "off", "detail": "Voice inactive.", "backend": ""},
            },
            "vision_active": False,
            "reconcile_thread_alive": False,
            "vision_service_url": "",
            "vision_service_health": False,
            "vision_service_managed": False,
            "vision_service_external": False,
            "vision_service_state": "stopped",
            "vision_service_autostart": False,
            "vision_service_mode": "camera",
            "vision_mock_replay": "",
            "archive_only": True,
            "archive_only_reason": "Archive-only mode is active. Load archives to inspect them; live conversation is disabled.",
        },
    )
    page = browser_page
    page.goto(f"{dashboard_url}?archive=sess-archive-vision-raw-ui&boot=live")

    page.wait_for_function("() => !!window.__dashboardTest.visionRawEventPoint('vision_snapshot_read', 0)")
    raw_point = page.evaluate("window.__dashboardTest.visionRawEventPoint('vision_snapshot_read', 0)")
    assert raw_point is not None
    page.mouse.click(raw_point["clientX"], raw_point["clientY"])
    expect(page.locator("#inspector-frame-panel")).to_be_visible()
    expect(page.locator("#detail-frame-image")).to_be_visible()
    expect(page.locator("#detail-frame-meta")).to_contain_text("Nearest archived frame")
    expect(page.locator("#detail-frame-open")).to_have_attribute("href", re.compile(r"session_id=sess-archive-vision-raw-ui"))


def test_archive_vlm_answer_shows_target_crop_in_inspector(
    browser_page: Page,
    dashboard_server,
):
    collector, _, dashboard_url, _ = dashboard_server
    collector.push(
        "runtime_controls",
        {
            "controls": {
                "reconcile": True,
                "vision": True,
                "auto_beat": False,
                "filler": False,
            },
            "conversation_paused": True,
            "session_stopped": True,
            "startup_pause_pending": False,
            "session_state": "ready",
            "voice_active": False,
            "voice_status": {
                "active": False,
                "output_only": False,
                "filler_enabled": False,
                "tts_backend": "",
                "mic_ready": False,
                "speaker_ready": False,
                "stt": {"state": "off", "detail": "Voice inactive."},
                "tts": {"state": "off", "detail": "Voice inactive.", "backend": ""},
            },
            "vision_active": False,
            "reconcile_thread_alive": False,
            "vision_service_url": "",
            "vision_service_health": False,
            "vision_service_managed": False,
            "vision_service_external": False,
            "vision_service_state": "stopped",
            "vision_service_autostart": False,
            "vision_service_mode": "camera",
            "vision_mock_replay": "",
            "archive_only": True,
            "archive_only_reason": "Archive-only mode is active. Load archives to inspect them; live conversation is disabled.",
        },
    )
    page = browser_page
    page.goto(f"{dashboard_url}?archive=sess-archive-vision-raw-ui&boot=live")

    page.wait_for_function("() => !!window.__dashboardTest.visionRawEventPoint('vlm_answer', 0)")
    raw_point = page.evaluate("window.__dashboardTest.visionRawEventPoint('vlm_answer', 0)")
    assert raw_point is not None
    page.mouse.click(raw_point["clientX"], raw_point["clientY"])
    expect(page.locator("#detail-frame-image")).to_be_visible()
    expect(page.locator("#detail-target-crop-wrap")).to_be_visible()
    expect(page.locator("#detail-target-crop-image")).to_be_visible()
    expect(page.locator("#detail-target-crop-meta")).to_contain_text("Visitor")
    expect(page.locator("#detail-target-crop-meta")).to_contain_text("x 40 y 50 w 120 h 140")


def test_archive_turn_rail_and_context_show_turn_metadata(
    browser_page: Page,
    dashboard_server,
):
    _, _, dashboard_url, _ = dashboard_server
    page = browser_page
    page.goto(f"{dashboard_url}?archive=sess-turn-summary&boot=live")

    page.wait_for_function("() => (window.__dashboardTest.turnRailsState() || []).length > 0")
    rails = page.evaluate("window.__dashboardTest.turnRailsState()")
    assert rails
    assert any(rail["startLabel"] == "user / dialogue" for rail in rails)
    assert any("replied" in (rail["endLabel"] or "") for rail in rails)

    reply_card = page.locator(".stream-card[data-event-type='assistant_reply']").first
    reply_card.click()
    expect(page.locator("#detail-type")).to_have_text("assistant_reply")
    expect(page.locator("#detail-open-reply-chat")).to_be_visible()
    expect(page.locator("#detail-turn-context")).to_have_class(re.compile(r"\bactive\b"))
    expect(page.locator("#detail-turn-badge")).to_contain_text("user / dialogue")
    expect(page.locator("#detail-turn-grid")).to_contain_text("replied")
    page.locator("#detail-open-reply-chat").click()
    expect(page.locator("#prompt-modal")).to_have_class(re.compile(r"\bopen\b"))
    expect(page.locator("#prompt-modal")).to_contain_text("Keep it short.")
    expect(page.locator("#prompt-modal")).to_contain_text("Do you have water?")
    page.locator("#prompt-modal-close").click()
    rails = page.evaluate("window.__dashboardTest.turnRailsState()")
    assert sum(1 for rail in rails if rail["selected"]) == 1
    expect(page.locator("#detail-prompts")).to_contain_text("Chat prompt")
    expect(page.locator("#detail-prompts")).to_contain_text("Free water. Want one?")
    page.locator(".stream-card[data-event-type='user_transcript_final']").first.click()
    expect(page.locator("#detail-type")).to_have_text("user_transcript_final")
    expect(page.locator("#detail-open-reply-chat")).to_be_hidden()
    reply_card.click()
    page.locator("#detail-open-json").click()
    expect(page.locator("#detail-payload")).to_contain_text("\"full_text\": \"Free water. Want one?\"")


def test_json_default_inspector_stacks_prompt_and_payload_sections(
    browser_page: Page,
    dashboard_server,
):
    collector, _, dashboard_url, _ = dashboard_server
    collector.push(
        "runtime_controls",
        {
            "controls": {
                "reconcile": True,
                "vision": True,
                "auto_beat": False,
                "filler": False,
            },
            "conversation_paused": True,
            "session_stopped": True,
            "startup_pause_pending": False,
            "session_state": "ready",
            "voice_active": False,
            "voice_status": {
                "active": False,
                "output_only": False,
                "filler_enabled": False,
                "tts_backend": "",
                "mic_ready": False,
                "speaker_ready": False,
                "stt": {"state": "off", "detail": "Voice inactive."},
                "tts": {"state": "off", "detail": "Voice inactive.", "backend": ""},
            },
            "vision_active": False,
            "reconcile_thread_alive": False,
            "vision_service_url": "",
            "vision_service_health": False,
            "vision_service_managed": False,
            "vision_service_external": False,
            "vision_service_state": "stopped",
            "vision_service_autostart": False,
            "vision_service_mode": "camera",
            "vision_mock_replay": "",
            "archive_only": True,
            "archive_only_reason": "Archive-only mode is active. Load archives to inspect them; live conversation is disabled.",
        },
    )
    page = browser_page
    page.goto(f"{dashboard_url}?archive=sess-archive-playback&boot=live")
    page.wait_for_function(
        "() => document.querySelectorAll('.stream-card').length > 0",
        timeout=5000,
    )

    selected = page.evaluate(
        """() => {
            const target =
              document.querySelector(".stream-card[data-event-type='vision_focus']") ||
              document.querySelector(".stream-card[data-event-type='user_transcript_final']") ||
              document.querySelector(".stream-card");
            if (!target) return false;
            target.click();
            return true;
        }"""
    )
    assert selected is True
    metrics = page.evaluate(
        """() => {
            const annotations = document.getElementById('inspector-annotations-panel');
            const promptPane = document.getElementById('detail-prompts-pane');
            const payloadPane = document.getElementById('detail-payload-pane');
            return {
                annotationsDisplay: window.getComputedStyle(annotations).display,
                promptBottom: promptPane.getBoundingClientRect().bottom,
                payloadTop: payloadPane.getBoundingClientRect().top,
            };
        }"""
    )
    assert metrics["annotationsDisplay"] == "none"
    assert metrics["payloadTop"] >= metrics["promptBottom"] - 1


def test_prompt_preview_opens_modal_for_prompt_backed_event(
    browser_page: Page,
    dashboard_server,
    vision_service,
):
    collector, _, dashboard_url, _ = dashboard_server
    page = browser_page
    page.goto(dashboard_url)

    collector.push(
        "runtime_controls",
        {
            "controls": {
                "reconcile": True,
                "vision": True,
                "auto_beat": False,
                "filler": False,
            },
            "conversation_paused": False,
            "session_stopped": False,
            "startup_pause_pending": False,
            "session_state": "live",
            "voice_active": True,
            "voice_status": {
                "active": True,
                "output_only": False,
                "filler_enabled": False,
                "tts_backend": "pocket",
                "mic_ready": True,
                "speaker_ready": True,
                "stt": {"state": "ready", "detail": "Deepgram socket open."},
                "tts": {"state": "ready", "detail": "Pocket-TTS reachable.", "backend": "pocket"},
            },
            "vision_active": True,
            "reconcile_thread_alive": True,
            "vision_service_url": vision_service.url,
            "vision_service_health": True,
            "vision_service_managed": False,
            "vision_service_external": True,
            "vision_service_state": "external",
            "vision_service_autostart": True,
            "vision_service_mode": "camera",
            "vision_mock_replay": "",
        },
    )
    collector.push(
        "session_start",
        {
            "character": "greg",
            "model": "groq-llama-8b",
            "session_id": "sess-prompt-preview",
            "stage": "engaged",
        },
    )
    collector.push("prompt_trace", {
        "label": "chat",
        "title": "Chat prompt",
        "provider": "Groq",
        "model": "llama-test",
        "messages": [
            {"role": "system", "content": "Keep it short."},
            {"role": "user", "content": "Do you have water?"},
        ],
        "output": "Free water. Want one?",
        "started_at": time.time(),
        "finished_at": time.time(),
    })
    collector.push(
        "response_done",
        {
            "full_text": "Free water. Want one?",
            "input_source": "voice",
            "ttft_ms": 190,
            "llm_ms": 780,
        },
    )

    response_card = page.locator(".stream-card[data-event-type='assistant_reply']").first
    expect(response_card).to_be_visible()
    response_card.click()
    expect(page.locator(".prompt-preview-card")).to_contain_text("Chat prompt")
    expect(page.locator(".prompt-preview-card")).to_contain_text("Free water. Want one?")
    page.locator("[data-prompt-open='0']").click()
    expect(page.locator("#prompt-modal")).to_have_class(re.compile(r"\bopen\b"))
    expect(page.locator("#prompt-modal")).to_contain_text("Rendered Prompt")
    expect(page.locator("#prompt-modal")).to_contain_text("Free water. Want one?")
    page.locator("#prompt-modal-copy").click()
    expect(page.locator("#prompt-modal-copy-status")).to_contain_text("Copied prompt JSON.")
    copied = page.evaluate("JSON.parse(window.__dashboardTest.lastCopiedText())")
    assert copied["session_id"] == "sess-prompt-preview"
    assert copied["event"]["type"] == "assistant_reply"
    assert re.match(r"^#\d+$", copied["reply_id"] or "")
    assert copied["prompt_block"]["title"] == "Chat prompt"
    assert copied["prompt_block"]["messages"][0]["content"] == "Keep it short."
    assert copied["prompt_block"]["output"] == "Free water. Want one?"


def test_beat_delivered_reply_infers_planner_prompt_trace(
    browser_page: Page,
    dashboard_server,
    vision_service,
):
    collector, _, dashboard_url, _ = dashboard_server
    page = browser_page
    page.goto(dashboard_url)

    now = time.time()
    collector.push(
        "runtime_controls",
        {
            "controls": {
                "reconcile": True,
                "vision": True,
                "auto_beat": False,
                "filler": False,
            },
            "conversation_paused": False,
            "session_stopped": False,
            "startup_pause_pending": False,
            "session_state": "live",
            "voice_active": True,
            "voice_status": {
                "active": True,
                "output_only": False,
                "filler_enabled": False,
                "tts_backend": "pocket",
                "mic_ready": True,
                "speaker_ready": True,
                "stt": {"state": "ready", "detail": "Deepgram socket open."},
                "tts": {"state": "ready", "detail": "Pocket-TTS reachable.", "backend": "pocket"},
            },
            "vision_active": True,
            "reconcile_thread_alive": True,
            "vision_service_url": vision_service.url,
            "vision_service_health": True,
            "vision_service_managed": False,
            "vision_service_external": True,
            "vision_service_state": "external",
            "vision_service_autostart": True,
            "vision_service_mode": "camera",
            "vision_mock_replay": "",
        },
    )
    collector.push(
        "session_start",
        {
            "character": "greg",
            "model": "groq-llama-8b",
            "session_id": "sess-prompt-beat",
            "stage": "watching",
        },
    )
    collector.push(
        "turn_start",
        {
            "input_text": "",
            "input_type": "beat",
            "trigger_source": "vision",
            "turn_kind": "reaction",
        },
    )
    collector.push("prompt_trace", {
        "label": "next_beat",
        "title": "Next beat prompt",
        "provider": "Groq",
        "model": "llama-test",
        "messages": [
            {"role": "system", "content": "Plan exactly one beat."},
            {"role": "user", "content": "=== RECENT CONVERSATION ===\nUSER: hey there"},
        ],
        "output": '{"beats":[{"line":"Hey there. What time is it?","intent":"ask for the time","condition":""}]}',
        "started_at": now,
        "finished_at": now,
    })
    collector.push(
        "response_done",
        {
            "full_text": "Hey there. What time is it?",
            "turn_kind": "reaction",
            "turn_outcome": "beat_delivered",
            "trigger_source": "vision",
            "ttft_ms": 0,
            "llm_ms": 0,
        },
    )

    response_card = page.locator(".stream-card[data-event-type='assistant_reply']").first
    expect(response_card).to_be_visible()
    response_card.click()
    expect(page.locator(".prompt-preview-card")).to_contain_text("Next beat prompt")
    expect(page.locator(".prompt-preview-card")).to_contain_text("Hey there. What time is it?")
    page.locator("[data-prompt-open='0']").click()
    expect(page.locator("#prompt-modal")).to_have_class(re.compile(r"\bopen\b"))
    expect(page.locator("#prompt-modal")).to_contain_text("Plan exactly one beat.")
    expect(page.locator("#prompt-modal")).to_contain_text("=== RECENT CONVERSATION ===")
    expect(page.locator("#prompt-modal")).to_contain_text("USER: hey there")


def test_play_rearms_follow_and_jumps_to_latest(
    browser_page: Page,
    dashboard_server,
):
    collector, input_queue, dashboard_url, _ = dashboard_server
    page = browser_page
    page.goto(dashboard_url)

    now = time.time()
    _collector_push_at(
        collector,
        "session_start",
        {
            "character": "greg",
            "model": "groq-llama-8b",
            "session_id": "sess-follow-live",
            "stage": "watching",
        },
        now - 90,
    )
    _collector_push_at(
        collector,
        "runtime_controls",
        {
            "controls": {
                "reconcile": True,
                "vision": True,
                "auto_beat": False,
                "filler": False,
            },
            "conversation_paused": True,
            "session_stopped": False,
            "startup_pause_pending": False,
            "session_state": "paused",
            "voice_active": True,
            "voice_status": {
                "active": True,
                "output_only": False,
                "filler_enabled": False,
                "tts_backend": "pocket",
                "mic_ready": True,
                "speaker_ready": True,
                "stt": {"state": "ready", "detail": "Deepgram socket open."},
                "tts": {"state": "ready", "detail": "Pocket-TTS reachable.", "backend": "pocket"},
            },
            "vision_active": True,
            "reconcile_thread_alive": True,
            "vision_service_url": "",
            "vision_service_health": False,
            "vision_service_managed": False,
            "vision_service_external": False,
            "vision_service_state": "stopped",
            "vision_service_autostart": False,
            "vision_service_mode": "camera",
            "vision_mock_replay": "",
        },
        now - 1,
    )
    _collector_push_at(
        collector,
        "response_done",
        {
            "full_text": "Latest reply in the stream.",
            "input_source": "voice",
            "ttft_ms": 190,
            "llm_ms": 780,
        },
        now,
    )

    expect(page.locator(".stream-card[data-event-type='assistant_reply']")).to_contain_text("Latest reply in the stream")
    page.evaluate(
        """() => {
            const viewport = document.getElementById('stream-viewport');
            const follow = document.getElementById('follow-live');
            viewport.scrollLeft = 0;
            follow.checked = false;
            follow.dispatchEvent(new Event('change', {bubbles: true}));
        }"""
    )

    before = page.evaluate(
        """() => {
            const viewport = document.getElementById('stream-viewport');
            return {
                scrollLeft: Number(viewport.scrollLeft || 0),
                maxScrollLeft: Number((viewport.scrollWidth || 0) - (viewport.clientWidth || 0)),
                follow: document.getElementById('follow-live').checked,
            };
        }"""
    )
    assert before["scrollLeft"] == 0
    assert before["maxScrollLeft"] > 0
    assert before["follow"] is False

    page.locator("#runtime-toggle").click()
    assert input_queue.get(timeout=2) == "/play"

    after = page.evaluate(
        """() => {
            const viewport = document.getElementById('stream-viewport');
            return {
                scrollLeft: Number(viewport.scrollLeft || 0),
                maxScrollLeft: Number((viewport.scrollWidth || 0) - (viewport.clientWidth || 0)),
                follow: document.getElementById('follow-live').checked,
            };
        }"""
    )
    assert after["follow"] is True
    assert after["scrollLeft"] >= after["maxScrollLeft"] - 4


def test_turn_start_cards_use_explanatory_beat_labels(
    browser_page: Page,
    dashboard_server,
):
    collector, _, dashboard_url, _ = dashboard_server
    page = browser_page
    page.goto(dashboard_url)

    collector.push(
        "runtime_controls",
        {
            "controls": {
                "reconcile": True,
                "vision": True,
                "auto_beat": False,
                "filler": False,
            },
            "conversation_paused": False,
            "session_stopped": False,
            "startup_pause_pending": False,
            "session_state": "live",
            "voice_active": True,
            "voice_status": {
                "active": True,
                "output_only": False,
                "filler_enabled": False,
                "tts_backend": "pocket",
                "mic_ready": True,
                "speaker_ready": True,
                "stt": {"state": "ready", "detail": "Deepgram socket open."},
                "tts": {"state": "ready", "detail": "Pocket-TTS reachable.", "backend": "pocket"},
            },
            "vision_active": True,
            "reconcile_thread_alive": True,
            "vision_service_url": "",
            "vision_service_health": False,
            "vision_service_managed": False,
            "vision_service_external": False,
            "vision_service_state": "stopped",
            "vision_service_autostart": False,
            "vision_service_mode": "camera",
            "vision_mock_replay": "",
        },
    )
    collector.push(
        "session_start",
        {
            "character": "greg",
            "model": "groq-llama-8b",
            "session_id": "sess-turn-start-copy",
            "stage": "watching",
        },
    )
    collector.push(
        "turn_start",
        {
            "input_type": "beat",
            "input_text": "/beat",
            "trigger_source": "vision",
            "turn_kind": "reaction",
            "trigger_reason": "visual_stage_exit",
            "trigger_signals": ["person_visible", "person_visible_stable"],
            "trigger_stage_from": "watching",
            "trigger_stage_from_goal": "Greg is indoors in a room, looking around and hoping to learn what time it is.",
            "trigger_stage_to": "spotted",
            "trigger_stage_to_goal": "Greg has noticed someone nearby. He makes one short grounded opener and quickly asks what time it is.",
            "trigger_stage_exit_label": "notices",
            "trigger_stage_exit_condition": "Someone becomes visible nearby or appears to notice Greg",
            "trigger_stage_exit_visual_signals": ["person_visible", "person_visible_stable"],
        },
    )

    card = page.locator(".stream-card[data-event-type='turn_start']").first
    expect(card).to_be_visible()
    expect(card).to_contain_text("Vision stage-advance beat")
    expect(card).to_contain_text("watching -> spotted")
    expect(card).not_to_contain_text("/beat")


def test_visual_stage_exit_turn_start_inspector_shows_scenario_match(
    browser_page: Page,
    dashboard_server,
):
    collector, _, dashboard_url, _ = dashboard_server
    page = browser_page
    page.goto(dashboard_url)

    collector.push(
        "runtime_controls",
        {
            "controls": {"reconcile": True, "vision": True, "auto_beat": False, "filler": False},
            "conversation_paused": False,
            "session_stopped": False,
            "startup_pause_pending": False,
            "session_state": "live",
            "voice_active": True,
            "voice_status": {
                "active": True,
                "output_only": False,
                "filler_enabled": False,
                "tts_backend": "pocket",
                "mic_ready": True,
                "speaker_ready": True,
                "stt": {"state": "ready", "detail": "Deepgram socket open."},
                "tts": {"state": "ready", "detail": "Pocket-TTS reachable.", "backend": "pocket"},
            },
            "vision_active": True,
            "reconcile_thread_alive": True,
            "vision_service_url": "",
            "vision_service_health": False,
            "vision_service_managed": False,
            "vision_service_external": False,
            "vision_service_state": "stopped",
            "vision_service_autostart": False,
            "vision_service_mode": "camera",
            "vision_mock_replay": "",
        },
    )
    collector.push(
        "session_start",
        {"character": "greg", "model": "groq-llama-8b", "session_id": "sess-turn-stage", "stage": "watching"},
    )
    collector.push(
        "turn_start",
        {
            "input_type": "beat",
            "input_text": "",
            "trigger_source": "vision",
            "turn_kind": "reaction",
            "trigger_reason": "visual_stage_exit",
            "trigger_signals": ["person_visible", "person_visible_stable"],
            "trigger_stage_from": "watching",
            "trigger_stage_from_goal": "Greg is indoors in a room, looking around and hoping to learn what time it is.",
            "trigger_stage_to": "spotted",
            "trigger_stage_to_goal": "Greg has noticed someone nearby. He makes one short grounded opener and quickly asks what time it is.",
            "trigger_stage_exit_label": "notices",
            "trigger_stage_exit_condition": "Someone becomes visible nearby or appears to notice Greg",
            "trigger_stage_exit_visual_signals": ["person_visible", "person_visible_stable"],
        },
    )

    page.locator(".stream-card[data-event-type='turn_start']").first.click()
    expect(page.locator("#detail-turn-context")).not_to_have_class(re.compile(r"active"))
    expect(page.locator("#inspector-summary-title")).to_have_text(re.compile(r"turn_start #\d+"))
    expect(page.locator("#detail-payload")).to_contain_text('"trigger_stage_from": "watching"')
    expect(page.locator("#detail-payload")).to_contain_text('"trigger_stage_to": "spotted"')
    expect(page.locator("#detail-payload")).to_contain_text('"trigger_stage_exit_condition": "Someone becomes visible nearby or appears to notice Greg"')
    expect(page.locator("#detail-payload")).to_contain_text('"trigger_stage_exit_visual_signals": [')
    expect(page.locator("#detail-payload")).to_contain_text('"person_visible_stable"')


def test_vision_state_update_falls_back_from_no_updates_summary(
    browser_page: Page,
    dashboard_server,
):
    collector, _, dashboard_url, _ = dashboard_server
    page = browser_page
    page.goto(dashboard_url)

    collector.push(
        "runtime_controls",
        {
            "controls": {"reconcile": True, "vision": True, "auto_beat": False, "filler": False},
            "conversation_paused": False,
            "session_stopped": False,
            "startup_pause_pending": False,
            "session_state": "live",
            "voice_active": True,
            "voice_status": {
                "active": True,
                "output_only": False,
                "filler_enabled": False,
                "tts_backend": "pocket",
                "mic_ready": True,
                "speaker_ready": True,
                "stt": {"state": "ready", "detail": "Deepgram socket open."},
                "tts": {"state": "ready", "detail": "Pocket-TTS reachable.", "backend": "pocket"},
            },
            "vision_active": True,
            "reconcile_thread_alive": True,
            "vision_service_url": "",
            "vision_service_health": False,
            "vision_service_managed": False,
            "vision_service_external": False,
            "vision_service_state": "stopped",
            "vision_service_autostart": False,
            "vision_service_mode": "camera",
            "vision_mock_replay": "",
        },
    )
    collector.push(
        "session_start",
        {"character": "greg", "model": "groq-llama-8b", "session_id": "sess-vision-summary", "stage": "watching"},
    )
    collector.push(
        "vision_state_update",
        {
            "summary": "No updates.",
            "task_answers": [
                {
                    "task_id": "world_state",
                    "label": "World State",
                    "question": "What changed in the room?",
                    "answer": "The visible room has light-colored walls and a stackable ladder in the back left.",
                    "interpret_as": "world_state",
                    "target": "scene",
                }
            ],
            "add_facts": [
                "The visible room has light-colored walls and a stackable ladder in the back left."
            ],
            "remove_facts": [],
            "events": [],
            "person_updates": [],
        },
    )

    dot = page.locator(".stream-dot[data-event-type='vision_state_update']").first
    expect(dot).to_be_visible()
    dot.click()
    expect(page.locator("#inspector-summary-title")).to_have_text(re.compile(r"vision_state_update #\d+"))
    expect(page.locator("#detail-label")).to_contain_text("The visible room has light-colored walls")
    expect(page.locator("#detail-label")).not_to_contain_text("No updates.")
    expect(page.locator("#detail-payload")).to_contain_text("The visible room has light-colored walls and a stackable ladder in the back left.")


def test_world_state_inspector_shows_raw_event_json_and_empty_prompt_state(
    browser_page: Page,
    dashboard_server,
):
    collector, _, dashboard_url, _ = dashboard_server
    page = browser_page
    page.goto(dashboard_url)

    collector.push(
        "runtime_controls",
        {
            "controls": {
                "reconcile": True,
                "vision": True,
                "auto_beat": False,
                "filler": False,
            },
            "conversation_paused": False,
            "session_stopped": False,
            "startup_pause_pending": False,
            "session_state": "live",
            "voice_active": True,
            "voice_status": {
                "active": True,
                "output_only": False,
                "filler_enabled": False,
                "tts_backend": "pocket",
                "mic_ready": True,
                "speaker_ready": True,
                "stt": {"state": "ready", "detail": "Deepgram socket open."},
                "tts": {"state": "ready", "detail": "Pocket-TTS reachable.", "backend": "pocket"},
            },
            "vision_active": True,
            "reconcile_thread_alive": True,
            "vision_service_url": "",
            "vision_service_health": False,
            "vision_service_managed": False,
            "vision_service_external": False,
            "vision_service_state": "stopped",
            "vision_service_autostart": False,
            "vision_service_mode": "camera",
            "vision_mock_replay": "",
        },
    )

    collector.push(
        "session_start",
        {
            "character": "greg",
            "model": "groq-llama-8b",
            "session_id": "sess-world-update",
            "stage": "watching",
        },
    )
    collector.push(
        "world_state",
        {
            "static_facts": [
                {"id": "f1", "text": "Greg is indoors in a room with a bookshelf and a man wearing a dark shirt."},
            ],
            "dynamic_facts": [],
            "pending": [],
        },
    )

    world_dot = page.locator(".stream-dot[data-event-type='world_state']").first
    expect(world_dot).to_be_visible()
    world_dot.click()

    expect(page.locator("#detail-label")).to_contain_text("Greg is indoors in a room")
    expect(page.locator("#detail-detail")).to_contain_text("Greg is indoors in a room")
    expect(page.locator("#detail-payload")).to_contain_text("\"static_facts\"")
    expect(page.locator("#detail-payload")).to_contain_text("Greg is indoors in a room")
    expect(page.locator("#detail-prompts")).to_contain_text("No prompt snapshot captured for this event.")

    with page.expect_popup() as popup_info:
        page.locator("#detail-open-json").evaluate("(node) => node.click()")
    popup = popup_info.value
    popup.wait_for_load_state()
    expect(popup.locator("body")).to_contain_text("Greg is indoors in a room")
    expect(popup.locator("body")).to_contain_text("\"type\": \"world_state\"")
    expect(popup.locator("body")).to_contain_text("\"static_facts\"")


def test_dashboard_preserves_unfinished_reply_when_new_turn_starts(
    browser_page: Page,
    dashboard_server,
):
    collector, _, dashboard_url, _ = dashboard_server
    page = browser_page
    page.goto(dashboard_url)

    collector.push(
        "session_start",
        {
            "character": "greg",
            "model": "groq-llama-8b",
            "session_id": "sess-ghost-reply",
            "stage": "engaged",
        },
    )
    collector.push(
        "runtime_controls",
        {
            "controls": {
                "reconcile": True,
                "vision": True,
                "auto_beat": False,
                "filler": False,
            },
            "conversation_paused": False,
            "session_stopped": False,
            "startup_pause_pending": False,
            "session_state": "live",
            "voice_active": True,
            "voice_status": {
                "active": True,
                "output_only": False,
                "filler_enabled": False,
                "tts_backend": "pocket",
                "mic_ready": True,
                "speaker_ready": True,
                "stt": {"state": "ready", "detail": "Deepgram socket open."},
                "tts": {"state": "ready", "detail": "Pocket-TTS reachable.", "backend": "pocket"},
            },
            "vision_active": False,
            "reconcile_thread_alive": False,
            "vision_service_url": "",
            "vision_service_health": False,
            "vision_service_managed": False,
            "vision_service_external": False,
            "vision_service_state": "stopped",
            "vision_service_autostart": False,
            "vision_service_mode": "camera",
            "vision_mock_replay": "",
        },
    )

    collector.push("user_transcript_final", {"text": "well hm"})
    collector.push("turn_start", {
        "input_type": "send",
        "input_text": "well hm",
        "input_source": "voice",
    })
    collector.push("response_chunk", {"text": "Let me think"})
    expect(page.locator(".stream-card[data-event-type='assistant_reply']")).to_have_count(1)
    expect(page.locator(".stream-card").filter(has_text="Let me think")).to_have_count(1)

    collector.push("user_transcript_final", {"text": "i don't think so"})
    collector.push("turn_start", {
        "input_type": "send",
        "input_text": "well hm i don't think so",
        "input_source": "voice",
    })
    collector.push("response_chunk", {"text": "Okay, got it."})
    collector.push("response_done", {
        "full_text": "Okay, got it.",
        "input_source": "voice",
        "ttft_ms": 120,
        "llm_ms": 260,
        "total_ms": 900,
    })

    expect(page.locator(".stream-card").filter(has_text="Let me think")).to_have_count(1)
    expect(page.locator(".stream-card").filter(has_text="Let me think").locator(".reply-marker", has_text="Interrupted")).to_have_count(1)
    expect(page.locator(".stream-card[data-event-type='assistant_reply']")).to_have_count(2)
    expect(page.locator(".stream-card").filter(has_text="Okay, got it.")).to_have_count(1)

def test_dashboard_marks_reply_as_interrupted_when_runtime_emits_interruption_event(
    browser_page: Page,
    dashboard_server,
):
    collector, _, dashboard_url, _ = dashboard_server
    page = browser_page
    page.goto(dashboard_url)

    collector.push(
        "session_start",
        {
            "character": "greg",
            "model": "groq-llama-8b",
            "session_id": "sess-ghost-reply",
            "stage": "engaged",
        },
    )
    collector.push(
        "runtime_controls",
        {
            "controls": {
                "reconcile": True,
                "vision": True,
                "auto_beat": False,
                "filler": False,
            },
            "conversation_paused": False,
            "session_stopped": False,
            "startup_pause_pending": False,
            "session_state": "live",
            "voice_active": True,
            "voice_status": {
                "active": True,
                "output_only": False,
                "filler_enabled": False,
                "tts_backend": "pocket",
                "mic_ready": True,
                "speaker_ready": True,
                "stt": {"state": "ready", "detail": "Deepgram socket open."},
                "tts": {"state": "ready", "detail": "Pocket-TTS reachable.", "backend": "pocket"},
            },
            "vision_active": False,
            "reconcile_thread_alive": False,
            "vision_service_url": "",
            "vision_service_health": False,
            "vision_service_managed": False,
            "vision_service_external": False,
            "vision_service_state": "stopped",
            "vision_service_autostart": False,
            "vision_service_mode": "camera",
            "vision_mock_replay": "",
        },
    )

    collector.push("user_transcript_final", {"text": "well hm"})
    collector.push("turn_start", {
        "input_type": "send",
        "input_text": "well hm",
        "input_source": "voice",
    })
    collector.push("response_chunk", {"text": "Let me think"})
    expect(page.locator(".stream-card[data-event-type='assistant_reply']")).to_have_count(1)
    expect(page.locator(".stream-card").filter(has_text="Let me think")).to_have_count(1)

    collector.push("user_transcript_final", {"text": "i don't think so"})
    collector.push("turn_start", {
        "input_type": "send",
        "input_text": "well hm i don't think so",
        "input_source": "voice",
    })
    collector.push("response_interrupted", {
        "reason": "barge_in",
        "raw_full_text": "Let me think about that for a second.",
        "truncated_text": "Let me think —",
        "spoken_chars": 13,
        "playback_ratio": 0.33,
    })
    collector.push("response_chunk", {"text": "Okay, got it."})
    collector.push("response_done", {
        "full_text": "Okay, got it.",
        "input_source": "voice",
        "ttft_ms": 120,
        "llm_ms": 260,
        "total_ms": 900,
    })

    expect(page.locator(".stream-card[data-event-type='response_interrupted']")).to_have_count(1)
    interrupted_reply = page.locator(".stream-card[data-event-type='assistant_reply']").filter(has_text="Let me think").first
    expect(interrupted_reply.locator(".reply-marker", has_text="Interrupted")).to_have_count(1)
    interrupted_event = page.locator(".stream-card[data-event-type='response_interrupted']").first
    expect(interrupted_event).to_contain_text("Assistant cut off after: Let me think —")
    expect(interrupted_event).to_contain_text("spoken: Let me think —")
    expect(interrupted_event).to_contain_text("cut off by: i don't think so")
    interrupted_event.evaluate("(node) => node.click()")
    expect(page.locator("#detail-payload")).to_contain_text("\"raw_full_text\": \"Let me think about that for a second.\"")
    expect(page.locator("#detail-payload")).to_contain_text("\"interruption_text\": \"i don't think so\"")
    expect(page.locator(".stream-card[data-event-type='assistant_reply']")).to_have_count(2)
    expect(page.locator(".stream-card").filter(has_text="Okay, got it.")).to_have_count(1)


def test_continue_from_beat_generated_assistant_line_rewinds_before_the_beat(
    browser_page: Page,
    dashboard_server,
):
    collector, input_queue, dashboard_url, _ = dashboard_server
    page = browser_page
    page.goto(dashboard_url)

    collector.push(
        "session_start",
        {
            "character": "greg",
            "model": "groq-llama-8b",
            "session_id": "sess-playwright",
            "stage": "watching",
        },
    )
    collector.push(
        "runtime_controls",
        {
            "controls": {
                "reconcile": True,
                "vision": True,
                "auto_beat": False,
                "filler": False,
            },
            "conversation_paused": True,
            "session_stopped": False,
            "startup_pause_pending": False,
            "session_state": "paused",
            "voice_active": False,
            "voice_status": {
                "active": False,
                "output_only": False,
                "filler_enabled": False,
                "tts_backend": "",
                "mic_ready": False,
                "speaker_ready": False,
                "stt": {"state": "off", "detail": "Voice inactive."},
                "tts": {"state": "off", "detail": "Voice inactive.", "backend": ""},
            },
            "vision_active": False,
            "reconcile_thread_alive": False,
            "vision_service_url": "",
            "vision_service_health": False,
            "vision_service_managed": False,
            "vision_service_external": False,
            "vision_service_state": "stopped",
            "vision_service_autostart": False,
            "vision_service_mode": "camera",
            "vision_mock_replay": "",
        },
    )
    collector.push("turn_start", {"input_type": "beat", "input_text": ""})
    collector.push("response_done", {"full_text": "Nice scarf, what brings you here?"})

    card = page.locator(".stream-card").filter(has_text="Nice scarf, what brings you here?").first
    card.evaluate("(node) => node.click()")
    page.locator("#detail-continue-from-event").evaluate("(node) => node.click()")

    assert input_queue.get(timeout=2) == "/rewind 0"
    assert input_queue.get(timeout=2) == "/play"
    assert input_queue.get(timeout=2) == "/beat"


def test_archive_load_shows_timeline_and_scrubbable_media_player(
    browser_page: Page,
    dashboard_server,
):
    collector, _, dashboard_url, _ = dashboard_server
    page = browser_page
    page.goto(dashboard_url)

    collector.push(
        "runtime_controls",
        {
            "controls": {
                "reconcile": True,
                "vision": True,
                "auto_beat": False,
                "filler": False,
            },
            "conversation_paused": True,
            "session_stopped": True,
            "startup_pause_pending": False,
            "session_state": "ready",
            "voice_active": False,
            "voice_status": {
                "active": False,
                "output_only": False,
                "filler_enabled": False,
                "tts_backend": "",
                "mic_ready": False,
                "speaker_ready": False,
                "stt": {"state": "off", "detail": "Voice inactive."},
                "tts": {"state": "off", "detail": "Voice inactive.", "backend": ""},
            },
            "vision_active": True,
            "reconcile_thread_alive": False,
            "vision_service_url": "",
            "vision_service_health": False,
            "vision_service_managed": False,
            "vision_service_external": False,
            "vision_service_state": "stopped",
            "vision_service_autostart": False,
            "vision_service_mode": "camera",
            "vision_mock_replay": "",
        },
    )

    page.locator("#archive-load-button").evaluate("(node) => node.click()")
    page.locator(".archive-row").filter(has_text="sess-archive-playback").locator(".archive-load-action").click()

    expect(page.locator("#archive-load-status")).to_contain_text("Browsing archive:")
    expect(page.locator("#runtime-toggle")).to_have_text("Play")
    expect(page.locator("#runtime-toggle")).to_be_disabled()
    expect(page.locator("#archive-transport")).to_have_class(re.compile(r"\bvisible\b"))
    expect(page.locator("#archive-transport-toggle")).to_have_text("Play")
    expect(page.locator("#archive-media-player")).to_have_class(re.compile(r"\bvisible\b"))
    expect(page.locator("#stream-ruler")).to_have_class(re.compile(r"\bscrubbable\b"))
    assert "00:00" in page.evaluate("window.__dashboardTest.rulerState().labels")
    expect(page.locator("#current-playhead-value")).to_have_text("00:00.0")
    expect(page.locator(".stream-card[data-event-type='session_start']")).to_have_count(0)
    expect(page.locator(".stream-card[data-event-type='user_transcript_final']")).to_have_count(1)
    expect(page.locator(".stream-card[data-event-type='history_status']")).to_have_count(0)
    expect(page.locator(".stream-card[data-event-type='runtime_controls']")).to_have_count(0)
    expect(page.locator(".stream-card").filter(has_text="Hey, what time is it to you?")).to_have_count(1)

    page.locator("#archive-transport-toggle").evaluate("(node) => node.click()")
    page.wait_for_function(
        "() => { const node = document.getElementById('archive-media-player'); return !!node && node.currentTime > 0.1; }",
        timeout=5000,
    )
    player_time = page.locator("#archive-media-player").evaluate("(node) => node.currentTime")
    assert player_time > 0.1
    expect(page.locator("#archive-transport-toggle")).to_have_text("Pause")
    expect(page.locator("#stream-playhead")).to_have_class(re.compile(r"\bvisible\b"))
    expect(page.locator("#stream-playhead")).to_have_attribute("data-time", re.compile(r"\d{2}:\d{2}(?:\.\d)?"))
    expect(page.locator("#current-playhead-value")).to_have_text(re.compile(r"\d{2}:\d{2}\.\d"))

    page.locator("#archive-media-player").evaluate("(node) => { node.pause(); node.currentTime = 0; }")
    page.wait_for_timeout(100)
    reply_card = page.locator(".stream-card").filter(has_text="Hey, what time is it to you?").first
    reply_card.evaluate("(node) => node.click()")
    page.wait_for_timeout(200)
    selected_time = page.locator("#archive-media-player").evaluate("(node) => node.currentTime")
    assert selected_time >= 0.9

    ruler_seek_time = page.evaluate(
        """() => {
            const player = document.getElementById('archive-media-player');
            const ruler = document.getElementById('stream-ruler');
            player.pause();
            player.currentTime = 0;
            const rect = ruler.getBoundingClientRect();
            const x = rect.left + rect.width * 0.75;
            const y = rect.top + rect.height / 2;
            ruler.dispatchEvent(new PointerEvent('pointerdown', {
                bubbles: true,
                clientX: x,
                clientY: y,
                pointerId: 1,
                isPrimary: true,
            }));
            window.dispatchEvent(new PointerEvent('pointerup', {
                bubbles: true,
                clientX: x,
                clientY: y,
                pointerId: 1,
                isPrimary: true,
            }));
            return player.currentTime;
        }"""
    )
    assert ruler_seek_time > 0.5

    page.locator("#time-window").evaluate(
        """(node) => {
            node.value = "0";
            node.dispatchEvent(new Event("input", {bubbles: true}));
        }"""
    )
    drag_result = page.evaluate(
        """() => {
            const player = document.getElementById('archive-media-player');
            const viewport = document.getElementById('stream-viewport');
            player.pause();
            player.currentTime = 0;
            viewport.scrollLeft = 0;
            const rect = viewport.getBoundingClientRect();
            const x = rect.left + rect.width * 0.7;
            const y = rect.top + Math.min(rect.height - 30, 120);
            viewport.dispatchEvent(new MouseEvent('mousedown', {
                bubbles: true,
                clientX: x,
                clientY: y,
                button: 0,
            }));
            window.dispatchEvent(new MouseEvent('mousemove', {
                bubbles: true,
                clientX: x - 180,
                clientY: y,
                buttons: 1,
            }));
            window.dispatchEvent(new MouseEvent('mouseup', {
                bubbles: true,
                clientX: x - 180,
                clientY: y,
                button: 0,
            }));
            return {
                currentTime: player.currentTime,
                scrollLeft: viewport.scrollLeft,
            };
        }"""
    )
    assert drag_result["currentTime"] <= 0.05
    assert drag_result["scrollLeft"] > 0


def test_archive_ruler_scrubbing_updates_playhead_across_longer_spans(
    browser_page: Page,
    dashboard_server,
):
    collector, _, dashboard_url, _ = dashboard_server
    page = browser_page
    page.goto(dashboard_url)

    collector.push(
        "runtime_controls",
        {
            "controls": {
                "reconcile": True,
                "vision": True,
                "auto_beat": False,
                "filler": False,
            },
            "conversation_paused": True,
            "session_stopped": True,
            "startup_pause_pending": False,
            "session_state": "ready",
            "voice_active": False,
            "voice_status": {
                "active": False,
                "output_only": False,
                "filler_enabled": False,
                "tts_backend": "",
                "mic_ready": False,
                "speaker_ready": False,
                "stt": {"state": "off", "detail": "Voice inactive."},
                "tts": {"state": "off", "detail": "Voice inactive.", "backend": ""},
            },
            "vision_active": True,
            "reconcile_thread_alive": False,
            "vision_service_url": "",
            "vision_service_health": False,
            "vision_service_managed": False,
            "vision_service_external": False,
            "vision_service_state": "stopped",
            "vision_service_autostart": False,
            "vision_service_mode": "camera",
            "vision_mock_replay": "",
        },
    )

    page.locator("#archive-load-button").evaluate("(node) => node.click()")
    page.locator(".archive-row").filter(has_text="sess-archive-long-playback").locator(".archive-load-action").click()

    expect(page.locator("#archive-load-status")).to_contain_text("Browsing archive:")
    expect(page.locator("#archive-transport")).to_have_class(re.compile(r"\bvisible\b"))
    expect(page.locator("#stream-ruler")).to_have_class(re.compile(r"\bscrubbable\b"))

    page.locator("#time-window").evaluate(
        """(node) => {
            node.value = "0";
            node.dispatchEvent(new Event("input", {bubbles: true}));
        }"""
    )
    page.locator("#archive-media-player").evaluate("(node) => { node.pause(); node.currentTime = 0; }")

    targets = [
        ("post_response", 3.8),
        ("world_update", 8.6),
    ]

    results = []
    for event_type, target in targets:
        page.evaluate(
            """(time) => {
                const slider = document.getElementById('archive-transport-scrubber');
                slider.value = String(time);
                slider.dispatchEvent(new Event('input', {bubbles: true}));
            }""",
            target,
        )
        page.wait_for_timeout(50)
        readout = page.locator("#current-playhead-value").inner_text()
        playhead = page.locator("#stream-playhead").get_attribute("data-time") or ""
        media_time = page.locator("#archive-media-player").evaluate("(node) => Number(node.currentTime || 0)")
        scroll_left = page.locator("#stream-viewport").evaluate("(node) => Number(node.scrollLeft || 0)")
        readout_seconds = page.evaluate(
            """(text) => {
                const parts = String(text || '').split(':');
                if (parts.length !== 2) return NaN;
                return (Number(parts[0]) * 60) + Number(parts[1]);
            }""",
            readout,
        )
        playhead_seconds = page.evaluate(
            """(text) => {
                const parts = String(text || '').split(':');
                if (parts.length !== 2) return NaN;
                return (Number(parts[0]) * 60) + Number(parts[1]);
            }""",
            playhead,
        )
        results.append(
            {
                "event_type": event_type,
                "target": target,
                "readout": readout,
                "readoutSeconds": readout_seconds,
                "playhead": playhead,
                "playheadSeconds": playhead_seconds,
                "mediaTime": media_time,
                "selectedType": page.locator("#detail-type").inner_text(),
                "scrollLeft": scroll_left,
            }
        )

    for result in results:
        assert abs(result["readoutSeconds"] - result["target"]) < 0.35, result
        assert abs(result["playheadSeconds"] - result["target"]) < 0.35, result
        assert abs(result["mediaTime"] - result["target"]) < 0.35, result
        assert result["selectedType"] == result["event_type"], result
    assert results[-1]["mediaTime"] > 8.0


def test_archive_virtualizes_dense_card_lanes_across_long_spans(
    browser_page: Page,
    dashboard_server,
):
    collector, _, dashboard_url, _ = dashboard_server
    page = browser_page
    page.goto(dashboard_url)

    collector.push(
        "runtime_controls",
        {
            "controls": {
                "reconcile": True,
                "vision": True,
                "auto_beat": False,
                "filler": False,
            },
            "conversation_paused": True,
            "session_stopped": True,
            "startup_pause_pending": False,
            "session_state": "ready",
            "voice_active": False,
            "voice_status": {
                "active": False,
                "output_only": False,
                "filler_enabled": False,
                "tts_backend": "",
                "mic_ready": False,
                "speaker_ready": False,
                "stt": {"state": "off", "detail": "Voice inactive."},
                "tts": {"state": "off", "detail": "Voice inactive.", "backend": ""},
            },
            "vision_active": True,
            "reconcile_thread_alive": False,
            "vision_service_url": "",
            "vision_service_health": False,
            "vision_service_managed": False,
            "vision_service_external": False,
            "vision_service_state": "stopped",
            "vision_service_autostart": False,
            "vision_service_mode": "camera",
            "vision_mock_replay": "",
        },
    )

    page.locator("#archive-load-button").evaluate("(node) => node.click()")
    page.locator(".archive-row").filter(has_text="sess-archive-dense-cards").locator(".archive-load-action").click()

    page.wait_for_function(
        """() => {
            const state = window.__dashboardTest.timelineState();
            return state.streamEventCount >= 470 && state.visibleCardCount > 0;
        }""",
        timeout=10_000,
    )
    state = page.evaluate("window.__dashboardTest.timelineState()")
    assert state["streamEventCount"] >= 470
    assert state["visibleCardCount"] < 120

    page.evaluate("window.__dashboardTest.scrollTimelineToStart()")
    page.wait_for_function(
        """() => {
            const labels = Array.from(document.querySelectorAll(".stream-card[data-event-type='assistant_reply'] .stream-card-label"))
                .map((node) => node.textContent || "");
            return labels.some((text) => text.includes("Dense reply turn 0"))
                && !labels.some((text) => text.includes("Dense reply turn 95"));
        }""",
        timeout=10_000,
    )

    page.evaluate("window.__dashboardTest.scrollTimelineToEnd()")
    page.wait_for_function(
        """() => {
            const labels = Array.from(document.querySelectorAll(".stream-card[data-event-type='assistant_reply'] .stream-card-label"))
                .map((node) => node.textContent || "");
            return labels.some((text) => text.includes("Dense reply turn 95"))
                && !labels.some((text) => text.includes("Dense reply turn 0"));
        }""",
        timeout=10_000,
    )
    end_state = page.evaluate("window.__dashboardTest.timelineState()")
    assert end_state["visibleCardCount"] < 120
    lane_heights = page.evaluate("window.__dashboardTest.laneTrackHeights()")
    assert lane_heights
    sparse_lanes = [item for item in lane_heights if item["visibleCardCount"] <= 1]
    assert all(item["height"] <= 140 for item in sparse_lanes), lane_heights


def test_archive_playhead_reports_media_cap_when_timeline_moves_past_video(
    browser_page: Page,
    dashboard_server,
):
    collector, _, dashboard_url, _ = dashboard_server
    page = browser_page
    page.goto(dashboard_url)

    collector.push(
        "runtime_controls",
        {
            "controls": {
                "reconcile": True,
                "vision": True,
                "auto_beat": False,
                "filler": False,
            },
            "conversation_paused": True,
            "session_stopped": True,
            "startup_pause_pending": False,
            "session_state": "ready",
            "voice_active": False,
            "voice_status": {
                "active": False,
                "output_only": False,
                "filler_enabled": False,
                "tts_backend": "",
                "mic_ready": False,
                "speaker_ready": False,
                "stt": {"state": "off", "detail": "Voice inactive."},
                "tts": {"state": "off", "detail": "Voice inactive.", "backend": ""},
            },
            "vision_active": True,
            "reconcile_thread_alive": False,
            "vision_service_url": "",
            "vision_service_health": False,
            "vision_service_managed": False,
            "vision_service_external": False,
            "vision_service_state": "stopped",
            "vision_service_autostart": False,
            "vision_service_mode": "camera",
            "vision_mock_replay": "",
        },
    )

    page.locator("#archive-load-button").evaluate("(node) => node.click()")
    page.locator(".archive-row").filter(has_text="sess-archive-media-gap").locator(".archive-load-action").click()
    page.wait_for_function(
        "() => { const node = document.getElementById('archive-transport-scrubber'); return !!node && Number(node.max || 0) >= 4; }",
        timeout=5000,
    )

    page.evaluate("window.__dashboardTest.seekArchiveTime(8.6)")
    page.wait_for_timeout(100)

    state = page.evaluate("window.__dashboardTest.archivePlaybackState()")
    assert abs(state["playheadTime"] - 8.6) < 0.35, state
    assert abs(state["mediaTime"] - 4.2) < 0.2, state
    assert state["capped"] is True, state
    assert "4.2" in state["capNote"], state
    expect(page.locator("#current-playhead-readout")).to_have_class(re.compile(r"\bwarn\b"))
    expect(page.locator("#current-media-readout")).to_have_class(re.compile(r"\bwarn\b"))
    expect(page.locator("#archive-timeline-gap-note")).to_have_class(re.compile(r"\bvisible\b"))


def test_archive_scrub_cursor_uses_preview_on_hover_and_active_on_drag(
    browser_page: Page,
    dashboard_server,
):
    collector, _, dashboard_url, _ = dashboard_server
    page = browser_page
    page.goto(dashboard_url)

    collector.push(
        "runtime_controls",
        {
            "controls": {
                "reconcile": True,
                "vision": True,
                "auto_beat": False,
                "filler": False,
            },
            "conversation_paused": True,
            "session_stopped": True,
            "startup_pause_pending": False,
            "session_state": "ready",
            "voice_active": False,
            "voice_status": {
                "active": False,
                "output_only": False,
                "filler_enabled": False,
                "tts_backend": "",
                "mic_ready": False,
                "speaker_ready": False,
                "stt": {"state": "off", "detail": "Voice inactive."},
                "tts": {"state": "off", "detail": "Voice inactive.", "backend": ""},
            },
            "vision_active": True,
            "reconcile_thread_alive": False,
            "vision_service_url": "",
            "vision_service_health": False,
            "vision_service_managed": False,
            "vision_service_external": False,
            "vision_service_state": "stopped",
            "vision_service_autostart": False,
            "vision_service_mode": "camera",
            "vision_mock_replay": "",
        },
    )

    page.locator("#archive-load-button").evaluate("(node) => node.click()")
    page.locator(".archive-row").filter(has_text="sess-archive-playback").locator(".archive-load-action").click()
    page.wait_for_function(
        "() => { const node = document.getElementById('archive-transport-scrubber'); return !!node && Number(node.max || 0) > 0; }",
        timeout=5000,
    )

    ruler_box = page.locator("#stream-ruler").bounding_box()
    assert ruler_box is not None
    x = ruler_box["x"] + (ruler_box["width"] * 0.4)
    y = ruler_box["y"] + (ruler_box["height"] / 2)
    page.evaluate(
        """({x, y}) => {
            const ruler = document.getElementById('stream-ruler');
            ruler.dispatchEvent(new PointerEvent('pointermove', {
                bubbles: true,
                clientX: x,
                clientY: y,
                pointerId: 1,
                isPrimary: true,
            }));
        }""",
        {"x": x, "y": y},
    )
    page.wait_for_timeout(50)
    hover_state = page.evaluate("window.__dashboardTest.scrubCursorState()")
    assert hover_state == {"visible": True, "preview": True, "active": False}

    page.evaluate(
        """({x, y}) => {
            const ruler = document.getElementById('stream-ruler');
            ruler.dispatchEvent(new PointerEvent('pointerdown', {
                bubbles: true,
                clientX: x,
                clientY: y,
                pointerId: 1,
                isPrimary: true,
            }));
        }""",
        {"x": x, "y": y},
    )
    page.wait_for_timeout(50)
    active_state = page.evaluate("window.__dashboardTest.scrubCursorState()")
    assert active_state == {"visible": True, "preview": False, "active": True}

    page.evaluate(
        """({x, y}) => {
            window.dispatchEvent(new PointerEvent('pointerup', {
                bubbles: true,
                clientX: x,
                clientY: y,
                pointerId: 1,
                isPrimary: true,
            }));
        }""",
        {"x": x, "y": y},
    )


def test_rewind_keeps_new_active_path_visible_and_greys_out_superseded_events(
    browser_page: Page,
    dashboard_server,
):
    collector, _, dashboard_url, _ = dashboard_server
    page = browser_page
    page.goto(dashboard_url)

    collector.push(
        "session_start",
        {
            "character": "greg",
            "model": "groq-llama-8b",
            "session_id": "sess-playwright",
            "stage": "watching",
        },
    )
    collector.push(
        "runtime_controls",
        {
            "controls": {
                "reconcile": True,
                "vision": True,
                "auto_beat": False,
                "filler": False,
            },
            "conversation_paused": False,
            "session_stopped": False,
            "startup_pause_pending": False,
            "session_state": "live",
            "voice_active": False,
            "voice_status": {
                "active": False,
                "output_only": False,
                "filler_enabled": False,
                "tts_backend": "",
                "mic_ready": False,
                "speaker_ready": False,
                "stt": {"state": "off", "detail": "Voice inactive."},
                "tts": {"state": "off", "detail": "Voice inactive.", "backend": ""},
            },
            "vision_active": False,
            "reconcile_thread_alive": False,
            "vision_service_url": "",
            "vision_service_health": False,
            "vision_service_managed": False,
            "vision_service_external": False,
            "vision_service_state": "stopped",
            "vision_service_autostart": False,
            "vision_service_mode": "camera",
            "vision_mock_replay": "",
        },
    )
    collector.push("turn_start", {"input_type": "beat", "input_text": ""})
    collector.push("response_done", {"full_text": "Nice scarf, what brings you here?"})
    collector.push("turn_start", {"input_type": "send", "input_text": "what time is it"})
    collector.push("response_done", {"full_text": "Right, my question was about the time."})

    events = collector.get_all()
    first_turn_timestamp = next(event["timestamp"] for event in events if event["type"] == "turn_start")

    collector.push(
        "history_status",
        {
            "enabled": True,
            "free_gib": 100.0,
            "sessions_count": 1,
            "pinned_count": 0,
            "moments_count": 0,
            "current_session": {
                "session_id": "sess-playwright",
                "record_mode": "debug_only",
                "replay_capture_enabled": False,
                "debug_only_reason": "rewind made this run exploratory; replay/snippet capture is now off",
            },
        },
    )
    collector.push(
        "history_rewind",
        {
            "source_session_id": "sess-playwright",
            "checkpoint_index": 0,
            "label": "beat",
            "mode": "rewind",
            "checkpoint_timestamp": first_turn_timestamp,
            "debug_only_reason": "rewind made this run exploratory; replay/snippet capture is now off",
        },
    )
    collector.push("turn_start", {"input_type": "beat", "input_text": ""})
    collector.push("response_done", {"full_text": "Nice scarf, what brings you here? (rerun)"})

    expect(page.locator(".stream-card.superseded").filter(has_text="Right, my question was about the time.")).to_have_count(1)
    expect(page.locator(".stream-card").filter(has_text="Nice scarf, what brings you here? (rerun)")).to_have_count(1)
    expect(page.locator(".stream-card.superseded").filter(has_text="Nice scarf, what brings you here? (rerun)")).to_have_count(0)
    expect(
        page.locator(".stream-card").filter(has_text="Nice scarf, what brings you here? (rerun)").locator(".reply-marker", has_text="Debug Only")
    ).to_have_count(0)


def test_save_stop_fallback_opens_modal_without_session_end(
    browser_page: Page,
    dashboard_server,
    vision_service: VisionServiceController,
):
    collector, input_queue, dashboard_url, _report_dir = dashboard_server
    vision_service_url = vision_service.url
    page = browser_page
    page.goto(dashboard_url)

    collector.push(
        "session_start",
        {
            "character": "greg",
            "model": "groq-llama-8b",
            "session_id": "sess-playwright",
            "stage": "greeting",
        },
    )
    collector.push(
        "runtime_controls",
        {
            "controls": {
                "reconcile": True,
                "vision": True,
                "auto_beat": False,
                "filler": False,
            },
            "conversation_paused": False,
            "session_stopped": False,
            "startup_pause_pending": False,
            "session_state": "live",
            "voice_active": True,
            "voice_status": {
                "active": True,
                "output_only": False,
                "filler_enabled": False,
                "tts_backend": "pocket",
                "mic_ready": True,
                "speaker_ready": True,
                "stt": {"state": "ready", "detail": "Deepgram socket open."},
                "tts": {"state": "ready", "detail": "Pocket-TTS reachable.", "backend": "pocket"},
            },
            "vision_active": True,
            "reconcile_thread_alive": True,
            "vision_service_url": vision_service_url,
            "vision_service_health": True,
            "vision_service_managed": False,
            "vision_service_external": True,
            "vision_service_state": "external",
            "vision_service_autostart": True,
            "vision_service_mode": "camera",
            "vision_mock_replay": "",
        },
    )
    collector.push(
        "runtime_controls",
        {
            "controls": {
                "reconcile": True,
                "vision": True,
                "auto_beat": False,
                "filler": False,
            },
            "conversation_paused": False,
            "session_stopped": False,
            "startup_pause_pending": False,
            "session_state": "live",
            "voice_active": True,
            "voice_status": {
                "active": True,
                "output_only": False,
                "filler_enabled": False,
                "tts_backend": "pocket",
                "mic_ready": True,
                "speaker_ready": True,
                "stt": {"state": "ready", "detail": "Deepgram socket open."},
                "tts": {"state": "ready", "detail": "Pocket-TTS reachable.", "backend": "pocket"},
            },
            "vision_active": True,
            "reconcile_thread_alive": True,
            "vision_service_url": vision_service_url,
            "vision_service_health": True,
            "vision_service_managed": False,
            "vision_service_external": True,
            "vision_service_state": "external",
            "vision_service_autostart": True,
            "vision_service_mode": "camera",
            "vision_mock_replay": "",
        },
    )
    collector.push("turn_start", {"input_type": "beat", "input_text": ""})
    collector.push("response_done", {"full_text": "Finished session content"})

    page.locator("#runtime-stop").click()
    expect(page.locator("#save-complete-modal")).to_have_class(re.compile(r"\bopen\b"))
    expect(page.locator("#save-complete-heading")).to_contain_text("Opening Stop Review")
    expect(page.locator("#save-complete-status")).to_contain_text("Pausing before review")
    assert input_queue.get(timeout=2) == "/review"
    collector.push(
        "runtime_controls",
        {
            "controls": {
                "reconcile": True,
                "vision": True,
                "auto_beat": False,
                "filler": False,
            },
            "conversation_paused": True,
            "session_stopped": False,
            "startup_pause_pending": False,
            "session_state": "paused",
            "runtime_phase": "review",
            "voice_active": True,
            "voice_status": {
                "active": True,
                "output_only": False,
                "filler_enabled": False,
                "tts_backend": "pocket",
                "mic_ready": True,
                "speaker_ready": True,
                "stt": {"state": "ready", "detail": "Deepgram socket open."},
                "tts": {"state": "ready", "detail": "Pocket-TTS reachable.", "backend": "pocket"},
            },
            "vision_active": True,
            "reconcile_thread_alive": True,
            "vision_service_url": vision_service_url,
            "vision_service_health": True,
            "vision_service_managed": False,
            "vision_service_external": True,
            "vision_service_state": "external",
            "vision_service_autostart": True,
            "vision_service_mode": "camera",
            "vision_mock_replay": "",
        },
    )
    page.locator("#save-complete-title").fill("Fallback save flow")
    page.locator("#save-complete-save").click()
    expect(page.locator("#save-complete-status")).to_contain_text("Saving...")
    expect(page.locator("#runtime-stop")).to_have_text("Saving...")
    assert input_queue.get(timeout=2) == "/save"

    collector.push(
        "runtime_controls",
        {
            "controls": {
                "reconcile": True,
                "vision": True,
                "auto_beat": False,
                "filler": False,
            },
            "conversation_paused": True,
            "session_stopped": True,
            "startup_pause_pending": True,
            "session_state": "ready",
            "voice_active": True,
            "voice_status": {
                "active": True,
                "output_only": False,
                "filler_enabled": False,
                "tts_backend": "pocket",
                "mic_ready": True,
                "speaker_ready": True,
                "stt": {"state": "ready", "detail": "Deepgram socket open."},
                "tts": {"state": "ready", "detail": "Pocket-TTS reachable.", "backend": "pocket"},
            },
            "vision_active": True,
            "reconcile_thread_alive": True,
            "vision_service_url": vision_service_url,
            "vision_service_health": True,
            "vision_service_managed": False,
            "vision_service_external": True,
            "vision_service_state": "external",
            "vision_service_autostart": True,
            "vision_service_mode": "camera",
            "vision_mock_replay": "",
        },
    )
    collector.push(
        "history_status",
        {
            "enabled": True,
            "free_gib": 100.0,
            "sessions_count": 1,
            "pinned_count": 0,
            "moments_count": 0,
            "current_session": {
                "session_id": "sess-playwright",
                "title": "Fallback save flow",
                "session_path": "/tmp/character-eng-dashboard-playwright-history/sessions/sess-playwright",
                "record_mode": "full",
                "replay_capture_enabled": True,
            },
            "playback": {"active": False},
        },
    )

    expect(page.locator("body")).not_to_have_class(re.compile(r"\bbooting\b"))
    expect(page.locator("#timeline")).to_contain_text("Finished session content")
    expect(page.locator("#save-complete-modal")).to_have_class(re.compile(r"\bopen\b"))
    expect(page.locator("#save-complete-session-id")).to_contain_text("sess-playwright")
    expect(page.locator("#save-complete-heading")).to_contain_text("Session Saved")
    expect(page.locator("#save-complete-title")).not_to_have_value("")


def test_paused_session_review_exports_full_window_without_marks(
    browser_page: Page,
    dashboard_server,
):
    collector, _, dashboard_url, _ = dashboard_server
    page = browser_page
    page.goto(dashboard_url)

    collector.push(
        "session_start",
        {
            "character": "greg",
            "model": "groq-llama-8b",
            "session_id": "sess-playwright",
            "stage": "watching",
        },
    )
    collector.push(
        "runtime_controls",
        {
            "controls": {
                "reconcile": True,
                "vision": True,
                "auto_beat": False,
                "filler": False,
            },
            "conversation_paused": False,
            "session_stopped": False,
            "startup_pause_pending": False,
            "session_state": "live",
            "voice_active": True,
            "voice_status": {
                "active": True,
                "output_only": False,
                "filler_enabled": False,
                "tts_backend": "pocket",
                "mic_ready": True,
                "speaker_ready": True,
                "stt": {"state": "ready", "detail": "Deepgram socket open."},
                "tts": {"state": "ready", "detail": "Pocket-TTS reachable.", "backend": "pocket"},
            },
            "vision_active": True,
            "reconcile_thread_alive": True,
            "vision_service_url": "http://127.0.0.1:7875",
            "vision_service_health": True,
            "vision_service_managed": False,
            "vision_service_external": True,
            "vision_service_state": "external",
            "vision_service_autostart": True,
            "vision_service_mode": "camera",
            "vision_mock_replay": "",
        },
    )
    collector.push(
        "history_status",
        {
            "enabled": True,
            "free_gib": 100.0,
            "sessions_count": 1,
            "pinned_count": 0,
            "moments_count": 0,
            "current_session": {
                "session_id": "sess-playwright",
                "title": "Playwright Session",
                "session_path": "/tmp/character-eng-dashboard-playwright-history/sessions/sess-playwright",
                "record_mode": "full",
                "replay_capture_enabled": True,
            },
            "playback": {"active": False},
        },
    )
    collector.push("pause", {"startup": False})

    expect(page.locator("#archive-load-button")).to_have_text("Return To Session")
    expect(page.locator("#archive-export-status")).to_contain_text("export the full paused session")
    page.locator("#archive-export-button").evaluate("(node) => node.click()")
    expect(page.locator("#save-complete-modal")).to_have_class(re.compile(r"\bopen\b"))
    expect(page.locator("#save-complete-heading")).to_contain_text("Export Archive")
    page.locator("#save-complete-close").click()
    expect(page.locator("#save-complete-modal")).not_to_have_class(re.compile(r"\bopen\b"))
    page.locator("#archive-export-button").evaluate("(node) => node.click()")
    expect(page.locator("#save-complete-heading")).to_contain_text("Export Archive")
    page.locator("#save-complete-title").fill("Paused review export")
    page.locator("#save-complete-save").click()
    expect(page.locator("#save-complete-heading")).to_contain_text("Archive Exported")
    expect(page.locator("#save-complete-session-id")).to_contain_text("sess-playwright")
    expect(page.locator("#archive-load-button")).to_have_text("Return To Session")


def test_discard_session_stays_in_discard_flow(
    browser_page: Page,
    dashboard_server,
    vision_service: VisionServiceController,
):
    collector, input_queue, dashboard_url, _report_dir = dashboard_server
    vision_service_url = vision_service.url
    page = browser_page
    page.goto(dashboard_url)

    collector.push(
        "session_start",
        {
            "character": "greg",
            "model": "groq-llama-8b",
            "session_id": "sess-playwright",
            "stage": "greeting",
        },
    )
    collector.push(
        "runtime_controls",
        {
            "controls": {
                "reconcile": True,
                "vision": True,
                "auto_beat": False,
                "filler": False,
            },
            "conversation_paused": False,
            "session_stopped": False,
            "startup_pause_pending": False,
            "session_state": "live",
            "voice_active": True,
            "voice_status": {
                "active": True,
                "output_only": False,
                "filler_enabled": False,
                "tts_backend": "pocket",
                "mic_ready": True,
                "speaker_ready": True,
                "stt": {"state": "ready", "detail": "Deepgram socket open."},
                "tts": {"state": "ready", "detail": "Pocket-TTS reachable.", "backend": "pocket"},
            },
            "vision_active": True,
            "reconcile_thread_alive": True,
            "vision_service_url": vision_service_url,
            "vision_service_health": True,
            "vision_service_managed": False,
            "vision_service_external": True,
            "vision_service_state": "external",
            "vision_service_autostart": True,
            "vision_service_mode": "camera",
            "vision_mock_replay": "",
        },
    )
    collector.push("turn_start", {"input_type": "beat", "input_text": ""})
    collector.push("response_done", {"full_text": "Discard me"})

    page.locator("#runtime-stop").click()
    expect(page.locator("#save-complete-heading")).to_contain_text("Opening Stop Review")
    expect(page.locator("#save-complete-status")).to_contain_text("Pausing before review")
    assert input_queue.get(timeout=2) == "/review"
    collector.push(
        "runtime_controls",
        {
            "controls": {
                "reconcile": True,
                "vision": True,
                "auto_beat": False,
                "filler": False,
            },
            "conversation_paused": True,
            "session_stopped": False,
            "startup_pause_pending": False,
            "session_state": "paused",
            "runtime_phase": "review",
            "voice_active": True,
            "voice_status": {
                "active": True,
                "output_only": False,
                "filler_enabled": False,
                "tts_backend": "pocket",
                "mic_ready": True,
                "speaker_ready": True,
                "stt": {"state": "ready", "detail": "Deepgram socket open."},
                "tts": {"state": "ready", "detail": "Pocket-TTS reachable.", "backend": "pocket"},
            },
            "vision_active": True,
            "reconcile_thread_alive": True,
            "vision_service_url": vision_service_url,
            "vision_service_health": True,
            "vision_service_managed": False,
            "vision_service_external": True,
            "vision_service_state": "external",
            "vision_service_autostart": True,
            "vision_service_mode": "camera",
            "vision_mock_replay": "",
        },
    )
    expect(page.locator("#save-complete-heading")).to_contain_text("Stop Session")
    page.locator("#save-complete-discard").click()
    expect(page.locator("#save-complete-status")).to_contain_text("Discarding...")
    expect(page.locator("#runtime-stop")).to_have_text("Discarding...")
    assert input_queue.get(timeout=2) == "/discard"

    collector.push(
        "runtime_controls",
        {
            "controls": {
                "reconcile": True,
                "vision": True,
                "auto_beat": False,
                "filler": False,
            },
            "conversation_paused": True,
            "session_stopped": True,
            "startup_pause_pending": True,
            "session_state": "ready",
            "voice_active": True,
            "voice_status": {
                "active": True,
                "output_only": False,
                "filler_enabled": False,
                "tts_backend": "pocket",
                "mic_ready": True,
                "speaker_ready": True,
                "stt": {"state": "ready", "detail": "Deepgram socket open."},
                "tts": {"state": "ready", "detail": "Pocket-TTS reachable.", "backend": "pocket"},
            },
            "vision_active": True,
            "reconcile_thread_alive": True,
            "vision_service_url": vision_service_url,
            "vision_service_health": True,
            "vision_service_managed": False,
            "vision_service_external": True,
            "vision_service_state": "external",
            "vision_service_autostart": True,
            "vision_service_mode": "camera",
            "vision_mock_replay": "",
        },
    )
    collector.push(
        "history_status",
        {
            "enabled": True,
            "free_gib": 100.0,
            "sessions_count": 1,
            "pinned_count": 0,
            "moments_count": 0,
            "current_session": {},
            "playback": {"active": False},
        },
    )
    collector.push("session_end", {"session_id": "sess-playwright", "discarded": True})

    expect(page.locator("#save-complete-heading")).to_contain_text("Session Discarded")
    expect(page.locator("#save-complete-heading")).not_to_contain_text("Session Saved")
    expect(page.locator("#save-complete-heading")).not_to_contain_text("Stop Session")


def test_discard_session_resolves_without_session_end_event(
    browser_page: Page,
    dashboard_server,
    vision_service: VisionServiceController,
):
    collector, input_queue, dashboard_url, _report_dir = dashboard_server
    vision_service_url = vision_service.url
    page = browser_page
    page.goto(dashboard_url)

    collector.push(
        "session_start",
        {
            "character": "greg",
            "model": "groq-llama-8b",
            "session_id": "sess-playwright",
            "stage": "greeting",
        },
    )
    collector.push(
        "runtime_controls",
        {
            "controls": {
                "reconcile": True,
                "vision": True,
                "auto_beat": False,
                "filler": False,
            },
            "conversation_paused": False,
            "session_stopped": False,
            "startup_pause_pending": False,
            "session_state": "live",
            "voice_active": True,
            "voice_status": {
                "active": True,
                "output_only": False,
                "filler_enabled": False,
                "tts_backend": "pocket",
                "mic_ready": True,
                "speaker_ready": True,
                "stt": {"state": "ready", "detail": "Deepgram socket open."},
                "tts": {"state": "ready", "detail": "Pocket-TTS reachable.", "backend": "pocket"},
            },
            "vision_active": True,
            "reconcile_thread_alive": True,
            "vision_service_url": vision_service_url,
            "vision_service_health": True,
            "vision_service_managed": False,
            "vision_service_external": True,
            "vision_service_state": "external",
            "vision_service_autostart": True,
            "vision_service_mode": "camera",
            "vision_mock_replay": "",
        },
    )
    collector.push("turn_start", {"input_type": "beat", "input_text": ""})
    collector.push("response_done", {"full_text": "Discard me"})

    page.locator("#runtime-stop").click()
    assert input_queue.get(timeout=2) == "/review"
    collector.push(
        "runtime_controls",
        {
            "controls": {
                "reconcile": True,
                "vision": True,
                "auto_beat": False,
                "filler": False,
            },
            "conversation_paused": True,
            "session_stopped": False,
            "startup_pause_pending": False,
            "session_state": "paused",
            "runtime_phase": "review",
            "voice_active": True,
            "voice_status": {
                "active": True,
                "output_only": False,
                "filler_enabled": False,
                "tts_backend": "pocket",
                "mic_ready": True,
                "speaker_ready": True,
                "stt": {"state": "ready", "detail": "Deepgram socket open."},
                "tts": {"state": "ready", "detail": "Pocket-TTS reachable.", "backend": "pocket"},
            },
            "vision_active": True,
            "reconcile_thread_alive": True,
            "vision_service_url": vision_service_url,
            "vision_service_health": True,
            "vision_service_managed": False,
            "vision_service_external": True,
            "vision_service_state": "external",
            "vision_service_autostart": True,
            "vision_service_mode": "camera",
            "vision_mock_replay": "",
        },
    )
    page.locator("#save-complete-discard").click()
    assert input_queue.get(timeout=2) == "/discard"

    collector.push(
        "runtime_controls",
        {
            "controls": {
                "reconcile": True,
                "vision": True,
                "auto_beat": False,
                "filler": False,
            },
            "conversation_paused": True,
            "session_stopped": True,
            "startup_pause_pending": True,
            "session_state": "ready",
            "voice_active": True,
            "voice_status": {
                "active": True,
                "output_only": False,
                "filler_enabled": False,
                "tts_backend": "pocket",
                "mic_ready": True,
                "speaker_ready": True,
                "stt": {"state": "ready", "detail": "Deepgram socket open."},
                "tts": {"state": "ready", "detail": "Pocket-TTS reachable.", "backend": "pocket"},
            },
            "vision_active": True,
            "reconcile_thread_alive": True,
            "vision_service_url": vision_service_url,
            "vision_service_health": True,
            "vision_service_managed": False,
            "vision_service_external": True,
            "vision_service_state": "external",
            "vision_service_autostart": True,
            "vision_service_mode": "camera",
            "vision_mock_replay": "",
        },
    )
    collector.push(
        "history_status",
        {
            "enabled": True,
            "free_gib": 100.0,
            "sessions_count": 0,
            "pinned_count": 0,
            "moments_count": 0,
            "current_session": {},
            "playback": {"active": False},
        },
    )
    started = time.perf_counter()
    expect(page.locator("#save-complete-heading")).to_contain_text("Session Discarded")
    elapsed = time.perf_counter() - started
    expect(page.locator("#save-complete-status")).to_contain_text("Discarded successfully")
    assert elapsed < 1.0
