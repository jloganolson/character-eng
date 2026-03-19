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

from character_eng.dashboard.events import DashboardEventCollector
from character_eng.dashboard.server import start_dashboard
from character_eng.history import HistoryService
from character_eng.person import PeopleState
from character_eng.world import Goals, Script, WorldState


def _write_test_video(path: Path):
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=320x240:d=2:r=30",
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


def _seed_archive_session(
    history: HistoryService,
    *,
    session_id: str,
    title: str | None = None,
    response_text: str = "Hey, what time is it to you?",
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
    _write_test_video(archive.media_dir / "playback.mp4")
    archive_manifest = json.loads(archive.manifest_path.read_text())
    archive_media = dict(archive_manifest.get("media", {}))
    archive_media["playback_path"] = str(archive.media_dir / "playback.mp4")
    archive_media["video_path"] = str(archive.media_dir / "playback.mp4")
    updates = {"media": archive_media}
    if title:
        updates["title"] = title
    archive.update_manifest(**updates)
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
def dashboard_server():
    collector = DashboardEventCollector()
    input_queue: queue.Queue[str] = queue.Queue()
    report_dir = Path("/tmp/character-eng-dashboard-playwright-reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    for child in report_dir.iterdir():
        child.unlink()
    history_root = Path("/tmp/character-eng-dashboard-playwright-history")
    if history_root.exists():
        for child in sorted(history_root.rglob("*"), reverse=True):
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
    history = HistoryService(root=history_root, free_warning_gib=0.0)
    _seed_archive_session(history, session_id="sess-archive-playback")
    _seed_archive_session(
        history,
        session_id="greg-offline-canonical-v1",
        title="Greg Offline Canonical v1",
        response_text="Canonical archive is ready for offline UX work.",
    )
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
    _, port = start_dashboard(
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
    yield collector, input_queue, f"http://127.0.0.1:{port}/", report_dir
    collector.shutdown()


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
    expect(page.locator("#history-status")).to_contain_text("current:")
    expect(page.locator("#archive-load-button")).to_be_visible()
    expect(page.locator("#boot-grid")).to_contain_text("voice")
    expect(page.locator("#boot-grid")).to_contain_text("models")
    expect(page.locator("#runtime-status")).to_contain_text("auto-beat: off")
    expect(page.locator("#runtime-status")).to_contain_text("stt: ready")
    expect(page.locator("#runtime-status")).to_contain_text("tts: ready")
    expect(page.locator("#runtime-toggle")).to_have_text("Start")
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
    expect(page.locator("#runtime-toggle")).to_have_text("Start")

    page.locator("#runtime-toggle").evaluate("(node) => node.click()")
    expect(page.locator("#runtime-toggle")).to_have_text("Starting...")
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
        "transcript_final_ts": turn_base + 0.42,
        "stt_ms": 420,
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
        "transcript_final_ts": turn_base + 0.42,
        "stt_ms": 420,
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
    reply_card = page.locator(".stream-card").filter(has_text="Free water. Want one?").first
    reply_card.click()
    page.locator("#report-ref-button").click()
    expect(page.locator("#report-ref-status")).to_contain_text("Copied archive ref:")
    expect(page.locator("#report-ref-status")).to_contain_text("sess-playwright")
    expect(page.locator("#detail-type")).to_have_text("assistant_reply")
    expect(page.locator("#detail-label")).to_contain_text("Free water. Want one?")
    expect(page.locator("#detail-detail")).to_contain_text("STT 420ms")
    expect(page.locator("#detail-detail")).to_contain_text("TTFT 190ms")
    expect(page.locator("#detail-loop")).to_have_class(re.compile(r"\bactive\b"))
    expect(page.locator("#detail-loop-chips")).to_contain_text("speech 0ms")
    expect(page.locator("#detail-loop-chips")).to_contain_text("first audio")
    page.locator("#tab-prompts").click()
    expect(page.locator("#detail-prompts")).to_contain_text("Chat prompt")
    expect(page.locator("#detail-prompts")).to_contain_text("Keep it short.")
    expect(page.locator("#detail-prompts")).to_contain_text("Do you have water?")
    expect(page.locator(".prompt-message.role-system")).to_have_count(2)
    expect(page.locator(".prompt-message.role-user")).to_have_count(1)
    expect(page.locator(".prompt-message.role-assistant")).to_have_count(1)
    page.locator("#annotation-toggle").click()
    expect(page.locator("#annotation-form")).to_be_visible()
    page.locator("#annotation-tags").fill("latency, interruption")
    page.locator("#annotation-subsystem").fill("timing")
    page.locator("#annotation-note").fill("Assistant waited too long before answering.")
    page.locator("#annotation-desired").fill("Start speaking sooner after the transcript finalizes.")
    page.locator("#annotation-save").click()
    expect(page.locator("#annotation-status")).to_contain_text("Saved annotation:")
    expect(page.locator("#annotation-form")).not_to_be_visible()
    expect(page.locator("#annotation-list")).to_contain_text("Assistant waited too long before answering.")
    expect(reply_card.locator(".reply-marker", has_text="note")).to_have_count(1)
    page.locator("#snippet-toggle").click()
    expect(page.locator("#snippet-form")).to_be_visible()
    page.locator("#snippet-save").click()
    expect(page.locator("#annotation-status")).to_contain_text("Captured snippet:")
    page.locator("#history-pin").click()
    expect(page.locator("#annotation-status")).to_contain_text("Pinned session:")
    page.locator("#history-checkpoint").fill("2")
    page.locator("#history-rewind").click()
    assert input_queue.get(timeout=2) == "/rewind 2"
    page.locator("#archive-load-button").click()
    expect(page.locator("#archive-picker")).to_have_class(re.compile(r"\bopen\b"))
    page.locator(".archive-row").filter(has=page.locator(".archive-kind", has_text="session")).locator(".archive-load-action").first.click()
    expect(page.locator("#archive-load-button")).to_have_text("Return To Live")
    expect(page.locator("#archive-load-status")).to_contain_text("Browsing archive:")
    expect(page.locator("#runtime-toggle")).to_have_text("Replay")
    expect(page.locator("#runtime-stop")).to_be_disabled()
    expect(page.locator(".stream-card[data-event-type='session_start']")).to_have_count(1)
    page.locator("#runtime-toggle").click()
    assert input_queue.get(timeout=2) == "/replay sess-playwright"
    expect(page.locator("#archive-load-button")).to_have_text("Return To Live")
    expect(page.locator("#archive-load-status")).to_contain_text("sess-playwright")
    page.locator("#archive-load-button").click()
    expect(page.locator("#archive-load-button")).to_have_text("Load Archive...")
    page.locator("#archive-load-button").click()
    page.locator(".archive-row").filter(has=page.locator(".archive-kind", has_text="session")).locator(".archive-replay-action").first.click()
    assert input_queue.get(timeout=2) == "/replay sess-playwright"
    page.locator("#detail-open-json").click()
    expect(page.locator("#detail-payload")).to_contain_text("\"session_id\": \"sess-playwright\"")
    expect(page.locator("#detail-related")).to_contain_text("No linked turn context")
    expect(page.locator("#detail-related")).not_to_contain_text("post_response")
    expect(page.locator("#tab-chronology")).to_have_count(0)
    expect(page.locator("#show-chronology")).to_have_count(0)

    page.locator("#stream-density").select_option("compact")
    reply_card.click()
    expect(page.locator("#detail-type")).to_have_text("assistant_reply")

    perception_card = page.locator(".stream-card").filter(has_text="A person is now holding a flier").first
    perception_card.click()
    expect(page.locator("#detail-type")).to_have_text("perception")
    expect(page.locator("#detail-trace")).to_have_class(re.compile(r"\bactive\b"))
    expect(page.locator("#detail-structure")).to_have_class(re.compile(r"\bactive\b"))
    expect(page.locator("#detail-structure")).to_contain_text("Vision Input")
    expect(page.locator("#detail-structure")).to_contain_text("Vision Synthesis Inputs")
    expect(page.locator("#detail-structure")).to_contain_text("Claim Provenance")
    expect(page.locator("#detail-trace-summary")).to_contain_text("Vision synthesis LLM")
    expect(page.locator("#detail-trace-summary")).to_contain_text("vision_synthesis_llm")
    expect(page.locator("#detail-structure")).to_contain_text("Evidence")
    expect(page.locator("#detail-structure")).to_contain_text("Is anyone holding a flier?")
    expect(page.locator("#detail-structure")).to_contain_text("A flier.")
    expect(page.locator("#detail-structure")).to_contain_text("supports")
    expect(page.locator("#detail-trace-summary")).to_contain_text("source: visual")
    expect(page.locator("#detail-trace-chips")).to_contain_text("object flier")
    expect(page.locator("#detail-trace-chips")).to_contain_text("sam flier")
    expect(page.locator("#detail-trace-chips")).to_contain_text("supports What is the person holding?")

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

    with page.expect_popup() as vision_popup_info:
        page.locator("#vision-service-link").click()
    vision_popup = vision_popup_info.value
    vision_popup.wait_for_load_state("domcontentloaded")
    assert "Vision Test UI" in vision_popup.text_content("body")
    vision_popup.close()

    page.locator("#runtime-stop").click()
    expect(page.locator("#save-complete-modal")).to_have_class(re.compile(r"\bopen\b"))
    expect(page.locator("#save-complete-heading")).to_contain_text("Pausing Session")
    expect(page.locator("#save-complete-status")).to_contain_text("Pausing before review")
    assert input_queue.get(timeout=2) == "/pause"
    collector.push("pause", {})
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
    expect(page.locator("#save-complete-heading")).to_contain_text("Save Session")
    expect(page.locator("#save-complete-status")).to_contain_text("Paused. Save or discard this session.")
    expect(page.locator("#save-complete-title")).not_to_have_value("")
    expect(page.locator("#runtime-stop")).to_have_text("Save + Stop")
    page.locator("#save-complete-title").fill("Greg stop flow")
    page.locator("#save-complete-save").click()
    expect(page.locator("#save-complete-status")).to_contain_text("Saving...")
    expect(page.locator("#runtime-stop")).to_have_text("Saving...")
    assert input_queue.get(timeout=2) == "/stop"

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
    expect(page.locator("#runtime-toggle")).to_have_text("Start")
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
    expect(page.locator("#runtime-toggle")).to_have_text("Replay")
    page.locator("#archive-load-button").click()
    expect(page.locator("#archive-load-button")).to_have_text("Load Archive...")

    page.locator("#report-ref-button").click()
    expect(page.locator("#report-ref-status")).to_contain_text("Copied archive ref:")
    expect(page.locator("#report-ref-status")).to_contain_text("sess-playwright")

    page.locator("#runtime-toggle").click()
    expect(page.locator("#runtime-toggle")).to_have_text("Starting...")
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

    page.locator("#boot-restart").click()
    assert input_queue.get(timeout=2) == "/restart"
    expect(page.locator("#boot-summary")).to_contain_text("Restarting conversation...")
    expect(page.locator(".stream-card")).to_have_count(0)

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
    expect(page.locator("#runtime-toggle")).to_have_text("Start")


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
    expect(page.locator("#runtime-toggle")).to_have_text("Start")
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
    expect(page.locator("#runtime-toggle")).to_have_text("Live Disabled")
    expect(page.locator("#runtime-toggle")).to_be_disabled()
    expect(page.locator("#runtime-restart")).to_be_disabled()
    expect(page.locator("#boot-restart")).to_have_text("Live Disabled")
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


def test_vision_state_event_defaults_snippet_mode_in_browser(
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

    card = page.locator(".stream-card[data-event-type='vision_state_update']").first
    expect(card).to_be_visible()
    card.click()
    page.locator("#snippet-toggle").click()
    expect(page.locator("#snippet-form")).to_be_visible()
    expect(page.locator("#snippet-mode")).to_have_value("vision_state_turn")
    expect(page.locator("#snippet-tags")).to_have_value("vision-state")
    page.locator("#snippet-save").click()
    expect(page.locator("#annotation-status")).to_contain_text("Captured snippet:")


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

    raw_sam3_dot = page.locator("[data-stream-lane='vision_raw'] .stream-dot[data-event-type='sam3_detection']").first
    expect(raw_sam3_dot).to_be_visible()
    raw_sam3_dot.dispatch_event("mouseenter", {"clientX": 320, "clientY": 220})
    expect(page.locator("#stream-hover-card")).to_have_class(re.compile(r"\bactive\b"))
    expect(page.locator("#stream-hover-card")).to_contain_text("sam3_detection")
    expect(page.locator("[data-stream-lane='vision']")).to_contain_text("vision llm")


def test_sidebar_registry_layout_and_panel_collapse(
    browser_page: Page,
    dashboard_server,
):
    _, _, dashboard_url, _ = dashboard_server
    page = browser_page
    page.goto(dashboard_url)

    expect(page.locator(".sidebar-group-label").nth(0)).to_have_text("Session Control")
    expect(page.locator(".sidebar-group-label").nth(1)).to_have_text("Live State")
    expect(page.locator("#plan-panel")).to_be_visible()
    expect(page.locator("#plan-panel")).to_have_attribute("data-panel-group", "live_state")
    expect(page.locator("#plan-content")).to_contain_text("No active script plan yet")
    assert page.evaluate(
        """() => Array.from(document.querySelectorAll('#sidebar .sidebar-panel'))
        .map((node) => node.dataset.panelId)"""
    ) == ["runtime", "history", "prompt_assets", "vision", "world", "stage", "plan", "people"]

    toggle = page.locator("#plan-panel .sidebar-panel-toggle")
    expect(toggle).to_have_text("Collapse")
    toggle.evaluate("(node) => node.click()")
    expect(page.locator("#plan-panel")).to_have_class(re.compile(r"\bcollapsed\b"))
    expect(toggle).to_have_text("Expand")


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
    expect(interrupted_event).to_contain_text("Let me think —")
    interrupted_event.evaluate("(node) => node.click()")
    expect(page.locator("#detail-payload")).to_contain_text("\"raw_full_text\": \"Let me think about that for a second.\"")
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
    expect(page.locator("#runtime-toggle")).to_have_text("Archive Loaded")
    expect(page.locator("#archive-transport")).to_have_class(re.compile(r"\bvisible\b"))
    expect(page.locator("#archive-transport-toggle")).to_have_text("Play")
    expect(page.locator("#archive-media-player")).to_have_class(re.compile(r"\bvisible\b"))
    expect(page.locator(".stream-card[data-event-type='session_start']")).to_have_count(1)
    expect(page.locator(".stream-card[data-event-type='user_transcript_final']")).to_have_count(1)
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

    page.locator("#archive-media-player").evaluate("(node) => { node.pause(); node.currentTime = 0; }")
    page.wait_for_timeout(100)
    reply_card = page.locator(".stream-card").filter(has_text="Hey, what time is it to you?").first
    reply_card.evaluate("(node) => node.click()")
    page.wait_for_timeout(200)
    selected_time = page.locator("#archive-media-player").evaluate("(node) => node.currentTime")
    assert selected_time >= 0.9


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

    expect(page.locator("#history-recording-mode")).to_contain_text("Older future events are shown as superseded")
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
    expect(page.locator("#save-complete-heading")).to_contain_text("Pausing Session")
    expect(page.locator("#save-complete-status")).to_contain_text("Pausing before review")
    assert input_queue.get(timeout=2) == "/pause"
    collector.push("pause", {})
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
    assert input_queue.get(timeout=2) == "/stop"

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
    expect(page.locator("#save-complete-heading")).to_contain_text("Pausing Session")
    expect(page.locator("#save-complete-status")).to_contain_text("Pausing before review")
    assert input_queue.get(timeout=2) == "/pause"
    collector.push("pause", {})
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
    expect(page.locator("#save-complete-heading")).to_contain_text("Save Session")
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
    expect(page.locator("#save-complete-heading")).not_to_contain_text("Save Session")


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
    assert input_queue.get(timeout=2) == "/pause"
    collector.push("pause", {})
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
