from __future__ import annotations

import json
import queue
import re
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
    _, port = start_dashboard(collector, input_queue, port=0, report_dir=report_dir)
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

    page.locator("#runtime-toggle").click()
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

    page.locator("#runtime-toggle").click()
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
    collector.push("vision_snapshot", {
        "faces": 0,
        "persons": 1,
        "objects": 1,
        "vlm_answers": [{"question": "Is there a bottle visible?", "answer": "Yes, a water bottle is on the table."}],
        "object_labels": ["water bottle"],
    })
    collector.push("perception", {
        "source": "visual",
        "description": "A person is now holding a flier",
        "source_trace": {
            "kind": "vision_synthesis",
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
    expect(page.locator("#report-ref-status")).to_contain_text("intermediate session log:")
    assert list(report_dir.iterdir())
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
    with page.expect_popup() as detail_popup_info:
        page.locator("#detail-open-window").click()
    detail_popup = detail_popup_info.value
    detail_popup.wait_for_load_state("domcontentloaded")
    expect(detail_popup.locator("body")).to_contain_text("Prompt / IO")
    expect(detail_popup.locator("body")).to_contain_text("Free water. Want one?")
    detail_popup.close()
    with page.expect_download() as json_download_info:
        page.locator("#detail-export-json").click()
    assert json_download_info.value.suggested_filename.endswith(".json")
    with page.expect_download() as html_download_info:
        page.locator("#detail-export-html").click()
    assert html_download_info.value.suggested_filename.endswith(".html")
    page.locator("#tab-payload").click()
    expect(page.locator("#detail-payload")).to_contain_text("\"total_ms\": 1980")
    page.locator("#tab-chronology").click()
    expect(page.locator("#detail-related")).to_contain_text("turn_start")
    expect(page.locator("#detail-related")).to_contain_text("user_transcript_final")
    expect(page.locator("#detail-related")).not_to_contain_text("post_response")
    expect(page.locator(".chronology-row.selected")).to_have_count(1)

    page.locator("#stream-density").select_option("compact")
    reply_card.click()
    expect(page.locator("#detail-type")).to_have_text("assistant_reply")

    perception_card = page.locator(".stream-card").filter(has_text="A person is now holding a flier").first
    perception_card.click()
    expect(page.locator("#detail-type")).to_have_text("vision_pass")
    expect(page.locator("#detail-trace")).to_have_class(re.compile(r"\bactive\b"))
    expect(page.locator("#detail-structure")).to_have_class(re.compile(r"\bactive\b"))
    expect(page.locator("#detail-structure")).to_contain_text("Inputs")
    expect(page.locator("#detail-structure")).to_contain_text("Raw Answers")
    expect(page.locator("#detail-structure")).to_contain_text("Derived Claims")
    expect(page.locator("#detail-structure")).to_contain_text("Effects")
    expect(page.locator("#detail-structure")).to_contain_text("Is anyone holding a flier?")
    expect(page.locator("#detail-structure")).to_contain_text("A flier.")
    expect(page.locator("#detail-structure")).to_contain_text("support:")
    expect(page.locator("#detail-trace-summary")).to_contain_text("source: visual")
    expect(page.locator("#detail-trace-chips")).to_contain_text("object flier")
    expect(page.locator("#detail-trace-chips")).to_contain_text("sam flier")
    expect(page.locator("#detail-trace-chips")).to_contain_text("supports What is the person holding?")

    with page.expect_popup() as vision_detail_popup_info:
        page.locator("#detail-open-window").click()
    vision_detail_popup = vision_detail_popup_info.value
    vision_detail_popup.wait_for_load_state("domcontentloaded")
    expect(vision_detail_popup.locator("body")).to_contain_text("Event Breakdown")
    expect(vision_detail_popup.locator("body")).to_contain_text("Derived Claims")
    vision_detail_popup.close()

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

    page.locator("#show-chronology").uncheck()
    expect(page.locator("#tab-chronology")).to_be_hidden()
    page.locator("#show-chronology").check()
    expect(page.locator("#tab-chronology")).to_be_visible()

    with page.expect_popup() as vision_popup_info:
        page.locator("#vision-service-link").click()
    vision_popup = vision_popup_info.value
    vision_popup.wait_for_load_state("domcontentloaded")
    assert "Vision Test UI" in vision_popup.text_content("body")
    vision_popup.close()

    page.locator("#runtime-stop").click()
    expect(page.locator("#runtime-stop")).to_have_text("Stopping...")
    assert input_queue.get(timeout=2) == "/stop"

    collector.push("session_end", {"total_turns": 2})
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

    page.locator("#report-ref-button").click()
    expect(page.locator("#report-ref-status")).to_contain_text("Last stopped session: intermediate session log:")

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
