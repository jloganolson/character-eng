from __future__ import annotations

import json
import queue
import re
import threading
import time
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

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
    _, port = start_dashboard(collector, input_queue, port=0)
    for _ in range(20):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/state")
            break
        except Exception:
            time.sleep(0.05)
    yield collector, input_queue, f"http://127.0.0.1:{port}/"
    collector.shutdown()


def test_runtime_panel_interactions_in_browser(
    browser_page: Page,
    dashboard_server,
    vision_service: VisionServiceController,
):
    collector, input_queue, dashboard_url = dashboard_server
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
            "voice_active": True,
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

    expect(page.locator("#h-character")).to_have_text("greg")
    expect(page.locator("body")).not_to_have_class(re.compile(r"\bbooting\b"))
    expect(page.locator("#boot-overlay")).not_to_be_visible()
    expect(page.locator("#runtime-status")).to_contain_text("state: paused")
    expect(page.locator("#boot-summary")).to_contain_text("Everything is warm. Resume when you want to start.")
    expect(page.locator("#boot-overlay-summary")).to_contain_text("Everything is warm. Resume when you want to start.")
    expect(page.locator("#boot-grid")).to_contain_text("voice")
    expect(page.locator("#boot-grid")).to_contain_text("models")
    expect(page.locator("#runtime-status")).to_contain_text("reconcile: on")
    expect(page.locator("#runtime-status")).to_contain_text("auto-beat: off")
    expect(page.locator("button[data-control='vision']")).to_have_text("vision on")
    expect(page.locator("button[data-control='filler']")).to_have_text("filler off")
    expect(page.locator("#vision-service-status")).to_contain_text("source: external service")
    expect(page.locator("#vision-service-status")).to_contain_text(vision_service_url)
    expect(page.locator("#vision-model-status")).to_contain_text("vLLM")
    expect(page.locator("#vision-model-status")).to_contain_text("gemma-vision-test")
    expect(page.locator("#vision-model-status")).to_contain_text("GPU memory")
    expect(page.locator("#vision-service-link")).to_have_attribute("href", vision_service_url)
    expect(page.locator(".stream-card")).to_have_count(2)

    collector.push("turn_start", {"input_type": "send", "input_text": "Do you have water?"})
    collector.push("response_chunk", {"text": "Free water. "})
    collector.push("response_chunk", {"text": "Want one?"})
    collector.push("response_done", {"full_text": "Free water. Want one?", "ttft_ms": 190, "total_ms": 780})
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

    expect(page.locator(".stream-card")).to_have_count(7)
    reply_card = page.locator(".stream-card").filter(has_text="Free water. Want one?").first
    reply_card.click()
    expect(page.locator("#detail-type")).to_have_text("assistant_reply")
    expect(page.locator("#detail-label")).to_contain_text("Free water. Want one?")
    expect(page.locator("#detail-detail")).to_contain_text("TTFT 190ms")
    expect(page.locator("#detail-payload")).to_contain_text("\"total_ms\": 780")
    expect(page.locator("#detail-related")).to_contain_text("turn_start")
    expect(page.locator(".chronology-row.selected")).to_have_count(1)

    page.locator("#stream-density").select_option("compact")
    reply_card.click()
    expect(page.locator(".stream-card.compact")).to_have_count(7)
    expect(page.locator("#detail-type")).to_have_text("assistant_reply")

    page.locator("button[data-lane-toggle='vision']").click()
    expect(page.locator("[data-stream-lane='vision']")).to_have_class(re.compile(r"\bhidden\b"))
    page.locator("button[data-lane-toggle='vision']").click()
    expect(page.locator("[data-stream-lane='vision']")).not_to_have_class(re.compile(r"\bhidden\b"))
    vision_card = page.locator(".stream-card").filter(has_text="water bottle").first
    vision_card.click()
    expect(page.locator("#detail-type")).to_have_text("vision_snapshot")
    expect(page.locator("#detail-payload")).to_contain_text("\"question\": \"Is there a bottle visible?\"")

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
            "voice_active": True,
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

    expect(page.locator("button[data-control='reconcile']")).to_have_text("reconcile off")
    expect(page.locator("#runtime-status")).to_contain_text("state: live")
    expect(page.locator("#boot-summary")).to_contain_text("Runtime is live. You can speak or type now.")
    expect(page.locator("button[data-control='auto-beat']")).to_have_text("auto-beat on")
    expect(page.locator("#vision-service-status")).to_contain_text("source: managed by app")
    expect(page.locator("#vision-service-status")).to_contain_text("mock: walkup.json")

    with page.expect_popup() as vision_popup_info:
        page.locator("#vision-service-link").click()
    vision_popup = vision_popup_info.value
    vision_popup.wait_for_load_state("domcontentloaded")
    assert "Vision Test UI" in vision_popup.text_content("body")
    vision_popup.close()

    with page.expect_popup() as system_map_popup_info:
        page.get_by_text("Open system map").click()
    system_map_popup = system_map_popup_info.value
    system_map_popup.wait_for_load_state("domcontentloaded")
    assert "Character Runtime & Design Map" in system_map_popup.text_content("body")
    system_map_popup.close()

    page.locator("button[data-control='reconcile']").click()
    assert input_queue.get(timeout=2) == "/threads reconcile toggle"

    page.locator("#runtime-pause").click()
    assert input_queue.get(timeout=2) == "/pause"

    page.locator("#runtime-resume").click()
    assert input_queue.get(timeout=2) == "/resume"

    page.locator("#runtime-restart").click()
    assert input_queue.get(timeout=2) == "/restart"

    page.locator("#vision-service-start").click()
    assert input_queue.get(timeout=2) == "/vision-service start"

    page.locator("#vision-service-stop").click()
    assert input_queue.get(timeout=2) == "/vision-service stop"

    page.locator("#vision-service-autostart").click()
    assert input_queue.get(timeout=2) == "/vision-service autostart toggle"

    page.locator("#chat-input").fill("status check")
    page.locator("#send-btn").click()
    assert input_queue.get(timeout=2) == "status check"

    page.keyboard.press("KeyB")
    assert input_queue.get(timeout=2) == "/beat"

    page.keyboard.press("Digit1")
    assert input_queue.get(timeout=2) == "1"

    expect(page.locator("#vision-model-status")).to_contain_text(re.compile("ready|loading"))

    vision_service.set_fail(True)
    expect(page.locator("#vision-model-status")).to_contain_text("Vision status unavailable", timeout=7000)
    expect(page.locator("#vision-service-status")).to_contain_text("health: unreachable")
    expect(page.locator("#vision-service-status")).to_contain_text("service: error")
    expect(page.locator("body")).to_have_class(re.compile(r"\bbooting\b"))
