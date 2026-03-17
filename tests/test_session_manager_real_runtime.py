from __future__ import annotations

import json
import os
import time
import urllib.request

import pytest
from playwright.sync_api import Error, sync_playwright

from character_eng.session_manager import RuntimeSessionManager, start_session_manager


REQUIRED_ENV = ("GEMINI_API_KEY", "DEEPGRAM_API_KEY", "ELEVENLABS_API_KEY")


@pytest.mark.integration
def test_session_manager_can_spawn_real_runtime_and_start_session():
    missing = [key for key in REQUIRED_ENV if not os.environ.get(key)]
    if missing:
        pytest.skip(f"missing env for real runtime smoke: {', '.join(missing)}")

    base_env = {
        "CHARACTER_ENG_CHAT_MODEL": "gemini-3-flash-lite",
        "CHARACTER_ENG_MICRO_MODEL": "gemini-3-flash-lite",
        "CHARACTER_ENG_BIG_MODEL": "gemini-3-flash-lite",
    }
    manager = RuntimeSessionManager(public_host="127.0.0.1", startup_timeout_s=40.0, base_env=base_env)
    server = None
    session = None
    try:
        _, _, server = start_session_manager(manager, port=0)
        session = manager.create_session(character="greg", vision=False)
        base_url = session.url.removesuffix(f"?token={session.token}")
        token = session.token

        with urllib.request.urlopen(session.url, timeout=5) as resp:
            html = resp.read().decode("utf-8")
        assert "Character Dashboard" in html
        assert token in html

        req = urllib.request.Request(
            f"{base_url}/send?token={token}",
            data=json.dumps({"text": "/start"}).encode(),
            method="POST",
            headers={"Content-Type": "application/json", "X-Session-Token": token},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.status == 200

        deadline = time.time() + 20
        while time.time() < deadline:
            with urllib.request.urlopen(f"{base_url}/state?token={token}", timeout=5) as resp:
                events = json.loads(resp.read())
            if any(event.get("type") == "session_start" for event in events):
                break
            time.sleep(0.5)
        else:
            raise AssertionError("session_start event not observed from real runtime")
    finally:
        if session is not None:
            try:
                manager.stop_session(session.session_id)
            except Exception:
                pass
        if server is not None:
            server.shutdown()
            server.server_close()
        manager.close()


@pytest.mark.integration
def test_session_manager_real_browser_media_flow():
    missing = [key for key in REQUIRED_ENV if not os.environ.get(key)]
    if missing:
        pytest.skip(f"missing env for real runtime smoke: {', '.join(missing)}")

    base_env = {
        "CHARACTER_ENG_CHAT_MODEL": "gemini-3-flash-lite",
        "CHARACTER_ENG_MICRO_MODEL": "gemini-3-flash-lite",
        "CHARACTER_ENG_BIG_MODEL": "gemini-3-flash-lite",
    }
    manager = RuntimeSessionManager(public_host="127.0.0.1", startup_timeout_s=60.0, base_env=base_env)
    server = None
    session = None
    try:
        _, _, server = start_session_manager(manager, port=0)
        session = manager.create_session(character="greg", vision=True)
        base_url = session.url.removesuffix(f"?token={session.token}")
        token = session.token

        with sync_playwright() as playwright:
            try:
                browser = playwright.chromium.launch(
                    headless=True,
                    args=[
                        "--use-fake-ui-for-media-stream",
                        "--use-fake-device-for-media-stream",
                    ],
                )
            except Error as exc:
                pytest.skip(f"Playwright browser not available: {exc}")
            context = browser.new_context(
                viewport={"width": 1600, "height": 1100},
                permissions=["camera", "microphone"],
            )
            page = context.new_page()
            page.goto(session.url, wait_until="domcontentloaded", timeout=120000)
            page.wait_for_timeout(5000)

            page.locator("#runtime-toggle").evaluate("(node) => node.click()")
            page.wait_for_timeout(5000)
            if page.locator("#runtime-toggle").inner_text().strip() == "Play":
                page.locator("#runtime-toggle").evaluate("(node) => node.click()")
                page.wait_for_timeout(12000)
            else:
                page.wait_for_timeout(7000)

            deadline = time.time() + 20
            while time.time() < deadline:
                with urllib.request.urlopen(f"{base_url}/state?token={token}", timeout=5) as resp:
                    events = json.loads(resp.read())
                if any(event.get("type") == "session_start" for event in events):
                    break
                time.sleep(0.5)
            else:
                raise AssertionError("session_start event not observed from browser start flow")

            page.locator("#bridge-mic-toggle").evaluate("(node) => node.click()")
            page.wait_for_timeout(2500)
            page.locator("#bridge-cam-toggle").evaluate("(node) => node.click()")
            page.wait_for_timeout(7000)

            page.wait_for_function(
                "() => window.__bridgeMetrics && window.__bridgeMetrics.audioChunksPlayed > 0",
                timeout=30000,
            )
            page.wait_for_function(
                "() => window.__bridgeMetrics && window.__bridgeMetrics.micUploads > 0",
                timeout=15000,
            )
            page.wait_for_function(
                "() => window.__bridgeMetrics && window.__bridgeMetrics.cameraUploads > 0",
                timeout=15000,
            )
            page.wait_for_function(
                "() => window.__bridgeMetrics && window.__bridgeMetrics.visionNaturalWidth > 0 && window.__bridgeMetrics.visionNaturalHeight > 0",
                timeout=30000,
            )

            metrics = page.evaluate("() => window.__bridgeMetrics")
            with urllib.request.urlopen(f"{base_url}/bridge/status?token={token}", timeout=5) as resp:
                bridge_status = json.loads(resp.read())

            assert metrics["audioChunksPlayed"] > 0
            assert metrics["micUploads"] > 0
            assert metrics["cameraUploads"] > 0
            assert metrics["visionNaturalWidth"] > 0
            assert metrics["visionNaturalHeight"] > 0
            assert "bridge/video_feed" in metrics["visionFeedSrc"]
            assert bridge_status["last_input_audio_at"] > 0
            assert bridge_status["last_video_frame_at"] > 0
            assert bridge_status["client"]["audio_armed"] is True
            assert bridge_status["client"]["mic"] is True
            assert bridge_status["client"]["camera"] is True

            browser.close()
    finally:
        if session is not None:
            try:
                manager.stop_session(session.session_id)
            except Exception:
                pass
        if server is not None:
            server.shutdown()
            server.server_close()
        manager.close()
