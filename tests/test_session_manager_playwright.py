from __future__ import annotations

import json
import sys
import textwrap
import urllib.request
from pathlib import Path

import pytest
from playwright.sync_api import Error, Page, expect, sync_playwright

from character_eng.session_manager import RuntimeSessionManager, start_session_manager


def _stub_runtime_script(tmp_path: Path) -> Path:
    script = tmp_path / "stub_runtime_playwright.py"
    script.write_text(
        textwrap.dedent(
            """
            import json
            import os
            from http import HTTPStatus
            from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
            from urllib.parse import parse_qs, urlparse

            PORT = int(os.environ["CHARACTER_ENG_BRIDGE_PORT"])
            TOKEN = os.environ.get("CHARACTER_ENG_BRIDGE_TOKEN", "")

            class Handler(BaseHTTPRequestHandler):
                def do_GET(self):
                    parsed = urlparse(self.path)
                    token = (parse_qs(parsed.query).get("token") or [""])[0]
                    if TOKEN and token != TOKEN:
                        body = json.dumps({"ok": False, "error": "unauthorized"}).encode()
                        self.send_response(HTTPStatus.UNAUTHORIZED)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)
                        return
                    if parsed.path == "/bridge/status":
                        body = json.dumps({"enabled": True, "client": {}, "last_text_input_at": 0}).encode()
                        self.send_response(HTTPStatus.OK)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)
                        return
                    if parsed.path == "/" or parsed.path == "/index.html":
                        body = b"<html><body><h1>Stub Session</h1></body></html>"
                        self.send_response(HTTPStatus.OK)
                        self.send_header("Content-Type", "text/html; charset=utf-8")
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)
                        return
                    self.send_error(HTTPStatus.NOT_FOUND)

                def log_message(self, fmt, *args):
                    return

            server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
            """
        ),
        encoding="utf-8",
    )
    return script


@pytest.fixture()
def browser_page() -> Page:
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            yield page
            browser.close()
    except Error as exc:
        pytest.skip(f"Playwright browser not available: {exc}")


def test_session_manager_page_creates_and_stops_session(tmp_path, browser_page: Page):
    script = _stub_runtime_script(tmp_path)

    def command_factory(session):
        return [sys.executable, str(script)]

    manager = RuntimeSessionManager(
        public_host="127.0.0.1",
        log_root=tmp_path / "logs",
        startup_timeout_s=5.0,
        command_factory=command_factory,
    )
    _, port, server = start_session_manager(manager, port=0)
    base_url = f"http://127.0.0.1:{port}"
    try:
        browser_page.goto(base_url, wait_until="networkidle")
        expect(browser_page.locator("h1")).to_have_text("Character Session Manager")
        browser_page.get_by_role("button", name="Create Session").click()
        expect(browser_page.locator(".session-card")).to_have_count(1)
        expect(browser_page.locator(".session-card .badge")).to_have_text("ready")

        session_link_text = browser_page.locator(".session-card .session-actions button").first
        with browser_page.context.expect_page() as new_page_info:
            session_link_text.click()
        session_page = new_page_info.value
        session_page.wait_for_load_state("networkidle")
        expect(session_page.locator("h1")).to_have_text("Stub Session")
        session_page.close()

        browser_page.get_by_role("button", name="Stop").click()
        expect(browser_page.locator(".empty")).to_have_text("No live sessions yet.")
    finally:
        server.shutdown()
        server.server_close()
        manager.close()


def test_session_manager_page_uses_admin_token_query(tmp_path, browser_page: Page):
    script = _stub_runtime_script(tmp_path)
    admin_token = "manager-secret"

    def command_factory(session):
        return [sys.executable, str(script)]

    manager = RuntimeSessionManager(
        public_host="127.0.0.1",
        log_root=tmp_path / "logs",
        startup_timeout_s=5.0,
        command_factory=command_factory,
    )
    _, port, server = start_session_manager(manager, port=0, admin_token=admin_token)
    base_url = f"http://127.0.0.1:{port}"
    try:
        browser_page.goto(f"{base_url}/?token={admin_token}", wait_until="networkidle")
        expect(browser_page.locator("#admin-token-input")).to_have_value(admin_token)
        browser_page.get_by_role("button", name="Create Session").click()
        expect(browser_page.locator(".session-card")).to_have_count(1)
        expect(browser_page.locator(".session-card .badge")).to_have_text("ready")
    finally:
        server.shutdown()
        server.server_close()
        manager.close()
