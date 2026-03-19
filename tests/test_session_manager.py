from __future__ import annotations

import json
import os
import queue
import sys
import textwrap
import time
import urllib.error
import urllib.request
from pathlib import Path

from character_eng.session_manager import RuntimeSessionManager, start_session_manager


def _stub_runtime_script(tmp_path: Path) -> Path:
    script = tmp_path / "stub_runtime.py"
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
                        body = json.dumps({"enabled": True, "client": {}}).encode()
                        self.send_response(HTTPStatus.OK)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)
                        return
                    if parsed.path == "/" or parsed.path == "/index.html":
                        body = b"<html><body>stub runtime</body></html>"
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


def _json_get(url: str, token: str = "", headers: dict[str, str] | None = None) -> dict:
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read())


def _json_post(
    url: str,
    payload: dict,
    token: str = "",
    headers: dict[str, str] | None = None,
) -> dict:
    req = urllib.request.Request(
        url,
        method="POST",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def test_runtime_session_manager_spawns_and_stops_stub_runtime(tmp_path):
    script = _stub_runtime_script(tmp_path)

    def command_factory(session):
        return [sys.executable, str(script)]

    manager = RuntimeSessionManager(
        public_host="127.0.0.1",
        log_root=tmp_path / "logs",
        startup_timeout_s=5.0,
        command_factory=command_factory,
    )
    session = manager.create_session(character="greg", vision=True)
    try:
        assert session.status == "ready"
        assert session.vision_enabled is True
        assert f"token={session.token}" in session.url
        payload = manager.get_session(session.session_id)
        assert payload is not None
        assert payload["session_id"] == session.session_id
        assert payload["bridge_port"] == session.bridge_port
        assert payload["vision_port"] == session.vision_port
        with urllib.request.urlopen(f"http://127.0.0.1:{session.bridge_port}/bridge/status?token={session.token}", timeout=3) as resp:
            status = json.loads(resp.read())
        assert status["enabled"] is True
        assert manager.stop_session(session.session_id) is True
    finally:
        manager.cleanup_dead_sessions()


def test_session_manager_http_api(tmp_path):
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
    created_session_id = ""
    try:
        health = _json_get(f"{base_url}/health", token=admin_token)
        assert health["ok"] is True

        created = _json_post(f"{base_url}/sessions", {"character": "greg", "vision": False}, token=admin_token)
        created_session_id = created["session"]["session_id"]
        assert created["session"]["status"] == "ready"
        assert created["session"]["url"].startswith(f"{base_url}/?token=")

        listing = _json_get(f"{base_url}/sessions", token=admin_token)
        assert any(item["session_id"] == created_session_id for item in listing["sessions"])

        detail = _json_get(f"{base_url}/sessions/{created_session_id}", token=admin_token)
        assert detail["session"]["session_id"] == created_session_id

        with urllib.request.urlopen(created["session"]["url"], timeout=5) as resp:
            html = resp.read().decode()
        assert "stub runtime" in html

        stopped = _json_post(f"{base_url}/sessions/{created_session_id}/stop", {}, token=admin_token)
        assert stopped["stopped"] is True
    finally:
        server.shutdown()
        server.server_close()
        manager.close()
        if created_session_id:
            manager.stop_session(created_session_id)


def test_session_manager_http_api_accepts_query_token(tmp_path):
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
        with urllib.request.urlopen(f"{base_url}/health?token={admin_token}", timeout=5) as resp:
            payload = json.loads(resp.read())
        assert payload["ok"] is True

        req = urllib.request.Request(f"{base_url}/sessions?admin_token={admin_token}")
        req.add_header("Content-Type", "application/json")
        req.method = "POST"
        req.data = json.dumps({"character": "greg", "vision": False}).encode()
        with urllib.request.urlopen(req, timeout=10) as resp:
            created = json.loads(resp.read())
        assert created["session"]["status"] == "ready"
    finally:
        server.shutdown()
        server.server_close()
        manager.close()


def test_session_manager_http_api_uses_forwarded_host_for_session_urls(tmp_path):
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
        created = _json_post(
            f"{base_url}/sessions",
            {"character": "greg", "vision": False},
            headers={
                "Host": "manager.example.test",
                "X-Forwarded-Proto": "https",
            },
        )
        assert created["session"]["url"].startswith("https://manager.example.test/?token=")
    finally:
        server.shutdown()
        server.server_close()
        manager.close()


def test_runtime_session_manager_uses_shared_vision_service_env(tmp_path):
    manager = RuntimeSessionManager(
        public_host="127.0.0.1",
        log_root=tmp_path / "logs",
        startup_timeout_s=5.0,
        command_factory=lambda session: [sys.executable, "-c", "print('noop')"],
        base_env={"CHARACTER_ENG_SHARED_VISION_URL": "http://127.0.0.1:7860"},
    )
    probe = manager._runtime_env(  # noqa: SLF001
        type(
            "SessionProbe",
            (),
            {
                "bridge_port": 12345,
                "token": "abc",
                "vision_enabled": True,
                "vision_port": 54321,
            },
        )()
    )
    assert probe["CHARACTER_ENG_VISION_URL"] == "http://127.0.0.1:7860"
    assert probe["CHARACTER_ENG_VISION_PORT"] == "7860"
    assert probe["CHARACTER_ENG_VISION_AUTO_LAUNCH"] == "false"


def test_session_manager_config_endpoint_reports_vision_settings(tmp_path):
    script = _stub_runtime_script(tmp_path)

    def command_factory(session):
        return [sys.executable, str(script)]

    manager = RuntimeSessionManager(
        public_host="127.0.0.1",
        log_root=tmp_path / "logs",
        startup_timeout_s=5.0,
        command_factory=command_factory,
        default_vision_enabled=False,
        allow_vision=False,
    )
    _, port, server = start_session_manager(manager, port=0)
    base_url = f"http://127.0.0.1:{port}"
    try:
        payload = _json_get(f"{base_url}/config")
        assert payload["config"]["default_vision_enabled"] is False
        assert payload["config"]["allow_vision"] is False
    finally:
        server.shutdown()
        server.server_close()
        manager.close()


def test_session_manager_http_api_rejects_vision_when_disabled(tmp_path):
    script = _stub_runtime_script(tmp_path)

    def command_factory(session):
        return [sys.executable, str(script)]

    manager = RuntimeSessionManager(
        public_host="127.0.0.1",
        log_root=tmp_path / "logs",
        startup_timeout_s=5.0,
        command_factory=command_factory,
        default_vision_enabled=False,
        allow_vision=False,
    )
    _, port, server = start_session_manager(manager, port=0)
    base_url = f"http://127.0.0.1:{port}"
    try:
        req = urllib.request.Request(
            f"{base_url}/sessions",
            data=json.dumps({"character": "greg", "vision": True}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(req, timeout=10)
            raise AssertionError("expected disabled vision request to fail")
        except urllib.error.HTTPError as exc:
            assert exc.code == 400
            payload = json.loads(exc.read())
            assert payload["ok"] is False
            assert "disabled" in payload["error"]
    finally:
        server.shutdown()
        server.server_close()
        manager.close()


def test_session_manager_http_api_rejects_missing_token(tmp_path):
    script = _stub_runtime_script(tmp_path)

    def command_factory(session):
        return [sys.executable, str(script)]

    manager = RuntimeSessionManager(
        public_host="127.0.0.1",
        log_root=tmp_path / "logs",
        startup_timeout_s=5.0,
        command_factory=command_factory,
    )
    _, port, server = start_session_manager(manager, port=0, admin_token="manager-secret")
    base_url = f"http://127.0.0.1:{port}"
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=5):
            raise AssertionError("expected unauthorized health request to fail")
    except urllib.error.HTTPError as exc:
        assert exc.code == 401
    finally:
        server.shutdown()
        server.server_close()
        manager.close()


def test_session_manager_http_api_returns_json_on_startup_timeout(tmp_path):
    script = tmp_path / "hang_runtime.py"
    script.write_text(
        "import time\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )

    def command_factory(session):
        return [sys.executable, str(script)]

    manager = RuntimeSessionManager(
        public_host="127.0.0.1",
        log_root=tmp_path / "logs",
        startup_timeout_s=0.1,
        command_factory=command_factory,
    )
    _, port, server = start_session_manager(manager, port=0)
    base_url = f"http://127.0.0.1:{port}"
    try:
        req = urllib.request.Request(
            f"{base_url}/sessions",
            data=json.dumps({"character": "greg"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(req, timeout=10)
        except urllib.error.HTTPError as exc:
            assert exc.code == 504
            payload = json.loads(exc.read())
            assert payload["ok"] is False
            assert "did not become ready before timeout" in payload["error"]
        else:
            raise AssertionError("expected startup timeout to return an HTTP error")
    finally:
        server.shutdown()
        server.server_close()
        manager.close()


def test_session_manager_reaps_idle_sessions(tmp_path):
    script = _stub_runtime_script(tmp_path)

    def command_factory(session):
        return [sys.executable, str(script)]

    def status_fetcher(session):
        return {
            "enabled": True,
            "client": {"last_control_at": 0.0},
            "last_input_audio_at": 0.0,
            "last_video_frame_at": 0.0,
            "last_text_input_at": 0.0,
        }

    manager = RuntimeSessionManager(
        public_host="127.0.0.1",
        log_root=tmp_path / "logs",
        startup_timeout_s=5.0,
        command_factory=command_factory,
        status_fetcher=status_fetcher,
        idle_timeout_s=0.01,
        reap_interval_s=60.0,
    )
    session = manager.create_session(character="greg", vision=False)
    try:
        time.sleep(0.05)
        expired = manager.reap_idle_sessions()
        assert session.session_id in expired
        assert manager.get_session(session.session_id) is None
    finally:
        manager.close()
