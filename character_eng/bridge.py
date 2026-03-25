from __future__ import annotations

import base64
import json
import queue
import threading
import time
import urllib.request
from collections import deque
from dataclasses import dataclass
from http import HTTPStatus
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Callable
import urllib.error
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from character_eng.config import load_config
from character_eng.dashboard.events import DashboardEventCollector
from character_eng.dashboard.server import DashboardHandler, REPORTS_DIR, _find_free_port, _port_available
from character_eng.vision.client import VisionClient


@dataclass
class _BridgeAudioChunk:
    pcm: bytes
    sample_rate: int
    created_at: float


class _BridgeHandler(DashboardHandler):
    bridge_server: "BridgeServer | None" = None

    def _bridge(self) -> "BridgeServer":
        bridge = type(self).bridge_server
        if bridge is None:
            raise RuntimeError("bridge server not configured")
        return bridge

    def _serve_html(self):
        bridge = self._bridge()
        try:
            content = bridge.render_dashboard_html(token=self._request_token())
        except FileNotFoundError:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "index.html not found")
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        if not self._authorize():
            return
        parsed = urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/index.html":
            self._serve_html()
            return
        if parsed.path == "/bridge/status":
            payload = self._bridge().status_payload()
            body = json.dumps(payload).encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/bridge/audio":
            payload = {"chunks": self._bridge().drain_audio_chunks()}
            body = json.dumps(payload).encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/bridge/video_feed":
            return self._bridge().stream_vision_feed(self)
        if parsed.path == "/bridge/vision-status":
            payload = self._bridge().vision_status_payload()
            body = json.dumps(payload).encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
            return
        original_path = self.path
        try:
            self.path = parsed.path
            super().do_GET()
        finally:
            self.path = original_path

    def do_POST(self):
        if not self._authorize():
            return
        parsed = urlparse(self.path)
        if parsed.path == "/bridge/audio":
            length = int(self.headers.get("Content-Length", 0))
            payload = self.rfile.read(length)
            self._bridge().handle_input_audio(payload)
            self._send_ok({"ok": True, "bytes": len(payload)})
            return
        if parsed.path == "/bridge/frame":
            length = int(self.headers.get("Content-Length", 0))
            payload = self.rfile.read(length)
            self._bridge().handle_video_frame(payload)
            self._send_ok({"ok": True, "bytes": len(payload)})
            return
        if parsed.path == "/bridge/control":
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            self._bridge().set_client_state(payload)
            self._send_ok({"ok": True, "state": self._bridge().status_payload()["client"]})
            return
        if parsed.path == "/send":
            self._bridge().touch_text_input()
        original_path = self.path
        try:
            self.path = parsed.path
            super().do_POST()
        finally:
            self.path = original_path

    def _send_ok(self, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _request_token(self) -> str:
        auth = self.headers.get("X-Session-Token", "").strip()
        if auth:
            return auth
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        return str((query.get("token") or [""])[0]).strip()

    def _authorize(self) -> bool:
        bridge = self._bridge()
        expected = bridge.access_token
        if not expected:
            return True
        if self._request_token() == expected:
            return True
        payload = json.dumps({"ok": False, "error": "unauthorized"}).encode()
        self.send_response(HTTPStatus.UNAUTHORIZED)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)
        return False


class BridgeServer:
    """Serve the dashboard plus browser-media bridge endpoints on one HTTP port."""

    def __init__(self, port: int = 7862, access_token: str = ""):
        self._requested_port = int(port)
        self._actual_port = 0
        self._access_token = str(access_token or "").strip()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._started = False
        self._lock = threading.Lock()
        self._audio_chunks: deque[_BridgeAudioChunk] = deque()
        self._max_audio_chunks = 256
        self._collector = DashboardEventCollector()
        self._input_queue: queue.Queue = queue.Queue()
        self._report_dir = REPORTS_DIR
        self._history_api = None
        self._history_status_provider = None
        self._default_character = ""
        self._prompt_asset_open_target = "vscode"
        self._prompt_asset_vscode_cmd = "code"
        self._on_audio: Callable[[bytes], None] | None = None
        self._on_video_frame: Callable[[bytes], None] | None = None
        self._last_input_audio_at = 0.0
        self._last_video_frame_at = 0.0
        self._last_text_input_at = 0.0
        self._client_state = {
            "audio_armed": False,
            "mic": False,
            "camera": False,
            "last_control_at": 0.0,
        }

    @property
    def port(self) -> int:
        return self._actual_port or self._requested_port

    @property
    def access_token(self) -> str:
        return self._access_token

    def set_dashboard(
        self,
        collector: DashboardEventCollector,
        input_queue: queue.Queue,
        *,
        report_dir: Path | None = None,
        history_api=None,
        history_status_provider=None,
        livekit_status_provider=None,
        livekit_token_provider=None,
        vision_state_provider=None,
        vision_action_handler=None,
        default_character: str = "",
        prompt_asset_open_target: str = "vscode",
        prompt_asset_vscode_cmd: str = "code",
    ) -> None:
        self._collector = collector
        self._input_queue = input_queue
        self._report_dir = report_dir or REPORTS_DIR
        self._history_api = history_api
        self._history_status_provider = history_status_provider
        self._default_character = default_character
        self._prompt_asset_open_target = prompt_asset_open_target
        self._prompt_asset_vscode_cmd = prompt_asset_vscode_cmd
        _BridgeHandler.collector = collector
        _BridgeHandler.input_queue = input_queue
        _BridgeHandler.report_dir = self._report_dir
        _BridgeHandler.history_api = history_api
        _BridgeHandler.history_status_provider = history_status_provider
        _BridgeHandler.livekit_status_provider = livekit_status_provider
        _BridgeHandler.livekit_token_provider = livekit_token_provider
        _BridgeHandler.vision_state_provider = vision_state_provider
        _BridgeHandler.vision_action_handler = vision_action_handler
        _BridgeHandler.default_character = default_character
        _BridgeHandler.prompt_asset_open_target = prompt_asset_open_target
        _BridgeHandler.prompt_asset_vscode_cmd = prompt_asset_vscode_cmd
        _BridgeHandler.bridge_server = self

    def start(
        self,
        *,
        on_audio: Callable[[bytes], None] | None = None,
        on_video_frame: Callable[[bytes], None] | None = None,
    ) -> None:
        self._on_audio = on_audio
        self._on_video_frame = on_video_frame
        if self._started:
            return
        port = self._requested_port
        if port <= 0 or not _port_available(port):
            port = _find_free_port()
        self._server = ThreadingHTTPServer(("0.0.0.0", port), _BridgeHandler)
        self._server.daemon_threads = True
        self._server.timeout = 1
        self._actual_port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self._started = True

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        self._started = False

    def render_dashboard_html(self, token: str = "") -> bytes:
        html_path = Path(__file__).parent / "dashboard" / "index.html"
        content = html_path.read_text(encoding="utf-8")
        safe_token = json.dumps(str(token or ""))
        injected = """
<script>
window.__bridgeConfig = {
  enabled: true,
  statusPath: "/bridge/status",
  audioPath: "/bridge/audio",
  framePath: "/bridge/frame",
  controlPath: "/bridge/control",
  visionFeedPath: "/bridge/video_feed",
  visionStatusPath: "/bridge/vision-status",
  accessToken: __ACCESS_TOKEN__
};
</script>
"""
        injected = injected.replace("__ACCESS_TOKEN__", safe_token)
        if "</head>" in content:
            content = content.replace("</head>", f"{injected}</head>", 1)
        else:
            content = injected + content
        return content.encode("utf-8")

    def push_audio(self, pcm: bytes, *, sample_rate: int = 24000) -> None:
        if not pcm:
            return
        with self._lock:
            self._audio_chunks.append(
                _BridgeAudioChunk(pcm=bytes(pcm), sample_rate=int(sample_rate), created_at=time.time())
            )
            while len(self._audio_chunks) > self._max_audio_chunks:
                self._audio_chunks.popleft()

    def drain_audio_chunks(self) -> list[dict]:
        with self._lock:
            chunks = list(self._audio_chunks)
            self._audio_chunks.clear()
        return [
            {
                "sample_rate": chunk.sample_rate,
                "created_at": chunk.created_at,
                "data_b64": base64.b64encode(chunk.pcm).decode("ascii"),
            }
            for chunk in chunks
        ]

    def handle_input_audio(self, pcm: bytes) -> None:
        if not pcm:
            return
        self._last_input_audio_at = time.time()
        if self._on_audio is not None:
            self._on_audio(pcm)

    def handle_video_frame(self, jpeg_bytes: bytes) -> None:
        if not jpeg_bytes:
            return
        self._last_video_frame_at = time.time()
        if self._on_video_frame is not None:
            self._on_video_frame(jpeg_bytes)

    def touch_text_input(self) -> None:
        self._last_text_input_at = time.time()

    def set_client_state(self, payload: dict) -> None:
        with self._lock:
            for key in ("audio_armed", "mic", "camera"):
                if key in payload:
                    self._client_state[key] = bool(payload.get(key))
            self._client_state["last_control_at"] = time.time()

    def status_payload(self) -> dict:
        with self._lock:
            client = dict(self._client_state)
            pending_audio = len(self._audio_chunks)
        return {
            "enabled": True,
            "client": client,
            "pending_audio_chunks": pending_audio,
            "last_input_audio_at": self._last_input_audio_at,
            "last_video_frame_at": self._last_video_frame_at,
            "last_text_input_at": self._last_text_input_at,
        }

    def vision_status_payload(self) -> dict:
        base_url = self._vision_base_url()
        if not base_url:
            return {"enabled": False, "status": None, "memory": None}
        client = VisionClient(base_url)
        try:
            status = client.model_status()
            try:
                memory = client.memory_status()
            except (urllib.error.URLError, OSError, json.JSONDecodeError):
                memory = None
            return {"enabled": True, "status": status, "memory": memory}
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            return {"enabled": True, "error": str(exc)}

    def _vision_base_url(self) -> str:
        cfg = load_config()
        if not cfg.vision.enabled:
            return ""
        return str(cfg.vision.service_url or "").rstrip("/")

    def stream_vision_feed(self, handler: _BridgeHandler) -> None:
        base_url = self._vision_base_url()
        if not base_url:
            handler.send_error(HTTPStatus.NOT_FOUND, "vision service not configured")
            return
        req = urllib.request.Request(f"{base_url}/video_feed")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                handler.send_response(resp.status)
                handler.send_header("Content-Type", resp.headers.get("Content-Type", "multipart/x-mixed-replace"))
                handler.send_header("Cache-Control", "no-store")
                handler.send_header("Access-Control-Allow-Origin", "*")
                handler.end_headers()
                while True:
                    chunk = resp.read(64 * 1024)
                    if not chunk:
                        break
                    handler.wfile.write(chunk)
                    handler.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return
        except (urllib.error.URLError, OSError) as exc:
            payload = json.dumps({"ok": False, "error": str(exc)}).encode()
            try:
                handler.send_response(HTTPStatus.BAD_GATEWAY)
                handler.send_header("Content-Type", "application/json")
                handler.send_header("Content-Length", str(len(payload)))
                handler.end_headers()
                handler.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                return
