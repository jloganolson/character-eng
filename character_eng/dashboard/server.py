"""Dashboard HTTP server — serves index.html, SSE events, state snapshots, and input injection."""

from __future__ import annotations

import json
import queue
import socket
import threading
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from pathlib import Path

from character_eng.dashboard.events import DashboardEventCollector

HTML_PATH = Path(__file__).parent / "index.html"
SYSTEM_MAP_PATH = Path(__file__).parent / "system_map.html"
STREAM_SCHEMA_PATH = Path(__file__).parent / "stream_schema.json"
REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "logs" / "intermediate"


class DashboardHandler(BaseHTTPRequestHandler):
    collector: DashboardEventCollector
    input_queue: queue.Queue
    report_dir: Path

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self._serve_html()
        elif self.path == "/system-map.html":
            self._serve_system_map()
        elif self.path == "/stream-schema.json":
            self._serve_stream_schema()
        elif self.path == "/events":
            self._serve_sse()
        elif self.path == "/state":
            self._serve_state()
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self):
        if self.path == "/send":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                data = json.loads(body)
                text = data.get("text", "").strip()
                if text:
                    self.input_queue.put(text)
                resp = json.dumps({"ok": True}).encode()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)
            except Exception as e:
                resp = json.dumps({"ok": False, "error": str(e)}).encode()
                self.send_response(HTTPStatus.BAD_REQUEST)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)
        elif self.path == "/report-snapshot":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                data = json.loads(body)
                session_id = str(data.get("session_id") or "unknown")
                safe_session = "".join(ch for ch in session_id if ch.isalnum() or ch in {"-", "_"}) or "unknown"
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_dir = getattr(self, "report_dir", REPORTS_DIR)
                report_dir.mkdir(parents=True, exist_ok=True)
                path = report_dir / f"{safe_session}_{timestamp}.json"
                path.write_text(json.dumps(data, indent=2), encoding="utf-8")
                paste_text = f"intermediate session log: {path}"
                resp = json.dumps({"ok": True, "path": str(path), "paste_text": paste_text}).encode()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)
            except Exception as e:
                resp = json.dumps({"ok": False, "error": str(e)}).encode()
                self.send_response(HTTPStatus.BAD_REQUEST)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def _serve_html(self):
        try:
            content = HTML_PATH.read_bytes()
        except FileNotFoundError:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "index.html not found")
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _serve_system_map(self):
        try:
            content = SYSTEM_MAP_PATH.read_bytes()
        except FileNotFoundError:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "system_map.html not found")
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _serve_stream_schema(self):
        try:
            content = STREAM_SCHEMA_PATH.read_bytes()
        except FileNotFoundError:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "stream_schema.json not found")
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(content)

    def _serve_sse(self):
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        q = self.collector.subscribe()
        try:
            while True:
                try:
                    event = q.get(timeout=30)
                except queue.Empty:
                    # Send keepalive comment
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    continue
                if event is None:
                    break
                line = f"data: {event.to_json()}\n\n"
                self.wfile.write(line.encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            self.collector.unsubscribe(q)

    def _serve_state(self):
        events = self.collector.get_all()
        body = json.dumps(events).encode()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("", port))
            return True
        except OSError:
            return False


def start_dashboard(collector: DashboardEventCollector, input_queue: queue.Queue,
                    port: int = 7862, report_dir: Path | None = None) -> tuple[threading.Thread, int]:
    """Start the dashboard HTTP server as a daemon thread. Returns (thread, actual_port)."""
    if port <= 0 or not _port_available(port):
        port = _find_free_port()

    DashboardHandler.collector = collector
    DashboardHandler.input_queue = input_queue
    DashboardHandler.report_dir = report_dir or REPORTS_DIR

    server = ThreadingHTTPServer(("0.0.0.0", port), DashboardHandler)
    server.daemon_threads = True
    server.timeout = 1
    actual_port = server.server_address[1]

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread, actual_port
