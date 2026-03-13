"""Dashboard HTTP server — serves index.html, SSE events, state snapshots, and input injection."""

from __future__ import annotations

import json
import queue
import shutil
import socket
import subprocess
import threading
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from character_eng.dashboard.events import DashboardEventCollector
from character_eng.prompts import mutable_prompt_inventory

HTML_PATH = Path(__file__).parent / "index.html"
SYSTEM_MAP_PATH = Path(__file__).parent / "system_map.html"
STREAM_SCHEMA_PATH = Path(__file__).parent / "stream_schema.json"
REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "logs" / "intermediate"


class DashboardHandler(BaseHTTPRequestHandler):
    collector: DashboardEventCollector
    input_queue: queue.Queue
    report_dir: Path
    history_api = None
    history_status_provider = None
    default_character: str = ""
    prompt_asset_open_target: str = "vscode"
    prompt_asset_vscode_cmd: str = "code"

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
        elif self.path == "/history/state":
            self._serve_history_state()
        elif self.path == "/history/list":
            self._serve_history_list()
        elif self.path.startswith("/history/checkpoint"):
            self._serve_history_checkpoint()
        elif self.path.startswith("/history/events"):
            self._serve_history_events()
        elif self.path.startswith("/history/frame"):
            self._serve_history_frame()
        elif self.path.startswith("/history/audio"):
            self._serve_history_audio()
        elif self.path.startswith("/history/media"):
            self._serve_history_media()
        elif self.path.startswith("/history/annotations"):
            self._serve_history_annotations()
        elif self.path.startswith("/prompt-assets"):
            self._serve_prompt_assets()
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
        elif self.path == "/history/annotation":
            self._handle_history_annotation()
        elif self.path == "/history/moment" or self.path == "/history/snippet":
            self._handle_history_snippet()
        elif self.path == "/history/promote":
            self._handle_history_promote()
        elif self.path == "/history/rename":
            self._handle_history_rename()
        elif self.path == "/history/prune":
            self._handle_history_prune()
        elif self.path == "/prompt-assets/action":
            self._handle_prompt_asset_action()
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

    def _serve_history_state(self):
        provider = getattr(type(self), "history_status_provider", None)
        history_api = getattr(self, "history_api", None)
        if callable(provider):
            payload = provider()
        elif history_api is not None:
            payload = history_api.status()
        else:
            payload = {
                "enabled": False,
                "warning": "",
                "free_gib": 0.0,
            }
        body = json.dumps(payload).encode()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _serve_history_list(self):
        history_api = getattr(self, "history_api", None)
        payload = history_api.list_archives() if history_api is not None else []
        body = json.dumps(payload).encode()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _serve_history_annotations(self):
        history_api = getattr(self, "history_api", None)
        payload = []
        if history_api is not None:
            query = parse_qs(urlparse(self.path).query)
            session_id = (query.get("session_id") or [None])[0]
            try:
                payload = history_api.list_annotations(session_id=session_id)
            except FileNotFoundError:
                payload = []
        body = json.dumps(payload).encode()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _serve_prompt_assets(self):
        query = parse_qs(urlparse(self.path).query)
        character = (query.get("character") or [None])[0] or self.default_character or None
        payload = mutable_prompt_inventory(character=character)
        body = json.dumps(payload).encode()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _serve_history_checkpoint(self):
        history_api = getattr(self, "history_api", None)
        if history_api is None:
            self._respond_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "history disabled"})
            return
        query = parse_qs(urlparse(self.path).query)
        session_id = (query.get("session_id") or [None])[0]
        event_time_raw = (query.get("event_time_s") or [None])[0]
        try:
            event_time_s = None if event_time_raw in (None, "") else float(event_time_raw)
            payload = history_api.resolve_checkpoint_for_event(session_id, event_time_s=event_time_s)
            self._respond_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "checkpoint_index": int(payload.get("checkpoint_index", 0)),
                    "label": str(payload.get("label", "checkpoint")),
                    "session_id": str(payload.get("session_id", session_id or "")),
                },
            )
        except Exception as e:
            self._respond_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(e)})

    def _serve_history_events(self):
        history_api = getattr(self, "history_api", None)
        if history_api is None:
            self._respond_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "history disabled"})
            return
        query = parse_qs(urlparse(self.path).query)
        session_id = (query.get("session_id") or [None])[0]
        try:
            payload = history_api.list_events(session_id)
            self._respond_json(HTTPStatus.OK, {"ok": True, **payload})
        except Exception as e:
            self._respond_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(e)})

    def _serve_history_frame(self):
        history_api = getattr(self, "history_api", None)
        if history_api is None:
            self._respond_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "history disabled"})
            return
        query = parse_qs(urlparse(self.path).query)
        session_id = (query.get("session_id") or [None])[0]
        event_time_raw = (query.get("event_time_s") or [None])[0]
        frame_timestamp_raw = (query.get("frame_timestamp") or [None])[0]
        raw = (query.get("raw") or ["0"])[0] == "1"
        try:
            event_time_s = None if event_time_raw in (None, "") else float(event_time_raw)
            frame_timestamp = None if frame_timestamp_raw in (None, "") else float(frame_timestamp_raw)
            if frame_timestamp is not None:
                payload = history_api.nearest_frame_for_timestamp(session_id, frame_timestamp=frame_timestamp)
            else:
                payload = history_api.nearest_frame_for_event(session_id, event_time_s=event_time_s)
            if raw:
                frame_path = Path(str(payload.get("abs_path", "")))
                if not frame_path.exists():
                    self.send_error(HTTPStatus.NOT_FOUND, "history frame not found")
                    return
                content = frame_path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(content)
                return
            self._respond_json(HTTPStatus.OK, {"ok": True, "frame": payload})
        except Exception as e:
            self._respond_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(e)})

    def _serve_history_audio(self):
        history_api = getattr(self, "history_api", None)
        if history_api is None:
            self._respond_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "history disabled"})
            return
        query = parse_qs(urlparse(self.path).query)
        session_id = (query.get("session_id") or [None])[0]
        try:
            content = history_api.audio_bytes_for_session(session_id)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self._respond_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(e)})

    def _serve_history_media(self):
        history_api = getattr(self, "history_api", None)
        if history_api is None:
            self._respond_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "history disabled"})
            return
        query = parse_qs(urlparse(self.path).query)
        session_id = (query.get("session_id") or [None])[0]
        try:
            media_path = history_api.playback_media_path_for_session(session_id)
            if media_path is None or not media_path.exists():
                self.send_error(HTTPStatus.NOT_FOUND, "history media not found")
                return
            size = media_path.stat().st_size
            range_header = self.headers.get("Range", "").strip()
            start = 0
            end = size - 1
            status = HTTPStatus.OK
            if range_header.startswith("bytes="):
                spec = range_header.split("=", 1)[1]
                start_text, _, end_text = spec.partition("-")
                if start_text:
                    start = max(0, int(start_text))
                if end_text:
                    end = min(size - 1, int(end_text))
                if start > end or start >= size:
                    self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    return
                status = HTTPStatus.PARTIAL_CONTENT
            length = end - start + 1
            self.send_response(status)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-store")
            if status == HTTPStatus.PARTIAL_CONTENT:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with media_path.open("rb") as fh:
                fh.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = fh.read(min(64 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except Exception as e:
            self._respond_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(e)})

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        return json.loads(body or b"{}")

    def _respond_json(self, status: HTTPStatus, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_history_annotation(self):
        history_api = getattr(self, "history_api", None)
        if history_api is None:
            self._respond_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "history disabled"})
            return
        try:
            data = self._read_json_body()
            path = history_api.save_annotation(data, session_id=data.get("session_id"))
            self._respond_json(HTTPStatus.OK, {"ok": True, "path": str(path), "status": history_api.status()})
        except Exception as e:
            self._respond_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(e)})

    def _handle_history_snippet(self):
        history_api = getattr(self, "history_api", None)
        if history_api is None:
            self._respond_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "history disabled"})
            return
        try:
            data = self._read_json_body()
            path = history_api.capture_snippet(data)
            self._respond_json(HTTPStatus.OK, {"ok": True, "path": str(path), "status": history_api.status()})
        except Exception as e:
            self._respond_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(e)})

    def _handle_history_promote(self):
        history_api = getattr(self, "history_api", None)
        if history_api is None:
            self._respond_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "history disabled"})
            return
        try:
            data = self._read_json_body()
            path = history_api.promote(session_id=data.get("session_id"), kind=data.get("kind", "pinned"))
            self._respond_json(HTTPStatus.OK, {"ok": True, "path": str(path), "status": history_api.status()})
        except Exception as e:
            self._respond_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(e)})

    def _handle_history_rename(self):
        history_api = getattr(self, "history_api", None)
        if history_api is None:
            self._respond_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "history disabled"})
            return
        try:
            data = self._read_json_body()
            manifest = history_api.rename(session_id=data.get("session_id"), title=str(data.get("title") or ""))
            self._respond_json(HTTPStatus.OK, {"ok": True, "manifest": manifest, "status": history_api.status()})
        except Exception as e:
            self._respond_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(e)})

    def _handle_history_prune(self):
        history_api = getattr(self, "history_api", None)
        if history_api is None:
            self._respond_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "history disabled"})
            return
        try:
            data = self._read_json_body()
            result = history_api.prune_unpromoted(
                older_than_days=data.get("older_than_days"),
                until_free_gib=data.get("until_free_gib"),
            )
            self._respond_json(HTTPStatus.OK, {"ok": True, **result, "status": history_api.status()})
        except Exception as e:
            self._respond_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(e)})

    def _handle_prompt_asset_action(self):
        try:
            data = self._read_json_body()
            path = Path(str(data.get("path") or "")).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(path)
            action = str(data.get("action") or "open")
            if action == "open":
                target = str(getattr(self, "prompt_asset_open_target", "vscode") or "vscode")
                if target == "folder":
                    self._open_folder(path.parent)
                else:
                    self._open_in_vscode(path)
            elif action == "folder":
                self._open_folder(path.parent)
            else:
                raise ValueError(f"unknown prompt asset action: {action}")
            self._respond_json(HTTPStatus.OK, {"ok": True})
        except Exception as e:
            self._respond_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(e)})

    def _open_in_vscode(self, path: Path) -> None:
        cmd = str(getattr(self, "prompt_asset_vscode_cmd", "code") or "code")
        binary = shutil.which(cmd) or cmd
        subprocess.Popen([binary, str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    @staticmethod
    def _open_folder(path: Path) -> None:
        opener = shutil.which("xdg-open")
        if not opener:
            raise RuntimeError("xdg-open not found")
        subprocess.Popen([opener, str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

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
                    port: int = 7862, report_dir: Path | None = None, history_api=None,
                    history_status_provider=None,
                    default_character: str = "",
                    prompt_asset_open_target: str = "vscode",
                    prompt_asset_vscode_cmd: str = "code") -> tuple[threading.Thread, int]:
    """Start the dashboard HTTP server as a daemon thread. Returns (thread, actual_port)."""
    if port <= 0 or not _port_available(port):
        port = _find_free_port()

    DashboardHandler.collector = collector
    DashboardHandler.input_queue = input_queue
    DashboardHandler.report_dir = report_dir or REPORTS_DIR
    DashboardHandler.history_api = history_api
    DashboardHandler.history_status_provider = history_status_provider
    DashboardHandler.default_character = default_character
    DashboardHandler.prompt_asset_open_target = prompt_asset_open_target
    DashboardHandler.prompt_asset_vscode_cmd = prompt_asset_vscode_cmd

    server = ThreadingHTTPServer(("0.0.0.0", port), DashboardHandler)
    server.daemon_threads = True
    server.timeout = 1
    actual_port = server.server_address[1]

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread, actual_port
