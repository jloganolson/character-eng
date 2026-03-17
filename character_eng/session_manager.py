from __future__ import annotations

import http.client
import json
import os
import secrets
import signal
import subprocess
import sys
import threading
import time
import uuid
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, urlparse

from character_eng.dashboard.server import _find_free_port


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SESSION_LOG_ROOT = PROJECT_ROOT / "logs" / "session_manager"
SESSION_MANAGER_HTML = Path(__file__).with_name("session_manager.html")


@dataclass
class ManagedSession:
    session_id: str
    token: str
    character: str
    bridge_port: int
    vision_port: int | None
    vision_enabled: bool
    created_at: float
    url: str
    log_path: str
    command: list[str]
    process: subprocess.Popen | None = field(repr=False, default=None)
    status: str = "starting"
    pid: int | None = None
    last_activity_at: float = 0.0

    def payload(self) -> dict:
        return {
            "session_id": self.session_id,
            "token": self.token,
            "character": self.character,
            "bridge_port": self.bridge_port,
            "vision_port": self.vision_port,
            "vision_enabled": self.vision_enabled,
            "created_at": self.created_at,
            "url": self.url,
            "log_path": self.log_path,
            "status": self.status,
            "pid": self.pid,
            "last_activity_at": self.last_activity_at,
            "command": list(self.command),
        }


class RuntimeSessionManager:
    def __init__(
        self,
        *,
        public_host: str = "127.0.0.1",
        log_root: Path = SESSION_LOG_ROOT,
        startup_timeout_s: float = 180.0,
        command_factory: Callable[[ManagedSession], list[str]] | None = None,
        base_env: dict[str, str] | None = None,
        status_fetcher: Callable[[ManagedSession], dict] | None = None,
        idle_timeout_s: float = 0.0,
        reap_interval_s: float = 10.0,
    ):
        self._public_host = public_host
        self._public_port = 0
        self._log_root = Path(log_root)
        self._startup_timeout_s = float(startup_timeout_s)
        self._command_factory = command_factory or self._default_command
        self._base_env = dict(os.environ)
        if base_env:
            self._base_env.update(base_env)
        self._status_fetcher = status_fetcher or self._fetch_bridge_status
        self._idle_timeout_s = float(idle_timeout_s)
        self._reap_interval_s = float(reap_interval_s)
        self._sessions: dict[str, ManagedSession] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._reaper_thread: threading.Thread | None = None
        self._log_root.mkdir(parents=True, exist_ok=True)
        if self._idle_timeout_s > 0:
            self._reaper_thread = threading.Thread(target=self._reaper_loop, daemon=True, name="session-manager-reaper")
            self._reaper_thread.start()

    def _default_command(self, session: ManagedSession) -> list[str]:
        cmd = ["uv", "run", "-m", "character_eng", "--browser", "--character", session.character, "--start-paused"]
        if session.vision_enabled:
            cmd.append("--vision")
        return cmd

    def set_public_endpoint(self, *, host: str | None = None, port: int | None = None) -> None:
        if host:
            self._public_host = host
        if port is not None:
            self._public_port = int(port)

    def _session_url(self, token: str, bridge_port: int) -> str:
        if self._public_port > 0:
            return f"http://{self._public_host}:{self._public_port}/?token={token}"
        return f"http://{self._public_host}:{bridge_port}/?token={token}"

    def _runtime_env(self, session: ManagedSession) -> dict[str, str]:
        env = dict(self._base_env)
        env["PYTHONUNBUFFERED"] = "1"
        env["CHARACTER_ENG_BRIDGE_ENABLED"] = "true"
        env["CHARACTER_ENG_BRIDGE_PORT"] = str(session.bridge_port)
        env["CHARACTER_ENG_BRIDGE_TOKEN"] = session.token
        env["CHARACTER_ENG_DASHBOARD_ENABLED"] = "true"
        env["CHARACTER_ENG_DASHBOARD_PORT"] = str(session.bridge_port)
        env["CHARACTER_ENG_VISION_ENABLED"] = "true" if session.vision_enabled else "false"
        if session.vision_enabled and session.vision_port is not None:
            env["CHARACTER_ENG_VISION_PORT"] = str(session.vision_port)
            env["CHARACTER_ENG_VISION_URL"] = f"http://127.0.0.1:{session.vision_port}"
        return env

    def _fetch_bridge_status(self, session: ManagedSession) -> dict:
        status_url = f"http://127.0.0.1:{session.bridge_port}/bridge/status?token={session.token}"
        with urllib.request.urlopen(status_url, timeout=1.5) as resp:
            return json.loads(resp.read())

    def _activity_ts_from_status(self, session: ManagedSession, payload: dict) -> float:
        client = payload.get("client") or {}
        return max(
            session.created_at,
            float(payload.get("last_input_audio_at") or 0.0),
            float(payload.get("last_video_frame_at") or 0.0),
            float(payload.get("last_text_input_at") or 0.0),
            float(client.get("last_control_at") or 0.0),
        )

    def _refresh_status(self, session: ManagedSession) -> dict:
        payload = self._status_fetcher(session)
        session.last_activity_at = self._activity_ts_from_status(session, payload)
        return payload

    def _wait_until_ready(self, session: ManagedSession) -> None:
        deadline = time.time() + self._startup_timeout_s
        while time.time() < deadline:
            proc = session.process
            if proc is not None and proc.poll() is not None:
                session.status = "failed"
                raise RuntimeError(f"session runtime exited with code {proc.returncode}")
            try:
                payload = self._refresh_status(session)
                if payload.get("enabled") is True:
                    session.status = "ready"
                    return
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
                time.sleep(0.25)
        session.status = "timeout"
        raise RuntimeError("session runtime did not become ready before timeout")

    def create_session(self, *, character: str = "greg", vision: bool = True) -> ManagedSession:
        session_id = uuid.uuid4().hex[:10]
        token = secrets.token_urlsafe(24)
        bridge_port = _find_free_port()
        vision_port = _find_free_port() if vision else None
        url = self._session_url(token, bridge_port)
        log_path = self._log_root / f"{session_id}.log"
        session = ManagedSession(
            session_id=session_id,
            token=token,
            character=character,
            bridge_port=bridge_port,
            vision_port=vision_port,
            vision_enabled=vision,
            created_at=time.time(),
            url=url,
            log_path=str(log_path),
            command=[],
        )
        session.command = list(self._command_factory(session))
        env = self._runtime_env(session)
        with log_path.open("wb") as log_file:
            proc = subprocess.Popen(
                session.command,
                cwd=str(PROJECT_ROOT),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        session.process = proc
        session.pid = proc.pid
        session.last_activity_at = session.created_at
        with self._lock:
            self._sessions[session.session_id] = session
        try:
            self._wait_until_ready(session)
        except Exception:
            self.stop_session(session.session_id)
            raise
        return session

    def get_session_by_token(self, token: str) -> ManagedSession | None:
        token = str(token or "").strip()
        if not token:
            return None
        self.cleanup_dead_sessions()
        with self._lock:
            sessions = list(self._sessions.values())
        for session in sessions:
            if session.token == token:
                return session
        return None

    def list_sessions(self) -> list[dict]:
        self.cleanup_dead_sessions()
        with self._lock:
            sessions = list(self._sessions.values())
        for session in sessions:
            try:
                self._refresh_status(session)
            except Exception:
                pass
        return [session.payload() for session in sessions]

    def get_session(self, session_id: str) -> dict | None:
        self.cleanup_dead_sessions()
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            return None
        try:
            self._refresh_status(session)
        except Exception:
            pass
        return session.payload()

    def stop_session(self, session_id: str) -> bool:
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            return False
        proc = session.process
        session.status = "stopped"
        if proc is None:
            return True
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return True
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if proc.poll() is not None:
                return True
            time.sleep(0.1)
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            return True
        proc.wait(timeout=2)
        return True

    def cleanup_dead_sessions(self) -> int:
        removed: list[str] = []
        with self._lock:
            for session_id, session in self._sessions.items():
                proc = session.process
                if proc is not None and proc.poll() is not None:
                    session.status = "exited"
                    removed.append(session_id)
            for session_id in removed:
                self._sessions.pop(session_id, None)
        return len(removed)

    def reap_idle_sessions(self) -> list[str]:
        if self._idle_timeout_s <= 0:
            return []
        with self._lock:
            sessions = list(self._sessions.values())
        now = time.time()
        expired: list[str] = []
        for session in sessions:
            try:
                self._refresh_status(session)
            except Exception:
                pass
            if session.last_activity_at and (now - session.last_activity_at) > self._idle_timeout_s:
                expired.append(session.session_id)
        for session_id in expired:
            self.stop_session(session_id)
        return expired

    def _reaper_loop(self) -> None:
        while not self._stop_event.wait(timeout=self._reap_interval_s):
            self.cleanup_dead_sessions()
            self.reap_idle_sessions()

    def close(self) -> None:
        self._stop_event.set()
        if self._reaper_thread is not None:
            self._reaper_thread.join(timeout=2)
            self._reaper_thread = None
        for session in self.list_sessions():
            self.stop_session(session["session_id"])


class _ManagerHandler(BaseHTTPRequestHandler):
    manager: RuntimeSessionManager | None = None
    admin_token: str = ""

    def _manager(self) -> RuntimeSessionManager:
        manager = type(self).manager
        if manager is None:
            raise RuntimeError("manager not configured")
        return manager

    def _request_token(self) -> str:
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        query_token = (params.get("token") or params.get("session_token") or [""])[0].strip()
        if query_token:
            return query_token
        return self.headers.get("X-Session-Token", "").strip()

    def _session_for_request(self) -> ManagedSession | None:
        return self._manager().get_session_by_token(self._request_token())

    def _authorized(self) -> bool:
        expected = type(self).admin_token.strip()
        if not expected:
            return True
        auth = self.headers.get("Authorization", "").strip()
        if auth == f"Bearer {expected}":
            return True
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        query_token = (params.get("token") or params.get("admin_token") or [""])[0].strip()
        if query_token == expected:
            return True
        payload = json.dumps({"ok": False, "error": "unauthorized"}).encode()
        self.send_response(HTTPStatus.UNAUTHORIZED)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
        return False

    def _proxy_request(self, session: ManagedSession) -> None:
        body = b""
        if self.command in {"POST", "PUT", "PATCH"}:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length > 0 else b""

        headers: dict[str, str] = {}
        for key, value in self.headers.items():
            lower = key.lower()
            if lower in {"host", "content-length", "connection"}:
                continue
            headers[key] = value
        headers.setdefault("X-Session-Token", session.token)

        conn = http.client.HTTPConnection("127.0.0.1", session.bridge_port, timeout=60)
        try:
            conn.request(self.command, self.path, body=body, headers=headers)
            resp = conn.getresponse()
            self.send_response(resp.status, resp.reason)
            for key, value in resp.getheaders():
                lower = key.lower()
                if lower in {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers", "transfer-encoding", "upgrade"}:
                    continue
                self.send_header(key, value)
            self.end_headers()
            try:
                while True:
                    chunk = resp.read(64 * 1024)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return
        finally:
            conn.close()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        session = self._session_for_request()
        if session is not None and path not in {"/health"} and not path.startswith("/sessions"):
            return self._proxy_request(session)
        if not self._authorized():
            return
        if path == "/" or path == "/index.html":
            try:
                body = SESSION_MANAGER_HTML.read_bytes()
            except FileNotFoundError:
                return self._send_json({"ok": False, "error": "session manager html missing"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/health":
            return self._send_json({"ok": True, "status": "ok"})
        if path == "/sessions":
            return self._send_json({"ok": True, "sessions": self._manager().list_sessions()})
        if path.startswith("/sessions/"):
            session_id = path.rsplit("/", 1)[-1].strip()
            payload = self._manager().get_session(session_id)
            if payload is None:
                return self._send_json({"ok": False, "error": "session not found"}, status=HTTPStatus.NOT_FOUND)
            return self._send_json({"ok": True, "session": payload})
        return self._send_json({"ok": False, "error": "not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        session = self._session_for_request()
        if session is not None and not path.startswith("/sessions"):
            return self._proxy_request(session)
        if not self._authorized():
            return
        if path == "/sessions":
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            try:
                session = self._manager().create_session(
                    character=str(payload.get("character") or "greg"),
                    vision=bool(payload.get("vision", True)),
                )
            except Exception as exc:
                return self._send_json(
                    {"ok": False, "error": str(exc)},
                    status=HTTPStatus.GATEWAY_TIMEOUT,
                )
            return self._send_json({"ok": True, "session": session.payload()})
        if path.startswith("/sessions/") and path.endswith("/stop"):
            session_id = path.split("/")[-2]
            stopped = self._manager().stop_session(session_id)
            if not stopped:
                return self._send_json({"ok": False, "error": "session not found"}, status=HTTPStatus.NOT_FOUND)
            return self._send_json({"ok": True, "session_id": session_id, "stopped": True})
        return self._send_json({"ok": False, "error": "not found"}, status=HTTPStatus.NOT_FOUND)

    def log_message(self, format, *args):  # noqa: A003
        return

    def _send_json(self, payload: dict, *, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def start_session_manager(
    manager: RuntimeSessionManager,
    *,
    port: int = 7870,
    admin_token: str = "",
) -> tuple[threading.Thread, int, ThreadingHTTPServer]:
    if port <= 0:
        port = _find_free_port()
    _ManagerHandler.manager = manager
    _ManagerHandler.admin_token = admin_token.strip()
    server = ThreadingHTTPServer(("0.0.0.0", port), _ManagerHandler)
    server.daemon_threads = True
    manager.set_public_endpoint(port=server.server_address[1])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread, server.server_address[1], server


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Spawn one browser runtime per user session.")
    parser.add_argument("--port", type=int, default=7870, help="HTTP port for the manager API")
    parser.add_argument("--public-host", default="127.0.0.1", help="Host used in returned session URLs")
    parser.add_argument("--admin-token", default=os.environ.get("CHARACTER_ENG_MANAGER_TOKEN", ""), help="Optional bearer token for manager API")
    args = parser.parse_args()

    manager = RuntimeSessionManager(public_host=args.public_host)
    _, actual_port, server = start_session_manager(manager, port=args.port, admin_token=args.admin_token)
    print(f"session manager listening on http://0.0.0.0:{actual_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        manager.close()


if __name__ == "__main__":
    main()
