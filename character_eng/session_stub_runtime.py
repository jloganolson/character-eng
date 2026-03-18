from __future__ import annotations

import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse


PORT = int(os.environ["CHARACTER_ENG_BRIDGE_PORT"])
TOKEN = os.environ.get("CHARACTER_ENG_BRIDGE_TOKEN", "").strip()


class _StubHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        token = (parse_qs(parsed.query).get("token") or [""])[0].strip()
        if TOKEN and token != TOKEN:
            body = json.dumps({"ok": False, "error": "unauthorized"}).encode()
            self.send_response(HTTPStatus.UNAUTHORIZED)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path == "/bridge/status":
            body = json.dumps({
                "enabled": True,
                "client": {},
                "mode": "stub",
                "last_input_audio_at": 0,
                "last_video_frame_at": 0,
                "last_text_input_at": 0,
            }).encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path in {"/", "/index.html"}:
            body = (
                "<!DOCTYPE html><html><body>"
                "<h1>character-eng stub session</h1>"
                "<p>Manager-lite validation runtime is active.</p>"
                "</body></html>"
            ).encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        body = json.dumps({"ok": False, "error": "not found"}).encode()
        self.send_response(HTTPStatus.NOT_FOUND)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):  # noqa: A003
        return


def main() -> None:
    server = ThreadingHTTPServer(("0.0.0.0", PORT), _StubHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
