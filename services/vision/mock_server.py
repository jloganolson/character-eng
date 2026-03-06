#!/usr/bin/env python3
"""Mock vision server for testing without a camera.

Serves scripted visual snapshots from a JSON replay file on the same
endpoints as the real vision service. The app can't tell the difference.

Usage:
    # Start mock server with a replay file
    python mock_server.py replays/walkup.json

    # Then run character-eng normally with --vision
    uv run -m character_eng --vision

Replay file format (JSON):
{
  "loop": true,           // restart from beginning when done
  "duration": 40.0,       // optional: override total duration for looping
  "snapshots": [
    {
      "at": 0.0,          // seconds from start
      "faces": [...],     // same format as real /snapshot
      "persons": [...],
      "objects": [...],
      "vlm_answers": [...]
    },
    ...
  ]
}

Each snapshot is served until the next one's timestamp is reached.
"""

import argparse
import json
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path


class MockVisionHandler(BaseHTTPRequestHandler):
    replay_data: dict = {}
    start_time: float = 0.0

    def _current_snapshot(self) -> dict:
        snapshots = self.replay_data.get("snapshots", [])
        if not snapshots:
            return {"faces": [], "persons": [], "objects": [], "vlm_answers": [], "timestamp": time.time()}

        elapsed = time.time() - self.start_time
        duration = self.replay_data.get("duration")
        if duration is None and snapshots:
            duration = snapshots[-1]["at"] + 5.0  # 5s after last snapshot

        if self.replay_data.get("loop", False) and duration > 0:
            elapsed = elapsed % duration

        # Find the latest snapshot at or before current time
        current = snapshots[0]
        for snap in snapshots:
            if snap["at"] <= elapsed:
                current = snap
            else:
                break

        return {
            "faces": current.get("faces", []),
            "persons": current.get("persons", []),
            "objects": current.get("objects", []),
            "vlm_answers": current.get("vlm_answers", []),
            "timestamp": time.time(),
        }

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json({"status": "ok", "mock": True, "timestamp": time.time()})
        elif self.path == "/snapshot":
            self._send_json(self._current_snapshot())
        elif self.path == "/":
            elapsed = time.time() - self.start_time
            snap = self._current_snapshot()
            html = f"""<html><head><title>Mock Vision</title>
            <meta http-equiv="refresh" content="2">
            <style>body {{ font-family: monospace; background: #1a1a1a; color: #e0e0e0; padding: 20px; }}
            pre {{ background: #2a2a2a; padding: 15px; border-radius: 8px; }}</style></head>
            <body><h2>Mock Vision Server</h2>
            <p>Elapsed: {elapsed:.1f}s | Faces: {len(snap['faces'])} | Persons: {len(snap['persons'])} | Objects: {len(snap['objects'])}</p>
            <pre>{json.dumps(snap, indent=2)}</pre></body></html>"""
            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        # Accept set_questions and set_sam_targets silently
        if self.path in ("/set_questions", "/set_sam_targets"):
            content_len = int(self.headers.get("Content-Length", 0))
            self.rfile.read(content_len)  # consume body
            self._send_json({"status": "ok"})
        else:
            self._send_json({"error": "not found"}, 404)

    def log_message(self, format, *args):
        # Quieter logging
        if self.path != "/snapshot":
            super().log_message(format, *args)


def main():
    parser = argparse.ArgumentParser(description="Mock vision server for testing")
    parser.add_argument("replay_file", help="Path to replay JSON file")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    replay_path = Path(args.replay_file)
    if not replay_path.exists():
        print(f"Replay file not found: {replay_path}")
        return

    replay_data = json.loads(replay_path.read_text())
    n_snapshots = len(replay_data.get("snapshots", []))
    loop = replay_data.get("loop", False)
    print(f"Loaded {n_snapshots} snapshots from {replay_path} (loop={loop})")

    MockVisionHandler.replay_data = replay_data
    MockVisionHandler.start_time = time.time()

    server = HTTPServer(("", args.port), MockVisionHandler)
    print(f"Mock vision server at http://localhost:{args.port}")
    print(f"Debug UI at http://localhost:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
