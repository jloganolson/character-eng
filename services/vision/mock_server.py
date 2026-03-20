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
import io
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from PIL import Image, ImageDraw, ImageFont


class MockVisionHandler(BaseHTTPRequestHandler):
    replay_data: dict = {}
    start_time: float = 0.0
    input_mode: str = "camera"
    _font = None

    @classmethod
    def _load_font(cls):
        if cls._font is None:
            try:
                cls._font = ImageFont.load_default()
            except Exception:
                cls._font = None
        return cls._font

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

    @staticmethod
    def _send_common_headers(handler, content_type: str, content_length: int | None = None):
        handler.send_header("Access-Control-Allow-Origin", "*")
        handler.send_header("Cache-Control", "no-store")
        handler.send_header("Content-Type", content_type)
        if content_length is not None:
            handler.send_header("Content-Length", str(content_length))

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self._send_common_headers(self, "application/json", len(body))
        self.end_headers()
        self.wfile.write(body)

    @staticmethod
    def _clamp_bbox(raw_bbox, width: int, height: int) -> tuple[int, int, int, int] | None:
        if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
            return None
        try:
            x, y, w, h = [int(float(v)) for v in raw_bbox]
        except (TypeError, ValueError):
            return None
        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))
        w = max(4, min(width - x, w))
        h = max(4, min(height - y, h))
        return x, y, w, h

    def _draw_label(self, draw: ImageDraw.ImageDraw, x: int, y: int, text: str, fill: str):
        font = self._load_font()
        bbox = draw.textbbox((x, y), text, font=font)
        draw.rounded_rectangle(
            (bbox[0] - 4, bbox[1] - 2, bbox[2] + 4, bbox[3] + 2),
            radius=6,
            fill=(18, 18, 18, 220),
        )
        draw.text((x, y), text, fill=fill, font=font)

    def _render_frame_bytes(self, snapshot: dict, *, annotated: bool = False, max_width: int = 0) -> bytes:
        width = 800
        height = 600
        img = Image.new("RGB", (width, height), "#d8edf6")
        draw = ImageDraw.Draw(img, "RGBA")
        font = self._load_font()

        draw.rectangle((0, 0, width, int(height * 0.56)), fill="#c9e8ff")
        draw.rectangle((0, int(height * 0.56), width, height), fill="#d4c39b")
        draw.ellipse((560, 40, 700, 180), fill=(255, 232, 153, 160))
        draw.rectangle((0, 410, width, 445), fill="#cfc087")

        for idx, obj in enumerate(snapshot.get("objects", [])):
            bbox = self._clamp_bbox(obj.get("bbox"), width, height)
            if bbox is None:
                continue
            x, y, w, h = bbox
            label = str(obj.get("label", "object"))
            color = {
                "lemonade stand": (241, 197, 92, 255),
                "cooler": (116, 169, 214, 255),
                "sign": (230, 244, 214, 255),
            }.get(label.lower(), (180, 210, 160, 255))
            draw.rounded_rectangle((x, y, x + w, y + h), radius=10, outline=color, width=4, fill=(color[0], color[1], color[2], 55))
            self._draw_label(draw, x + 8, max(8, y - 18), label, "#f8f6ef")
            if annotated:
                conf = obj.get("confidence")
                if conf is not None:
                    self._draw_label(draw, x + 8, y + 10, f"{float(conf):.2f}", "#d7f7ff")

        for idx, person in enumerate(snapshot.get("persons", [])):
            bbox = self._clamp_bbox(person.get("bbox"), width, height)
            if bbox is None:
                continue
            x, y, w, h = bbox
            tint = (196, 126, 86, 255) if idx % 2 == 0 else (130, 108, 198, 255)
            draw.rounded_rectangle((x, y, x + w, y + h), radius=14, outline=tint, width=4, fill=(tint[0], tint[1], tint[2], 45))
            identity = str(person.get("identity") or f"Person {idx + 1}")
            self._draw_label(draw, x + 8, max(10, y - 20), identity, "#fff5dd")

        for face in snapshot.get("faces", []):
            bbox = self._clamp_bbox(face.get("bbox"), width, height)
            if bbox is None:
                continue
            x, y, w, h = bbox
            draw.ellipse((x, y, x + w, y + h), outline=(255, 105, 105, 255), width=3, fill=(255, 170, 170, 35))
            gaze = str(face.get("gaze_direction") or "").strip()
            if gaze:
                self._draw_label(draw, x + 4, y + h + 6, gaze, "#ffd7d7")

        answer = ""
        if snapshot.get("vlm_answers"):
            answer = str(snapshot["vlm_answers"][0].get("answer", "")).strip()
        draw.rounded_rectangle((18, 18, width - 18, 74), radius=14, fill=(14, 20, 28, 196))
        header = f"mock vision feed  faces {len(snapshot.get('faces', []))}  people {len(snapshot.get('persons', []))}  objects {len(snapshot.get('objects', []))}"
        draw.text((34, 28), header, fill="#f4efe2", font=font)
        if answer:
            draw.text((34, 48), answer[:100], fill="#d7edf4", font=font)

        if max_width > 0 and img.width > max_width:
            ratio = max_width / float(img.width)
            img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.Resampling.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=88)
        return buf.getvalue()

    def do_GET(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        if parsed.path == "/health":
            self._send_json({"status": "ok", "mock": True, "timestamp": time.time()})
        elif parsed.path == "/snapshot":
            self._send_json(self._current_snapshot())
        elif parsed.path == "/model_status":
            self._send_json({
                "vllm": "ready",
                "vllm_model": "mock replay",
                "sam3": {"status": "ready", "error": ""},
                "face": {"status": "ready", "error": ""},
                "person": {"status": "ready", "error": ""},
            })
        elif parsed.path == "/memory_status":
            self._send_json({
                "gpu": {"allocated_mb": 512, "total_mb": 8192},
                "mock": True,
            })
        elif parsed.path == "/input_mode":
            self._send_json({"mode": self.input_mode, "mock": True})
        elif parsed.path == "/frame.jpg":
            snapshot = self._current_snapshot()
            annotated = query.get("annotated", ["0"])[0] == "1"
            try:
                max_width = int(query.get("max_width", ["0"])[0] or "0")
            except ValueError:
                max_width = 0
            body = self._render_frame_bytes(snapshot, annotated=annotated, max_width=max_width)
            self.send_response(200)
            self._send_common_headers(self, "image/jpeg", len(body))
            self.end_headers()
            self.wfile.write(body)
        elif parsed.path == "/video_feed":
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    snapshot = self._current_snapshot()
                    body = self._render_frame_bytes(snapshot, annotated=True, max_width=960)
                    payload = (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        + f"Content-Length: {len(body)}\r\n\r\n".encode()
                        + body
                        + b"\r\n"
                    )
                    self.wfile.write(payload)
                    self.wfile.flush()
                    time.sleep(0.2)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
        elif parsed.path == "/":
            elapsed = time.time() - self.start_time
            snap = self._current_snapshot()
            html = f"""<html><head><title>Mock Vision</title>
            <meta http-equiv="refresh" content="2">
            <style>body {{ font-family: monospace; background: #1a1a1a; color: #e0e0e0; padding: 20px; }}
            pre {{ background: #2a2a2a; padding: 15px; border-radius: 8px; }}</style></head>
            <body><h2>Mock Vision Server</h2>
            <p><img src="/frame.jpg?annotated=1&max_width=640" style="max-width:100%;border-radius:12px;border:1px solid #444"></p>
            <p>Elapsed: {elapsed:.1f}s | Faces: {len(snap['faces'])} | Persons: {len(snap['persons'])} | Objects: {len(snap['objects'])}</p>
            <pre>{json.dumps(snap, indent=2)}</pre></body></html>"""
            body = html.encode()
            self.send_response(200)
            self._send_common_headers(self, "text/html", len(body))
            self.end_headers()
            self.wfile.write(body)
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        # Accept runtime configuration endpoints silently in mock mode.
        if self.path in ("/set_questions", "/set_sam_targets", "/inject_frame"):
            content_len = int(self.headers.get("Content-Length", 0))
            self.rfile.read(content_len)  # consume body
            self._send_json({"status": "ok"})
        elif self.path == "/set_input_mode":
            content_len = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(content_len) or b"{}")
            self.input_mode = str(payload.get("mode", "camera") or "camera")
            self._send_json({"status": "ok", "mode": self.input_mode})
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

    server = ThreadingHTTPServer(("", args.port), MockVisionHandler)
    print(f"Mock vision server at http://localhost:{args.port}")
    print(f"Debug UI at http://localhost:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
