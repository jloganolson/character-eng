"""Serve an HTML report with annotation save support.

Starts a tiny HTTP server that serves the report and handles POST
requests to save annotations directly to logs/annotated/. Opens the
browser automatically.

Usage:
    uv run -m character_eng.open_report              # latest HTML in logs/
    uv run -m character_eng.open_report <path>        # specific file
"""

from __future__ import annotations

import json
import platform
import shutil
import socket
import subprocess
import sys
import threading
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
ANNOTATED_DIR = LOGS_DIR / "annotated"


def _is_wsl() -> bool:
    try:
        return "microsoft" in platform.uname().release.lower()
    except Exception:
        return False


def open_in_browser(url_or_path: Path | str):
    """Open a URL or file in the default browser, handling WSL."""
    target = str(url_or_path)
    if _is_wsl():
        if target.startswith("http"):
            # URL — use cmd.exe start directly
            subprocess.Popen(
                ["cmd.exe", "/c", "start", "", target],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return
        # File path — convert to Windows path
        try:
            win_path = subprocess.check_output(
                ["wslpath", "-w", target], text=True
            ).strip()
            subprocess.Popen(
                ["cmd.exe", "/c", "start", "", win_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return
        except Exception:
            pass

    # macOS / Linux
    if sys.platform == "darwin":
        opener = "open"
    elif shutil.which("xdg-open"):
        opener = "xdg-open"
    else:
        print(f"Cannot detect browser opener. Target: {target}")
        return

    subprocess.Popen(
        [opener, target],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def find_latest_html() -> Path | None:
    """Find the most recently modified HTML file in logs/."""
    if not LOGS_DIR.is_dir():
        return None
    htmls = sorted(LOGS_DIR.glob("*.html"), key=lambda p: p.stat().st_mtime)
    return htmls[-1] if htmls else None


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class ReportHandler(BaseHTTPRequestHandler):
    """Serves the HTML report and handles annotation saves."""

    def __init__(self, report_path: Path, *args, **kwargs):
        self.report_path = report_path
        super().__init__(*args, **kwargs)

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            content = self.report_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self):
        if self.path == "/save-annotations":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                data = json.loads(body)
                report_name = data.get("report", "unknown")
                # Sanitize filename
                safe_name = "".join(
                    c if c.isalnum() or c in "-_." else "_" for c in report_name
                )
                ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
                out_path = ANNOTATED_DIR / f"{safe_name}.annotations.json"
                out_path.write_text(json.dumps(data, indent=2))

                response = json.dumps({"ok": True, "path": str(out_path)}).encode()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)
                print(f"  Saved: {out_path}")
            except Exception as e:
                response = json.dumps({"ok": False, "error": str(e)}).encode()
                self.send_response(HTTPStatus.INTERNAL_SERVER_ERROR)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)
        elif self.path == "/shutdown":
            response = json.dumps({"ok": True}).encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)
            print("  Shutdown requested by browser.")
            # Shutdown in a separate thread so the response finishes first
            threading.Thread(target=self.server.shutdown, daemon=True).start()
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def log_message(self, format, *args):
        # Suppress default access logs
        pass


def serve_report(report_path: Path):
    """Start HTTP server, open browser, block until Ctrl+C."""
    port = _find_free_port()
    handler = partial(ReportHandler, report_path)
    server = HTTPServer(("0.0.0.0", port), handler)
    url = f"http://127.0.0.1:{port}/"

    print(f"Serving: {report_path}")
    print(f"URL:     {url}")
    print(f"Annotations save to: {ANNOTATED_DIR}/")
    print("Press Ctrl+C to stop\n")

    # Open browser after a short delay
    def _open():
        import time
        time.sleep(0.3)
        open_in_browser(url)
    threading.Thread(target=_open, daemon=True).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


def generate_test_report() -> Path:
    """Generate a spoofed report using the real generation codepath."""
    from datetime import datetime

    from character_eng.qa_personas import (
        PersonaConfig,
        PersonaResult,
        TurnRecord,
        generate_html_report,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    personas = [
        PersonaResult(
            persona=PersonaConfig(
                name="Test Alice",
                description="Normal user — asks questions",
                turns=4,
                persona_prompt="",
            ),
            boot_log=[{
                "type": "plan",
                "beats": 3,
                "intents": ["Greet visitor", "Mention the orb", "Ask for help"],
            }],
            turns=[
                TurnRecord(turn=0, action="message", input_text="Hey, what's up?",
                           response="Not much — just sitting here on this desk, staring at an orb. Story of my life."),
                TurnRecord(turn=1, action="message", input_text="What's the orb do?",
                           response="Wish I knew. A wizard left it here. Hasn't done a thing since.",
                           eval_entry={"thought": "They seem curious about the orb", "gaze": "orb",
                                       "expression": "curious", "script_status": "advance"}),
                TurnRecord(turn=2, action="world", input_text="The orb starts glowing faintly",
                           response="Whoa — did you see that? It's actually doing something!"),
                TurnRecord(turn=3, action="message", input_text="Yeah it's glowing!",
                           response="I've waited so long for this. What do you think it means?",
                           eval_entry={"thought": "The orb is finally active", "gaze": "orb",
                                       "expression": "excited", "script_status": "off_book",
                                       "plan_request": "Explore the orb's new behavior"}),
            ],
        ),
        PersonaResult(
            persona=PersonaConfig(
                name="Test Bob",
                description="Beat runner — scripted delivery",
                turns=3,
                persona_prompt="",
            ),
            boot_log=[{
                "type": "plan",
                "beats": 3,
                "intents": ["Note the orb", "Lament lack of hands", "Call for help"],
            }],
            turns=[
                TurnRecord(turn=0, action="beat", input_text="",
                           response="Still nothing? You're a very patient piece of glass, aren't you?"),
                TurnRecord(turn=1, action="beat", input_text="",
                           response="I wish I had hands so I could give you a shake.",
                           eval_entry={"thought": "Maybe if I had a body...", "gaze": "orb",
                                       "expression": "wistful", "script_status": "advance"}),
                TurnRecord(turn=2, action="beat", input_text="",
                           response="Hey, is anyone out there? This orb needs poking.",
                           stale_discard=True),
            ],
        ),
        PersonaResult(
            persona=PersonaConfig(
                name="Test Carol",
                description="Chaos agent — rapid world changes",
                turns=3,
                persona_prompt="",
            ),
            boot_log=[],
            turns=[
                TurnRecord(turn=0, action="world", input_text="A cat jumps on the desk",
                           response="What the — there's a cat here now? On MY desk?"),
                TurnRecord(turn=1, action="message", input_text="lol",
                           response="It's not funny, this cat is sitting on my face."),
                TurnRecord(turn=2, action="nothing", input_text="",
                           response="(idle)"),
            ],
            error=None,
        ),
        PersonaResult(
            persona=PersonaConfig(
                name="Test Error",
                description="Simulated error case",
                turns=1,
                persona_prompt="",
            ),
            boot_log=[],
            turns=[],
            error="Connection timeout after 30s",
        ),
    ]

    LOGS_DIR.mkdir(exist_ok=True)
    (LOGS_DIR / "annotated").mkdir(exist_ok=True)

    html = generate_html_report(
        results=personas,
        model_key="test-model",
        model_name="Test Model (Spoofed)",
        character="greg",
        timestamp=timestamp,
    )
    report_path = LOGS_DIR / f"test_report_{timestamp}.html"
    report_path.write_text(html)
    print(f"Generated: {report_path}")
    return report_path


def main():
    if "--test" in sys.argv:
        path = generate_test_report()
        serve_report(path)
        return

    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        if not path.exists():
            print(f"File not found: {path}")
            sys.exit(1)
    else:
        path = find_latest_html()
        if path is None:
            print("No HTML reports found in logs/")
            sys.exit(1)

    serve_report(path)


if __name__ == "__main__":
    main()
