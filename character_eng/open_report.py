"""Open an HTML report in the system browser.

Usage:
    uv run -m character_eng.open_report              # latest HTML in logs/
    uv run -m character_eng.open_report <path>        # specific file
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"


def _is_wsl() -> bool:
    try:
        return "microsoft" in platform.uname().release.lower()
    except Exception:
        return False


def open_in_browser(path: Path):
    """Open a file in the default browser, handling WSL."""
    resolved = str(path.resolve())
    if _is_wsl():
        # Convert to Windows path and open via cmd.exe
        try:
            win_path = subprocess.check_output(
                ["wslpath", "-w", resolved], text=True
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
        print(f"Cannot detect browser opener. File: {resolved}")
        return

    subprocess.Popen(
        [opener, resolved],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def find_latest_html() -> Path | None:
    """Find the most recently modified HTML file in logs/."""
    if not LOGS_DIR.is_dir():
        return None
    htmls = sorted(LOGS_DIR.glob("*.html"), key=lambda p: p.stat().st_mtime)
    return htmls[-1] if htmls else None


def main():
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

    print(f"Opening: {path}")
    open_in_browser(path)


if __name__ == "__main__":
    main()
