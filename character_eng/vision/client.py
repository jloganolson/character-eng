"""HTTP client for the vision service."""

from __future__ import annotations

import json
from urllib.parse import urlencode
import urllib.request
import urllib.error

from character_eng.vision.context import RawVisualSnapshot


class VisionClient:
    """Thin HTTP client for the vision service (stdlib only, no new deps)."""

    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")

    def health(self) -> bool:
        try:
            data = self._get_json("/health", timeout=3)
            return data.get("status") == "ok"
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            return False

    def snapshot(self) -> RawVisualSnapshot:
        data = self._get_json("/snapshot", timeout=5)
        return RawVisualSnapshot.from_json(data)

    def model_status(self) -> dict:
        return self._get_json("/model_status", timeout=3)

    def memory_status(self) -> dict:
        return self._get_json("/memory_status", timeout=3)

    def capture_frame_jpeg(self, *, annotated: bool = False, max_width: int = 0) -> bytes:
        query = urlencode({
            "annotated": "1" if annotated else "0",
            "max_width": str(max_width or 0),
        })
        req = urllib.request.Request(f"{self.base_url}/frame.jpg?{query}")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.read()

    @staticmethod
    def _question_suffix(text: str) -> str:
        suffix = "(1-2 sentences max)"
        stripped = text.strip()
        if not stripped:
            return stripped
        if stripped.endswith(suffix):
            return stripped
        return f"{stripped} {suffix}"

    def set_questions(self, constant: list[str], ephemeral: list[str]) -> None:
        body = json.dumps({
            "constant": [self._question_suffix(item) for item in constant if str(item).strip()],
            "ephemeral": [self._question_suffix(item) for item in ephemeral if str(item).strip()],
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/set_questions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5):
            pass

    def set_sam_targets(self, constant: list[str], ephemeral: list[str]) -> None:
        body = json.dumps({"constant": constant, "ephemeral": ephemeral}).encode()
        req = urllib.request.Request(
            f"{self.base_url}/set_sam_targets",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5):
            pass

    def inject_frame(self, jpeg_bytes: bytes) -> None:
        """Send a JPEG frame to the vision service (browser camera mode)."""
        req = urllib.request.Request(
            f"{self.base_url}/inject_frame",
            data=jpeg_bytes,
            headers={"Content-Type": "application/octet-stream"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5):
            pass

    def set_input_mode(self, mode: str) -> None:
        body = json.dumps({"mode": str(mode or "camera")}).encode()
        req = urllib.request.Request(
            f"{self.base_url}/set_input_mode",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5):
            pass

    def _get_json(self, path: str, timeout: float = 5) -> dict:
        req = urllib.request.Request(f"{self.base_url}{path}")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
