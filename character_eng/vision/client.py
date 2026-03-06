"""HTTP client for the vision service."""

from __future__ import annotations

import json
import urllib.request
import urllib.error

from character_eng.vision.context import RawVisualSnapshot


class VisionClient:
    """Thin HTTP client for the vision service (stdlib only, no new deps)."""

    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")

    def health(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.base_url}/health")
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read())
                return data.get("status") == "ok"
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            return False

    def snapshot(self) -> RawVisualSnapshot:
        req = urllib.request.Request(f"{self.base_url}/snapshot")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        return RawVisualSnapshot.from_json(data)

    def set_questions(self, constant: list[str], ephemeral: list[str]) -> None:
        body = json.dumps({"constant": constant, "ephemeral": ephemeral}).encode()
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
