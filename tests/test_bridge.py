from __future__ import annotations

import base64
import json
import queue
import time
import urllib.error
import urllib.request

from character_eng.bridge import BridgeServer
from character_eng.config import AppConfig
from character_eng.dashboard.events import DashboardEventCollector


def _json_get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=3) as resp:
        return json.loads(resp.read())


def _post(url: str, body: bytes, content_type: str) -> dict:
    req = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": content_type})
    with urllib.request.urlopen(req, timeout=3) as resp:
        return json.loads(resp.read())


def test_bridge_serves_dashboard_and_media_endpoints():
    collector = DashboardEventCollector()
    input_queue: queue.Queue[str] = queue.Queue()
    audio_payloads: list[bytes] = []
    frame_payloads: list[bytes] = []

    bridge = BridgeServer(port=0)
    bridge.set_dashboard(collector, input_queue)
    bridge.start(
        on_audio=lambda payload: audio_payloads.append(payload),
        on_video_frame=lambda payload: frame_payloads.append(payload),
    )
    base_url = f"http://127.0.0.1:{bridge.port}"
    try:
        with urllib.request.urlopen(f"{base_url}/", timeout=3) as resp:
            html = resp.read().decode("utf-8")
        assert "window.__bridgeConfig" in html

        _post(f"{base_url}/send", json.dumps({"text": "hello bridge"}).encode(), "application/json")
        assert input_queue.get(timeout=1) == "hello bridge"
        status = _json_get(f"{base_url}/bridge/status")
        assert status["last_text_input_at"] > 0

        pcm = b"\x01\x02\x03\x04"
        result = _post(f"{base_url}/bridge/audio", pcm, "application/octet-stream")
        assert result["ok"] is True
        assert audio_payloads == [pcm]

        jpeg = b"\xff\xd8\xff"
        result = _post(f"{base_url}/bridge/frame", jpeg, "application/octet-stream")
        assert result["ok"] is True
        assert frame_payloads == [jpeg]

        bridge.push_audio(b"\x10\x00\x20\x00", sample_rate=24000)
        payload = _json_get(f"{base_url}/bridge/audio")
        assert len(payload["chunks"]) == 1
        chunk = payload["chunks"][0]
        assert chunk["sample_rate"] == 24000
        assert base64.b64decode(chunk["data_b64"]) == b"\x10\x00\x20\x00"
    finally:
        bridge.stop()


def test_bridge_control_and_status_round_trip():
    bridge = BridgeServer(port=0)
    bridge.set_dashboard(DashboardEventCollector(), queue.Queue())
    bridge.start()
    base_url = f"http://127.0.0.1:{bridge.port}"
    try:
        _post(
            f"{base_url}/bridge/control",
            json.dumps({"audio_armed": True, "mic": True, "camera": False}).encode(),
            "application/json",
        )
        status = _json_get(f"{base_url}/bridge/status")
        assert status["enabled"] is True
        assert status["client"]["audio_armed"] is True
        assert status["client"]["mic"] is True
        assert status["client"]["camera"] is False
    finally:
        bridge.stop()


def test_bridge_serves_proxied_vision_status():
    bridge = BridgeServer(port=0)
    bridge.set_dashboard(DashboardEventCollector(), queue.Queue())
    bridge.vision_status_payload = lambda: {  # type: ignore[method-assign]
        "enabled": True,
        "status": {"vllm": "ready"},
        "memory": {"gpu": {"allocated_mb": 123, "total_mb": 456}},
    }
    bridge.start()
    base_url = f"http://127.0.0.1:{bridge.port}"
    try:
        payload = _json_get(f"{base_url}/bridge/vision-status")
        assert payload["enabled"] is True
        assert payload["status"]["vllm"] == "ready"
        assert payload["memory"]["gpu"]["allocated_mb"] == 123
    finally:
        bridge.stop()


def test_bridge_vision_status_uses_service_url(monkeypatch):
    called = {}

    class StubVisionClient:
        def __init__(self, base_url):
            called["base_url"] = base_url

        def model_status(self):
            return {"vllm": "ready"}

        def memory_status(self):
            return {"gpu": {"allocated_mb": 1, "total_mb": 2}}

    cfg = AppConfig()
    cfg.vision.enabled = True
    cfg.vision.service_url = "http://127.0.0.1:9999"

    monkeypatch.setattr("character_eng.bridge.load_config", lambda: cfg)
    monkeypatch.setattr("character_eng.bridge.VisionClient", StubVisionClient)

    bridge = BridgeServer(port=0)
    payload = bridge.vision_status_payload()
    assert called["base_url"] == "http://127.0.0.1:9999"
    assert payload["status"]["vllm"] == "ready"


def test_bridge_token_required_when_configured():
    bridge = BridgeServer(port=0, access_token="top-secret")
    bridge.set_dashboard(DashboardEventCollector(), queue.Queue())
    bridge.start()
    base_url = f"http://127.0.0.1:{bridge.port}"
    try:
        try:
            urllib.request.urlopen(f"{base_url}/bridge/status", timeout=3)
        except urllib.error.HTTPError as exc:
            assert exc.code == 401
        else:
            raise AssertionError("expected unauthorized bridge status without token")

        status = _json_get(f"{base_url}/bridge/status?token=top-secret")
        assert status["enabled"] is True

        with urllib.request.urlopen(f"{base_url}/?token=top-secret", timeout=3) as resp:
            html = resp.read().decode("utf-8")
        assert "accessToken" in html
        assert "top-secret" in html
    finally:
        bridge.stop()


def test_bridge_query_token_allows_dashboard_routes():
    collector = DashboardEventCollector()
    collector.push("session_start", {"session_id": "sess-1"})
    input_queue: queue.Queue[str] = queue.Queue()
    bridge = BridgeServer(port=0, access_token="query-secret")
    bridge.set_dashboard(collector, input_queue)
    bridge.start()
    base_url = f"http://127.0.0.1:{bridge.port}"
    try:
        _post(
            f"{base_url}/send?token=query-secret",
            json.dumps({"text": "hello"}).encode(),
            "application/json",
        )
        assert input_queue.get(timeout=1) == "hello"
        state = _json_get(f"{base_url}/state?token=query-secret")
        assert any(event["type"] == "session_start" for event in state)
    finally:
        bridge.stop()
