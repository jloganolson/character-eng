"""Tests for vision module dataclasses and client."""

import threading
import json
import socket
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from character_eng.vision.context import (
    FaceObservation,
    ObjectDetection,
    PersonObservation,
    RawVisualSnapshot,
    VisualContext,
    VLMAnswer,
)
from character_eng.vision.client import VisionClient


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_raw_visual_snapshot_from_json():
    data = {
        "faces": [
            {
                "identity": "Person 1",
                "bbox": [10, 20, 100, 150],
                "age": 30,
                "gender": "M",
                "confidence": 0.95,
                "gaze_direction": "at camera",
                "looking_at_camera": True,
            }
        ],
        "persons": [
            {"identity": "Person 1", "bbox": [5, 10, 200, 300], "confidence": 0.8}
        ],
        "objects": [
            {"label": "cup", "bbox": [50, 60, 30, 40], "confidence": 0.7}
        ],
        "vlm_answers": [
            {
                "question": "How many people?",
                "answer": "One person",
                "elapsed": 0.5,
                "slot_type": "constant",
                "timestamp": 1234567890.0,
            }
        ],
        "timestamp": 1234567890.0,
    }

    snap = RawVisualSnapshot.from_json(data)
    assert len(snap.faces) == 1
    assert snap.faces[0].identity == "Person 1"
    assert snap.faces[0].age == 30
    assert snap.faces[0].looking_at_camera is True
    assert len(snap.persons) == 1
    assert snap.persons[0].identity == "Person 1"
    assert len(snap.objects) == 1
    assert snap.objects[0].label == "cup"
    assert len(snap.vlm_answers) == 1
    assert snap.vlm_answers[0].answer == "One person"


def test_raw_visual_snapshot_from_empty_json():
    snap = RawVisualSnapshot.from_json({})
    assert snap.faces == []
    assert snap.persons == []
    assert snap.objects == []
    assert snap.vlm_answers == []


def test_visual_context_render_for_prompt():
    ctx = VisualContext()
    assert ctx.render_for_prompt() == ""

    snap = RawVisualSnapshot(
        faces=[FaceObservation(identity="Person 1", bbox=(0, 0, 100, 100), age=25, gender="M", gaze_direction="at camera")],
        persons=[PersonObservation(identity="Person 1", bbox=(0, 0, 200, 300))],
        objects=[ObjectDetection(label="cup", bbox=(50, 60, 30, 40))],
        vlm_answers=[VLMAnswer(question="What's happening?", answer="A person is standing")],
    )
    ctx.update(snap)
    text = ctx.render_for_prompt()
    assert "Person 1" in text
    assert "cup" in text
    assert "What's happening?" in text


def test_visual_context_gaze_targets():
    ctx = VisualContext()
    assert ctx.get_gaze_targets() == []

    snap = RawVisualSnapshot(
        persons=[
            PersonObservation(identity="Person 1", bbox=(0, 0, 100, 200)),
            PersonObservation(identity="Person 2", bbox=(100, 0, 100, 200)),
        ],
        objects=[ObjectDetection(label="cup"), ObjectDetection(label="sign")],
    )
    ctx.update(snap)
    targets = ctx.get_gaze_targets()
    assert "Person 1" in targets
    assert "Person 2" in targets
    assert "cup" in targets
    assert "sign" in targets


def test_visual_context_thread_safety():
    """Ensure update and read don't crash under concurrent access."""
    import threading

    ctx = VisualContext()
    snap = RawVisualSnapshot(
        persons=[PersonObservation(identity="Test", bbox=(0, 0, 100, 100))],
    )

    errors = []

    def writer():
        for _ in range(100):
            try:
                ctx.update(snap)
            except Exception as e:
                errors.append(e)

    def reader():
        for _ in range(100):
            try:
                ctx.render_for_prompt()
                ctx.get_gaze_targets()
            except Exception as e:
                errors.append(e)

    threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors


def test_vision_client_init():
    client = VisionClient("http://localhost:9999")
    assert client.base_url == "http://localhost:9999"
    # Health check should return False for non-existent server
    assert client.health() is False


def test_vision_client_capture_frame_jpeg():
    payload = b"fake-jpeg-bytes"

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if not self.path.startswith("/frame.jpg"):
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):
            pass

    port = _free_port()
    server = HTTPServer(("127.0.0.1", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = VisionClient(f"http://127.0.0.1:{port}")
        assert client.capture_frame_jpeg() == payload
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_mock_vision_server_contract():
    root = Path(__file__).resolve().parent.parent
    replay = root / "services" / "vision" / "replays" / "walkup.json"
    script = root / "services" / "vision" / "mock_server.py"
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, str(script), str(replay), "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    client = VisionClient(f"http://127.0.0.1:{port}")
    try:
        for _ in range(20):
            if client.health():
                break
            if proc.poll() is not None:
                raise AssertionError("mock vision server exited during startup")
            time.sleep(0.1)
        else:
            raise AssertionError("mock vision server did not become healthy")

        snapshot = client.snapshot()
        assert snapshot is not None
        client.set_questions(["How many people are visible?"], ["What is the person doing?"])
        client.set_sam_targets(["person"], ["cup"])
    finally:
        proc.kill()
        proc.wait(timeout=5)


def test_config_vision():
    from character_eng.config import VisionConfig, AppConfig

    vc = VisionConfig()
    assert vc.enabled is False
    assert vc.service_url == "http://localhost:7860"
    assert vc.auto_launch is True
    assert vc.synthesis_min_interval == 0.75

    cfg = AppConfig()
    assert cfg.vision.enabled is False


def test_person_get_or_create():
    from character_eng.person import PeopleState

    ps = PeopleState()
    pid1 = ps.get_or_create("Person 1")
    assert pid1 == "p1"
    # Second call returns same person
    pid2 = ps.get_or_create("Person 1")
    assert pid2 == "p1"
    # Different name creates new person
    pid3 = ps.get_or_create("Person 2")
    assert pid3 == "p2"


def test_vision_manager_event_dedup():
    from character_eng.vision.manager import VisionManager

    mgr = VisionManager()

    # Normalize lowers case and collapses whitespace only.
    assert mgr._normalize_event("A person is standing at the stand") == "a person is standing at the stand"
    assert mgr._normalize_event(" A person is   standing at the stand ") == "a person is standing at the stand"

    # Different events normalize differently
    assert mgr._normalize_event("Person walks away") != mgr._normalize_event("Person sits down")


def test_vision_manager_people_resolution():
    from character_eng.person import PeopleState
    from character_eng.vision.manager import VisionManager

    mgr = VisionManager()
    mgr._people = PeopleState()

    snap = RawVisualSnapshot(
        persons=[PersonObservation(identity="Person 1", bbox=(0, 0, 100, 200))],
    )
    mgr._resolve_people(snap)

    # Should create person and emit arrival event
    assert "Person 1" in mgr._known_persons
    assert len(mgr._events) == 1
    assert "appeared" in mgr._events[0].description
    assert mgr._events[0].kind == "person_presence"
    assert mgr._events[0].payload["signals"] == ["person_visible"]
    assert mgr._people.get_or_create("Person 1") == "p1"

    # Same person again — no new event
    mgr._resolve_people(snap)
    assert len(mgr._events) == 1

    # Person disappears
    empty_snap = RawVisualSnapshot()
    mgr._resolve_people(empty_snap)
    assert len(mgr._events) == 2
    assert "no longer visible" in mgr._events[1].description
    assert mgr._events[1].payload["signals"] == ["person_departed"]


def test_vision_manager_update_focus_pushes_dashboard_event(monkeypatch):
    from character_eng.vision.focus import VisualFocusResult
    from character_eng.vision.manager import VisionManager
    from character_eng.dashboard.events import DashboardEventCollector

    collector = DashboardEventCollector()
    mgr = VisionManager()
    mgr._collector = collector

    pushed_questions = {}
    pushed_targets = {}

    monkeypatch.setattr(
        "character_eng.vision.manager.visual_focus_call",
        lambda *args, **kwargs: VisualFocusResult(
            constant_questions=["Who is here?"],
            ephemeral_questions=["Are they holding a bottle?"],
            constant_sam_targets=["person"],
            ephemeral_sam_targets=["bottle"],
        ),
    )
    monkeypatch.setattr(mgr._client, "set_questions", lambda constant, ephemeral: pushed_questions.update({
        "constant": list(constant),
        "ephemeral": list(ephemeral),
    }))
    monkeypatch.setattr(mgr._client, "set_sam_targets", lambda constant, ephemeral: pushed_targets.update({
        "constant": list(constant),
        "ephemeral": list(ephemeral),
    }))

    mgr.update_focus(
        beat=None,
        stage_goal="Notice the bottle.",
        thought="Look for obvious props.",
        world=None,
        people=None,
        model_config={},
    )

    assert pushed_questions["constant"] == ["Who is here?"]
    assert pushed_questions["ephemeral"] == ["Are they holding a bottle?"]
    assert pushed_targets["constant"] == ["person"]
    assert pushed_targets["ephemeral"] == ["bottle"]
    event = collector.get_all()[-1]
    assert event["type"] == "vision_focus"
    assert event["data"]["ephemeral_sam_targets"] == ["bottle"]


def test_visual_focus_call_keeps_person_constants(monkeypatch):
    from character_eng.vision.focus import CORE_CONSTANT_QUESTIONS, CORE_CONSTANT_SAM_TARGETS, visual_focus_call

    class Message:
        content = json.dumps({
            "ephemeral_questions": ["Is the person gesturing toward anything?"],
            "ephemeral_sam_targets": ["phone"],
        })

    class Choice:
        message = Message()

    class Response:
        choices = [Choice()]

    monkeypatch.setattr("character_eng.world._llm_call", lambda *args, **kwargs: Response())

    result = visual_focus_call(
        beat=None,
        stage_goal="Notice the person.",
        thought="Look for behavior.",
        world=None,
        people=None,
        model_config={},
    )

    assert result.constant_questions == CORE_CONSTANT_QUESTIONS
    assert result.constant_sam_targets == CORE_CONSTANT_SAM_TARGETS
    assert result.ephemeral_questions == ["Is the person gesturing toward anything?"]
    assert result.ephemeral_sam_targets == ["phone"]


def test_vision_manager_dashboard_snapshot_includes_objects_and_focus(monkeypatch):
    from character_eng.vision.focus import VisualFocusResult
    from character_eng.vision.manager import VisionManager
    from character_eng.dashboard.events import DashboardEventCollector

    collector = DashboardEventCollector()
    mgr = VisionManager()
    mgr._collector = collector
    mgr._model_config = {"name": "test"}
    mgr._last_dashboard_push = 0.0
    mgr._last_focus = VisualFocusResult(
        constant_questions=["Who is here?"],
        ephemeral_questions=["Are they holding a bottle?"],
        constant_sam_targets=["person"],
        ephemeral_sam_targets=["bottle"],
    )

    snapshot = RawVisualSnapshot(
        persons=[PersonObservation(identity="Person 1", bbox=(0, 0, 100, 200))],
        objects=[ObjectDetection(label="bottle"), ObjectDetection(label="chair")],
        vlm_answers=[VLMAnswer(question="Who is here?", answer="One person near a chair", slot_type="constant")],
    )

    monkeypatch.setattr(mgr._client, "snapshot", lambda: snapshot)
    monkeypatch.setattr(
        "character_eng.vision.manager.vision_synthesis_call",
        lambda **kwargs: type("Synth", (), {"events": []})(),
    )

    mgr._tick()

    event = collector.get_all()[-1]
    assert event["type"] == "vision_snapshot"
    assert event["data"]["objects"] == 2
    assert event["data"]["object_labels"] == ["bottle", "chair"]
    assert event["data"]["vlm_answers"][0]["question"] == "Who is here?"
    assert event["data"]["focus"]["ephemeral_questions"] == ["Are they holding a bottle?"]


def test_vision_manager_carries_structured_synthesis_signals(monkeypatch):
    from character_eng.vision.manager import VisionManager

    mgr = VisionManager()
    mgr._model_config = {"name": "test"}
    snapshot = RawVisualSnapshot(
        persons=[PersonObservation(identity="Person 1", bbox=(0, 0, 100, 200))],
        vlm_answers=[
            VLMAnswer(
                question="Is anyone holding a water bottle?",
                answer="No, there does not appear to be anyone holding a water bottle.",
                slot_type="ephemeral",
            ),
        ],
    )

    monkeypatch.setattr(mgr._client, "snapshot", lambda: snapshot)
    monkeypatch.setattr(
        "character_eng.vision.manager.vision_synthesis_call",
        lambda **kwargs: type("Synth", (), {"events": ["Someone is holding a water bottle"], "signals": ["water_bottle_visible"]})(),
    )

    mgr._tick()

    drained = mgr.drain_events()
    synthesis_event = next(event for event in drained if event.description == "Someone is holding a water bottle")
    assert synthesis_event.kind == "visual_claim"
    assert synthesis_event.payload["signals"] == ["water_bottle_visible"]


def test_vision_manager_only_adds_synthesis_history_on_drain(monkeypatch):
    from character_eng.vision.manager import VisionManager

    mgr = VisionManager()
    mgr._model_config = {"name": "test"}
    snapshot = RawVisualSnapshot(
        persons=[PersonObservation(identity="Person 1", bbox=(0, 0, 100, 200))],
        vlm_answers=[
            VLMAnswer(
                question="Is anyone approaching the stand?",
                answer="Yes, one person is near the stand.",
                slot_type="ephemeral",
            ),
        ],
    )

    monkeypatch.setattr(mgr._client, "snapshot", lambda: snapshot)
    monkeypatch.setattr(
        "character_eng.vision.manager.vision_synthesis_call",
        lambda **kwargs: type("Synth", (), {"events": ["A person approaches the stand"], "signals": ["person_at_stand"]})(),
    )

    mgr._tick()

    assert list(mgr._event_history) == []
    drained = mgr.drain_events()
    descriptions = [event.description for event in drained]
    assert "A person approaches the stand" in descriptions
    assert "Person 1 appeared in view" in descriptions
    assert list(mgr._event_history) == descriptions


def test_disabled_character_is_hidden_from_selection():
    from character_eng.prompts import list_characters

    assert "mara" not in list_characters()
