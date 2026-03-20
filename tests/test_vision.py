"""Tests for vision module dataclasses and client."""

import threading
import json
import socket
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from character_eng.history import HistoryService
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
                "answer_id": "vlm-1",
            }
        ],
        "cycle_id": "vision-cycle-1",
        "timestamp": 1234567890.0,
        "trace": {"cycle_id": "vision-cycle-1"},
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
    assert snap.vlm_answers[0].answer_id == "vlm-1"
    assert snap.cycle_id == "vision-cycle-1"
    assert snap.trace["cycle_id"] == "vision-cycle-1"


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


def test_vision_client_set_questions_appends_suffix():
    received = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/set_questions":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            received["body"] = json.loads(self.rfile.read(length))
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, format, *args):
            pass

    port = _free_port()
    server = HTTPServer(("127.0.0.1", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = VisionClient(f"http://127.0.0.1:{port}")
        client.set_questions(
            ["What stands out about the nearest person's appearance?"],
            ["What notable object are they clearly using? (1-2 sentences max)"],
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert received["body"]["constant"][0]["question"] == "What stands out about the nearest person's appearance? (1-2 sentences max)"
    assert received["body"]["constant"][0]["task_id"] == "what_stands_out_about_the_nearest_person_s_appearance"
    assert received["body"]["ephemeral"][0]["question"] == "What notable object are they clearly using? (1-2 sentences max)"


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
    assert vc.raw_poll_interval == 0.2
    assert vc.synthesis_min_interval == 0.75

    cfg = AppConfig()
    assert cfg.vision.enabled is False


def test_person_get_or_create():
    from character_eng.person import PeopleState
    from character_eng.world import PersonUpdate

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
    assert mgr._events[0].description == "A person appeared in view"
    assert mgr._events[0].kind == "person_presence"
    assert mgr._events[0].payload["signals"] == ["person_visible"]
    assert mgr._events[0].trace["provenance"]["label"] == "Presence tracker"
    assert mgr._events[0].trace["provenance"]["primary"] == "person_tracker"
    assert mgr._people.get_or_create("Person 1") == "p1"

    # Same person again — no new event
    mgr._resolve_people(snap)
    assert len(mgr._events) == 1

    # Identity churn while a person is still visible should not produce fake leave/re-enter events.
    churned_snap = RawVisualSnapshot(
        persons=[PersonObservation(identity="Person 2", bbox=(0, 0, 100, 200))],
    )
    mgr._resolve_people(churned_snap)
    assert len(mgr._events) == 1

    # Person disappears
    empty_snap = RawVisualSnapshot()
    mgr._resolve_people(empty_snap)
    assert len(mgr._events) == 2
    assert mgr._events[1].description == "A person is no longer visible"
    assert mgr._events[1].payload["signals"] == ["person_departed"]
    assert mgr._events[1].trace["provenance"]["label"] == "Presence tracker"


def test_vision_manager_presence_uses_face_or_sam_person_backup():
    from character_eng.vision.context import FaceObservation, ObjectDetection
    from character_eng.vision.manager import VisionManager

    mgr = VisionManager()

    face_only = RawVisualSnapshot(
        faces=[FaceObservation(identity="Visitor", bbox=(0, 0, 100, 100))],
    )
    mgr._resolve_people(face_only)
    assert len(mgr._events) == 1
    assert mgr._events[0].payload["signals"] == ["person_visible"]

    sam_person_only = RawVisualSnapshot(
        objects=[ObjectDetection(label="person", bbox=(0, 0, 50, 50), confidence=0.9)],
    )
    mgr._resolve_people(sam_person_only)
    assert len(mgr._events) == 1


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
            constant_sam_targets=["person", "rude gesture"],
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
    assert pushed_targets["constant"] == ["person", "rude gesture"]
    assert pushed_targets["ephemeral"] == ["bottle"]
    event = collector.get_all()[-1]
    assert event["type"] == "vision_focus"
    assert event["data"]["ephemeral_sam_targets"] == ["bottle"]


def test_visual_focus_call_keeps_person_constants(monkeypatch):
    from character_eng.vision.focus import CORE_CONSTANT_QUESTIONS, CORE_CONSTANT_SAM_TARGETS, CORE_CONSTANT_VLM_SPECS, visual_focus_call

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
    assert [spec.task_id for spec in result.constant_vlm_specs] == [spec.task_id for spec in CORE_CONSTANT_VLM_SPECS]
    assert result.ephemeral_vlm_specs[0].task_id.startswith("ephemeral_")


def test_visual_focus_call_includes_scenario_authored_constants(monkeypatch):
    from types import SimpleNamespace
    from character_eng.vision.focus import CORE_CONSTANT_QUESTIONS, CORE_CONSTANT_SAM_TARGETS, visual_focus_call

    class Message:
        content = json.dumps({
            "ephemeral_questions": [],
            "ephemeral_sam_targets": [],
        })

    class Choice:
        message = Message()

    class Response:
        choices = [Choice()]

    monkeypatch.setattr("character_eng.world._llm_call", lambda *args, **kwargs: Response())

    scenario = SimpleNamespace(
        active_visual_requirements=lambda: SimpleNamespace(
            constant_questions=["Is there a visible clock, watch, phone, or screen that could tell Greg the time?"],
            constant_sam_targets=["phone", "clock", "screen"],
        )
    )

    result = visual_focus_call(
        beat=None,
        stage_goal="Find the time.",
        thought="Look for a time source.",
        world=None,
        people=None,
        model_config={},
        scenario=scenario,
    )

    assert result.constant_questions == CORE_CONSTANT_QUESTIONS + [
        "Is there a visible clock, watch, phone, or screen that could tell Greg the time?"
    ]
    assert result.constant_sam_targets == CORE_CONSTANT_SAM_TARGETS + ["phone", "clock", "screen"]


def test_visual_focus_call_merges_stage_specific_constants(monkeypatch):
    from types import SimpleNamespace
    from character_eng.vision.focus import visual_focus_call

    class Message:
        content = json.dumps({
            "ephemeral_questions": [],
            "ephemeral_sam_targets": [],
        })

    class Choice:
        message = Message()

    class Response:
        choices = [Choice()]

    monkeypatch.setattr("character_eng.world._llm_call", lambda *args, **kwargs: Response())

    scenario = SimpleNamespace(
        active_visual_requirements=lambda: SimpleNamespace(
            constant_questions=["Is there a visible clock?", "If the person is showing Greg a phone, what time does it show?"],
            constant_sam_targets=["clock", "phone"],
        )
    )

    result = visual_focus_call(
        beat=None,
        stage_goal="Find the time.",
        thought="",
        world=None,
        people=None,
        model_config={},
        scenario=scenario,
    )

    assert "Is there a visible clock?" in result.constant_questions
    assert "If the person is showing Greg a phone, what time does it show?" in result.constant_questions
    assert "clock" in result.constant_sam_targets
    assert "phone" in result.constant_sam_targets


def test_vision_manager_no_longer_emits_dashboard_snapshot_rollup(monkeypatch):
    from character_eng.vision.focus import VisualFocusResult
    from character_eng.vision.manager import VisionManager
    from character_eng.dashboard.events import DashboardEventCollector

    collector = DashboardEventCollector()
    mgr = VisionManager()
    mgr._collector = collector
    mgr._model_config = {"name": "test"}
    mgr._last_focus = VisualFocusResult(
        constant_questions=["Who is here?"],
        ephemeral_questions=["Are they holding a bottle?"],
        constant_sam_targets=["person", "rude gesture"],
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

    event_types = [event["type"] for event in collector.get_all()]
    assert "vision_snapshot" not in event_types
    assert "vision_snapshot_read" in event_types
    assert "sam3_detection" in event_types


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
                answer_id="vlm-2",
            ),
        ],
        cycle_id="vision-cycle-9",
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
    assert synthesis_event.trace["provenance"]["label"] == "Vision synthesis LLM"
    assert synthesis_event.trace["provenance"]["primary"] == "vision_synthesis_llm"
    assert "person_tracker" in synthesis_event.trace["provenance"]["components"]
    assert "vlm_answers" in synthesis_event.trace["provenance"]["components"]
    assert synthesis_event.payload["input_event_ids"] == ["vision-cycle-9:snapshot", "vision-cycle-9:sam3", "vlm-2"]


def test_vision_manager_emits_raw_dashboard_stage_events(monkeypatch):
    from character_eng.vision.manager import VisionManager
    from character_eng.dashboard.events import DashboardEventCollector

    collector = DashboardEventCollector()
    mgr = VisionManager()
    mgr._collector = collector
    mgr._model_config = {"name": "test"}
    snapshot = RawVisualSnapshot(
        faces=[FaceObservation(identity="Visitor", bbox=(1, 2, 30, 40), confidence=0.91, gaze_direction="left")],
        persons=[PersonObservation(identity="Visitor", bbox=(0, 0, 100, 200), confidence=0.8)],
        objects=[ObjectDetection(label="bottle", bbox=(50, 60, 30, 40), confidence=0.7)],
        vlm_answers=[
            VLMAnswer(
                question="Is the visitor holding a bottle?",
                answer="Yes.",
                elapsed=0.42,
                slot_type="ephemeral",
                timestamp=123.0,
                answer_id="vlm-3",
            ),
        ],
        cycle_id="vision-cycle-42",
        trace={
            "face_tracking": {
                "timing": {"total": 0.11, "fps": 9.1},
                "faces": [{"identity": "Visitor", "bbox": [1, 2, 30, 40], "confidence": 0.91, "gaze_direction": "left", "track_id": 7}],
            },
            "sam3_detection": {
                "timing": {"sam3": 0.28, "total": 0.36, "prompts": "person, bottle"},
                "persons": [{"identity": "Visitor", "bbox": [0, 0, 100, 200], "confidence": 0.8, "track_id": 7}],
                "objects": [{"label": "bottle", "bbox": [50, 60, 30, 40], "confidence": 0.7}],
            },
            "reid_tracking": {
                "timing": {"reid": 0.05, "total": 0.36},
                "persons": [{"identity": "Visitor", "bbox": [0, 0, 100, 200], "confidence": 0.8, "track_id": 7, "identity_source": "reid"}],
            },
        },
    )

    monkeypatch.setattr(mgr._client, "snapshot", lambda: snapshot)
    monkeypatch.setattr(
        "character_eng.vision.manager.vision_synthesis_call",
        lambda **kwargs: type("Synth", (), {"events": ["Visitor is holding a bottle"], "signals": ["bottle_visible"]})(),
    )

    mgr._tick()

    events = collector.get_all()
    event_types = [event["type"] for event in events]
    assert "vision_snapshot_read" in event_types
    assert "face_tracking" in event_types
    assert "sam3_detection" in event_types
    assert "reid_track" in event_types
    assert "vlm_answer" in event_types
    reid_event = next(event for event in events if event["type"] == "reid_track")
    assert reid_event["data"]["input_event_ids"] == ["vision-cycle-42:snapshot", "vision-cycle-42:sam3"]
    drained = mgr.drain_events()
    claim = next(event for event in drained if event.kind == "visual_claim")
    assert claim.payload["input_event_ids"] == [
        "vision-cycle-42:snapshot",
        "vision-cycle-42:face",
        "vision-cycle-42:sam3",
        "vision-cycle-42:reid:7",
        "vlm-3",
    ]


def test_vision_manager_emits_raw_events_to_history_when_runtime_emitter_is_present(monkeypatch, tmp_path):
    from character_eng.dashboard.events import DashboardEventCollector
    from character_eng.vision.manager import VisionManager

    collector = DashboardEventCollector()
    history = HistoryService(root=tmp_path, free_warning_gib=0.0)
    history.start_session(session_id="sess-vision-history", character="greg", model="test")

    def runtime_emit(event_type: str, data: dict) -> None:
        history.record_event(event_type, data)
        collector.push(event_type, data)

    mgr = VisionManager()
    mgr._collector = collector
    mgr._emitter = runtime_emit
    mgr._model_config = {"name": "test"}
    snapshot = RawVisualSnapshot(
        faces=[FaceObservation(identity="Visitor", bbox=(1, 2, 30, 40), confidence=0.91, gaze_direction="left")],
        persons=[PersonObservation(identity="Visitor", bbox=(0, 0, 100, 200), confidence=0.8)],
        objects=[ObjectDetection(label="bottle", bbox=(50, 60, 30, 40), confidence=0.7)],
        vlm_answers=[
            VLMAnswer(
                question="Is the visitor holding a bottle?",
                answer="Yes.",
                elapsed=0.42,
                slot_type="ephemeral",
                timestamp=123.0,
                answer_id="vlm-3",
            ),
        ],
        cycle_id="vision-cycle-history",
        trace={
            "face_tracking": {
                "timing": {"total": 0.11, "fps": 9.1},
                "faces": [{"identity": "Visitor", "bbox": [1, 2, 30, 40], "confidence": 0.91, "gaze_direction": "left", "track_id": 7}],
            },
            "sam3_detection": {
                "timing": {"sam3": 0.28, "total": 0.36, "prompts": "person, bottle"},
                "persons": [{"identity": "Visitor", "bbox": [0, 0, 100, 200], "confidence": 0.8, "track_id": 7}],
                "objects": [{"label": "bottle", "bbox": [50, 60, 30, 40], "confidence": 0.7}],
            },
            "reid_tracking": {
                "timing": {"reid": 0.05, "total": 0.36},
                "persons": [{"identity": "Visitor", "bbox": [0, 0, 100, 200], "confidence": 0.8, "track_id": 7, "identity_source": "reid"}],
            },
        },
    )

    monkeypatch.setattr(mgr._client, "snapshot", lambda: snapshot)
    monkeypatch.setattr(
        "character_eng.vision.manager.vision_synthesis_call",
        lambda **kwargs: type("Synth", (), {"events": [], "signals": []})(),
    )

    mgr._tick()

    history_events = history.list_events("sess-vision-history")["events"]
    history_types = [event["type"] for event in history_events]
    assert "vision_snapshot_read" in history_types
    assert "face_tracking" in history_types
    assert "sam3_detection" in history_types
    assert "reid_track" in history_types
    assert "vlm_answer" in history_types
    collector_types = [event["type"] for event in collector.get_all()]
    assert collector_types == history_types


def test_vision_manager_raw_poll_is_faster_than_synthesis(monkeypatch):
    from character_eng.dashboard.events import DashboardEventCollector
    from character_eng.vision.manager import VisionManager

    collector = DashboardEventCollector()
    mgr = VisionManager(min_interval=0.75, raw_poll_interval=0.2)
    mgr._collector = collector
    mgr._model_config = {"name": "test"}
    snapshots = iter([
        RawVisualSnapshot(
            persons=[PersonObservation(identity="Visitor", bbox=(0, 0, 100, 200), confidence=0.8)],
            cycle_id="vision-cycle-fast-1",
        ),
        RawVisualSnapshot(
            persons=[PersonObservation(identity="Visitor", bbox=(0, 0, 100, 200), confidence=0.8)],
            cycle_id="vision-cycle-fast-2",
        ),
    ])
    synth_calls: list[str] = []

    monkeypatch.setattr(mgr._client, "snapshot", lambda: next(snapshots))
    monkeypatch.setattr(
        "character_eng.vision.manager.vision_synthesis_call",
        lambda **kwargs: synth_calls.append(kwargs["visual_context"].snapshot.cycle_id) or type("Synth", (), {"events": [], "signals": []})(),
    )

    mgr._tick()
    mgr._tick()

    events = collector.get_all()
    snapshot_reads = [event for event in events if event["type"] == "vision_snapshot_read"]
    sam3_events = [event for event in events if event["type"] == "sam3_detection"]
    assert [event["data"]["vision_cycle_id"] for event in snapshot_reads] == ["vision-cycle-fast-1", "vision-cycle-fast-2"]
    assert [event["data"]["vision_cycle_id"] for event in sam3_events] == ["vision-cycle-fast-1", "vision-cycle-fast-2"]
    assert synth_calls == ["vision-cycle-fast-1"]


def test_vision_manager_emits_stage_trigger_events(monkeypatch):
    from character_eng.dashboard.events import DashboardEventCollector
    from character_eng.scenario import ScenarioScript, Stage, VisionTrigger
    from character_eng.vision.manager import VisionManager

    collector = DashboardEventCollector()
    mgr = VisionManager()
    mgr._collector = collector
    mgr._model_config = {"name": "test"}
    mgr.update_context(
        scenario=ScenarioScript(
            name="test",
            start="watching",
            current_stage="watching",
            stages={
                "watching": Stage(
                    "watching",
                    "Look for a clock",
                    vision_triggers=[
                        VisionTrigger(signal="clock_visible", source="sam3", label="clock", frames=2),
                    ],
                ),
            },
        )
    )
    snapshot = RawVisualSnapshot(
        objects=[ObjectDetection(label="clock", bbox=(10, 20, 30, 40), confidence=0.9)],
        cycle_id="vision-cycle-77",
    )

    monkeypatch.setattr(mgr._client, "snapshot", lambda: snapshot)
    monkeypatch.setattr(
        "character_eng.vision.manager.vision_synthesis_call",
        lambda **kwargs: type("Synth", (), {"events": [], "signals": []})(),
    )

    mgr._tick()
    assert not any(event.kind == "vision_trigger" for event in mgr.drain_events())

    mgr._tick()
    drained = mgr.drain_events()
    trigger = next(event for event in drained if event.kind == "vision_trigger")
    assert trigger.payload["signals"] == ["clock_visible"]
    assert trigger.trace["trigger"]["source"] == "sam3"
    raw_event = next(event for event in collector.get_all() if event["type"] == "vision_trigger")
    assert raw_event["data"]["signal"] == "clock_visible"


def test_vision_manager_emits_direct_state_updates_from_system_tasks(monkeypatch):
    from character_eng.dashboard.events import DashboardEventCollector
    from character_eng.person import PeopleState
    from character_eng.vision.interpret import VisionStateUpdateResult
    from character_eng.vision.manager import VisionManager
    from character_eng.world import PersonUpdate, WorldUpdate

    collector = DashboardEventCollector()
    mgr = VisionManager()
    mgr._collector = collector
    mgr._model_config = {"name": "test"}
    mgr._people = PeopleState()
    person = mgr._people.add_person(name="Visitor", presence="present")
    mgr._visual_to_person["Visitor"] = person.person_id
    snapshot = RawVisualSnapshot(
        persons=[PersonObservation(identity="Visitor", bbox=(0, 0, 100, 200))],
        vlm_answers=[
            VLMAnswer(
                task_id="person_description_static",
                label="Person Description Static",
                question="Describe the nearest visible person's appearance.",
                answer="The visitor is wearing a blue jacket.",
                interpret_as="person_description_static",
                target="nearest_person",
                target_identity="Visitor",
                answer_id="vlm-77",
            ),
        ],
        cycle_id="vision-cycle-88",
    )

    monkeypatch.setattr(mgr._client, "snapshot", lambda: snapshot)
    monkeypatch.setattr(
        "character_eng.vision.manager.vision_synthesis_call",
        lambda **kwargs: type("Synth", (), {"events": [], "signals": []})(),
    )
    monkeypatch.setattr(
        "character_eng.vision.manager.vision_state_update_call",
        lambda **kwargs: VisionStateUpdateResult(
            update=WorldUpdate(
                person_updates=[
                    PersonUpdate(
                        person_id=person.person_id,
                        add_facts=["Wearing a blue jacket"],
                    )
                ],
            ),
            summary="Visitor is wearing a blue jacket",
        ),
    )

    mgr._tick()

    drained = mgr.drain_events()
    update_event = next(event for event in drained if event.kind == "vision_state_update")
    assert update_event.payload["person_updates"][0]["person_id"] == person.person_id
    assert update_event.payload["task_answers"][0]["target_person_id"] == person.person_id
    raw_event = next(event for event in collector.get_all() if event["type"] == "vision_state_update")
    assert raw_event["data"]["summary"] == "Visitor is wearing a blue jacket"


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
    assert "A person appeared in view" in descriptions
    assert list(mgr._event_history) == descriptions


def test_disabled_character_is_hidden_from_selection():
    from character_eng.prompts import list_characters

    assert "mara" not in list_characters()
