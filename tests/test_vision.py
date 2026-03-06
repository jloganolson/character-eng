"""Tests for vision module dataclasses and client."""

import json

from character_eng.vision.context import (
    FaceObservation,
    ObjectDetection,
    PersonObservation,
    RawVisualSnapshot,
    VisualContext,
    VLMAnswer,
)
from character_eng.vision.client import VisionClient


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

    # Normalize strips articles and common verbs
    assert mgr._normalize_event("A person is standing at the stand") == "person standing at stand"
    assert mgr._normalize_event("The person is standing at the stand") == "person standing at stand"
    assert mgr._normalize_event("A person is now standing at the stand") == "person standing at stand"

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
    assert mgr._people.get_or_create("Person 1") == "p1"

    # Same person again — no new event
    mgr._resolve_people(snap)
    assert len(mgr._events) == 1

    # Person disappears
    empty_snap = RawVisualSnapshot()
    mgr._resolve_people(empty_snap)
    assert len(mgr._events) == 2
    assert "no longer visible" in mgr._events[1].description


def test_mara_character_loads():
    """Verify the Mara character loads correctly."""
    from character_eng.prompts import load_prompt
    from character_eng.world import load_world_state, load_goals
    from character_eng.scenario import load_scenario_script
    from character_eng.perception import load_sim_script
    from character_eng.person import PeopleState

    world = load_world_state("mara")
    assert len(world.static) == 10
    assert len(world.dynamic) == 6

    goals = load_goals("mara")
    assert "see themselves" in goals.long_term.lower() or "insight" in goals.long_term.lower()

    scenario = load_scenario_script("mara")
    assert scenario is not None
    assert scenario.name == "Mystic Mara — Carnival Reading"
    assert len(scenario.stages) == 5
    assert scenario.current_stage == "idle"

    prompt = load_prompt("mara", world_state=world, people_state=PeopleState())
    assert "Mara" in prompt
    assert "fortune" in prompt.lower() or "carnival" in prompt.lower()

    sim = load_sim_script("mara", "curious")
    assert len(sim.events) == 15
