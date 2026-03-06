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
