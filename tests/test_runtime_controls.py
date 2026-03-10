"""Tests for runtime control helpers in the main loop."""

import queue
from unittest.mock import MagicMock

import character_eng.__main__ as app
from character_eng.config import AppConfig
from character_eng.person import PeopleState


def test_normalize_runtime_control_name_handles_aliases():
    assert app._normalize_runtime_control_name("auto-beat") == "auto_beat"
    assert app._normalize_runtime_control_name("autobeat") == "auto_beat"
    assert app._normalize_runtime_control_name("vision") == "vision"
    assert app._normalize_runtime_control_name("guardrail") == "guardrails"
    assert app._normalize_runtime_control_name("guardrails") == "guardrails"
    assert app._normalize_runtime_control_name("thought") == "thinker"
    assert app._normalize_runtime_control_name("guidance") == "beat_guidance"
    assert app._normalize_runtime_control_name("unknown") is None


def test_set_runtime_control_updates_voice_side_effects():
    previous = dict(app._runtime_controls)
    voice = MagicMock()
    try:
        app._set_runtime_control("filler", False, voice_io=voice)
        assert app._runtime_controls["filler"] is False
        voice.set_filler_enabled.assert_called_once_with(False)
        voice.cancel_latency_filler.assert_called_once()

        app._set_runtime_control("auto_beat", False, voice_io=voice)
        assert app._runtime_controls["auto_beat"] is False
        voice._cancel_auto_beat.assert_called_once()
        voice.cancel_and_drain_auto_beat.assert_called_once()
    finally:
        app._runtime_controls.update(previous)


def test_runtime_snapshot_includes_vision_service_metadata(monkeypatch):
    cfg = AppConfig()
    cfg.vision.service_url = "http://localhost:7860"
    cfg.vision.auto_launch = True
    monkeypatch.setattr(app, "_vision_service_health", lambda vision_cfg=None: True)
    monkeypatch.setitem(app._vision_runtime, "proc", None)
    monkeypatch.setitem(app._vision_runtime, "cfg", cfg.vision)
    monkeypatch.setitem(app._vision_runtime, "mock_replay", None)
    monkeypatch.setitem(app._vision_runtime, "no_camera", False)

    state = app._runtime_controls_snapshot(vision_cfg=cfg.vision)

    assert state["vision_service_url"] == "http://localhost:7860"
    assert state["vision_service_health"] is True
    assert state["vision_service_state"] == "external"
    assert state["vision_service_autostart"] is True
    assert state["conversation_paused"] is False
    assert state["voice_status"]["stt"]["state"] == "off"


def test_runtime_snapshot_includes_paused_state(monkeypatch):
    previous = app._conversation_paused
    cfg = AppConfig()
    monkeypatch.setattr(app, "_vision_service_health", lambda vision_cfg=None: False)
    monkeypatch.setitem(app._vision_runtime, "cfg", cfg.vision)
    try:
        app._conversation_paused = True
        state = app._runtime_controls_snapshot(vision_cfg=cfg.vision)
        assert state["conversation_paused"] is True
    finally:
        app._conversation_paused = previous


def test_runtime_snapshot_includes_voice_status(monkeypatch):
    cfg = AppConfig()
    monkeypatch.setattr(app, "_vision_service_health", lambda vision_cfg=None: False)
    monkeypatch.setitem(app._vision_runtime, "cfg", cfg.vision)
    voice = MagicMock()
    voice.status_snapshot.return_value = {
        "active": True,
        "output_only": False,
        "filler_enabled": True,
        "tts_backend": "pocket",
        "mic_ready": True,
        "speaker_ready": True,
        "stt": {"state": "ready", "detail": "Deepgram socket open."},
        "tts": {"state": "ready", "detail": "Pocket-TTS reachable.", "backend": "pocket"},
    }

    state = app._runtime_controls_snapshot(voice_io=voice, vision_cfg=cfg.vision)

    assert state["voice_active"] is True
    assert state["voice_status"]["stt"]["state"] == "ready"
    assert state["voice_status"]["tts"]["backend"] == "pocket"


def test_vision_service_autostart_command_updates_config(monkeypatch):
    cfg = AppConfig()
    saved = {}
    monkeypatch.setattr(app, "_app_config", cfg)
    monkeypatch.setitem(app._vision_runtime, "cfg", cfg.vision)

    def fake_save_config(updated):
        saved["auto_launch"] = updated.vision.auto_launch

    monkeypatch.setattr(app, "save_config", fake_save_config)

    handled, _ = app.handle_vision_service_command(
        "/vision-service autostart off",
        vision_mgr=None,
        vision_cfg=cfg.vision,
        model_config={"name": "test"},
        world=None,
        people=None,
        stage_goal="",
    )

    assert handled is True
    assert cfg.vision.auto_launch is False
    assert saved["auto_launch"] is False


def test_get_live_input_prefers_dashboard_queue(monkeypatch):
    previous = app._dashboard_input_queue
    dashboard_q = queue.Queue()
    dashboard_q.put("/restart")
    monkeypatch.setattr(app, "_dashboard_input_queue", dashboard_q)
    monkeypatch.setattr(app.console, "input", lambda prompt="": "")

    class FakeVoice:
        def wait_for_input(self, timeout=None):
            raise queue.Empty

    try:
        assert app._get_live_input("sess", FakeVoice()) == "/restart"
    finally:
        app._dashboard_input_queue = previous


def test_get_live_input_falls_back_to_voice(monkeypatch):
    previous = app._dashboard_input_queue
    monkeypatch.setattr(app, "_dashboard_input_queue", queue.Queue())
    monkeypatch.setattr(app.console, "input", lambda prompt="": "")

    class FakeVoice:
        def __init__(self):
            self.calls = 0

        def wait_for_input(self, timeout=None):
            self.calls += 1
            if self.calls < 2:
                raise queue.Empty
            return "/resume"

    try:
        assert app._get_live_input("sess", FakeVoice()) == "/resume"
    finally:
        app._dashboard_input_queue = previous


def test_first_visible_guardrails_only_fire_once_until_people_clear():
    previous = app._had_visible_people
    people = PeopleState()
    try:
        app._had_visible_people = False
        assert app._should_apply_first_visible_guardrails(people) is False

        people.add_person(name="Visitor", presence="present")
        assert app._should_apply_first_visible_guardrails(people) is True
        assert app._should_apply_first_visible_guardrails(people) is False

        people.people["p1"].presence = "gone"
        assert app._should_apply_first_visible_guardrails(people) is False

        people.people["p1"].presence = "present"
        assert app._should_apply_first_visible_guardrails(people) is True
    finally:
        app._had_visible_people = previous


def test_visual_turn_trigger_is_edge_based_and_cooldown_gated(monkeypatch):
    previous_seen = app._had_visible_people
    previous_at = app._last_visual_turn_at
    people = PeopleState()
    try:
        app._had_visible_people = False
        app._last_visual_turn_at = 0.0

        people.add_person(name="Visitor", presence="present")
        monkeypatch.setattr(app.time, "time", lambda: 100.0)
        assert app._should_trigger_visual_turn({"person_visible"}, people) is True
        assert app._should_trigger_visual_turn({"person_visible"}, people) is False

        monkeypatch.setattr(app.time, "time", lambda: 103.0)
        assert app._should_trigger_visual_turn({"rude_gesture"}, people) is False

        monkeypatch.setattr(app.time, "time", lambda: 108.0)
        assert app._should_trigger_visual_turn({"rude_gesture"}, people) is True
    finally:
        app._had_visible_people = previous_seen
        app._last_visual_turn_at = previous_at
