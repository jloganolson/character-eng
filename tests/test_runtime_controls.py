"""Tests for runtime control helpers in the main loop."""

import json
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


def test_vision_requires_external_frames_only_for_browser_and_webrtc_remote_hot():
    assert app._vision_requires_external_frames(browser_mode=True, transport_mode="local") is True
    assert app._vision_requires_external_frames(browser_mode=False, transport_mode="remote_hot_webrtc") is True
    assert app._vision_requires_external_frames(browser_mode=False, transport_mode="local") is False
    assert app._vision_requires_external_frames(browser_mode=False, transport_mode="remote_hot") is False


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


def test_runtime_snapshot_includes_ready_state(monkeypatch):
    previous_paused = app._conversation_paused
    previous_stopped = app._session_stopped
    previous_startup = app._startup_pause_pending
    cfg = AppConfig()
    monkeypatch.setattr(app, "_vision_service_health", lambda vision_cfg=None: False)
    monkeypatch.setitem(app._vision_runtime, "cfg", cfg.vision)
    try:
        app._conversation_paused = False
        app._session_stopped = True
        app._startup_pause_pending = False
        state = app._runtime_controls_snapshot(vision_cfg=cfg.vision)
        assert state["conversation_paused"] is False
        assert state["session_stopped"] is True
        assert state["startup_pause_pending"] is False
        assert state["session_state"] == "ready"
    finally:
        app._conversation_paused = previous_paused
        app._session_stopped = previous_stopped
        app._startup_pause_pending = previous_startup


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


def test_runtime_snapshot_includes_transport_metrics(tmp_path, monkeypatch):
    cfg = AppConfig()
    video_path = tmp_path / "video.json"
    audio_path = tmp_path / "audio.json"
    video_path.write_text(json.dumps({"avg_upload_ms": 88.1, "target_fps": 4.0}))
    audio_path.write_text(json.dumps({"avg_request_ms": 141.0, "avg_first_chunk_ms": 222.0}))
    monkeypatch.setenv("CHARACTER_ENG_TRANSPORT_MODE", "remote_hot")
    monkeypatch.setenv("CHARACTER_ENG_TRANSPORT_VIDEO_METRICS_PATH", str(video_path))
    monkeypatch.setenv("CHARACTER_ENG_TRANSPORT_AUDIO_METRICS_PATH", str(audio_path))
    monkeypatch.setattr(app, "_vision_service_health", lambda vision_cfg=None: False)
    monkeypatch.setitem(app._vision_runtime, "cfg", cfg.vision)

    state = app._runtime_controls_snapshot(vision_cfg=cfg.vision)

    assert state["transport"]["mode"] == "remote_hot"
    assert state["transport"]["video"]["avg_upload_ms"] == 88.1
    assert state["transport"]["audio"]["avg_first_chunk_ms"] == 222.0


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


def test_normalize_interrupted_reply_drops_tiny_fragments():
    assert app._normalize_interrupted_reply("N") == ""
    assert app._normalize_interrupted_reply("You're look") == ""
    assert app._normalize_interrupted_reply("Nice to meet you. What brings") == "Nice to meet you. What —"


def test_turn_guardrails_add_short_fragment_reference_bias():
    class DummySession:
        def __init__(self):
            self.messages = []

        def upsert_system(self, tag, content):
            self.messages.append((tag, content))

        def remove_tagged_system(self, tag):
            self.messages = [item for item in self.messages if item[0] != tag]

    session = DummySession()
    previous_controls = dict(app._runtime_controls)
    try:
        app._runtime_controls["guardrails"] = True
        app._inject_runtime_turn_guardrails(session, "outside world")
    finally:
        app._runtime_controls.clear()
        app._runtime_controls.update(previous_controls)

    runtime_note = next(content for tag, content in session.messages if tag == "runtime_turn_guardrails")
    assert "short fragment or repeated phrase" in runtime_note
    assert "immediately previous line or question" in runtime_note


def test_voice_trace_is_suppressed_until_live(monkeypatch):
    pushed = []
    previous_paused = app._conversation_paused
    previous_stopped = app._session_stopped

    class FakeVoice:
        def __init__(self, **kwargs):
            self.trace_hook = kwargs["trace_hook"]

    monkeypatch.setattr(app, "_push", lambda event_type, data: pushed.append((event_type, data)))
    try:
        app._session_stopped = True
        app._conversation_paused = False
        voice = app._create_voice_io(None, FakeVoice)
        voice.trace_hook("user_speech_started", {})
        assert pushed == []

        app._session_stopped = False
        app._conversation_paused = True
        voice.trace_hook("user_transcript_final", {"text": "hello"})
        assert pushed == []

        app._conversation_paused = False
        voice.trace_hook("user_transcript_final", {"text": "hello"})
        assert pushed == [("user_transcript_final", {"text": "hello"})]
    finally:
        app._conversation_paused = previous_paused
        app._session_stopped = previous_stopped
