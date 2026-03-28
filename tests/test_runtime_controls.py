"""Tests for runtime control helpers in the main loop."""

import json
import queue
from unittest.mock import MagicMock

import character_eng.__main__ as app
from character_eng.config import AppConfig
from character_eng.person import PeopleState
from character_eng.rust_eyes_bridge import RustEyesBridgeError


def test_normalize_runtime_control_name_handles_aliases():
    assert app._normalize_runtime_control_name("auto-beat") == "auto_beat"
    assert app._normalize_runtime_control_name("autobeat") == "auto_beat"
    assert app._normalize_runtime_control_name("vision") == "vision"
    assert app._normalize_runtime_control_name("guardrail") == "guardrails"
    assert app._normalize_runtime_control_name("guardrails") == "guardrails"
    assert app._normalize_runtime_control_name("thought") == "thinker"
    assert app._normalize_runtime_control_name("guidance") == "beat_guidance"
    assert app._normalize_runtime_control_name("unknown") is None


def test_vision_requires_external_frames_only_for_browser_and_webrtc():
    assert app._vision_requires_external_frames(browser_mode=True, transport_mode="local") is True
    assert app._vision_requires_external_frames(browser_mode=False, transport_mode="remote_hot_webrtc") is True
    assert app._vision_requires_external_frames(browser_mode=False, transport_mode="local") is False


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
    previous_phase = app._runtime_phase
    cfg = AppConfig()
    monkeypatch.setattr(app, "_vision_service_health", lambda vision_cfg=None: False)
    monkeypatch.setitem(app._vision_runtime, "cfg", cfg.vision)
    try:
        app._set_runtime_phase(app.RuntimePhase.READY)
        state = app._runtime_controls_snapshot(vision_cfg=cfg.vision)
        assert state["conversation_paused"] is False
        assert state["session_stopped"] is True
        assert state["startup_pause_pending"] is False
        assert state["session_state"] == "ready"
        assert state["runtime_phase"] == "ready"
    finally:
        app._set_runtime_phase(previous_phase)


def test_runtime_snapshot_exposes_startup_paused_phase(monkeypatch):
    previous_phase = app._runtime_phase
    cfg = AppConfig()
    monkeypatch.setattr(app, "_vision_service_health", lambda vision_cfg=None: False)
    monkeypatch.setitem(app._vision_runtime, "cfg", cfg.vision)
    try:
        app._set_runtime_phase(app.RuntimePhase.STARTUP_PAUSED)
        state = app._runtime_controls_snapshot(vision_cfg=cfg.vision)
        assert state["conversation_paused"] is True
        assert state["session_stopped"] is False
        assert state["startup_pause_pending"] is True
        assert state["session_state"] == "paused"
        assert state["runtime_phase"] == "startup_paused"
    finally:
        app._set_runtime_phase(previous_phase)


def test_runtime_snapshot_exposes_review_phase(monkeypatch):
    previous_phase = app._runtime_phase
    cfg = AppConfig()
    monkeypatch.setattr(app, "_vision_service_health", lambda vision_cfg=None: False)
    monkeypatch.setitem(app._vision_runtime, "cfg", cfg.vision)
    try:
        app._set_runtime_phase(app.RuntimePhase.REVIEW)
        state = app._runtime_controls_snapshot(vision_cfg=cfg.vision)
        assert state["conversation_paused"] is True
        assert state["session_stopped"] is False
        assert state["startup_pause_pending"] is False
        assert state["session_state"] == "paused"
        assert state["runtime_phase"] == "review"
    finally:
        app._set_runtime_phase(previous_phase)


def test_runtime_phase_should_pause_vision_for_preplay_and_review_states():
    assert app._runtime_phase_should_pause_vision(app.RuntimePhase.READY) is True
    assert app._runtime_phase_should_pause_vision(app.RuntimePhase.STARTUP_PAUSED) is True
    assert app._runtime_phase_should_pause_vision(app.RuntimePhase.PAUSED) is True
    assert app._runtime_phase_should_pause_vision(app.RuntimePhase.REVIEW) is True
    assert app._runtime_phase_should_pause_vision(app.RuntimePhase.LIVE) is False


def test_sync_vision_pause_state_uses_runtime_phase_policy():
    vision_mgr = MagicMock()

    app._sync_vision_pause_state(vision_mgr, app.RuntimePhase.READY)
    vision_mgr.set_paused.assert_called_with(True)

    app._sync_vision_pause_state(vision_mgr, app.RuntimePhase.LIVE)
    vision_mgr.set_paused.assert_called_with(False)


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
    video_path.write_text(json.dumps({"video_frames_sent": 54, "last_frame_bytes": 20480}))
    audio_path.write_text(json.dumps({"audio_frames_in": 17, "audio_frames_out": 9}))
    monkeypatch.setenv("CHARACTER_ENG_TRANSPORT_MODE", "remote_hot_webrtc")
    monkeypatch.setenv("CHARACTER_ENG_TRANSPORT_VIDEO_METRICS_PATH", str(video_path))
    monkeypatch.setenv("CHARACTER_ENG_TRANSPORT_AUDIO_METRICS_PATH", str(audio_path))
    monkeypatch.setattr(app, "_vision_service_health", lambda vision_cfg=None: False)
    monkeypatch.setitem(app._vision_runtime, "cfg", cfg.vision)

    state = app._runtime_controls_snapshot(vision_cfg=cfg.vision)

    assert state["transport"]["mode"] == "remote_hot_webrtc"
    assert state["transport"]["video"]["video_frames_sent"] == 54
    assert state["transport"]["audio"]["audio_frames_out"] == 9


def test_sync_robot_sim_expression_drives_rust_eyes_bridge(monkeypatch):
    previous_bridge = app._rust_eyes_bridge
    previous_status = dict(app._rust_eyes_bridge_status)
    pushed = []

    class FakeBridge:
        def push_face(self, update):
            pushed.append(update)
            return {"ok": True, "expression_key": "curious"}

    monkeypatch.setattr(app, "_rust_eyes_bridge", FakeBridge())

    try:
        state = app._sync_robot_sim_expression(
            gaze="person",
            gaze_type="hold",
            expression="curious",
            source="test",
        )
    finally:
        app._rust_eyes_bridge = previous_bridge
        app._rust_eyes_bridge_status = previous_status

    assert state["face"]["expression"] == "curious"
    assert state["rust_eyes"]["enabled"] is True
    assert state["rust_eyes"]["state"] == "ok"
    assert state["rust_eyes"]["expression_key"] == "curious"
    assert len(pushed) == 1
    assert pushed[0].gaze == "person"
    assert pushed[0].expression == "curious"


def test_sync_robot_sim_expression_reports_rust_eyes_bridge_error(monkeypatch):
    previous_bridge = app._rust_eyes_bridge
    previous_status = dict(app._rust_eyes_bridge_status)
    notices = []

    class FakeBridge:
        def push_face(self, update):
            raise RustEyesBridgeError("no rust-eyes expression mapping for 'happy'; update /tmp/map.toml")

    monkeypatch.setattr(app, "_rust_eyes_bridge", FakeBridge())
    monkeypatch.setattr(app, "_push", lambda event_type, data: notices.append((event_type, data)))
    monkeypatch.setattr(app.console, "print", lambda *args, **kwargs: None)

    try:
        app._sync_robot_sim_expression(
            gaze="scene",
            gaze_type="hold",
            expression="happy",
            source="test",
        )
    finally:
        app._rust_eyes_bridge = previous_bridge
        app._rust_eyes_bridge_status = previous_status

    assert notices
    event_type, payload = next(
        (event_type, data) for event_type, data in notices if event_type == "robot_preview_bridge"
    )
    assert event_type == "robot_preview_bridge"
    assert payload["status"] == "error"
    assert payload["target"] == "rust_eyes"
    assert "update /tmp/map.toml" in payload["message"]


def test_dashboard_robot_sim_state_payload_includes_optional_rust_eyes_status(monkeypatch):
    previous_status = dict(app._rust_eyes_bridge_status)
    monkeypatch.setattr(app, "_rust_eyes_bridge_status", {
        "enabled": True,
        "state": "configured",
        "message": "http://127.0.0.1:5555",
        "expression_key": "",
        "updated_at": 123.0,
    })

    try:
        payload = app._dashboard_robot_sim_state_payload()
    finally:
        app._rust_eyes_bridge_status = previous_status

    assert payload["rust_eyes"]["enabled"] is True
    assert payload["rust_eyes"]["state"] == "configured"


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


def test_apply_expression_runtime_context_noops_for_prompt_history():
    class DummySession:
        def __init__(self):
            self.messages = []

        def upsert_system(self, tag, content):
            self.messages.append((tag, content))

        def remove_tagged_system(self, tag):
            self.messages = [item for item in self.messages if item[0] != tag]

    session = DummySession()
    expr = app.ExpressionResult(gaze="Person 1", gaze_type="hold", expression="curious")

    app._apply_expression_runtime_context(session, expr)

    assert session.messages == []


def test_voice_trace_is_suppressed_until_live(monkeypatch):
    pushed = []
    previous_phase = app._runtime_phase

    class FakeVoice:
        def __init__(self, **kwargs):
            self.trace_hook = kwargs["trace_hook"]

    monkeypatch.setattr(app, "_push", lambda event_type, data: pushed.append((event_type, data)))
    try:
        app._set_runtime_phase(app.RuntimePhase.READY)
        voice = app._create_voice_io(None, FakeVoice)
        voice.trace_hook("user_speech_started", {})
        assert pushed == []

        app._set_runtime_phase(app.RuntimePhase.PAUSED)
        voice.trace_hook("user_transcript_final", {"text": "hello"})
        assert pushed == []

        app._set_runtime_phase(app.RuntimePhase.LIVE)
        voice.trace_hook("user_transcript_final", {"text": "hello"})
        assert pushed == [("user_transcript_final", {"text": "hello"})]
    finally:
        app._set_runtime_phase(previous_phase)
