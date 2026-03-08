"""Tests for runtime control helpers in the main loop."""

from unittest.mock import MagicMock

import character_eng.__main__ as app
from character_eng.config import AppConfig


def test_normalize_runtime_control_name_handles_aliases():
    assert app._normalize_runtime_control_name("auto-beat") == "auto_beat"
    assert app._normalize_runtime_control_name("autobeat") == "auto_beat"
    assert app._normalize_runtime_control_name("vision") == "vision"
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
