"""Tests for config load/save helpers."""

from character_eng import config as config_mod
from character_eng.config import AppConfig, load_config, save_config


def test_save_config_round_trip(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    cfg = AppConfig()
    cfg.voice.enabled = True
    cfg.voice.input_device = "Mic"
    cfg.voice.output_device = 4
    cfg.voice.tts_backend = "pocket"
    cfg.voice.filler_enabled = False
    cfg.voice.filler_lead_ms = 320
    cfg.vision.enabled = True
    cfg.vision.auto_launch = False
    cfg.vision.service_url = "http://127.0.0.1:9999"
    cfg.dashboard.port = 9000
    cfg.dashboard.prompt_asset_open_target = "folder"
    cfg.dashboard.prompt_asset_vscode_cmd = "cursor"
    cfg.bridge.enabled = True
    cfg.bridge.token = "secret-token"
    cfg.history.enabled = True
    cfg.history.free_warning_gib = 12.5
    cfg.history.vision_capture_fps = 4.0

    save_config(cfg, path=path)
    monkeypatch.setenv("CHARACTER_ENG_CONFIG_PATH", str(path))

    loaded = load_config()

    assert loaded.voice.enabled is True
    assert loaded.voice.input_device == "Mic"
    assert loaded.voice.output_device == 4
    assert loaded.voice.tts_backend == "pocket"
    assert loaded.voice.filler_enabled is False
    assert loaded.voice.filler_lead_ms == 320
    assert loaded.vision.enabled is True
    assert loaded.vision.auto_launch is False
    assert loaded.vision.service_url == "http://127.0.0.1:9999"
    assert loaded.dashboard.port == 9000
    assert loaded.dashboard.prompt_asset_open_target == "folder"
    assert loaded.dashboard.prompt_asset_vscode_cmd == "cursor"
    assert loaded.bridge.enabled is True
    assert loaded.bridge.token == "secret-token"
    assert loaded.history.enabled is True
    assert loaded.history.free_warning_gib == 12.5
    assert loaded.history.vision_capture_fps == 4.0


def test_load_config_env_overrides(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    save_config(AppConfig(), path=path)
    monkeypatch.setenv("CHARACTER_ENG_CONFIG_PATH", str(path))
    monkeypatch.setenv("CHARACTER_ENG_TTS_BACKEND", "pocket")
    monkeypatch.setenv("CHARACTER_ENG_TTS_SERVER_URL", "http://127.0.0.1:9800")
    monkeypatch.setenv("CHARACTER_ENG_VISION_URL", "http://127.0.0.1:9001")
    monkeypatch.setenv("CHARACTER_ENG_VISION_PORT", "9001")
    monkeypatch.setenv("CHARACTER_ENG_VISION_AUTO_LAUNCH", "false")
    monkeypatch.setenv("CHARACTER_ENG_DASHBOARD_PORT", "9010")
    monkeypatch.setenv("CHARACTER_ENG_BRIDGE_ENABLED", "true")
    monkeypatch.setenv("CHARACTER_ENG_BRIDGE_PORT", "9011")
    monkeypatch.setenv("CHARACTER_ENG_BRIDGE_TOKEN", "bridge-secret")

    loaded = load_config()

    assert loaded.voice.tts_backend == "pocket"
    assert loaded.voice.tts_server_url == "http://127.0.0.1:9800"
    assert loaded.vision.service_url == "http://127.0.0.1:9001"
    assert loaded.vision.service_port == 9001
    assert loaded.vision.auto_launch is False
    assert loaded.dashboard.port == 9010
    assert loaded.bridge.enabled is True
    assert loaded.bridge.port == 9011
    assert loaded.bridge.token == "bridge-secret"
