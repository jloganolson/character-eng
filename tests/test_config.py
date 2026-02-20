"""Tests for character_eng.config."""

from pathlib import Path
from unittest.mock import patch

from character_eng.config import AppConfig, VoiceConfig, load_config


def test_missing_file_returns_defaults():
    """No config.toml → all defaults."""
    with patch("character_eng.config.CONFIG_PATH", Path("/nonexistent/config.toml")):
        cfg = load_config()
    assert isinstance(cfg, AppConfig)
    assert cfg.voice.input_device is None
    assert cfg.voice.output_device is None
    assert cfg.voice.enabled is False


def test_full_config(tmp_path):
    """All fields present → all parsed."""
    toml = tmp_path / "config.toml"
    toml.write_text(
        "[voice]\n"
        "input_device = 14\n"
        "output_device = 7\n"
        "enabled = true\n"
    )
    with patch("character_eng.config.CONFIG_PATH", toml):
        cfg = load_config()
    assert cfg.voice.input_device == 14
    assert cfg.voice.output_device == 7
    assert cfg.voice.enabled is True


def test_partial_config(tmp_path):
    """Only some fields → rest are defaults."""
    toml = tmp_path / "config.toml"
    toml.write_text("[voice]\ninput_device = 3\n")
    with patch("character_eng.config.CONFIG_PATH", toml):
        cfg = load_config()
    assert cfg.voice.input_device == 3
    assert cfg.voice.output_device is None
    assert cfg.voice.enabled is False


def test_empty_config(tmp_path):
    """Empty file → all defaults."""
    toml = tmp_path / "config.toml"
    toml.write_text("")
    with patch("character_eng.config.CONFIG_PATH", toml):
        cfg = load_config()
    assert cfg.voice.input_device is None
    assert cfg.voice.output_device is None
    assert cfg.voice.enabled is False
