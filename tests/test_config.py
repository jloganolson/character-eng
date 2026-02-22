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


def test_tts_backend_defaults():
    """Missing config.toml → TTS backend defaults to elevenlabs."""
    with patch("character_eng.config.CONFIG_PATH", Path("/nonexistent/config.toml")):
        cfg = load_config()
    assert cfg.voice.tts_backend == "elevenlabs"
    assert cfg.voice.ref_audio == ""
    assert cfg.voice.tts_model == "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    assert cfg.voice.tts_device == "cuda:0"


def test_tts_backend_local(tmp_path):
    """Local TTS config fields parse correctly."""
    toml = tmp_path / "config.toml"
    toml.write_text(
        "[voice]\n"
        'tts_backend = "local"\n'
        'ref_audio = "/path/to/ref.wav"\n'
        'tts_model = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"\n'
        'tts_device = "cuda:1"\n'
    )
    with patch("character_eng.config.CONFIG_PATH", toml):
        cfg = load_config()
    assert cfg.voice.tts_backend == "local"
    assert cfg.voice.ref_audio == "/path/to/ref.wav"
    assert cfg.voice.tts_model == "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    assert cfg.voice.tts_device == "cuda:1"


def test_tts_backend_partial_defaults(tmp_path):
    """Only tts_backend set → other TTS fields use defaults."""
    toml = tmp_path / "config.toml"
    toml.write_text('[voice]\ntts_backend = "local"\n')
    with patch("character_eng.config.CONFIG_PATH", toml):
        cfg = load_config()
    assert cfg.voice.tts_backend == "local"
    assert cfg.voice.ref_audio == ""
    assert cfg.voice.tts_model == "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    assert cfg.voice.tts_device == "cuda:0"


def test_string_device_names(tmp_path):
    """Device names as strings are accepted alongside integers."""
    toml = tmp_path / "config.toml"
    toml.write_text(
        "[voice]\n"
        'input_device = "reSpeaker"\n'
        'output_device = "reSpeaker"\n'
    )
    with patch("character_eng.config.CONFIG_PATH", toml):
        cfg = load_config()
    assert cfg.voice.input_device == "reSpeaker"
    assert cfg.voice.output_device == "reSpeaker"
