"""Load user settings from config.toml (project root)."""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.toml"


@dataclass
class VoiceConfig:
    input_device: int | None = None
    output_device: int | None = None
    enabled: bool = False
    mic_mute_during_playback: bool = True  # False for devices with hardware AEC
    tts_backend: str = "elevenlabs"  # "elevenlabs" or "local"
    ref_audio: str = ""  # path to reference audio WAV for local voice cloning
    tts_model: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"  # HuggingFace model ID
    tts_device: str = "cuda:0"  # GPU device for local TTS


@dataclass
class AppConfig:
    voice: VoiceConfig = field(default_factory=VoiceConfig)


def load_config() -> AppConfig:
    """Read config.toml and return AppConfig. Returns defaults if file missing."""
    if not CONFIG_PATH.exists():
        return AppConfig()

    with open(CONFIG_PATH, "rb") as f:
        data = tomllib.load(f)

    voice_data = data.get("voice", {})
    voice = VoiceConfig(
        input_device=voice_data.get("input_device"),
        output_device=voice_data.get("output_device"),
        enabled=voice_data.get("enabled", False),
        mic_mute_during_playback=voice_data.get("mic_mute_during_playback", True),
        tts_backend=voice_data.get("tts_backend", "elevenlabs"),
        ref_audio=voice_data.get("ref_audio", ""),
        tts_model=voice_data.get("tts_model", "Qwen/Qwen3-TTS-12Hz-0.6B-Base"),
        tts_device=voice_data.get("tts_device", "cuda:0"),
    )
    return AppConfig(voice=voice)
