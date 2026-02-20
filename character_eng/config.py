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
    )
    return AppConfig(voice=voice)
