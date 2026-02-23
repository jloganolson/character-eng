"""Load user settings from config.toml (project root)."""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.toml"


@dataclass
class VoiceConfig:
    input_device: int | str | None = None
    output_device: int | str | None = None
    enabled: bool = False
    mic_mute_during_playback: bool = True  # False for devices with hardware AEC
    tts_backend: str = "elevenlabs"  # "elevenlabs", "qwen", "pocket", or "local" (alias for qwen)
    ref_audio: str = ""  # path to reference audio WAV for voice cloning
    ref_text: str = ""  # transcript of ref_audio for qwen ICL mode (higher quality cloning)
    tts_model: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"  # HuggingFace model ID
    tts_device: str = "cuda:0"  # GPU device for local TTS
    tts_server_url: str = ""  # server URL for pocket backend
    pocket_voice: str = "voices/greg.safetensors"  # path to .safetensors or WAV for pocket-tts serve --voice


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
        ref_text=voice_data.get("ref_text", ""),
        tts_model=voice_data.get("tts_model", "Qwen/Qwen3-TTS-12Hz-0.6B-Base"),
        tts_device=voice_data.get("tts_device", "cuda:0"),
        tts_server_url=voice_data.get("tts_server_url", ""),
        pocket_voice=voice_data.get("pocket_voice", "voices/greg.safetensors"),
    )
    return AppConfig(voice=voice)
