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
    aec: bool = True  # WebRTC AEC3 echo cancellation; False to fall back to mic-muting
    tts_backend: str = "elevenlabs"  # "elevenlabs", "qwen", "pocket", or "local" (alias for qwen)
    ref_audio: str = ""  # path to reference audio WAV for voice cloning
    ref_text: str = ""  # transcript of ref_audio for qwen ICL mode (higher quality cloning)
    tts_model: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"  # HuggingFace model ID
    tts_device: str = "cuda:0"  # GPU device for local TTS
    tts_server_url: str = ""  # server URL for pocket backend
    pocket_voice: str = "voices/greg.safetensors"  # path to .safetensors or WAV for pocket-tts serve --voice


@dataclass
class VisionConfig:
    enabled: bool = False
    service_url: str = "http://localhost:7860"
    service_port: int = 7860
    auto_launch: bool = True
    synthesis_min_interval: float = 0.75


@dataclass
class DashboardConfig:
    enabled: bool = True
    port: int = 7862


@dataclass
class AppConfig:
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)


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
        aec=voice_data.get("aec", True),
        tts_backend=voice_data.get("tts_backend", "elevenlabs"),
        ref_audio=voice_data.get("ref_audio", ""),
        ref_text=voice_data.get("ref_text", ""),
        tts_model=voice_data.get("tts_model", "Qwen/Qwen3-TTS-12Hz-0.6B-Base"),
        tts_device=voice_data.get("tts_device", "cuda:0"),
        tts_server_url=voice_data.get("tts_server_url", ""),
        pocket_voice=voice_data.get("pocket_voice", "voices/greg.safetensors"),
    )

    vision_data = data.get("vision", {})
    vision = VisionConfig(
        enabled=vision_data.get("enabled", False),
        service_url=vision_data.get("service_url", "http://localhost:7860"),
        service_port=vision_data.get("service_port", 7860),
        auto_launch=vision_data.get("auto_launch", True),
        synthesis_min_interval=vision_data.get("synthesis_min_interval", 0.75),
    )

    dashboard_data = data.get("dashboard", {})
    dashboard = DashboardConfig(
        enabled=dashboard_data.get("enabled", True),
        port=dashboard_data.get("port", 7862),
    )

    return AppConfig(voice=voice, vision=vision, dashboard=dashboard)
