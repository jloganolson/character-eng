"""Load and persist user settings from config.toml (project root)."""

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
    filler_enabled: bool = True  # use short Pocket-TTS filler clips while primary TTS spins up
    filler_lead_ms: int = 180  # wait this long for real audio before inserting filler


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
    prompt_asset_open_target: str = "vscode"  # "vscode" or "folder"
    prompt_asset_vscode_cmd: str = "code"


@dataclass
class BridgeConfig:
    enabled: bool = False
    port: int = 7862  # shares dashboard port (HTTP + WS on single port)


@dataclass
class HistoryConfig:
    enabled: bool = True
    free_warning_gib: float = 50.0
    vision_capture_fps: float = 2.0


@dataclass
class AppConfig:
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    bridge: BridgeConfig = field(default_factory=BridgeConfig)
    history: HistoryConfig = field(default_factory=HistoryConfig)


def _toml_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def save_config(config: AppConfig, path: Path = CONFIG_PATH) -> None:
    """Persist AppConfig to config.toml."""
    lines = [
        "[voice]",
        f"enabled = {'true' if config.voice.enabled else 'false'}",
        f"aec = {'true' if config.voice.aec else 'false'}",
        f"tts_backend = {_toml_string(config.voice.tts_backend)}",
        f"ref_audio = {_toml_string(config.voice.ref_audio)}",
        f"ref_text = {_toml_string(config.voice.ref_text)}",
        f"tts_model = {_toml_string(config.voice.tts_model)}",
        f"tts_device = {_toml_string(config.voice.tts_device)}",
        f"tts_server_url = {_toml_string(config.voice.tts_server_url)}",
        f"pocket_voice = {_toml_string(config.voice.pocket_voice)}",
        f"filler_enabled = {'true' if config.voice.filler_enabled else 'false'}",
        f"filler_lead_ms = {int(config.voice.filler_lead_ms)}",
        "",
        "[vision]",
        f"enabled = {'true' if config.vision.enabled else 'false'}",
        f"service_url = {_toml_string(config.vision.service_url)}",
        f"service_port = {int(config.vision.service_port)}",
        f"auto_launch = {'true' if config.vision.auto_launch else 'false'}",
        f"synthesis_min_interval = {config.vision.synthesis_min_interval}",
        "",
        "[dashboard]",
        f"enabled = {'true' if config.dashboard.enabled else 'false'}",
        f"port = {int(config.dashboard.port)}",
        f"prompt_asset_open_target = {_toml_string(config.dashboard.prompt_asset_open_target)}",
        f"prompt_asset_vscode_cmd = {_toml_string(config.dashboard.prompt_asset_vscode_cmd)}",
        "",
        "[bridge]",
        f"enabled = {'true' if config.bridge.enabled else 'false'}",
        f"port = {int(config.bridge.port)}",
        "",
        "[history]",
        f"enabled = {'true' if config.history.enabled else 'false'}",
        f"free_warning_gib = {config.history.free_warning_gib}",
        f"vision_capture_fps = {config.history.vision_capture_fps}",
        "",
    ]

    if config.voice.input_device is not None:
        value = config.voice.input_device
        lines.insert(1, f"input_device = {value}" if isinstance(value, int) else f"input_device = {_toml_string(str(value))}")
    if config.voice.output_device is not None:
        value = config.voice.output_device
        insert_at = 2 if config.voice.input_device is not None else 1
        lines.insert(insert_at, f"output_device = {value}" if isinstance(value, int) else f"output_device = {_toml_string(str(value))}")

    path.write_text("\n".join(lines), encoding="utf-8")


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
        filler_enabled=voice_data.get("filler_enabled", True),
        filler_lead_ms=voice_data.get("filler_lead_ms", 180),
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
        prompt_asset_open_target=dashboard_data.get("prompt_asset_open_target", "vscode"),
        prompt_asset_vscode_cmd=dashboard_data.get("prompt_asset_vscode_cmd", "code"),
    )

    bridge_data = data.get("bridge", {})
    bridge = BridgeConfig(
        enabled=bridge_data.get("enabled", False),
        port=bridge_data.get("port", 7862),
    )

    history_data = data.get("history", {})
    history = HistoryConfig(
        enabled=history_data.get("enabled", True),
        free_warning_gib=history_data.get("free_warning_gib", 50.0),
        vision_capture_fps=history_data.get("vision_capture_fps", 2.0),
    )

    return AppConfig(voice=voice, vision=vision, dashboard=dashboard, bridge=bridge, history=history)
