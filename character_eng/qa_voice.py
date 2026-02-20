"""Voice integration test — verifies Deepgram STT and ElevenLabs TTS connectivity.

Sends a short audio clip to Deepgram, sends text to ElevenLabs, verifies responses.
No microphone/speaker needed — tests API connectivity only.

Usage:
    uv run -m character_eng.qa_voice
"""

import base64
import json
import os
import struct
import sys
import threading
import time

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

load_dotenv()

console = Console()


def _generate_sine_pcm(freq: float = 440.0, duration: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Generate a sine wave as 16-bit PCM bytes (for testing audio pipelines)."""
    import math

    samples = int(sample_rate * duration)
    data = []
    for i in range(samples):
        t = i / sample_rate
        value = int(32767 * 0.5 * math.sin(2 * math.pi * freq * t))
        data.append(struct.pack("<h", value))
    return b"".join(data)


def test_deepgram_connectivity() -> tuple[bool, str]:
    """Test Deepgram STT connectivity by sending audio and checking for a connection.

    Uses the v5 SDK: client.listen.v1.connect() as a context manager.
    Returns (passed, detail) tuple.
    """
    api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not api_key:
        return False, "DEEPGRAM_API_KEY not set"

    try:
        from deepgram import DeepgramClient
        from deepgram.core.events import EventType
        from deepgram.extensions.types.sockets.listen_v1_control_message import ListenV1ControlMessage
    except ImportError:
        return False, "deepgram-sdk not installed"

    got_response = threading.Event()
    ws_ready = threading.Event()
    error_msg = []

    try:
        client = DeepgramClient(api_key=api_key)
        ctx = client.listen.v1.connect(
            model="nova-3",
            encoding="linear16",
            sample_rate="16000",
            channels="1",
            interim_results="false",
        )
        socket = ctx.__enter__()
    except Exception as e:
        return False, f"Connection failed: {e}"

    def on_open(_):
        ws_ready.set()

    def on_message(msg):
        got_response.set()

    def on_error(err):
        error_msg.append(str(err))

    socket.on(EventType.OPEN, on_open)
    socket.on(EventType.MESSAGE, on_message)
    socket.on(EventType.ERROR, on_error)

    # start_listening() BLOCKS (infinite WebSocket receive loop) — run on daemon thread
    listener_thread = threading.Thread(target=socket.start_listening, daemon=True)
    listener_thread.start()

    if not ws_ready.wait(timeout=5):
        ctx.__exit__(None, None, None)
        return False, "WebSocket did not open within 5 seconds"

    # Send audio in small chunks (simulating real-time mic stream)
    pcm = _generate_sine_pcm(duration=1.0)
    chunk_size = 3200  # 100ms at 16kHz 16-bit mono
    try:
        for i in range(0, len(pcm), chunk_size):
            socket.send_media(pcm[i:i + chunk_size])
            time.sleep(0.05)
    except Exception as e:
        ctx.__exit__(None, None, None)
        return False, f"Send failed: {e}"

    # Wait for any response
    got_response.wait(timeout=5)

    try:
        socket.send_control(ListenV1ControlMessage(type="CloseStream"))
    except Exception:
        pass
    try:
        ctx.__exit__(None, None, None)
    except Exception:
        pass

    if error_msg:
        return False, f"Deepgram error: {error_msg[0]}"

    if got_response.is_set():
        return True, "Connected, sent audio, got response"
    else:
        return True, "Connected, sent audio (no transcript from sine wave, expected)"


def test_elevenlabs_connectivity() -> tuple[bool, str]:
    """Test ElevenLabs TTS connectivity by sending text and receiving audio.

    Returns (passed, detail) tuple.
    """
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        return False, "ELEVENLABS_API_KEY not set"

    try:
        import websocket
    except ImportError:
        return False, "websocket-client not installed"

    voice_id = "pNInz6obpgDQGcFmaJgB"  # Adam
    model_id = "eleven_flash_v2_5"
    url = (
        f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        f"/stream-input?model_id={model_id}"
        f"&output_format=pcm_24000"
    )

    audio_chunks = []
    recv_done = threading.Event()
    error_msg = []

    try:
        ws = websocket.WebSocket()
        ws.settimeout(10)
        ws.connect(url, header=[f"xi-api-key: {api_key}"])
    except Exception as e:
        return False, f"WebSocket connect failed: {e}"

    def recv_loop():
        try:
            while True:
                msg = ws.recv()
                if not msg:
                    break
                data = json.loads(msg)
                if "audio" in data and data["audio"]:
                    audio_chunks.append(base64.b64decode(data["audio"]))
                if data.get("isFinal"):
                    break
        except Exception as e:
            if "closed" not in str(e).lower():
                error_msg.append(str(e))
        finally:
            recv_done.set()

    recv_thread = threading.Thread(target=recv_loop, daemon=True)
    recv_thread.start()

    # Send init
    init_msg = {
        "text": " ",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        "generation_config": {"chunk_length_schedule": [120, 160, 250, 290]},
    }
    try:
        ws.send(json.dumps(init_msg))
    except Exception as e:
        ws.close()
        return False, f"Init send failed: {e}"

    # Send test text
    try:
        ws.send(json.dumps({"text": "Hello, this is a test. ", "try_trigger_generation": True}))
        ws.send(json.dumps({"text": ""}))  # flush
    except Exception as e:
        ws.close()
        return False, f"Text send failed: {e}"

    # Wait for audio
    recv_done.wait(timeout=10)

    try:
        ws.close()
    except Exception:
        pass

    if error_msg:
        return False, f"ElevenLabs error: {error_msg[0]}"

    total_bytes = sum(len(c) for c in audio_chunks)
    if total_bytes > 0:
        duration_ms = int(total_bytes / (24000 * 2) * 1000)  # 24kHz 16-bit mono
        return True, f"Got {total_bytes} bytes of audio (~{duration_ms}ms)"
    else:
        return False, "Connected but received no audio"


def test_audio_devices() -> tuple[bool, str]:
    """Test that sounddevice can enumerate audio devices."""
    try:
        from character_eng.voice import list_audio_devices, get_default_devices
    except ImportError:
        return False, "Voice packages not installed"

    devices = list_audio_devices()
    if not devices:
        return False, "No audio devices found"

    input_dev, output_dev = get_default_devices()
    parts = [f"{len(devices)} device(s) found"]
    if input_dev:
        parts.append(f"input: [{input_dev['index']}] {input_dev['name']}")
    else:
        parts.append("no default input device")
    if output_dev:
        parts.append(f"output: [{output_dev['index']}] {output_dev['name']}")
    else:
        parts.append("no default output device")

    return True, ", ".join(parts)


def main():
    console.print(Panel("[bold]Voice QA — Integration Test[/bold]", border_style="green"))
    console.print()

    tests = [
        ("Audio devices", test_audio_devices),
        ("Deepgram STT", test_deepgram_connectivity),
        ("ElevenLabs TTS", test_elevenlabs_connectivity),
    ]

    results = []
    all_passed = True
    for name, test_fn in tests:
        console.print(f"  Testing {name}...", end=" ")
        try:
            passed, detail = test_fn()
        except Exception as e:
            passed, detail = False, f"Exception: {e}"

        results.append((name, passed, detail))
        if passed:
            console.print(f"[green]PASS[/green]  {detail}")
        else:
            console.print(f"[red]FAIL[/red]  {detail}")
            all_passed = False

    console.print()
    if all_passed:
        console.print("[bold green]All voice tests passed.[/bold green]")
    else:
        console.print("[bold red]Some voice tests failed.[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
