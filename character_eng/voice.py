"""Voice I/O layer — Deepgram STT + ElevenLabs TTS + sounddevice mic/speaker.

Wraps around the existing text pipeline as an I/O layer. Text mode continues
to work unchanged. Local Qwen TTS still needs: uv sync --extra local-tts
"""

from __future__ import annotations

import base64
import json
import os
import queue
import select
import struct
import subprocess
import sys
import termios
import threading
import time
import tty
from pathlib import Path
from typing import Callable, Iterator
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()

# Sentinel strings returned by wait_for_input()
VOICE_OFF = "__VOICE_OFF__"
EXIT = "__EXIT__"
VOICE_ERROR = "__VOICE_ERROR__"

# Key mappings for hotkeys
_KEY_MAP = {
    "i": "/info",
    "b": "/beat",
    "t": "/trace",
    "1": "1",
    "2": "2",
    "3": "3",
    "4": "4",
    "q": EXIT,
    "\x1b": VOICE_OFF,  # Escape
    "\x03": EXIT,  # Ctrl+C
}

from character_eng.utils import ts as _ts

AUTO_BEAT_DELAY = 4.0  # seconds after TTS playback finishes


def list_audio_devices() -> list[dict]:
    """List available audio devices with index, name, and input/output channel counts.

    Returns a list of dicts: {index, name, inputs, outputs, is_default_input, is_default_output}
    """
    import sounddevice as sd

    devices = sd.query_devices()
    defaults = sd.default.device  # (input_idx, output_idx)
    result = []
    for i, dev in enumerate(devices):
        result.append({
            "index": i,
            "name": dev["name"],
            "inputs": dev["max_input_channels"],
            "outputs": dev["max_output_channels"],
            "is_default_input": i == defaults[0],
            "is_default_output": i == defaults[1],
        })
    return result


def get_default_devices() -> tuple[dict | None, dict | None]:
    """Return (input_device, output_device) info dicts for current defaults."""
    devices = list_audio_devices()
    input_dev = next((d for d in devices if d["is_default_input"]), None)
    output_dev = next((d for d in devices if d["is_default_output"]), None)
    return input_dev, output_dev


def resolve_device(spec: int | str | None, kind: str = "output") -> int | None:
    """Resolve a device spec to a sounddevice index.

    Args:
        spec: int (use directly), str (match by name substring, case-insensitive),
              or None (use system default).
        kind: "input" or "output" — filters to devices with matching capability.

    Returns:
        Device index, or None to use system default.

    Raises:
        RuntimeError: if string spec matches no device.
    """
    if spec is None:
        return None
    if isinstance(spec, int):
        return spec
    # String: search by name
    import sounddevice as sd

    for i, d in enumerate(sd.query_devices()):
        if spec.lower() in d["name"].lower():
            if kind == "output" and d["max_output_channels"] > 0:
                return i
            if kind == "input" and d["max_input_channels"] > 0:
                return i
    raise RuntimeError(f"No {kind} device matching '{spec}'")


def check_voice_available(tts_backend: str = "elevenlabs") -> tuple[bool, str]:
    """Check if voice dependencies and API keys are available.

    Args:
        tts_backend: "elevenlabs", "qwen", "local" (alias for qwen), or "pocket".

    Returns (available, reason) tuple.
    """
    # Base voice deps (sounddevice, numpy, deepgram, websocket-client) are in the
    # main environment. Only local-tts (qwen) needs extra packages.
    if tts_backend in ("local", "qwen"):
        missing = []
        try:
            import torch  # noqa: F401
        except ImportError:
            missing.append("torch")
        try:
            import transformers  # noqa: F401
        except ImportError:
            missing.append("transformers")
        if missing:
            return False, f"Missing packages: {', '.join(missing)}. Install with: uv sync --extra local-tts"

    if not os.environ.get("DEEPGRAM_API_KEY"):
        return False, "DEEPGRAM_API_KEY not set in .env"
    if tts_backend == "elevenlabs" and not os.environ.get("ELEVENLABS_API_KEY"):
        return False, "ELEVENLABS_API_KEY not set in .env"

    return True, ""


class MicStream:
    """Captures 16kHz/16-bit/mono PCM from the microphone via sounddevice.

    Callback puts raw bytes on a queue. A feeder daemon thread drains the queue
    and calls on_audio(bytes) — typically sending to Deepgram.

    If the device doesn't support mono, opens with the device's actual channel
    count and downmixes to mono (takes channel 0) in the callback.
    """

    SAMPLE_RATE = 16000
    CHANNELS = 1  # target: mono for Deepgram
    DTYPE = "int16"
    BLOCKSIZE = 1280  # ~80ms at 16kHz

    def __init__(self, on_audio: Callable[[bytes], None], device: int | None = None):
        self._on_audio = on_audio
        self._device = device
        self._queue: queue.Queue[bytes] = queue.Queue()
        self._stream = None
        self._feeder: threading.Thread | None = None
        self._running = False
        self._actual_channels = 1

    def _callback(self, indata, frames, time_info, status):
        """sounddevice InputStream callback — runs in audio thread."""
        raw = bytes(indata)
        if self._actual_channels > 1:
            # Downmix: take channel 0 only (every Nth int16 sample)
            import numpy as np

            samples = np.frombuffer(raw, dtype=np.int16)
            mono = samples[:: self._actual_channels].copy()
            self._queue.put(mono.tobytes())
        else:
            self._queue.put(raw)

    def _feed_loop(self):
        """Daemon thread that drains the queue and calls on_audio."""
        while self._running:
            try:
                data = self._queue.get(timeout=0.1)
                self._on_audio(data)
            except queue.Empty:
                continue

    def start(self):
        import sounddevice as sd

        self._running = True

        # Determine channel count: try mono first, fall back to device's count
        channels_to_try = [self.CHANNELS]
        if self._device is not None:
            dev_info = sd.query_devices(self._device)
            dev_channels = int(dev_info["max_input_channels"])
            if dev_channels > 1 and dev_channels not in channels_to_try:
                channels_to_try.append(dev_channels)

        for channels in channels_to_try:
            kwargs = dict(
                samplerate=self.SAMPLE_RATE,
                channels=channels,
                dtype=self.DTYPE,
                blocksize=self.BLOCKSIZE,
                callback=self._callback,
            )
            if self._device is not None:
                kwargs["device"] = self._device
            try:
                self._stream = sd.RawInputStream(**kwargs)
                self._stream.start()
                self._actual_channels = channels
                break
            except sd.PortAudioError:
                if channels == channels_to_try[-1]:
                    raise
                continue

        self._feeder = threading.Thread(target=self._feed_loop, daemon=True)
        self._feeder.start()

    def stop(self):
        self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._feeder is not None:
            self._feeder.join(timeout=1)
            self._feeder = None
        # Drain queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break


class SpeakerStream:
    """Plays PCM audio via sounddevice OutputStream.

    Input at 24kHz/16-bit/mono matching ElevenLabs PCM output format.
    If the device doesn't support 24kHz, falls back to 48kHz with 2x upsampling.
    enqueue(pcm) adds audio (resampled automatically if needed).
    flush() clears buffer (barge-in). wait_until_done() blocks until buffer drained.
    """

    NATIVE_RATE = 24000  # ElevenLabs PCM output rate
    CHANNELS = 1
    DTYPE = "int16"

    def __init__(self, device: int | None = None, on_output: Callable[[bytes], None] | None = None):
        self._device = device
        self._on_output = on_output
        self._buffer = bytearray()
        self._lock = threading.Lock()
        self._stream = None
        self._done_event = threading.Event()
        self._done_event.set()  # starts "done" (nothing to play)
        self._audio_started = threading.Event()
        self._actual_rate = self.NATIVE_RATE
        self._resample = False
        self._bytes_played = 0
        self._bytes_enqueued = 0

    def _callback(self, outdata, frames, time_info, status):
        """sounddevice OutputStream callback — runs in audio thread."""
        bytes_needed = frames * 2  # 16-bit = 2 bytes per sample
        with self._lock:
            available = len(self._buffer)
            if available >= bytes_needed:
                data = bytes(self._buffer[:bytes_needed])
                del self._buffer[:bytes_needed]
                self._bytes_played += bytes_needed
                outdata[:] = data
            else:
                # Partial data + silence
                if available > 0:
                    data = bytes(self._buffer[:available])
                    del self._buffer[:available]
                    self._bytes_played += available
                    outdata[:available] = data
                # Fill rest with silence
                outdata[available:] = b'\x00' * (len(outdata) - available)
                if available == 0:
                    self._done_event.set()
        # Report played audio at speaker's actual rate (for AEC reference)
        if self._on_output is not None:
            self._on_output(bytes(outdata))

    def enqueue(self, pcm: bytes):
        """Add PCM audio bytes to the playback buffer. Resamples if needed."""
        if self._resample and len(pcm) > 0:
            pcm = self._upsample(pcm)
        with self._lock:
            self._buffer.extend(pcm)
            self._bytes_enqueued += len(pcm)
            self._done_event.clear()
            if len(pcm) > 0:
                self._audio_started.set()

    def _upsample(self, pcm: bytes) -> bytes:
        """Resample 24kHz PCM to the actual output rate."""
        import numpy as np

        ratio = self._actual_rate / self.NATIVE_RATE
        samples = np.frombuffer(pcm, dtype=np.int16)
        int_ratio = self._actual_rate // self.NATIVE_RATE
        if self._actual_rate == int_ratio * self.NATIVE_RATE:
            # Integer ratio (e.g. 48000/24000 = 2x): simple repeat
            return np.repeat(samples, int_ratio).tobytes()
        # Non-integer ratio (e.g. 44100/24000): linear interpolation
        old_len = len(samples)
        new_len = round(old_len * ratio)
        old_idx = np.arange(old_len)
        new_idx = np.linspace(0, old_len - 1, new_len)
        return np.interp(new_idx, old_idx, samples.astype(np.float64)).astype(np.int16).tobytes()

    def reset_counters(self):
        """Reset played/enqueued byte counters for a new utterance."""
        with self._lock:
            self._bytes_played = 0
            self._bytes_enqueued = 0

    @property
    def playback_ratio(self) -> float:
        """Fraction of enqueued audio that has been played (0.0 to 1.0)."""
        with self._lock:
            if self._bytes_enqueued == 0:
                return 0.0
            return min(self._bytes_played / self._bytes_enqueued, 1.0)

    def flush(self):
        """Clear the playback buffer immediately (barge-in)."""
        with self._lock:
            self._buffer.clear()
            self._done_event.set()
        self._audio_started.clear()

    def wait_for_audio(self, timeout: float = 0.5) -> bool:
        """Block until first audio chunk arrives. Returns True if audio arrived, False on timeout."""
        return self._audio_started.wait(timeout=timeout)

    def wait_until_done(self, timeout: float | None = None) -> bool:
        """Block until buffer is drained. Returns True if done, False on timeout."""
        return self._done_event.wait(timeout=timeout)

    def is_done(self) -> bool:
        """Check if playback buffer is empty."""
        return self._done_event.is_set()

    def start(self):
        import sounddevice as sd

        # Build sample rate fallback chain: device default → native 24kHz → 48kHz
        # Device default first avoids noisy failed probes on non-24kHz devices
        rates_to_try = []
        if self._device is not None:
            dev_info = sd.query_devices(self._device)
            dev_rate = int(dev_info["default_samplerate"])
            rates_to_try.append(dev_rate)
        for r in [self.NATIVE_RATE, 48000]:
            if r not in rates_to_try:
                rates_to_try.append(r)

        last_error = None
        for rate in rates_to_try:
            kwargs = dict(
                samplerate=rate,
                channels=self.CHANNELS,
                dtype=self.DTYPE,
                blocksize=int(rate * 0.1),  # 100ms
                callback=self._callback,
            )
            if self._device is not None:
                kwargs["device"] = self._device
            try:
                # Suppress PortAudio C-level stderr noise during sample rate probing
                devnull = os.open(os.devnull, os.O_WRONLY)
                old_stderr = os.dup(2)
                os.dup2(devnull, 2)
                try:
                    self._stream = sd.RawOutputStream(**kwargs)
                    self._stream.start()
                finally:
                    os.dup2(old_stderr, 2)
                    os.close(devnull)
                    os.close(old_stderr)
                self._actual_rate = rate
                self._resample = (rate != self.NATIVE_RATE)
                return
            except sd.PortAudioError as e:
                last_error = e
                continue

        raise last_error

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        with self._lock:
            self._buffer.clear()
            self._done_event.set()
        self._audio_started.clear()


class DeepgramSTT:
    """Wraps Deepgram SDK v5 for streaming speech-to-text with turn detection.

    Uses nova-3 model with endpointing via v1 WebSocket API. Dispatches:
    - on_transcript(text) when a final transcript with speech_final is received
    - on_turn_start() when speech starts (barge-in trigger)

    The v5 SDK uses client.listen.v1.connect() as a context manager, returning a
    V1SocketClient with on(EventType, callback), send_media(bytes), and
    send_control(ListenV1ControlMessage) methods.
    """

    def __init__(
        self,
        on_transcript: Callable[[str], None],
        on_turn_start: Callable[[], None],
    ):
        self._on_transcript = on_transcript
        self._on_turn_start = on_turn_start
        self._socket = None
        self._ctx_manager = None
        self._listener_thread: threading.Thread | None = None
        self._is_speaking = False
        self._pending_transcript = ""
        self._ws_open = threading.Event()
        self._last_error = ""
        self._last_is_final_at: float | None = None

    def start(self):
        from deepgram import DeepgramClient
        from deepgram.core.events import EventType

        client = DeepgramClient(api_key=os.environ["DEEPGRAM_API_KEY"])
        self._ctx_manager = client.listen.v1.connect(
            model="nova-3",
            encoding="linear16",
            sample_rate=str(MicStream.SAMPLE_RATE),
            channels=str(MicStream.CHANNELS),
            interim_results="true",
            utterance_end_ms="1500",
            vad_events="true",
            endpointing="300",
        )
        self._socket = self._ctx_manager.__enter__()

        def _on_open(_):
            self._ws_open.set()

        # Register event handlers
        self._socket.on(EventType.OPEN, _on_open)
        self._socket.on(EventType.MESSAGE, self._on_message)
        self._socket.on(EventType.ERROR, self._on_error)

        # start_listening() BLOCKS (infinite WebSocket receive loop) — run on daemon thread
        def _listen():
            try:
                self._socket.start_listening()
            except Exception as exc:
                self._last_error = str(exc)
            finally:
                self._ws_open.clear()

        self._listener_thread = threading.Thread(
            target=_listen, daemon=True
        )
        self._listener_thread.start()
        self._ws_open.wait(timeout=5)

    def send_audio(self, data: bytes):
        if self._socket is not None:
            try:
                self._socket.send_media(data)
            except Exception as exc:
                self._last_error = str(exc)
                self._ws_open.clear()

    def _on_message(self, message):
        """Called for each message from Deepgram.

        Messages are dicts with a 'type' field. We handle:
        - 'Results': transcript data with is_final and speech_final flags
        - 'SpeechStarted': user started speaking
        - 'UtteranceEnd': backup endpoint for flushing pending transcript
        """
        if isinstance(message, dict):
            msg_type = message.get("type", "")
        else:
            # Try attribute access for SDK response objects
            msg_type = getattr(message, "type", "")

        if msg_type == "SpeechStarted":
            if not self._is_speaking:
                self._is_speaking = True
                self._on_turn_start()
            return

        if msg_type == "UtteranceEnd":
            if self._pending_transcript:
                text = self._pending_transcript.strip()
                self._pending_transcript = ""
                self._is_speaking = False
                if text:
                    self._on_transcript(text)
            else:
                if self._is_speaking:
                    self._is_speaking = False
            return

        if msg_type != "Results":
            return

        # Extract transcript from Results message
        transcript = ""
        is_final = False
        speech_final = False
        try:
            if isinstance(message, dict):
                channel = message.get("channel", {})
                alts = channel.get("alternatives", [])
                if alts:
                    transcript = alts[0].get("transcript", "")
                is_final = message.get("is_final", False)
                speech_final = message.get("speech_final", False)
            else:
                channel = message.channel
                if channel and channel.alternatives:
                    transcript = channel.alternatives[0].transcript
                is_final = message.is_final
                speech_final = getattr(message, "speech_final", False)
        except (AttributeError, IndexError, KeyError):
            return

        if transcript and is_final:
            self._pending_transcript = transcript
            self._last_is_final_at = time.time()

        if speech_final and self._pending_transcript:
            text = self._pending_transcript.strip()
            self._pending_transcript = ""
            self._is_speaking = False
            if text:
                self._on_transcript(text)

    def _on_error(self, error):
        """Called on Deepgram errors."""
        self._last_error = str(error)
        self._ws_open.clear()

    def status_snapshot(self) -> dict:
        return {
            "thread_alive": bool(self._listener_thread and self._listener_thread.is_alive()),
            "ws_open": self._ws_open.is_set(),
            "error": self._last_error,
            "speaking": self._is_speaking,
            "pending_transcript": bool(self._pending_transcript),
        }

    def stop(self):
        if self._socket is not None:
            try:
                from deepgram.extensions.types.sockets.listen_v1_control_message import ListenV1ControlMessage
                self._socket.send_control(ListenV1ControlMessage(type="CloseStream"))
            except Exception:
                pass
            self._socket = None
        if self._ctx_manager is not None:
            try:
                self._ctx_manager.__exit__(None, None, None)
            except Exception:
                pass
            self._ctx_manager = None
        if self._listener_thread is not None:
            self._listener_thread.join(timeout=2)
            self._listener_thread = None
        self._pending_transcript = ""
        self._is_speaking = False
        self._ws_open.clear()


class ElevenLabsTTS:
    """Streaming TTS via ElevenLabs WebSocket API.

    Sends text chunks as they arrive from the LLM, receives PCM audio
    in real-time. Uses eleven_flash_v2_5 for low latency.
    """

    MODEL = "eleven_flash_v2_5"
    DEFAULT_VOICE = "pNInz6obpgDQGcFmaJgB"  # Adam

    def __init__(self, on_audio: Callable[[bytes], None], voice_id: str = ""):
        self._on_audio = on_audio
        self._voice_id = voice_id or self.DEFAULT_VOICE
        self._ws = None
        self._recv_thread: threading.Thread | None = None
        self._running = False
        self._lock = threading.Lock()
        self._generation_done = threading.Event()

    def _connect(self):
        """Open WebSocket connection to ElevenLabs."""
        import websocket

        self._generation_done.clear()

        api_key = os.environ["ELEVENLABS_API_KEY"]
        url = (
            f"wss://api.elevenlabs.io/v1/text-to-speech/{self._voice_id}"
            f"/stream-input?model_id={self.MODEL}"
            f"&output_format=pcm_24000"
        )

        try:
            self._ws = websocket.WebSocket()
            self._ws.connect(url, header=[f"xi-api-key: {api_key}"])
        except Exception as e:
            sys.stderr.write(f"[TTS] Connection failed: {e}\n")
            sys.stderr.flush()
            self._ws = None
            self._generation_done.set()  # unblock wait_for_done()
            return

        # Send initial config
        init_msg = {
            "text": " ",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
            },
            "generation_config": {
                "chunk_length_schedule": [120, 160, 250, 290],
            },
        }
        self._ws.send(json.dumps(init_msg))

        # Start receive thread
        self._running = True
        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._recv_thread.start()

    def _recv_loop(self):
        """Receive audio chunks from ElevenLabs WebSocket."""
        chunks_received = 0
        try:
            while self._running:
                try:
                    with self._lock:
                        ws = self._ws
                    if ws is None:
                        break
                    msg = ws.recv()
                    if not msg:
                        break
                    data = json.loads(msg)
                    if data.get("isFinal"):
                        break
                    if "audio" in data and data["audio"]:
                        pcm = base64.b64decode(data["audio"])
                        chunks_received += 1
                        self._on_audio(pcm)
                    elif chunks_received == 0:
                        # Log non-audio messages to help debug silent failures
                        preview = str(data)[:200]
                        sys.stderr.write(f"[TTS] non-audio msg: {preview}\n")
                        sys.stderr.flush()
                except Exception as e:
                    if self._running:
                        sys.stderr.write(f"[TTS] recv error: {e}\n")
                        sys.stderr.flush()
                    break
        finally:
            if chunks_received == 0 and self._running:
                sys.stderr.write("[TTS] Warning: no audio chunks received\n")
                sys.stderr.flush()
            self._generation_done.set()

    def send_text(self, text: str):
        """Send a text chunk for TTS synthesis."""
        with self._lock:
            if self._ws is None:
                self._connect()
        try:
            msg = {"text": text, "try_trigger_generation": True}
            with self._lock:
                if self._ws is not None:
                    self._ws.send(json.dumps(msg))
        except Exception:
            pass

    def flush(self):
        """Signal end of text input — flush remaining audio."""
        try:
            msg = {"text": ""}
            with self._lock:
                if self._ws is not None:
                    self._ws.send(json.dumps(msg))
        except Exception:
            pass

    def wait_for_done(self, timeout: float = 15.0) -> bool:
        """Block until ElevenLabs sends isFinal. Returns True if done, False on timeout."""
        return self._generation_done.wait(timeout=timeout)

    def close(self):
        """Close the WebSocket connection."""
        self._running = False
        self._generation_done.set()
        with self._lock:
            if self._ws is not None:
                try:
                    self._ws.close()
                except Exception:
                    pass
                self._ws = None
        if self._recv_thread is not None:
            self._recv_thread.join(timeout=2)
            self._recv_thread = None


class KeyListener:
    """Non-blocking keystroke reader using tty.setcbreak().

    Daemon thread reads single keystrokes and puts mapped commands on the event queue.
    Saves terminal settings before entering cbreak mode and registers an atexit
    handler so settings are restored even on abrupt exit (SIGINT/Ctrl+C).
    """

    def __init__(self, event_queue: queue.Queue, on_voice_off: Callable[[], None] | None = None):
        self._event_queue = event_queue
        self._on_voice_off = on_voice_off
        self._thread: threading.Thread | None = None
        self._running = False
        self._old_settings = None

    def _restore_terminal(self):
        """Restore saved terminal settings immediately."""
        if self._old_settings is not None:
            try:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, self._old_settings)
            except Exception:
                pass

    def _listen_loop(self):
        """Read keystrokes in cbreak mode."""
        if not sys.stdin.isatty():
            return
        fd = sys.stdin.fileno()
        try:
            tty.setcbreak(fd)
            while self._running:
                # Use select to avoid blocking indefinitely
                ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                if ready:
                    ch = sys.stdin.read(1)
                    if ch in _KEY_MAP:
                        mapped = _KEY_MAP[ch]
                        self._event_queue.put(mapped)
                        if mapped == VOICE_OFF and self._on_voice_off is not None:
                            self._on_voice_off()
        except Exception:
            pass
        finally:
            self._restore_terminal()

    def start(self):
        import atexit

        # Save terminal settings *before* spawning the thread so stop() always has them
        if sys.stdin.isatty():
            self._old_settings = termios.tcgetattr(sys.stdin.fileno())
            atexit.register(self._restore_terminal)

        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        self._restore_terminal()
        if self._thread is not None:
            self._thread.join(timeout=1)
            self._thread = None


class VoiceIO:
    """Orchestrator — wires mic, speaker, STT, TTS, keys, and auto-beat timer.

    Usage:
        voice = VoiceIO()
        voice.start()
        while True:
            user_input = voice.wait_for_input()
            # ... process through ChatSession.send() ...
            voice.speak(llm_chunks)  # or voice.speak_text(beat.line)
        voice.stop()
    """

    def __init__(
        self,
        voice_id: str = "",
        input_device: int | None = None,
        output_device: int | None = None,
        aec: bool = True,
        tts_backend: str = "elevenlabs",
        ref_audio: str = "",
        ref_text: str = "",
        tts_model: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        tts_device: str = "cuda:0",
        tts_server_url: str = "",
        pocket_voice: str = "",
        filler_enabled: bool = True,
        filler_lead_ms: int = 180,
        output_only: bool = False,
        trace_hook: Callable[[str, dict], None] | None = None,
        input_audio_hook: Callable[[bytes], None] | None = None,
        output_audio_hook: Callable[[bytes], None] | None = None,
    ):
        self._event_queue: queue.Queue[str] = queue.Queue()
        self._cancelled = threading.Event()
        self._is_speaking = False
        self._barged_in = False
        self._auto_beat_timer: threading.Timer | None = None
        self._auto_beat_countdown_active = False
        self._aec_enabled = aec
        self._aec = None  # LiveKitAEC, created in start() if enabled

        # Components (created in start())
        self._speaker: SpeakerStream | None = None
        self._mic: MicStream | None = None
        self._stt: DeepgramSTT | None = None
        self._tts: ElevenLabsTTS | None = None
        self._keys: KeyListener | None = None
        self._voice_id = voice_id
        self._input_device = input_device
        self._output_device = output_device
        self._tts_backend = tts_backend
        self._ref_audio = ref_audio
        self._ref_text = ref_text
        self._tts_model = tts_model
        self._tts_device = tts_device
        self._tts_server_url = tts_server_url
        self._pocket_voice = pocket_voice
        self._pocket_server: subprocess.Popen | None = None
        self._pocket_we_started = False
        self._filler_enabled = filler_enabled
        self._filler_lead_ms = filler_lead_ms
        self._filler_bank = None
        self._filler_thread: threading.Thread | None = None
        self._filler_cancel = threading.Event()
        self._started = False
        self._output_only = output_only
        self._trace_hook = trace_hook
        self._input_audio_hook = input_audio_hook
        self._output_audio_hook = output_audio_hook
        self._last_speech_started_at: float | None = None
        self._last_transcript_final_at: float | None = None
        self._last_speech_ended_at: float | None = None
        self._last_transcript_text: str = ""
        self._deferred_input_prefix: str = ""

    def _trace(self, event_type: str, data: dict) -> None:
        if self._trace_hook is None:
            return
        try:
            self._trace_hook(event_type, dict(data))
        except Exception:
            pass

    def status_snapshot(self) -> dict:
        import requests

        stt = {"state": "off", "detail": "STT inactive."}
        if self._output_only:
            stt = {"state": "off", "detail": "Output-only mode."}
        elif self._stt is not None:
            stt_status = self._stt.status_snapshot()
            if stt_status["error"]:
                stt = {"state": "error", "detail": stt_status["error"]}
            elif stt_status["ws_open"]:
                stt = {"state": "ready", "detail": "Deepgram socket open."}
            elif stt_status["thread_alive"]:
                stt = {"state": "starting", "detail": "Deepgram socket connecting."}
            else:
                stt = {"state": "error", "detail": "Deepgram listener stopped."}

        tts = {"state": "off", "detail": "TTS inactive.", "backend": self._tts_backend}
        if self._tts is not None:
            tts = {"state": "ready", "detail": f"{self._tts_backend} ready.", "backend": self._tts_backend}
            if self._tts_backend == "pocket":
                server_url = self._tts_server_url or "http://localhost:8003"
                try:
                    resp = requests.get(server_url, timeout=1.5)
                    resp.raise_for_status()
                    tts["detail"] = f"Pocket-TTS reachable at {server_url}."
                except Exception as exc:
                    tts["state"] = "error"
                    tts["detail"] = f"Pocket-TTS unavailable: {exc}"
            elif self._tts_backend in ("local", "qwen"):
                tts["detail"] = "Local TTS model loaded."
            elif self._tts_backend == "elevenlabs":
                tts["detail"] = "ElevenLabs streaming client ready."

        return {
            "active": self._started,
            "output_only": self._output_only,
            "filler_enabled": self._filler_enabled,
            "tts_backend": self._tts_backend,
            "mic_ready": self._mic is not None,
            "speaker_ready": self._speaker is not None,
            "stt": stt,
            "tts": tts,
        }

    @property
    def audio_started(self) -> bool:
        return bool(self._speaker is not None and self._speaker._audio_started.is_set())

    def defer_next_user_turn(self, prefix: str) -> None:
        cleaned = (prefix or "").strip()
        if not cleaned:
            return
        self._deferred_input_prefix = cleaned

    def merge_deferred_input(self, text: str) -> str:
        cleaned = (text or "").strip()
        prefix = self._deferred_input_prefix.strip()
        self._deferred_input_prefix = ""
        if not prefix:
            return cleaned
        if not cleaned:
            return prefix
        return f"{prefix} {cleaned}"

    def prune_stale_transcripts(self, accepted_text: str) -> int:
        """Drop queued transcript fragments already covered by an accepted utterance."""
        normalized = " ".join((accepted_text or "").lower().split()).strip()
        if len(normalized) < 4:
            return 0

        drained = 0
        items = []
        while not self._event_queue.empty():
            try:
                item = self._event_queue.get_nowait()
            except queue.Empty:
                break
            if not isinstance(item, str):
                items.append(item)
                continue
            if item in (VOICE_OFF, EXIT, VOICE_ERROR) or item.startswith("/"):
                items.append(item)
                continue
            candidate = " ".join(item.lower().split()).strip()
            if len(candidate) >= 4 and candidate in normalized:
                drained += 1
                continue
            items.append(item)

        for item in items:
            self._event_queue.put(item)
        return drained

    def start(self):
        """Initialize and start all voice components."""
        # Resolve string device names to indices
        self._input_device = resolve_device(self._input_device, "input")
        self._output_device = resolve_device(self._output_device, "output")

        # Set up AEC if enabled — speaker reports played audio for echo reference
        on_output = None
        if self._aec_enabled:
            from character_eng.aec import LiveKitAEC, resample_to_16k

            self._aec = LiveKitAEC(stream_delay_ms=0)

            def _feed_aec(played_bytes: bytes):
                """Resample speaker output to 16kHz and feed AEC reference."""
                if self._aec is not None:
                    pcm_16k = resample_to_16k(played_bytes, self._speaker._actual_rate)
                    self._aec.feed_playback(pcm_16k)
                if self._output_audio_hook is not None:
                    self._output_audio_hook(played_bytes)

            on_output = _feed_aec
        elif self._output_audio_hook is not None:
            on_output = self._output_audio_hook

        self._speaker = SpeakerStream(device=self._output_device, on_output=on_output)
        self._speaker.start()

        if self._tts_backend in ("local", "qwen"):
            from character_eng.local_tts import LocalTTS, load_model

            # Eagerly load model + extract voice at startup
            load_model(self._tts_model, self._tts_device, self._ref_audio,
                        ref_text=self._ref_text)
            self._tts = LocalTTS(
                on_audio=self._speaker.enqueue,
                ref_audio_path=self._ref_audio,
                model_id=self._tts_model,
                device=self._tts_device,
                ref_text=self._ref_text,
            )
        elif self._tts_backend == "pocket":
            from character_eng.pocket_tts import PocketTTS

            server_url = self._tts_server_url or "http://localhost:8003"
            self._ensure_pocket_server(server_url)
            self._tts = PocketTTS(
                on_audio=self._speaker.enqueue,
                server_url=server_url,
                ref_audio=self._ref_audio,
                voice="" if self._pocket_voice else "alba",
            )
        else:
            self._tts = ElevenLabsTTS(
                on_audio=self._speaker.enqueue,
                voice_id=self._voice_id,
            )

        if self._filler_enabled:
            from character_eng.filler_audio import FillerBank

            server_url = self._tts_server_url or "http://localhost:8003"
            self._ensure_pocket_server(server_url)
            voice_path = ""
            if self._pocket_voice:
                voice_path = self._pocket_voice
                voice_candidate = Path(voice_path)
                if not voice_candidate.is_absolute():
                    voice_path = str(Path(__file__).resolve().parent.parent / voice_candidate)
            self._filler_bank = FillerBank(
                server_url,
                voice=voice_path,
                ref_audio=self._ref_audio,
            )
            try:
                warmed = self._filler_bank.prewarm()
                sys.stderr.write(f"[Pocket-TTS] Prewarmed {warmed} filler clips.\n")
                sys.stderr.flush()
            except Exception as exc:
                sys.stderr.write(f"[Pocket-TTS] Filler prewarm failed: {exc}\n")
                sys.stderr.flush()
                self._filler_enabled = False
                self._filler_bank = None

        if not self._output_only:
            self._stt = DeepgramSTT(
                on_transcript=self._on_transcript,
                on_turn_start=self._on_turn_start,
            )
            self._stt.start()

            self._mic = MicStream(on_audio=self._on_mic_audio, device=self._input_device)
            self._mic.start()

            self._keys = KeyListener(self._event_queue, on_voice_off=self.cancel_speech)
            self._keys.start()

        self._started = True

    def _ensure_pocket_server(self, server_url: str):
        """Check if Pocket-TTS server is running; start it if not."""
        import requests as _requests

        parsed = urlparse(server_url)
        port = parsed.port or 8003

        # Check if server is already running
        try:
            _requests.get(server_url, timeout=2)
            sys.stderr.write(f"[Pocket-TTS] Server already running on port {port}\n")
            sys.stderr.flush()
            return
        except Exception:
            pass

        # Build command — resolve binary path for venv compatibility
        from character_eng.pocket_runtime import ensure_pocket_tts_binary

        pocket_bin = ensure_pocket_tts_binary()
        if pocket_bin is None:
            raise RuntimeError("pocket-tts is unavailable and could not be bootstrapped")

        cmd = [pocket_bin, "serve", "--port", str(port)]
        if self._pocket_voice:
            # Resolve relative paths against project root
            voice_path = Path(self._pocket_voice)
            if not voice_path.is_absolute():
                voice_path = Path(__file__).resolve().parent.parent / voice_path
            cmd.extend(["--voice", str(voice_path)])

        sys.stderr.write(f"[Pocket-TTS] Starting server on port {port}...\n")
        sys.stderr.flush()

        self._pocket_server = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,  # own process group so we can kill children too
        )
        self._pocket_we_started = True

        # Poll until server responds
        deadline = time.time() + 15.0
        while time.time() < deadline:
            time.sleep(0.5)
            if self._pocket_server.poll() is not None:
                sys.stderr.write(
                    f"[Pocket-TTS] Server process exited with code {self._pocket_server.returncode}\n"
                )
                sys.stderr.flush()
                self._pocket_we_started = False
                self._pocket_server = None
                raise RuntimeError("Pocket-TTS server failed to start")
            try:
                _requests.get(server_url, timeout=1)
                sys.stderr.write("[Pocket-TTS] Server ready.\n")
                hint_cmd = f"pocket-tts serve --port {port}"
                if self._pocket_voice:
                    hint_cmd += f" --voice {voice_path}"
                sys.stderr.write(
                    f"[Pocket-TTS] Tip: run in background for faster startup next time:\n"
                    f"  {hint_cmd} &\n"
                )
                sys.stderr.flush()
                return
            except Exception:
                continue

        # Timeout
        self._stop_pocket_server()
        raise RuntimeError("Pocket-TTS server did not become ready within 15s")

    def _stop_pocket_server(self):
        """Stop the Pocket-TTS server if we started it."""
        if not self._pocket_we_started or self._pocket_server is None:
            return

        parsed = urlparse(self._tts_server_url or "http://localhost:8003")
        port = parsed.port or 8003

        sys.stderr.write("[Pocket-TTS] Stopping server...\n")
        sys.stderr.flush()
        # Kill the entire process group (server + uvicorn child)
        pgid = os.getpgid(self._pocket_server.pid)
        os.killpg(pgid, 15)  # SIGTERM
        try:
            self._pocket_server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(pgid, 9)  # SIGKILL
            self._pocket_server.wait(timeout=2)

        hint_cmd = f"pocket-tts serve --port {port}"
        if self._pocket_voice:
            voice_path = Path(self._pocket_voice)
            if not voice_path.is_absolute():
                voice_path = Path(__file__).resolve().parent.parent / voice_path
            hint_cmd += f" --voice {voice_path}"
        sys.stderr.write(
            f"[Pocket-TTS] Server stopped. For persistent server, run:\n"
            f"  {hint_cmd} &\n"
        )
        sys.stderr.flush()
        self._pocket_server = None
        self._pocket_we_started = False

    def stop(self):
        """Stop all voice components and cancel timers."""
        self._started = False
        self._cancel_auto_beat()
        self.cancel_latency_filler()

        if self._mic is not None:
            self._mic.stop()
            self._mic = None
        if self._stt is not None:
            self._stt.stop()
            self._stt = None
        if self._tts is not None:
            self._tts.close()
            self._tts = None
        if self._speaker is not None:
            self._speaker.stop()
            self._speaker = None
        if self._keys is not None:
            self._keys.stop()
            self._keys = None
        if self._aec is not None:
            self._aec.close()
            self._aec = None

        self._stop_pocket_server()

        # Drain event queue
        while not self._event_queue.empty():
            try:
                self._event_queue.get_nowait()
            except queue.Empty:
                break

    def set_filler_enabled(self, enabled: bool) -> None:
        self._filler_enabled = enabled
        if not enabled:
            self.cancel_latency_filler()

    def start_latency_filler(self, response_text: str = "", expression: str = "") -> None:
        if (
            not self._filler_enabled
            or self._filler_bank is None
            or self._speaker is None
            or self._cancelled.is_set()
        ):
            return
        self.cancel_latency_filler()
        self._filler_cancel.clear()

        def _run():
            if self._speaker is None:
                return
            if self._speaker.wait_for_audio(timeout=self._filler_lead_ms / 1000):
                return
            if self._filler_cancel.is_set() or self._cancelled.is_set():
                return
            try:
                choice = self._filler_bank.pick(
                    sentiment="neutral",
                    response_text=response_text,
                    expression=expression,
                )
                clip = self._filler_bank.clip_for_choice(choice)
            except Exception:
                return
            if clip and not self._filler_cancel.is_set() and not self._cancelled.is_set():
                self._trace("assistant_filler", {
                    "phrase": choice.phrase,
                    "sentiment": choice.sentiment,
                })
                self._speaker.enqueue(clip)

        self._filler_thread = threading.Thread(target=_run, daemon=True)
        self._filler_thread.start()

    def cancel_latency_filler(self) -> None:
        self._filler_cancel.set()
        if self._filler_thread is not None:
            self._filler_thread.join(timeout=1)
            self._filler_thread = None

    @property
    def is_started(self) -> bool:
        return self._started

    def wait_for_input(self, timeout: float | None = None) -> str:
        """Block until user speaks, auto-beat fires, or hotkey pressed.

        Returns:
            Transcript text, "/beat", or sentinel strings (VOICE_OFF, EXIT, VOICE_ERROR)
        """
        wait_timeout = 0.5 if timeout is None else timeout
        while True:
            try:
                return self._event_queue.get(timeout=wait_timeout)
            except queue.Empty:
                if timeout is not None:
                    raise
                if not self._started:
                    return EXIT

    def inject_input_audio(self, pcm: bytes) -> None:
        """Feed prerecorded mono PCM16 audio through the live mic/STT path."""
        if not pcm:
            return
        self._on_mic_audio(pcm)

    @property
    def speaker_playback_ratio(self) -> float:
        """Fraction of last utterance's audio that was actually played (0.0-1.0)."""
        if self._speaker is not None:
            return self._speaker.playback_ratio
        return 1.0

    def speak(self, text_chunks: Iterator[str]) -> None:
        """Feed LLM streaming chunks to TTS, wait for speaker to finish, start auto-beat.

        Checks _cancelled event between chunks so barge-in can interrupt.
        """
        self._barged_in = False
        self._cancelled.clear()
        self._is_speaking = True
        self.cancel_latency_filler()
        started_at = time.time()
        self._trace("assistant_tts_live_start", {})
        if self._speaker is not None:
            self._speaker.reset_counters()
        for chunk in text_chunks:
            if self._cancelled.is_set():
                break
            if self._tts is not None:
                self._tts.send_text(chunk)

        # Signal end of text input
        if not self._cancelled.is_set() and self._tts is not None:
            self._tts.flush()
            self.start_latency_filler()

        # Wait for ElevenLabs to finish generating
        if not self._cancelled.is_set() and self._tts is not None:
            self._tts.wait_for_done(timeout=15.0)
            self._trace("assistant_tts_live_synth_done", {
                "elapsed_ms": int((time.time() - started_at) * 1000),
            })

        # Wait for first audio to arrive at the speaker
        first_audio_at = None
        if not self._cancelled.is_set() and self._speaker is not None:
            if self._speaker.wait_for_audio(timeout=0.5):
                first_audio_at = time.time()
                self._trace("assistant_tts_live_first_audio", {
                    "elapsed_ms": int((first_audio_at - started_at) * 1000),
                })
        self.cancel_latency_filler()

        # Wait for speaker to finish playback
        if self._speaker is not None and not self._cancelled.is_set():
            while not self._speaker.is_done():
                if self._cancelled.is_set():
                    break
                time.sleep(0.05)

        # Close TTS connection to reset for next utterance
        if self._tts is not None:
            self._tts.close()
        self._is_speaking = False
        done_at = time.time()
        self._trace("assistant_tts_live_done", {
            "elapsed_ms": int((done_at - started_at) * 1000),
            "playback_ms": int((done_at - (first_audio_at or started_at)) * 1000),
        })

    def speak_text(self, text: str) -> None:
        """One-shot TTS for pre-rendered beats."""
        self._barged_in = False
        self._cancelled.clear()
        self._is_speaking = True
        self.cancel_latency_filler()
        started_at = time.time()
        self._trace("assistant_tts_live_start", {"text": text})
        if self._speaker is not None:
            self._speaker.reset_counters()
        if self._tts is not None:
            self._tts.send_text(text)
            self._tts.flush()
            self.start_latency_filler(text)

        # Wait for ElevenLabs to finish generating
        if not self._cancelled.is_set() and self._tts is not None:
            self._tts.wait_for_done(timeout=15.0)
            self._trace("assistant_tts_live_synth_done", {
                "elapsed_ms": int((time.time() - started_at) * 1000),
                "text": text,
            })

        # Wait for first audio to arrive at the speaker
        first_audio_at = None
        if not self._cancelled.is_set() and self._speaker is not None:
            if self._speaker.wait_for_audio(timeout=0.5):
                first_audio_at = time.time()
                self._trace("assistant_tts_live_first_audio", {
                    "elapsed_ms": int((first_audio_at - started_at) * 1000),
                    "text": text,
                })
        self.cancel_latency_filler()

        # Wait for speaker to finish playback
        if self._speaker is not None and not self._cancelled.is_set():
            while not self._speaker.is_done():
                if self._cancelled.is_set():
                    break
                time.sleep(0.05)

        # Close TTS for next utterance
        if self._tts is not None:
            self._tts.close()
        self._is_speaking = False
        done_at = time.time()
        self._trace("assistant_tts_live_done", {
            "elapsed_ms": int((done_at - started_at) * 1000),
            "playback_ms": int((done_at - (first_audio_at or started_at)) * 1000),
            "text": text,
        })

    def cancel_speech(self):
        """Barge-in: cancel current LLM+TTS+speaker, cancel auto-beat."""
        self._cancelled.set()
        self._barged_in = True
        self._is_speaking = False
        self._cancel_auto_beat()
        self.cancel_latency_filler()
        if self._tts is not None:
            self._tts.close()
        if self._speaker is not None:
            self._speaker.flush()

    def _on_mic_audio(self, data: bytes):
        """Forward mic audio to STT, processing through AEC if enabled.

        With AEC on: mic stays live always, audio cleaned via AEC before STT.
        With AEC off: mic muted during playback (legacy echo suppression).
        """
        if self._input_audio_hook is not None:
            self._input_audio_hook(data)
        if self._aec is not None:
            cleaned = self._aec.process_capture(data)
            if self._stt is not None and len(cleaned) > 0:
                self._stt.send_audio(cleaned)
        else:
            # Legacy: mute mic during playback
            if self._is_speaking:
                return
            if self._stt is not None:
                self._stt.send_audio(data)

    def _on_transcript(self, text: str):
        """Called by DeepgramSTT when a complete utterance is detected.

        With AEC, barge-in triggers here (on real transcript) instead of on
        SpeechStarted, since AEC residual can cause false VAD triggers.
        """
        self._last_transcript_final_at = time.time()
        self._last_speech_ended_at = self._stt._last_is_final_at if self._stt is not None else None
        self._last_transcript_text = text
        self._trace("user_transcript_final", {"text": text})
        if self._is_speaking and not self.audio_started:
            self._cancel_auto_beat()
            self.cancel_speech()
        if self._aec is not None and self._is_speaking:
            self._cancel_auto_beat()
            self.cancel_speech()
        self._event_queue.put(text)

    def _on_turn_start(self):
        """Called by DeepgramSTT when user starts speaking (SpeechStarted VAD).

        With AEC: ignored during playback (too sensitive to residual echo).
        Barge-in is handled in _on_transcript() instead.
        Without AEC: triggers barge-in during playback (legacy behavior).
        """
        self._last_speech_started_at = time.time()
        self._trace("user_speech_started", {})
        if self._is_speaking:
            if not self.audio_started:
                self._cancel_auto_beat()
                self.cancel_speech()
                return
            if self._aec is not None:
                # AEC mode: ignore SpeechStarted during playback — barge-in
                # only triggers on real transcript in _on_transcript()
                # But still cancel auto-beat — speech was genuinely detected
                self._cancel_auto_beat()
                return
            # Legacy: barge-in on SpeechStarted
            self._cancel_auto_beat()
            self.cancel_speech()
        else:
            # Real speech — cancel auto-beat
            self._cancel_auto_beat()

    def consume_input_timing(self, expected_text: str = "") -> dict:
        """Return and clear cached speech/transcript timing for the latest utterance."""
        started_at = self._last_speech_started_at
        final_at = self._last_transcript_final_at
        speech_ended_at = self._last_speech_ended_at
        transcript_text = self._last_transcript_text
        self._last_speech_started_at = None
        self._last_transcript_final_at = None
        self._last_speech_ended_at = None
        self._last_transcript_text = ""

        if started_at is None and final_at is None and not transcript_text:
            return {}

        payload: dict[str, object] = {
            "input_source": "voice",
        }
        if started_at is not None:
            payload["speech_started_ts"] = started_at
        if final_at is not None:
            payload["transcript_final_ts"] = final_at
        if speech_ended_at is not None:
            payload["speech_ended_ts"] = speech_ended_at
        if started_at is not None and final_at is not None:
            payload["stt_ms"] = round(max(0.0, final_at - started_at) * 1000)
        if speech_ended_at is not None and final_at is not None:
            payload["endpointing_ms"] = round(max(0.0, final_at - speech_ended_at) * 1000)
        cleaned_transcript = transcript_text.strip()
        cleaned_expected = expected_text.strip()
        if cleaned_transcript:
            payload["stt_text"] = cleaned_transcript
            payload["stt_text_match"] = cleaned_transcript == cleaned_expected if cleaned_expected else True
        return payload

    def _start_auto_beat(self):
        """Start auto-beat timer — fires /beat after AUTO_BEAT_DELAY seconds."""
        self._cancel_auto_beat()
        self._auto_beat_countdown_active = True
        self._auto_beat_timer = threading.Timer(
            AUTO_BEAT_DELAY,
            self._fire_auto_beat,
        )
        self._auto_beat_timer.daemon = True
        self._auto_beat_timer.start()

    def _cancel_auto_beat(self):
        """Cancel the auto-beat timer if running."""
        self._auto_beat_countdown_active = False
        if self._auto_beat_timer is not None:
            self._auto_beat_timer.cancel()
            self._auto_beat_timer = None

    def cancel_and_drain_auto_beat(self):
        """Cancel auto-beat timer and drain any stale /beat from the event queue.

        Call this when user input is received to prevent stale auto-beat
        commands from firing during or after command processing.
        """
        self._cancel_auto_beat()
        # Drain any /beat entries that the timer already put on the queue
        drained = 0
        items = []
        while not self._event_queue.empty():
            try:
                item = self._event_queue.get_nowait()
                if item == "/beat":
                    drained += 1
                else:
                    items.append(item)
            except queue.Empty:
                break
        # Put back non-beat items
        for item in items:
            self._event_queue.put(item)

    def _fire_auto_beat(self):
        """Auto-beat timer callback — puts /beat on the event queue."""
        self._auto_beat_countdown_active = False
        # Don't fire if user is mid-utterance or has a pending transcript
        if self._stt is not None and (
            self._stt._is_speaking or self._stt._pending_transcript
        ):
            return
        self._event_queue.put("/beat")
