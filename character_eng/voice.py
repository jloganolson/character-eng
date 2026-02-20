"""Voice I/O layer — Deepgram STT + ElevenLabs TTS + sounddevice mic/speaker.

Wraps around the existing text pipeline as an I/O layer. Text mode continues
to work unchanged. Install with: uv sync --extra voice
"""

from __future__ import annotations

import base64
import json
import os
import queue
import select
import struct
import sys
import termios
import threading
import time
import tty
from typing import Callable, Iterator

from dotenv import load_dotenv

load_dotenv()

# Sentinel strings returned by wait_for_input()
TOGGLE_VOICE = "__TOGGLE_VOICE__"
EXIT = "__EXIT__"
VOICE_ERROR = "__VOICE_ERROR__"

# Key mappings for hotkeys
_KEY_MAP = {
    "w": "/world",
    "b": "/beat",
    "p": "/plan",
    "g": "/goals",
    "t": "/trace",
    "\x1b": TOGGLE_VOICE,  # Escape
    "\x03": EXIT,  # Ctrl+C
}

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


def check_voice_available() -> tuple[bool, str]:
    """Check if voice dependencies and API keys are available.

    Returns (available, reason) tuple.
    """
    missing = []
    try:
        import sounddevice  # noqa: F401
    except ImportError:
        missing.append("sounddevice")
    try:
        import numpy  # noqa: F401
    except ImportError:
        missing.append("numpy")
    try:
        import deepgram  # noqa: F401
    except ImportError:
        missing.append("deepgram-sdk")
    try:
        import websocket  # noqa: F401
    except ImportError:
        missing.append("websocket-client")

    if missing:
        return False, f"Missing packages: {', '.join(missing)}. Install with: uv sync --extra voice"

    if not os.environ.get("DEEPGRAM_API_KEY"):
        return False, "DEEPGRAM_API_KEY not set in .env"
    if not os.environ.get("ELEVENLABS_API_KEY"):
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

    def __init__(self, device: int | None = None):
        self._device = device
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

        # Build sample rate fallback chain: native 24kHz → 48kHz → device default
        rates_to_try = [self.NATIVE_RATE, 48000]
        if self._device is not None:
            dev_info = sd.query_devices(self._device)
            dev_rate = int(dev_info["default_samplerate"])
            if dev_rate not in rates_to_try:
                rates_to_try.append(dev_rate)

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
                self._stream = sd.RawOutputStream(**kwargs)
                self._stream.start()
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

        # Wait for WebSocket to be fully open before accepting audio
        self._ws_open = threading.Event()

        def _on_open(_):
            self._ws_open.set()

        # Register event handlers
        self._socket.on(EventType.OPEN, _on_open)
        self._socket.on(EventType.MESSAGE, self._on_message)
        self._socket.on(EventType.ERROR, self._on_error)

        # start_listening() BLOCKS (infinite WebSocket receive loop) — run on daemon thread
        self._listener_thread = threading.Thread(
            target=self._socket.start_listening, daemon=True
        )
        self._listener_thread.start()
        self._ws_open.wait(timeout=5)

    def send_audio(self, data: bytes):
        if self._socket is not None:
            try:
                self._socket.send_media(data)
            except Exception:
                pass

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

        if speech_final and self._pending_transcript:
            text = self._pending_transcript.strip()
            self._pending_transcript = ""
            self._is_speaking = False
            if text:
                self._on_transcript(text)

    def _on_error(self, error):
        """Called on Deepgram errors."""
        pass

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
        except Exception:
            self._ws = None
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
                        self._on_audio(pcm)
                except Exception:
                    break
        finally:
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

    def __init__(self, event_queue: queue.Queue):
        self._event_queue = event_queue
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
                        self._event_queue.put(_KEY_MAP[ch])
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

    def __init__(self, voice_id: str = "", input_device: int | None = None, output_device: int | None = None, mic_mute_during_playback: bool = True):
        self._event_queue: queue.Queue[str] = queue.Queue()
        self._cancelled = threading.Event()
        self._is_speaking = False
        self._barged_in = False
        self._auto_beat_timer: threading.Timer | None = None
        self._auto_beat_countdown_active = False
        self._mic_mute_during_playback = mic_mute_during_playback

        # Components (created in start())
        self._speaker: SpeakerStream | None = None
        self._mic: MicStream | None = None
        self._stt: DeepgramSTT | None = None
        self._tts: ElevenLabsTTS | None = None
        self._keys: KeyListener | None = None
        self._voice_id = voice_id
        self._input_device = input_device
        self._output_device = output_device
        self._started = False

    def start(self):
        """Initialize and start all voice components."""
        self._speaker = SpeakerStream(device=self._output_device)
        self._speaker.start()

        self._tts = ElevenLabsTTS(
            on_audio=self._speaker.enqueue,
            voice_id=self._voice_id,
        )

        self._stt = DeepgramSTT(
            on_transcript=self._on_transcript,
            on_turn_start=self._on_turn_start,
        )
        self._stt.start()

        self._mic = MicStream(on_audio=self._on_mic_audio, device=self._input_device)
        self._mic.start()

        self._keys = KeyListener(self._event_queue)
        self._keys.start()

        self._started = True

    def stop(self):
        """Stop all voice components and cancel timers."""
        self._started = False
        self._cancel_auto_beat()

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

        # Drain event queue
        while not self._event_queue.empty():
            try:
                self._event_queue.get_nowait()
            except queue.Empty:
                break

    @property
    def is_started(self) -> bool:
        return self._started

    def wait_for_input(self) -> str:
        """Block until user speaks, auto-beat fires, or hotkey pressed.

        Returns:
            Transcript text, "/beat", or sentinel strings (TOGGLE_VOICE, EXIT, VOICE_ERROR)
        """
        while True:
            try:
                return self._event_queue.get(timeout=0.5)
            except queue.Empty:
                if not self._started:
                    return EXIT

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

        # Wait for ElevenLabs to finish generating
        if not self._cancelled.is_set() and self._tts is not None:
            self._tts.wait_for_done(timeout=15.0)

        # Wait for first audio to arrive at the speaker
        if not self._cancelled.is_set() and self._speaker is not None:
            self._speaker.wait_for_audio(timeout=0.5)

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

        # Start auto-beat timer if not cancelled
        if not self._cancelled.is_set():
            self._start_auto_beat()

    def speak_text(self, text: str) -> None:
        """One-shot TTS for pre-rendered beats."""
        self._barged_in = False
        self._cancelled.clear()
        self._is_speaking = True
        if self._speaker is not None:
            self._speaker.reset_counters()
        if self._tts is not None:
            self._tts.send_text(text)
            self._tts.flush()

        # Wait for ElevenLabs to finish generating
        if not self._cancelled.is_set() and self._tts is not None:
            self._tts.wait_for_done(timeout=15.0)

        # Wait for first audio to arrive at the speaker
        if not self._cancelled.is_set() and self._speaker is not None:
            self._speaker.wait_for_audio(timeout=0.5)

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

        if not self._cancelled.is_set():
            self._start_auto_beat()

    def cancel_speech(self):
        """Barge-in: cancel current LLM+TTS+speaker, cancel auto-beat."""
        self._cancelled.set()
        self._barged_in = True
        self._is_speaking = False
        self._cancel_auto_beat()
        if self._tts is not None:
            self._tts.close()
        if self._speaker is not None:
            self._speaker.flush()

    def _on_mic_audio(self, data: bytes):
        """Forward mic audio to STT, unless muted during TTS playback (echo suppression).

        When mic_mute_during_playback is False (e.g. device has hardware AEC),
        audio is always forwarded, enabling voice barge-in during playback.
        """
        if self._mic_mute_during_playback and self._is_speaking:
            return
        if self._stt is not None:
            self._stt.send_audio(data)

    def _on_transcript(self, text: str):
        """Called by DeepgramSTT when a complete utterance is detected."""
        self._event_queue.put(text)

    def _on_turn_start(self):
        """Called by DeepgramSTT when user starts speaking (barge-in trigger).

        Always cancels the auto-beat timer (user spoke, so don't auto-advance).
        Only cancels active speech when _is_speaking is True.
        Uses \\n prefix (not \\r) to preserve partial response text on the terminal.
        """
        self._cancel_auto_beat()
        if self._is_speaking:
            if self._started:
                sys.stdout.write("\n  [heard — interrupting]\n")
                sys.stdout.flush()
            self.cancel_speech()

    def _start_auto_beat(self):
        """Start auto-beat timer — fires /beat after AUTO_BEAT_DELAY seconds."""
        self._cancel_auto_beat()
        self._auto_beat_countdown_active = True
        if self._started:
            threading.Thread(target=self._countdown_loop, daemon=True).start()
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

    def _fire_auto_beat(self):
        """Auto-beat timer callback — puts /beat on the event queue."""
        self._auto_beat_countdown_active = False
        self._event_queue.put("/beat")

    def _countdown_loop(self):
        """Print auto-beat countdown dots to stdout. Clears line on cancel or fire."""
        sys.stdout.write("  auto-beat ")
        sys.stdout.flush()
        for _ in range(int(AUTO_BEAT_DELAY)):
            if not self._auto_beat_countdown_active:
                sys.stdout.write("\r\033[K")
                sys.stdout.flush()
                return
            time.sleep(1.0)
            if not self._auto_beat_countdown_active:
                sys.stdout.write("\r\033[K")
                sys.stdout.flush()
                return
            sys.stdout.write(".")
            sys.stdout.flush()
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()
