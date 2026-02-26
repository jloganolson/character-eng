"""AEC Live Demo — interactive robot-speak + barge-in test with real-time HUD.

Press SPACE to trigger robot speech via Pocket TTS.
AEC cleans the mic in real-time.
Deepgram STT + Silero VAD run on the cleaned signal.
Terminal HUD shows levels, VAD, STT events, and an event log.

Usage:
    uv run python experiments/aec/live_demo.py
    uv run python experiments/aec/live_demo.py --tts-url http://host:8003
"""

import atexit
import os
import queue
import random
import select
import shutil
import struct
import subprocess
import sys
import termios
import threading
import time
import tty
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import sounddevice as sd

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Ensure experiments/aec/ is on path for sibling imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from livekit_aec import LiveKitAEC  # noqa: E402

# ── Constants ────────────────────────────────────────────────────────────────

AEC_RATE = 16000       # AEC + STT sample rate
SPEAKER_RATE = 48000   # Speaker output rate
TTS_RATE = 24000       # Pocket TTS native rate
MIC_BLOCKSIZE = 160    # 10ms at 16kHz — matches AEC frame size
SPK_BLOCKSIZE = 480    # 10ms at 48kHz
REF_RATIO = AEC_RATE / SPEAKER_RATE  # 1/3

HUD_REFRESH = 0.1      # 10Hz
HUD_HEIGHT = 18         # total lines the HUD occupies
EVENT_LOG_SIZE = 8

ROBOT_LINES = [
    "Well howdy there, welcome to Greg's Lemonade Stand! Can I interest you in a refreshing glass of lemonade?",
    "You know what makes my lemonade special? It's the love I put into every squeeze. Also the secret robot arm strength.",
    "I've been standing here all day waiting for someone like you to come along. Business has been slower than my boot sequence.",
    "Between you and me, I think the hot dog cart across the street is getting jealous of my lemon game.",
    "Every morning I wake up and think, today's gonna be the day I sell a hundred glasses. Then reality kicks in.",
    "My circuits are telling me you look like a person who could use some vitamin C and a friendly conversation.",
]


# ── Audio utilities ──────────────────────────────────────────────────────────

def resample_linear(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Linear interpolation resample (from aec_poc.py)."""
    if orig_sr == target_sr:
        return audio
    ratio = target_sr / orig_sr
    n_out = int(len(audio) * ratio)
    indices = np.arange(n_out) / ratio
    idx = np.clip(indices.astype(int), 0, len(audio) - 2)
    frac = indices - idx
    if audio.dtype == np.int16:
        audio_f = audio.astype(np.float64)
        result = audio_f[idx] * (1 - frac) + audio_f[idx + 1] * frac
        return np.clip(result, -32768, 32767).astype(np.int16)
    return (audio[idx] * (1 - frac) + audio[idx + 1] * frac).astype(audio.dtype)


def rms_db(pcm_int16: np.ndarray) -> float:
    """RMS in dB (ref: full-scale int16)."""
    rms = np.sqrt(np.mean(pcm_int16.astype(np.float64) ** 2))
    if rms == 0:
        return -120.0
    return 20 * np.log10(rms / 32768.0)


# ── Deepgram live STT ───────────────────────────────────────────────────────

class LiveSTT:
    """Deepgram STT with live callbacks for transcript + interim text."""

    def __init__(self, on_transcript=None):
        self.transcripts: list[str] = []
        self.interim_text = ""
        self.speech_started = False
        self._on_transcript = on_transcript
        self._socket = None
        self._ctx = None
        self._thread = None

    def start(self):
        from deepgram import DeepgramClient
        from deepgram.core.events import EventType

        client = DeepgramClient(api_key=os.environ["DEEPGRAM_API_KEY"])
        self._ctx = client.listen.v1.connect(
            model="nova-3",
            encoding="linear16",
            sample_rate=str(AEC_RATE),
            channels="1",
            interim_results="true",
            utterance_end_ms="1500",
            vad_events="true",
            endpointing="300",
        )
        self._socket = self._ctx.__enter__()

        ws_open = threading.Event()

        def _on_open(_):
            ws_open.set()

        def _on_message(message):
            if isinstance(message, dict):
                msg_type = message.get("type", "")
            else:
                msg_type = getattr(message, "type", "")

            if msg_type == "SpeechStarted":
                self.speech_started = True
                return

            if msg_type != "Results":
                return

            try:
                if isinstance(message, dict):
                    alts = message.get("channel", {}).get("alternatives", [])
                    transcript = alts[0].get("transcript", "") if alts else ""
                    is_final = message.get("is_final", False)
                else:
                    transcript = message.channel.alternatives[0].transcript
                    is_final = message.is_final
            except (AttributeError, IndexError, KeyError):
                return

            if transcript:
                if is_final:
                    self.interim_text = ""
                    text = transcript.strip()
                    self.transcripts.append(text)
                    if self._on_transcript:
                        self._on_transcript(text)
                else:
                    self.interim_text = transcript.strip()

        self._socket.on(EventType.OPEN, _on_open)
        self._socket.on(EventType.MESSAGE, _on_message)

        self._thread = threading.Thread(target=self._socket.start_listening, daemon=True)
        self._thread.start()
        ws_open.wait(timeout=5)

    def send_audio(self, data: bytes):
        if self._socket:
            try:
                self._socket.send_media(data)
            except Exception:
                pass

    def stop(self):
        if self._socket:
            try:
                from deepgram.extensions.types.sockets.listen_v1_control_message import (
                    ListenV1ControlMessage,
                )
                self._socket.send_control(ListenV1ControlMessage(type="CloseStream"))
            except Exception:
                pass
            self._socket = None
        if self._ctx:
            try:
                self._ctx.__exit__(None, None, None)
            except Exception:
                pass
            self._ctx = None
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None


# ── Silero VAD (background thread) ──────────────────────────────────────────

class SileroVAD:
    """Runs Silero VAD on a background thread, exposes a live probability level."""

    def __init__(self):
        self.level = 0.0
        self._queue: queue.Queue[bytes] = queue.Queue()
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def feed(self, pcm_16k: bytes):
        self._queue.put(pcm_16k)

    def _run(self):
        import torch
        from silero_vad import load_silero_vad

        model = load_silero_vad()
        model.reset_states()

        buf = bytearray()
        window_bytes = 512 * 2  # 512 samples at 16kHz = 32ms

        while self._running:
            try:
                data = self._queue.get(timeout=0.5)
                buf.extend(data)
                while len(buf) >= window_bytes:
                    chunk = np.frombuffer(bytes(buf[:window_bytes]), dtype=np.int16)
                    del buf[:window_bytes]
                    f32 = chunk.astype(np.float32) / 32768.0
                    prob = model(torch.from_numpy(f32), 16000).item()
                    self.level = prob
            except queue.Empty:
                pass

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)


# ── TTS streaming ───────────────────────────────────────────────────────────

WAV_HEADER_MIN = 44


def stream_tts(text: str, server_url: str, voice: str, on_pcm: callable):
    """POST to Pocket TTS, parse WAV header, deliver 24kHz int16 PCM chunks."""
    import requests

    url = f"{server_url}/tts"
    data = {"text": text, "voice_url": voice}
    resp = requests.post(url, data=data, stream=True, timeout=120)
    resp.raise_for_status()

    header_buf = b""
    header_parsed = False

    for chunk in resp.iter_content(chunk_size=4800):
        if not header_parsed:
            header_buf += chunk
            if len(header_buf) >= WAV_HEADER_MIN:
                if header_buf[:4] == b"RIFF" and header_buf[8:12] == b"WAVE":
                    pos = 12
                    while pos < len(header_buf) - 8:
                        subchunk_id = header_buf[pos:pos + 4]
                        subchunk_size = struct.unpack_from("<I", header_buf, pos + 4)[0]
                        if subchunk_id == b"data":
                            data_start = pos + 8
                            header_parsed = True
                            pcm = header_buf[data_start:]
                            if pcm:
                                on_pcm(pcm)
                            break
                        pos += 8 + subchunk_size
                    else:
                        continue
                else:
                    header_parsed = True
                    on_pcm(header_buf)
        else:
            on_pcm(chunk)


# ── Pocket TTS server management ─────────────────────────────────────────────

_pocket_server = None
_pocket_we_started = False


def ensure_pocket_server(server_url: str, voice: str = ""):
    """Start Pocket TTS server if not already running."""
    global _pocket_server, _pocket_we_started
    import requests as _requests

    parsed = urlparse(server_url)
    port = parsed.port or 8003

    try:
        _requests.get(server_url, timeout=2)
        print(f"  Pocket TTS already running on port {port}")
        return
    except Exception:
        pass

    pocket_bin = shutil.which("pocket-tts")
    if pocket_bin is None:
        print("ERROR: pocket-tts not found in PATH. Install with: uv sync --extra voice")
        sys.exit(1)

    cmd = [pocket_bin, "serve", "--port", str(port)]
    if voice and voice != "alba":
        voice_path = Path(voice)
        if not voice_path.is_absolute():
            voice_path = Path(__file__).resolve().parent.parent / voice_path
        cmd.extend(["--voice", str(voice_path)])

    print(f"  Starting Pocket TTS on port {port}...")
    _pocket_server = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    _pocket_we_started = True

    deadline = time.time() + 15.0
    while time.time() < deadline:
        time.sleep(0.5)
        if _pocket_server.poll() is not None:
            _pocket_we_started = False
            _pocket_server = None
            print(f"ERROR: Pocket TTS server exited unexpectedly")
            sys.exit(1)
        try:
            _requests.get(server_url, timeout=1)
            print(f"  Pocket TTS ready on port {port}")
            return
        except Exception:
            continue

    stop_pocket_server()
    print("ERROR: Pocket TTS did not become ready within 15s")
    sys.exit(1)


def stop_pocket_server():
    """Stop Pocket TTS server if we started it."""
    global _pocket_server, _pocket_we_started
    if not _pocket_we_started or _pocket_server is None:
        return
    try:
        pgid = os.getpgid(_pocket_server.pid)
        os.killpg(pgid, 15)
        try:
            _pocket_server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(pgid, 9)
            _pocket_server.wait(timeout=2)
    except Exception:
        pass
    _pocket_server = None
    _pocket_we_started = False


# ── Keyboard listener ───────────────────────────────────────────────────────

class KeyListener:
    """Non-blocking keyboard reader using tty.setcbreak()."""

    def __init__(self, on_key: callable):
        self._on_key = on_key
        self._running = False
        self._thread: threading.Thread | None = None
        self._old_settings = None

    def start(self):
        if not sys.stdin.isatty():
            return
        self._old_settings = termios.tcgetattr(sys.stdin.fileno())
        atexit.register(self._restore)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _restore(self):
        if self._old_settings is not None:
            try:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, self._old_settings)
            except Exception:
                pass

    def _loop(self):
        fd = sys.stdin.fileno()
        try:
            tty.setcbreak(fd)
            while self._running:
                r, _, _ = select.select([fd], [], [], 0.1)
                if r:
                    ch = os.read(fd, 1).decode("utf-8", errors="ignore")
                    self._on_key(ch)
        finally:
            self._restore()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)
        self._restore()


# ── HUD ─────────────────────────────────────────────────────────────────────

def make_bar(value: float, lo: float, hi: float, width: int = 20) -> str:
    """Render a horizontal bar with ▓ and ░."""
    frac = max(0.0, min(1.0, (value - lo) / (hi - lo))) if hi != lo else 0.0
    filled = int(frac * width)
    return "▓" * filled + "░" * (width - filled)


def draw_hud(
    robot_speaking: bool,
    robot_line: str,
    mic_db: float,
    vad_level: float,
    stt_text: str,
    events: list[str],
    first_draw: bool = False,
):
    """Draw the HUD using ANSI escape codes."""
    lines: list[str] = []
    lines.append("══ AEC Live Demo ═══════════════════════════════════════════")

    if robot_speaking:
        short = robot_line[:48] + "..." if len(robot_line) > 48 else robot_line
        lines.append(f'  Robot:  \033[1;33mSPEAKING\033[0m  "{short}"')
    else:
        lines.append("  Robot:  (idle)")

    mic_bar = make_bar(mic_db, -90, -20)
    lines.append(f"  Mic:    {mic_bar}  {mic_db:6.1f} dB (cleaned)")

    vad_bar = make_bar(vad_level, 0, 1)
    lines.append(f"  VAD:    {vad_bar}  {vad_level:.2f}")

    if stt_text:
        lines.append(f'  STT:    \033[1;32m"{stt_text[:50]}"\033[0m')
    else:
        lines.append("  STT:    (idle)")

    lines.append("════════════════════════════════════════════════════════════")
    lines.append("  [SPACE] robot speak   [Q] quit")
    lines.append("════════════════════════════════════════════════════════════")

    # Event log — last N events, padded to fixed size
    tail = events[-EVENT_LOG_SIZE:]
    for evt in tail:
        lines.append(f"  {evt}")
    for _ in range(EVENT_LOG_SIZE - len(tail)):
        lines.append("")

    total = len(lines)

    # Move cursor up (except on first draw) and overwrite
    if not first_draw:
        sys.stdout.write(f"\033[{total}A")
    for line in lines:
        sys.stdout.write(f"\033[2K{line}\n")
    sys.stdout.flush()

    return total


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="AEC Live Demo")
    parser.add_argument("--tts-url", default="http://localhost:8003",
                        help="Pocket TTS server URL (default: http://localhost:8003)")
    parser.add_argument("--voice", default="alba",
                        help="Pocket TTS voice (default: alba)")
    args = parser.parse_args()

    if not os.environ.get("DEEPGRAM_API_KEY"):
        print("ERROR: DEEPGRAM_API_KEY not set")
        sys.exit(1)

    # ── Shared state ──
    speaker_buf = bytearray()
    ref_buf = bytearray()
    buf_lock = threading.Lock()

    robot_speaking = threading.Event()
    tts_stream_done = threading.Event()  # set after TTS HTTP stream finishes
    tts_stream_done.set()  # starts "done"
    robot_line = [""]
    robot_start_time = [0.0]
    transcript_count_at_start = [0]  # for "no false detection" check
    mic_rms = [-90.0]
    events: list[str] = []
    quit_event = threading.Event()
    tts_busy = threading.Lock()  # prevent overlapping TTS requests

    def log_event(msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        events.append(f"{ts}  {msg}")

    # ── AEC ──
    aec = LiveKitAEC(stream_delay_ms=0)

    # ── Deepgram STT ──
    def on_transcript(text: str):
        log_event(f'Human: "{text}"')

    stt = LiveSTT(on_transcript=on_transcript)

    # ── Silero VAD ──
    vad = SileroVAD()

    # ── Speaker output callback ──
    def speaker_callback(outdata, frames, time_info, status):
        bytes_needed = frames * 2
        ref_samples = int(frames * REF_RATIO)
        ref_bytes = ref_samples * 2

        with buf_lock:
            # Play audio
            avail = len(speaker_buf)
            if avail >= bytes_needed:
                outdata[:] = bytes(speaker_buf[:bytes_needed])
                del speaker_buf[:bytes_needed]
            else:
                if avail > 0:
                    outdata[:avail] = bytes(speaker_buf[:avail])
                    del speaker_buf[:avail]
                outdata[avail:] = b'\x00' * (len(outdata) - avail)
                # Mark done only when TTS stream finished AND buffer drained
                if avail == 0 and robot_speaking.is_set() and tts_stream_done.is_set():
                    elapsed = time.monotonic() - robot_start_time[0]
                    robot_speaking.clear()
                    # Check for false transcripts during robot speech
                    new_transcripts = len(stt.transcripts) - transcript_count_at_start[0]
                    if new_transcripts == 0:
                        log_event(f"Robot done ({elapsed:.1f}s) — no false detection \u2713")
                    else:
                        log_event(f"Robot done ({elapsed:.1f}s) — {new_transcripts} transcript(s)")

            # Feed AEC reference
            ref_avail = len(ref_buf)
            if ref_avail >= ref_bytes:
                ref_chunk = bytes(ref_buf[:ref_bytes])
                del ref_buf[:ref_bytes]
            else:
                ref_chunk = bytes(ref_buf[:ref_avail]) + b'\x00' * (ref_bytes - ref_avail)
                ref_buf.clear()

        aec.feed_playback(ref_chunk)

    # ── Mic input callback ──
    def mic_callback(indata, frames, time_info, status):
        raw = bytes(indata)
        cleaned = aec.process_capture(raw)
        if cleaned:
            arr = np.frombuffer(cleaned, dtype=np.int16)
            mic_rms[0] = rms_db(arr)
            stt.send_audio(cleaned)
            vad.feed(cleaned)

    # ── TTS trigger ──
    def trigger_robot_speak():
        if not tts_busy.acquire(blocking=False):
            return  # already speaking

        def _run():
            try:
                line = random.choice(ROBOT_LINES)
                robot_line[0] = line
                robot_start_time[0] = time.monotonic()
                transcript_count_at_start[0] = len(stt.transcripts)
                tts_stream_done.clear()
                robot_speaking.set()
                log_event("Robot started speaking")

                def on_pcm(pcm_24k: bytes):
                    """Receive 24kHz PCM from TTS, resample to 48kHz + 16kHz."""
                    samples = np.frombuffer(pcm_24k, dtype=np.int16)
                    if len(samples) == 0:
                        return
                    # 24k → 48k (2x repeat for integer ratio)
                    samples_48k = np.repeat(samples, 2)
                    # 24k → 16k (linear interpolation)
                    samples_16k = resample_linear(samples, TTS_RATE, AEC_RATE)
                    with buf_lock:
                        speaker_buf.extend(samples_48k.tobytes())
                        ref_buf.extend(samples_16k.tobytes())

                stream_tts(line, args.tts_url, args.voice, on_pcm)
                tts_stream_done.set()
            except Exception as e:
                log_event(f"TTS error: {e}")
                tts_stream_done.set()
                robot_speaking.clear()
            finally:
                tts_busy.release()

        threading.Thread(target=_run, daemon=True).start()

    # ── Keyboard handler ──
    def on_key(ch: str):
        if ch == " ":
            trigger_robot_speak()
        elif ch.lower() == "q" or ch == "\x03":
            quit_event.set()

    # ── Suppress PulseAudio stderr noise ──
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    os.dup2(devnull, 2)
    try:
        mic_stream = sd.RawInputStream(
            samplerate=AEC_RATE,
            channels=1,
            dtype="int16",
            blocksize=MIC_BLOCKSIZE,
            callback=mic_callback,
        )
        spk_stream = sd.RawOutputStream(
            samplerate=SPEAKER_RATE,
            channels=1,
            dtype="int16",
            blocksize=SPK_BLOCKSIZE,
            callback=speaker_callback,
        )
    finally:
        os.dup2(old_stderr, 2)
        os.close(devnull)
        os.close(old_stderr)

    # ── Start everything ──
    print("Starting AEC Live Demo...")
    print("  AEC: LiveKit WebRTC AEC3")
    print(f"  TTS: {args.tts_url} (voice: {args.voice})")
    print("  STT: Deepgram Nova-3")
    print("  VAD: Silero")

    ensure_pocket_server(args.tts_url, args.voice)
    print()

    stt.start()
    vad.start()
    mic_stream.start()
    spk_stream.start()

    keys = KeyListener(on_key)
    keys.start()

    log_event("Ready — press SPACE to make the robot speak")

    # ── HUD loop ──
    first = True
    try:
        while not quit_event.is_set():
            stt_display = stt.interim_text
            if not stt_display and stt.transcripts:
                # Show last transcript briefly
                stt_display = stt.transcripts[-1]

            draw_hud(
                robot_speaking=robot_speaking.is_set(),
                robot_line=robot_line[0],
                mic_db=mic_rms[0],
                vad_level=vad.level,
                stt_text=stt_display,
                events=events,
                first_draw=first,
            )
            first = False

            # Check if robot just finished with no false detection
            # (we log this in the speaker callback)

            quit_event.wait(timeout=HUD_REFRESH)
    except KeyboardInterrupt:
        pass

    # ── Cleanup ──
    print("\n\nShutting down...")
    keys.stop()
    mic_stream.stop()
    mic_stream.close()
    spk_stream.stop()
    spk_stream.close()
    stt.stop()
    vad.stop()
    aec.close()
    stop_pocket_server()
    print("Done.")


if __name__ == "__main__":
    main()
