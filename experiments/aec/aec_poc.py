"""AEC Proof of Concept — 4-Quadrant Test.

Tests echo cancellation with real audio hardware using LiveKit WebRTC AEC3.
Uses pre-generated WAVs for "robot" (AEC reference) and "human" (no reference).
Evaluates with both Silero VAD (offline) and Deepgram STT (real-time).

Usage:
    uv run python experiments/aec/aec_poc.py
"""

import argparse
import collections
import os
import sys
import time
import wave
import threading
import numpy as np
import sounddevice as sd
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

SAMPLES_DIR = Path(__file__).resolve().parent / "samples"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
TAIL_SECONDS = 2.0
VAD_THRESHOLD = 0.5
AEC_SAMPLE_RATE = 16000


def create_aec(delay_ms: int = 0):
    """Create LiveKit AEC instance."""
    from livekit_aec import LiveKitAEC
    return LiveKitAEC(stream_delay_ms=delay_ms)


# --- Audio utilities (from experiments.py) ---

def load_wav(path: Path) -> tuple[np.ndarray, int]:
    """Load WAV as int16 mono array + sample rate."""
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        n_ch = wf.getnchannels()
        data = wf.readframes(wf.getnframes())
    samples = np.frombuffer(data, dtype=np.int16)
    if n_ch > 1:
        samples = samples.reshape(-1, n_ch)[:, 0].copy()
    return samples, sr


def resample_linear(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Simple linear interpolation resample."""
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


def rms_db(audio: np.ndarray) -> float:
    """RMS level in dB (ref: full-scale int16)."""
    rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))
    if rms == 0:
        return -120.0
    return 20 * np.log10(rms / 32768.0)


# --- Silero VAD ---

def run_silero_vad(audio_int16: np.ndarray, sr: int) -> list[tuple[float, float, float]]:
    """Run Silero VAD. Returns [(start_sec, end_sec, prob), ...]."""
    import torch
    from silero_vad import load_silero_vad

    model = load_silero_vad()
    model.reset_states()

    audio_f32 = audio_int16.astype(np.float32) / 32768.0
    if sr != 16000:
        audio_f32 = resample_linear(
            audio_f32.astype(np.float64), sr, 16000
        ).astype(np.float32)
        sr = 16000

    results = []
    window = 512  # Silero requires 512 at 16kHz
    for start in range(0, len(audio_f32) - window + 1, window):
        chunk = torch.from_numpy(audio_f32[start:start + window])
        prob = model(chunk, sr).item()
        results.append((start / sr, (start + window) / sr, prob))
    return results


# --- Deepgram STT detector ---

class DeepgramDetector:
    """Lightweight Deepgram STT for detecting speech in AEC-processed audio."""

    def __init__(self):
        self.transcripts: list[str] = []
        self.speech_started = False
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
            sample_rate=str(AEC_SAMPLE_RATE),
            channels="1",
            interim_results="true",
            utterance_end_ms="1500",
            vad_events="true",
            endpointing="300",
        )
        self._socket = self._ctx.__enter__()

        self._ws_open = threading.Event()

        def _on_open(_):
            self._ws_open.set()

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

            # Extract transcript
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

            if transcript and is_final:
                self.transcripts.append(transcript.strip())

        self._socket.on(EventType.OPEN, _on_open)
        self._socket.on(EventType.MESSAGE, _on_message)

        self._thread = threading.Thread(target=self._socket.start_listening, daemon=True)
        self._thread.start()
        self._ws_open.wait(timeout=5)

    def send_audio(self, data: bytes):
        if self._socket:
            try:
                self._socket.send_media(data)
            except Exception:
                pass

    def stop(self):
        if self._socket:
            try:
                from deepgram.extensions.types.sockets.listen_v1_control_message import ListenV1ControlMessage
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

    def detected(self) -> bool:
        """Return True if any speech was detected (transcript or SpeechStarted)."""
        return self.speech_started or bool(self.transcripts)

    def has_transcript(self) -> bool:
        """Return True only if actual transcript text was produced."""
        return bool(self.transcripts)

    def summary(self) -> str:
        if self.transcripts:
            text = " ".join(self.transcripts)
            return f'"{text[:60]}"'
        if self.speech_started:
            return "(speech-start only, no transcript)"
        return "(none)"


# --- Test runner ---

def run_test_case(
    label: str,
    aec,
    robot_wav: tuple[np.ndarray, int] | None,
    human_wav: tuple[np.ndarray, int] | None,
    use_deepgram: bool = True,
) -> dict:
    """Run a single AEC test case.

    Uses a callback-based output stream so AEC reference is fed at the exact
    moment audio goes to the speaker (perfect synchronization).

    Args:
        label: Test case name
        robot_wav: (int16 array, sample_rate) for robot audio — gets AEC reference
        human_wav: (int16 array, sample_rate) for human audio — NO AEC reference
        use_deepgram: Whether to also test with Deepgram STT

    Returns dict with results.
    """
    print(f"\n--- {label} ---")

    aec.reset()

    mic_sr = AEC_SAMPLE_RATE  # 16kHz
    play_sr = 48000  # Use 48kHz for output (common device rate, avoids PulseAudio issues)

    # Prepare playback audio at play_sr AND reference at 16kHz
    # Robot audio: goes to speaker AND to AEC reference
    # Human audio: goes to speaker only (no AEC reference)
    robot_play = np.array([], dtype=np.int16)
    human_play = np.array([], dtype=np.int16)
    ref_16k = np.array([], dtype=np.int16)  # Robot reference at 16kHz

    if robot_wav is not None:
        robot_samples, robot_sr = robot_wav
        robot_play = resample_linear(robot_samples, robot_sr, play_sr)
        ref_16k = resample_linear(robot_samples, robot_sr, mic_sr)

    if human_wav is not None:
        human_samples, human_sr = human_wav
        human_play = resample_linear(human_samples, human_sr, play_sr)

    # Mix robot + human into speaker output; keep reference separate
    max_play_len = max(len(robot_play), len(human_play), 1)
    speaker_audio = np.zeros(max_play_len, dtype=np.int32)
    if len(robot_play) > 0:
        speaker_audio[:len(robot_play)] += robot_play.astype(np.int32)
    if len(human_play) > 0:
        speaker_audio[:len(human_play)] += human_play.astype(np.int32)
    speaker_audio = np.clip(speaker_audio, -32768, 32767).astype(np.int16)

    play_duration = max_play_len / play_sr if max_play_len > 1 else 0.0

    # Resample reference to match output timing: ref_16k samples correspond to
    # the same duration as speaker_audio. We need a version at play_sr rate for
    # the output callback to know how much ref to feed per output frame.
    # Strategy: output callback plays N samples at play_sr. That's N/play_sr seconds.
    # Feed N/play_sr * mic_sr = N * mic_sr / play_sr samples of ref_16k to AEC.
    ref_ratio = mic_sr / play_sr  # 16000/48000 = 1/3

    # State for output callback
    play_pos = [0]
    ref_pos = [0]
    play_done = threading.Event()

    def output_callback(outdata, frames, time_info, status):
        """Speaker callback: play audio AND feed AEC reference."""
        start = play_pos[0]
        end = start + frames

        if start >= len(speaker_audio):
            outdata[:] = b'\x00' * len(outdata)
            if not play_done.is_set():
                play_done.set()
            return

        avail = min(end, len(speaker_audio)) - start
        chunk = speaker_audio[start:start + avail]
        outdata[:avail * 2] = chunk.tobytes()
        if avail < frames:
            outdata[avail * 2:] = b'\x00' * ((frames - avail) * 2)
        play_pos[0] = end

        # Feed corresponding reference to AEC
        ref_samples = int(frames * ref_ratio)
        if len(ref_16k) > 0:
            rstart = ref_pos[0]
            rend = min(rstart + ref_samples, len(ref_16k))
            if rstart < len(ref_16k):
                ref_chunk = ref_16k[rstart:rend].tobytes()
                if len(ref_chunk) < ref_samples * 2:
                    ref_chunk += b'\x00' * (ref_samples * 2 - len(ref_chunk))
            else:
                ref_chunk = b'\x00' * (ref_samples * 2)
            ref_pos[0] = rend
        else:
            ref_chunk = b'\x00' * (ref_samples * 2)

        aec.feed_playback(ref_chunk)

    # Collected AEC-processed frames
    processed_chunks: list[np.ndarray] = []
    lock = threading.Lock()

    # Deepgram detector
    dg = None
    if use_deepgram:
        dg = DeepgramDetector()
        dg.start()

    def mic_callback(indata, frames, time_info, status):
        """Mic callback: process through AEC, collect + send to Deepgram."""
        raw = bytes(indata)
        cleaned = aec.process_capture(raw)
        if cleaned:
            arr = np.frombuffer(cleaned, dtype=np.int16)
            with lock:
                processed_chunks.append(arr.copy())
            if dg:
                dg.send_audio(cleaned)

    # Suppress PulseAudio stderr noise during stream open
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    os.dup2(devnull, 2)
    try:
        # Start mic
        mic_stream = sd.RawInputStream(
            samplerate=mic_sr,
            channels=1,
            dtype="int16",
            blocksize=160,  # 10ms — match AEC frame size for tight alignment
            callback=mic_callback,
        )
        mic_stream.start()

        # Start speaker output (callback-based for sync reference feeding)
        spk_stream = sd.RawOutputStream(
            samplerate=play_sr,
            channels=1,
            dtype="int16",
            blocksize=int(play_sr * 0.01),  # 10ms blocks
            callback=output_callback,
        )
        spk_stream.start()
    finally:
        os.dup2(old_stderr, 2)
        os.close(devnull)
        os.close(old_stderr)

    time.sleep(1.0)  # warm-up: let AEC adapt to room noise + speaker/mic latency

    if play_duration > 0:
        print(f"  Playing {play_duration:.1f}s audio...")
        play_done.wait(timeout=play_duration + 5)
    else:
        print(f"  Silence ({TAIL_SECONDS:.1f}s)...")

    print(f"  Recording {TAIL_SECONDS:.0f}s tail...")
    time.sleep(TAIL_SECONDS)

    # Stop streams
    mic_stream.stop()
    mic_stream.close()
    spk_stream.stop()
    spk_stream.close()

    # Give Deepgram a moment to process remaining audio
    if dg:
        time.sleep(1.0)
        dg.stop()

    # Combine processed audio
    with lock:
        if processed_chunks:
            processed = np.concatenate(processed_chunks)
        else:
            processed = np.array([], dtype=np.int16)

    # Save processed recording
    RESULTS_DIR.mkdir(exist_ok=True)
    safe_label = label.lower().replace(" ", "_").replace("(", "").replace(")", "")
    rec_path = RESULTS_DIR / f"poc_{safe_label}.wav"
    if len(processed) > 0:
        with wave.open(str(rec_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(mic_sr)
            wf.writeframes(processed.tobytes())
        print(f"  Saved: {rec_path.name} ({len(processed)/mic_sr:.1f}s, {rms_db(processed):.1f} dB)")

    # Silero VAD
    print("  Running Silero VAD...", end="", flush=True)
    if len(processed) > 0:
        vad_results = run_silero_vad(processed, mic_sr)
        vad_peak = max(p for _, _, p in vad_results) if vad_results else 0.0
        vad_avg = np.mean([p for _, _, p in vad_results]) if vad_results else 0.0
        vad_detected = vad_peak > VAD_THRESHOLD
    else:
        vad_peak = 0.0
        vad_avg = 0.0
        vad_detected = False
    print(f" avg={vad_avg:.3f}, peak={vad_peak:.3f}")

    # Deepgram result
    dg_detected = dg.detected() if dg else False
    dg_has_transcript = dg.has_transcript() if dg else False
    dg_summary = dg.summary() if dg else "n/a"

    return {
        "label": label,
        "vad_peak": vad_peak,
        "vad_avg": vad_avg,
        "vad_detected": vad_detected,
        "dg_detected": dg_detected,
        "dg_has_transcript": dg_has_transcript,
        "dg_summary": dg_summary,
        "rms_db": rms_db(processed) if len(processed) > 0 else -120.0,
    }


def main():
    parser = argparse.ArgumentParser(description="AEC 4-Quadrant Test")
    parser.add_argument("--delay", type=int, default=0,
                        help="AEC stream delay in ms (default: 0)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    # Check for test WAVs
    robot_path = SAMPLES_DIR / "robot_greg.wav"
    human_path = SAMPLES_DIR / "human_visitor.wav"

    if not robot_path.exists() or not human_path.exists():
        print("Test WAVs not found. Generate them first:")
        print("  uv run python experiments/aec/generate_pocket_wavs.py")
        sys.exit(1)

    robot_wav = load_wav(robot_path)
    human_wav = load_wav(human_path)

    print(f"Robot WAV: {robot_path.name} ({len(robot_wav[0])/robot_wav[1]:.1f}s @ {robot_wav[1]}Hz)")
    print(f"Human WAV: {human_path.name} ({len(human_wav[0])/human_wav[1]:.1f}s @ {human_wav[1]}Hz)")

    # Check Deepgram
    has_deepgram = bool(os.environ.get("DEEPGRAM_API_KEY"))
    if not has_deepgram:
        print("\nDEEPGRAM_API_KEY not set — running with Silero VAD only")

    print("\n" + "=" * 70)
    print("AEC Proof of Concept — 4 Quadrant Test")
    print(f"Backend: LiveKit WebRTC AEC3")
    print(f"Detectors: Silero VAD" + (" + Deepgram STT" if has_deepgram else ""))
    print("=" * 70)

    aec = create_aec(delay_ms=args.delay)

    results = []

    # Case 1: Silence
    results.append(run_test_case(
        "Case 1: Silence (baseline)",
        aec, robot_wav=None, human_wav=None,
        use_deepgram=has_deepgram,
    ))

    # Case 2: Robot only (with AEC reference)
    results.append(run_test_case(
        "Case 2: Robot speaks (AEC ref)",
        aec, robot_wav=robot_wav, human_wav=None,
        use_deepgram=has_deepgram,
    ))

    # Case 3: Human only (no AEC reference)
    results.append(run_test_case(
        "Case 3: Human speaks (no AEC ref)",
        aec, robot_wav=None, human_wav=human_wav,
        use_deepgram=has_deepgram,
    ))

    # Case 4: Barge-in (both, only robot has AEC ref)
    results.append(run_test_case(
        "Case 4: Barge-in (robot+human)",
        aec, robot_wav=robot_wav, human_wav=human_wav,
        use_deepgram=has_deepgram,
    ))

    aec.close()

    # --- Summary ---
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    # Expected: silence=no, robot=no, human=yes, bargein=yes
    expected_detect = [False, False, True, True]
    pass_count = 0
    strict_pass = 0  # transcript-only Deepgram scoring
    total = 0
    strict_total = 0

    for r, expect in zip(results, expected_detect):
        # Silero
        vad_pass = (r["vad_detected"] == expect)
        vad_mark = "PASS" if vad_pass else "FAIL"
        vad_label = f"peak={r['vad_peak']:.2f}"

        # Deepgram (any detection = SpeechStarted or transcript)
        # Strict = only actual transcripts count as detection
        if has_deepgram:
            dg_pass = (r["dg_detected"] == expect)
            dg_strict = (r["dg_has_transcript"] == expect)
            dg_mark = "PASS" if dg_pass else "FAIL"
            dg_strict_mark = "PASS" if dg_strict else "FAIL"
            dg_label = r["dg_summary"]
            total += 2
            strict_total += 2
            pass_count += vad_pass + dg_pass
            strict_pass += vad_pass + dg_strict
        else:
            dg_mark = ""
            dg_label = ""
            total += 1
            strict_total += 1
            pass_count += vad_pass
            strict_pass += vad_pass

        print(f"\n{r['label']}")
        print(f"  RMS: {r['rms_db']:.1f} dB")
        print(f"  Silero: {vad_label} → {vad_mark}")
        if has_deepgram:
            print(f"  Deepgram: {dg_label} → {dg_mark}")
            if dg_mark != dg_strict_mark:
                print(f"  Deepgram (transcript-only): → {dg_strict_mark}")
        expect_str = "speech expected" if expect else "silence expected"
        print(f"  ({expect_str})")

    print(f"\n{'=' * 70}")
    print(f"Score (any detection):      {pass_count}/{total} PASS")
    if has_deepgram:
        print(f"Score (transcript-only DG): {strict_pass}/{strict_total} PASS")
    if strict_pass == strict_total:
        print("\nAEC working: no false transcripts produced from echo.")
    if pass_count == total:
        print("Perfect: all detection criteria pass.")
    elif strict_pass == strict_total:
        soft_fails = total - pass_count
        print(f"\n{soft_fails} soft fail(s) = SpeechStarted without transcript (debounce-able).")
    else:
        print("\nSome tests failed. Check recordings in experiments/aec/results/")
    print("=" * 70)


if __name__ == "__main__":
    main()
