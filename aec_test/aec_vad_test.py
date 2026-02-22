"""AEC interruption test using Silero VAD.

Records continuously from ReSpeaker mic while playing audio segments
through alternating output paths (ReSpeaker vs Audioengine).
Runs Silero VAD on the recording and reports which segments triggered
speech detection.

If AEC works: segments played through ReSpeaker should NOT trigger VAD.
Segments played through Audioengine (no reference) SHOULD trigger VAD.

Usage:
    uv run --extra voice python aec_test/aec_vad_test.py
"""

import wave
import time
import threading
import numpy as np
import sounddevice as sd
import torch
from pathlib import Path

SAMPLES_DIR = Path(__file__).resolve().parent / "samples"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

GAP_SECONDS = 3.0  # silence between segments


def find_device(name_substr: str, kind: str = "output") -> int:
    for i, d in enumerate(sd.query_devices()):
        if name_substr.lower() in d["name"].lower():
            if kind == "output" and d["max_output_channels"] > 0:
                return i
            if kind == "input" and d["max_input_channels"] > 0:
                return i
    raise RuntimeError(f"Device matching '{name_substr}' ({kind}) not found")


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        n_ch = wf.getnchannels()
        data = wf.readframes(wf.getnframes())
    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    if n_ch > 1:
        samples = samples.reshape(-1, n_ch)[:, 0]
    return samples, sr


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    ratio = target_sr / orig_sr
    n_out = int(len(audio) * ratio)
    idx = np.arange(n_out) / ratio
    i = np.clip(idx.astype(int), 0, len(audio) - 2)
    f = idx - i
    return audio[i] * (1 - f) + audio[i + 1] * f


def play_on_device(audio_f32: np.ndarray, sr: int, dev: int):
    """Play float32 mono audio on a specific device (blocking)."""
    info = sd.query_devices(dev)
    dev_sr = int(info["default_samplerate"])
    play = resample(audio_f32, sr, dev_sr)
    ch = max(1, min(info["max_output_channels"], 2))
    if ch == 2:
        play = np.column_stack([play, play])
    else:
        play = play.reshape(-1, 1)
    sd.play(play, samplerate=dev_sr, device=dev)
    sd.wait()


def run_silero_vad(
    audio_int16: np.ndarray, sr: int,
) -> list[tuple[float, float, float]]:
    """Run Silero VAD on int16 mono audio.

    Returns list of (start_sec, end_sec, confidence) for each window.
    Silero requires exactly 512 samples at 16kHz (32ms windows).
    """
    model, utils = torch.hub.load(
        "snakers4/silero-vad", "silero_vad", trust_repo=True
    )
    model.reset_states()

    # Silero expects 16kHz float32
    audio_f32 = audio_int16.astype(np.float32) / 32768.0

    # Resample to 16kHz if needed
    if sr != 16000:
        audio_f32 = resample(audio_f32, sr, 16000)
        sr = 16000

    # Silero requires exactly 512 samples per chunk at 16kHz
    window_samples = 512
    results = []
    for start in range(0, len(audio_f32) - window_samples + 1, window_samples):
        chunk = torch.from_numpy(audio_f32[start : start + window_samples])
        prob = model(chunk, sr).item()
        t_start = start / sr
        t_end = (start + window_samples) / sr
        results.append((t_start, t_end, prob))

    return results


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    respeaker_out = find_device("reSpeaker", "output")
    respeaker_in = find_device("reSpeaker", "input")
    audioengine_out = find_device("Audioengine", "output")

    rec_info = sd.query_devices(respeaker_in)
    rec_sr = int(rec_info["default_samplerate"])
    rec_ch = min(rec_info["max_input_channels"], 2)

    print("=== AEC VAD Interruption Test ===")
    print(f"AEC output:    device {respeaker_out} ({sd.query_devices(respeaker_out)['name']})")
    print(f"No-AEC output: device {audioengine_out} ({sd.query_devices(audioengine_out)['name']})")
    print(f"Recording:     device {respeaker_in} ({rec_info['name']})")
    print()

    # Load all samples
    wavs = sorted(SAMPLES_DIR.glob("line_*.wav"))
    if len(wavs) < 2:
        print("Need at least 2 WAV samples. Run generate_wavs.py first.")
        return

    samples = [(p, *load_wav(p)) for p in wavs]

    # Build playback schedule: alternate AEC / no-AEC
    # Format: (label, device, wav_path, audio, sr)
    schedule = []
    routes = [
        ("AEC (ReSpeaker)", respeaker_out),
        ("NO-AEC (Audioengine)", audioengine_out),
    ]
    for i, (path, audio, sr) in enumerate(samples):
        label, dev = routes[i % 2]
        schedule.append((label, dev, path, audio, sr))

    # Print schedule
    print("Playback schedule:")
    t = 1.0  # start after 1s lead-in
    timeline = []  # (start, end, label)
    for label, dev, path, audio, sr in schedule:
        dur = len(audio) / sr
        print(f"  {t:5.1f}s - {t + dur:5.1f}s : {path.name} via {label}")
        timeline.append((t, t + dur, label))
        t += dur + GAP_SECONDS
    total_duration = t + 1.0  # 1s tail
    print(f"  Total recording: {total_duration:.1f}s")
    print()

    # Start continuous recording
    chunks: list[np.ndarray] = []

    def rec_cb(indata, frames, time_info, status):
        chunks.append(indata.copy())

    print("Recording + playing...")
    rec_stream = sd.InputStream(
        device=respeaker_in, samplerate=rec_sr, channels=rec_ch,
        dtype="int16", callback=rec_cb, blocksize=1024,
    )
    rec_stream.start()
    rec_start = time.monotonic()

    # Lead-in silence
    time.sleep(1.0)

    # Play each segment
    for i, (label, dev, path, audio, sr) in enumerate(schedule):
        elapsed = time.monotonic() - rec_start
        print(f"  [{elapsed:5.1f}s] Playing {path.name} via {label}...")
        play_on_device(audio, sr, dev)
        elapsed = time.monotonic() - rec_start
        print(f"  [{elapsed:5.1f}s] Done. Gap...")
        time.sleep(GAP_SECONDS)

    # Tail
    time.sleep(1.0)
    rec_stream.stop()
    rec_stream.close()
    print()

    # Combine recording
    recording = np.concatenate(chunks)
    # Take channel 0 for VAD (mono)
    if recording.ndim > 1:
        mono = recording[:, 0]
    else:
        mono = recording
    rec_duration = len(mono) / rec_sr

    # Save full recording
    rec_path = RESULTS_DIR / "vad_test_full_recording.wav"
    with wave.open(str(rec_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rec_sr)
        wf.writeframes(mono.tobytes())
    print(f"Saved recording: {rec_path} ({rec_duration:.1f}s)")

    # Run Silero VAD
    print("Running Silero VAD...")
    vad_results = run_silero_vad(mono, rec_sr)

    # Analyze: for each scheduled segment, compute average VAD probability
    print()
    print("=" * 75)
    print(f"{'Segment':<20} {'Route':<25} {'Avg VAD':>8} {'Peak VAD':>9} {'Detected':>9}")
    print("-" * 75)

    VAD_THRESHOLD = 0.5

    for seg_start, seg_end, label in timeline:
        # Find VAD windows overlapping this segment
        probs = [
            p for t0, t1, p in vad_results
            if t0 < seg_end and t1 > seg_start
        ]
        avg_p = np.mean(probs) if probs else 0.0
        peak_p = max(probs) if probs else 0.0
        detected = peak_p > VAD_THRESHOLD
        marker = "YES !!!" if detected else "no"

        # For AEC segments, detection = BAD (echo leaked)
        # For no-AEC segments, detection = EXPECTED
        if "NO-AEC" in label:
            note = "(expected)" if detected else "(AEC leaking to wrong path?)"
        else:
            note = "(ECHO LEAK)" if detected else "(AEC working)"

        print(f"  {seg_start:4.1f}-{seg_end:4.1f}s    {label:<25} {avg_p:>7.3f} {peak_p:>9.3f} {marker:>6}  {note}")

    print("=" * 75)

    # Also print a timeline visualization
    print()
    print("VAD Timeline (. = silent, # = speech detected, threshold=0.5):")
    print()
    line_chars = []
    time_labels = []
    for t0, t1, prob in vad_results:
        ch = "#" if prob > VAD_THRESHOLD else "."
        line_chars.append(ch)

    # Print in rows of 80 chars with time markers
    chars_per_sec = len(vad_results) / rec_duration
    row_width = 80
    for row_start in range(0, len(line_chars), row_width):
        row = "".join(line_chars[row_start : row_start + row_width])
        t_start = row_start / chars_per_sec
        t_end = min((row_start + row_width) / chars_per_sec, rec_duration)
        print(f"  {t_start:5.1f}s |{row}| {t_end:.1f}s")

    print()
    print("Expected: '#' only during NO-AEC segments, '.' during AEC segments")


if __name__ == "__main__":
    main()
