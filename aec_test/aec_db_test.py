"""Batch AEC dB comparison: play each sample through both output paths,
record from ReSpeaker mic, compare RMS levels.

Usage:
    uv run --extra voice python aec_test/aec_db_test.py
"""

import wave
import time
import numpy as np
import sounddevice as sd
from pathlib import Path

SAMPLES_DIR = Path(__file__).resolve().parent / "samples"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def find_device(name_substr: str, kind: str = "output") -> int:
    """Find device index by name substring."""
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


def rms_db(audio: np.ndarray) -> float:
    rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))
    if rms == 0:
        return -120.0
    return 20 * np.log10(rms / 32768.0)


def play_and_record(
    wav_audio: np.ndarray,
    wav_sr: int,
    play_dev: int,
    rec_dev: int,
    tail: float = 2.0,
) -> np.ndarray:
    """Play audio on play_dev while recording from rec_dev. Returns int16 array."""
    play_info = sd.query_devices(play_dev)
    rec_info = sd.query_devices(rec_dev)
    play_sr = int(play_info["default_samplerate"])
    rec_sr = int(rec_info["default_samplerate"])
    rec_ch = min(rec_info["max_input_channels"], 2)

    play_audio = resample(wav_audio, wav_sr, play_sr)
    play_ch = max(1, min(play_info["max_output_channels"], 2))
    if play_ch == 2:
        play_out = np.column_stack([play_audio, play_audio])
    else:
        play_out = play_audio.reshape(-1, 1)

    chunks: list[np.ndarray] = []

    def cb(indata, frames, time_info, status):
        chunks.append(indata.copy())

    with sd.InputStream(
        device=rec_dev, samplerate=rec_sr, channels=rec_ch,
        dtype="int16", callback=cb, blocksize=1024,
    ):
        time.sleep(0.2)
        sd.play(play_out, samplerate=play_sr, device=play_dev)
        sd.wait()
        time.sleep(tail)

    return np.concatenate(chunks)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    respeaker_out = find_device("reSpeaker", "output")
    respeaker_in = find_device("reSpeaker", "input")
    audioengine_out = find_device("Audioengine", "output")

    respeaker_info = sd.query_devices(respeaker_out)
    audioengine_info = sd.query_devices(audioengine_out)
    rec_info = sd.query_devices(respeaker_in)

    print("=== AEC dB Comparison ===")
    print(f"AEC path:    device {respeaker_out} ({respeaker_info['name']})")
    print(f"No-AEC path: device {audioengine_out} ({audioengine_info['name']})")
    print(f"Recording:   device {respeaker_in} ({rec_info['name']})")
    print()

    wavs = sorted(SAMPLES_DIR.glob("line_*.wav"))
    if not wavs:
        print("No sample WAVs found. Run generate_wavs.py first.")
        return

    results = []

    for wav_path in wavs:
        audio, sr = load_wav(wav_path)
        duration = len(audio) / sr
        print(f"--- {wav_path.name} ({duration:.1f}s) ---")

        # AEC path: play through ReSpeaker
        print(f"  Playing through ReSpeaker (AEC)...", end="", flush=True)
        rec_aec = play_and_record(audio, sr, respeaker_out, respeaker_in)
        db_aec = rms_db(rec_aec)
        aec_path = RESULTS_DIR / f"rec_{wav_path.stem}_AEC.wav"
        _save_wav(aec_path, rec_aec, rec_info)
        print(f" {db_aec:.1f} dB")

        # Pause between tests to let AEC state settle
        time.sleep(2)

        # No-AEC path: play through Audioengine
        print(f"  Playing through Audioengine (no AEC)...", end="", flush=True)
        rec_noaec = play_and_record(audio, sr, audioengine_out, respeaker_in)
        db_noaec = rms_db(rec_noaec)
        noaec_path = RESULTS_DIR / f"rec_{wav_path.stem}_noAEC.wav"
        _save_wav(noaec_path, rec_noaec, rec_info)
        print(f" {db_noaec:.1f} dB")

        results.append((wav_path.name, db_aec, db_noaec))
        time.sleep(2)

    # Summary table
    print()
    print("=" * 62)
    print(f"{'Sample':<30} {'AEC (dB)':>10} {'No AEC (dB)':>12} {'Delta':>8}")
    print("-" * 62)
    for name, aec, noaec in results:
        delta = aec - noaec
        marker = "<<<" if delta < -10 else ""
        print(f"{name:<30} {aec:>10.1f} {noaec:>12.1f} {delta:>+8.1f} {marker}")
    print("=" * 62)
    print()
    print("Delta < -10 dB = AEC is clearly cancelling echo")
    print(f"Recordings saved to {RESULTS_DIR}/")


def _save_wav(path: Path, data: np.ndarray, rec_info: dict):
    rec_sr = int(rec_info["default_samplerate"])
    rec_ch = data.shape[1] if data.ndim > 1 else 1
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(rec_ch)
        wf.setsampwidth(2)
        wf.setframerate(rec_sr)
        wf.writeframes(data.tobytes())


if __name__ == "__main__":
    main()
