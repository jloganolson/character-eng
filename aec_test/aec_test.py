"""AEC test: play WAV on chosen output while recording from ReSpeaker mic.

Usage:
    # Test A — play through ReSpeaker out (AEC has reference):
    uv run --extra voice python aec_test/aec_test.py aec_test/samples/line_1_rachel.wav --play-device 0

    # Test B — play through Audioengine/PulseAudio (no AEC reference):
    uv run --extra voice python aec_test/aec_test.py aec_test/samples/line_1_rachel.wav --play-device 14

    Recordings saved next to the input WAV with device name in filename.
"""

import argparse
import wave
import time
import numpy as np
import sounddevice as sd
from pathlib import Path


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    """Load WAV as float32 mono array + sample rate."""
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        n_ch = wf.getnchannels()
        data = wf.readframes(wf.getnframes())
    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    if n_ch > 1:
        samples = samples.reshape(-1, n_ch)[:, 0]
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
    return audio[idx] * (1 - frac) + audio[idx + 1] * frac


def rms_db(audio: np.ndarray) -> float:
    """RMS level in dB."""
    rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))
    if rms == 0:
        return -120.0
    return 20 * np.log10(rms / 32768.0)


def main():
    parser = argparse.ArgumentParser(description="AEC test: play + record simultaneously")
    parser.add_argument("wav", help="WAV file to play")
    parser.add_argument("--play-device", type=int, required=True, help="Output device index for playback")
    parser.add_argument("--rec-device", type=int, default=0, help="Input device index for recording (default: 0 = ReSpeaker)")
    parser.add_argument("--tail", type=float, default=2.0, help="Extra recording seconds after playback (default: 2)")
    parser.add_argument("--out", help="Output WAV path (auto-generated if omitted)")
    args = parser.parse_args()

    wav_path = Path(args.wav)
    audio, wav_sr = load_wav(wav_path)

    play_info = sd.query_devices(args.play_device)
    rec_info = sd.query_devices(args.rec_device)
    play_sr = int(play_info["default_samplerate"])
    rec_sr = int(rec_info["default_samplerate"])
    rec_ch = min(rec_info["max_input_channels"], 2)

    # Resample for playback device
    play_audio = resample_linear(audio, wav_sr, play_sr)

    # Make stereo if device needs it
    play_ch = max(1, min(play_info["max_output_channels"], 2))
    if play_ch == 2:
        play_audio_out = np.column_stack([play_audio, play_audio])
    else:
        play_audio_out = play_audio.reshape(-1, 1)

    play_duration = len(play_audio) / play_sr

    # Output path
    if args.out:
        out_path = Path(args.out)
    else:
        dev_label = play_info["name"].split(":")[0].strip().replace(" ", "_")
        out_path = wav_path.parent / f"rec_{wav_path.stem}_via_{dev_label}.wav"

    print(f"Source:  {wav_path.name} ({len(audio)/wav_sr:.1f}s @ {wav_sr}Hz)")
    print(f"Play:    device {args.play_device} ({play_info['name']}) @ {play_sr}Hz, {play_ch}ch")
    print(f"Record:  device {args.rec_device} ({rec_info['name']}) @ {rec_sr}Hz, {rec_ch}ch")
    print(f"Output:  {out_path.name}")
    print(f"Duration: {play_duration:.1f}s playback + {args.tail:.1f}s tail")
    print()

    # Collect recorded chunks
    rec_chunks: list[np.ndarray] = []

    def rec_callback(indata, frames, time_info, status):
        if status:
            print(f"  [rec] {status}")
        rec_chunks.append(indata.copy())

    # Open recording stream, then play
    with sd.InputStream(
        device=args.rec_device,
        samplerate=rec_sr,
        channels=rec_ch,
        dtype="int16",
        callback=rec_callback,
        blocksize=1024,
    ):
        time.sleep(0.2)  # let recording stabilize
        print(">>> Playing...")
        sd.play(play_audio_out, samplerate=play_sr, device=args.play_device)
        sd.wait()
        print(f">>> Playback done. Recording {args.tail:.0f}s tail...")
        time.sleep(args.tail)

    # Combine and save
    recording = np.concatenate(rec_chunks)
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(rec_ch)
        wf.setsampwidth(2)
        wf.setframerate(rec_sr)
        wf.writeframes(recording.tobytes())

    rec_duration = len(recording) / (rec_sr * rec_ch)
    level = rms_db(recording)
    print(f"\nSaved: {out_path}")
    print(f"  Duration: {rec_duration:.1f}s")
    print(f"  RMS level: {level:.1f} dB")
    print(f"\nListen to the recording to check echo cancellation.")
    print(f"Lower RMS = less echo picked up by mic.")


if __name__ == "__main__":
    main()
