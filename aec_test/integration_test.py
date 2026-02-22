"""Integration test: verify AEC works through the real voice pipeline.

Starts the full VoiceIO stack with ReSpeaker config from config.toml,
generates TTS audio, records what the mic picks up, and runs VAD.
If AEC is working, VAD should NOT detect speech from the speaker.

Usage:
    uv run --extra voice --extra local-tts python aec_test/integration_test.py
"""

import os
import sys
import time
import wave
import threading
import numpy as np
from pathlib import Path

# Load .env for API keys
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from character_eng.config import load_config
from character_eng.voice import (
    VoiceIO,
    SpeakerStream,
    MicStream,
    DeepgramSTT,
    resolve_device,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
TEST_TEXT = "Hello there! I've been waiting for you. The orb is glowing brighter than usual today."


def run_vad(audio_int16: np.ndarray, sr: int) -> list[tuple[float, float, float]]:
    """Run Silero VAD, return (start, end, prob) per 32ms window."""
    import torch

    model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
    model.reset_states()

    audio_f32 = audio_int16.astype(np.float32) / 32768.0

    # Resample to 16kHz if needed
    if sr != 16000:
        ratio = 16000 / sr
        n_out = int(len(audio_f32) * ratio)
        idx = np.arange(n_out) / ratio
        i = np.clip(idx.astype(int), 0, len(audio_f32) - 2)
        f = idx - i
        audio_f32 = audio_f32[i] * (1 - f) + audio_f32[i + 1] * f
        sr = 16000

    results = []
    window = 512  # Silero requires 512 samples at 16kHz
    for start in range(0, len(audio_f32) - window + 1, window):
        chunk = torch.from_numpy(audio_f32[start : start + window])
        prob = model(chunk, sr).item()
        results.append((start / sr, (start + window) / sr, prob))
    return results


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    cfg = load_config()

    input_idx = resolve_device(cfg.voice.input_device, "input")
    output_idx = resolve_device(cfg.voice.output_device, "output")

    import sounddevice as sd
    in_info = sd.query_devices(input_idx) if input_idx is not None else {"name": "default"}
    out_info = sd.query_devices(output_idx) if output_idx is not None else {"name": "default"}

    print("=== Voice Pipeline AEC Integration Test ===")
    print(f"Input:  [{input_idx}] {in_info['name']}")
    print(f"Output: [{output_idx}] {out_info['name']}")
    print(f"TTS:    {cfg.voice.tts_backend}")
    print(f"Mic mute: {cfg.voice.mic_mute_during_playback}")
    print()

    # --- Step 1: Start recording from ReSpeaker mic ---
    rec_sr = int(sd.query_devices(input_idx)["default_samplerate"]) if input_idx is not None else 16000
    rec_ch = min(int(sd.query_devices(input_idx)["max_input_channels"]), 2) if input_idx is not None else 1
    rec_chunks: list[np.ndarray] = []

    def rec_cb(indata, frames, time_info, status):
        rec_chunks.append(indata.copy())

    print("Starting mic recording...")
    rec_stream = sd.InputStream(
        device=input_idx, samplerate=rec_sr, channels=rec_ch,
        dtype="int16", callback=rec_cb, blocksize=1024,
    )
    rec_stream.start()
    time.sleep(0.5)  # let it stabilize

    # --- Step 2: Create VoiceIO and speak through it ---
    print(f"Creating VoiceIO (tts_backend={cfg.voice.tts_backend})...")
    voice_io = VoiceIO(
        input_device=cfg.voice.input_device,
        output_device=cfg.voice.output_device,
        mic_mute_during_playback=cfg.voice.mic_mute_during_playback,
        tts_backend=cfg.voice.tts_backend,
        ref_audio=cfg.voice.ref_audio,
        tts_model=cfg.voice.tts_model,
        tts_device=cfg.voice.tts_device,
    )

    # Only start speaker + TTS (skip STT/mic/keys to avoid conflicts with our test recorder)
    voice_io._output_device = resolve_device(cfg.voice.output_device, "output")
    voice_io._speaker = SpeakerStream(device=voice_io._output_device)
    voice_io._speaker.start()

    if cfg.voice.tts_backend == "local":
        from character_eng.local_tts import LocalTTS, load_model
        load_model(cfg.voice.tts_model, cfg.voice.tts_device, cfg.voice.ref_audio)
        voice_io._tts = LocalTTS(
            on_audio=voice_io._speaker.enqueue,
            ref_audio_path=cfg.voice.ref_audio,
            model_id=cfg.voice.tts_model,
            device=cfg.voice.tts_device,
        )
    else:
        from character_eng.voice import ElevenLabsTTS
        voice_io._tts = ElevenLabsTTS(
            on_audio=voice_io._speaker.enqueue,
        )

    voice_io._started = True

    print(f"Speaking: \"{TEST_TEXT[:60]}...\"")
    t0 = time.time()
    voice_io.speak_text(TEST_TEXT)
    t_spoken = time.time() - t0
    print(f"TTS done ({t_spoken:.1f}s)")

    # --- Step 3: Record tail silence ---
    print("Recording 2s tail...")
    time.sleep(2.0)
    rec_stream.stop()
    rec_stream.close()

    # Clean up voice
    if voice_io._tts is not None:
        voice_io._tts.close()
    if voice_io._speaker is not None:
        voice_io._speaker.stop()

    # --- Step 4: Analyze recording ---
    recording = np.concatenate(rec_chunks)
    if recording.ndim > 1:
        mono = recording[:, 0]
    else:
        mono = recording

    rec_duration = len(mono) / rec_sr

    # Save recording
    rec_path = RESULTS_DIR / "integration_test_recording.wav"
    with wave.open(str(rec_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rec_sr)
        wf.writeframes(mono.tobytes())
    print(f"\nSaved: {rec_path} ({rec_duration:.1f}s)")

    # RMS level
    rms = np.sqrt(np.mean(mono.astype(np.float64) ** 2))
    rms_db = 20 * np.log10(rms / 32768.0) if rms > 0 else -120.0
    print(f"RMS level: {rms_db:.1f} dB")

    # Run VAD
    print("\nRunning Silero VAD...")
    vad_results = run_vad(mono, rec_sr)

    # Check if any window has speech detected
    VAD_THRESHOLD = 0.5
    speech_windows = [(t0, t1, p) for t0, t1, p in vad_results if p > VAD_THRESHOLD]
    peak_vad = max(p for _, _, p in vad_results) if vad_results else 0.0
    avg_vad = np.mean([p for _, _, p in vad_results]) if vad_results else 0.0

    print(f"VAD: avg={avg_vad:.3f}, peak={peak_vad:.3f}, speech_windows={len(speech_windows)}/{len(vad_results)}")

    # --- Verdict ---
    print()
    if peak_vad < VAD_THRESHOLD and rms_db < -50:
        print("PASS — AEC is working. No speech detected from speaker in mic recording.")
        print("       The full voice pipeline correctly routes TTS through ReSpeaker")
        print("       output, and the hardware AEC cancels the echo.")
        return 0
    elif peak_vad < VAD_THRESHOLD:
        print("PASS (marginal) — VAD didn't detect speech, but RMS is higher than expected.")
        print(f"       RMS: {rms_db:.1f} dB (expected < -50 dB)")
        return 0
    else:
        print("FAIL — Speech detected in mic recording during TTS playback.")
        print("       AEC may not be working correctly.")
        print(f"       peak VAD: {peak_vad:.3f}, speech windows: {len(speech_windows)}")
        # Show when speech was detected
        for t0, t1, p in speech_windows[:10]:
            print(f"         {t0:.1f}-{t1:.1f}s: {p:.3f}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
