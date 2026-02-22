# Learnings

Hard-won knowledge from development. Things that took real debugging time or weren't obvious from docs.

## ReSpeaker XVF3800 AEC Integration (2026-02-21)

### The core insight

The XVF3800's acoustic echo cancellation only works when TTS audio is routed **through the ReSpeaker's own output**. The DSP uses its output as the reference signal to subtract from the mic input. If audio goes directly to speakers (e.g. via USB to Audioengine), the ReSpeaker has no reference and picks up the full echo.

- **Through ReSpeaker out**: mic records ~-71 dB (near silence)
- **Through speakers directly**: mic records ~-30 to -40 dB (clear echo)
- **Delta**: 30-40 dB of cancellation, confirmed across multiple samples and voices

### Config that works

```toml
[voice]
input_device = "reSpeaker"
output_device = "reSpeaker"       # THIS IS THE KEY - routes TTS through ReSpeaker for AEC reference
mic_mute_during_playback = false  # hardware AEC handles echo, no need for software mute
```

### Device enumeration is unstable

USB device indices change across reboots, suspend/resume cycles, and when devices are plugged/unplugged. Integer device IDs in config break silently. Solution: name-based matching (`"reSpeaker"` instead of `14`), resolved at runtime via `resolve_device()` in `voice.py`.

### SpeakerStream sample rate fallback

The ReSpeaker only supports 16kHz output. ElevenLabs and the `SpeakerStream` default to 24kHz PCM. The fallback chain matters:

- **Bad order** (24kHz -> 48kHz -> device default): PortAudio prints noisy C-level error spam to stderr for each failed rate probe, even though the fallback works correctly.
- **Good order** (device default -> 24kHz -> 48kHz): Hits 16kHz on the first try, no noise.
- The stderr noise is from PortAudio's C layer, not Python — suppressing it requires fd-level redirect (`os.dup2` to `/dev/null` during the probe).

### Validation approach

We built a standalone test suite (`aec_test/`) that doesn't need a human in the loop:

1. **dB comparison** (`aec_db_test.py`): Play WAVs through both paths, record from mic, compare RMS levels. Clear 30-40 dB delta = AEC working.
2. **VAD test** (`aec_vad_test.py`): Play segments through alternating paths, run Silero VAD on the recording. Speech should only be detected during no-AEC segments. Uses 512-sample windows at 16kHz (Silero's required chunk size — not 512ms, which was our first bug).
3. **Integration test** (`integration_test.py`): Runs TTS through the real `VoiceIO` pipeline, records from mic, checks VAD. Tests the full stack end-to-end.

### Silero VAD gotcha

Silero VAD requires **exactly 512 samples per chunk at 16kHz** (32ms windows). The window size is in samples, not milliseconds. Passing 512ms worth of samples (8192) crashes with a cryptic TorchScript error about "Provided number of samples is 8192 (Supported values: 256 for 8000 sample rate, 512 for 16000)".

### Post-suspend AEC behavior

After a laptop suspend/resume cycle, the AEC still works but RMS levels are slightly higher (~-43 dB vs ~-71 dB in a fresh session). VAD still doesn't detect speech — the higher RMS is ambient noise, not echo leakage. The AEC DSP may need a few seconds to recalibrate after wake.
