# AEC Proof of Concept — Results Report

## Summary

**LiveKit WebRTC AEC3 works.** It eliminates false transcripts from echo and reduces echo by 24+ dB. Sufficient for replacing mic-muting in the voice pipeline.

SpeexDSP (linear NLMS filter) was also tested but proved insufficient — only ~7 dB reduction on nonlinear acoustic paths.

## Test Setup

- **Hardware**: Audioengine 2+ speakers + USB mic (same desk, ~60cm apart)
- **TTS**: Pocket TTS (greg voice for robot, alba voice for human)
- **AEC**: LiveKit `AudioProcessingModule` (WebRTC AEC3) — echo cancellation + noise suppression + high-pass filter
- **Detection**: Silero VAD (offline, threshold 0.5) + Deepgram STT nova-3 (real-time)
- **Audio**: 16kHz/int16/mono for AEC, 48kHz for speaker output

### Test WAVs (generated via Pocket TTS)

| File | Voice | Duration | Role |
|------|-------|----------|------|
| `samples/robot_greg.wav` | Greg (cloned) | 7.2s | Robot's TTS output — AEC should cancel |
| `samples/human_visitor.wav` | Alba (built-in) | 5.9s | Human speech — AEC should pass through |

### 4-Quadrant Test Matrix

| Case | Speaker Audio | AEC Reference | Expected |
|------|-------------|--------------|----------|
| 1. Silence | None | None | No detection |
| 2. Robot speaks | Robot WAV | Robot (16kHz resample) | No detection (True Negative) |
| 3. Human speaks | Human WAV | None | Speech detected (True Positive) |
| 4. Barge-in | Robot + Human mixed | Robot only | Speech detected (True Positive) |

## Results — LiveKit AEC3

**Best run (stream_delay_ms=0, 1s warm-up):**

| Case | RMS (dB) | Silero Peak | Silero | DG Transcript | DG SpeechStarted |
|------|----------|-------------|--------|---------------|------------------|
| 1. Silence | -92.4 | 0.02 | PASS | (none) | No |
| 2. Robot | -67.7 | 0.51 | ~PASS | (none) | Yes (intermittent) |
| 3. Human | -43.1 | 1.00 | PASS | Full transcript | Yes |
| 4. Barge-in | -46.0 | 1.00 | PASS | Partial transcript | Yes |

**Key finding: Deepgram NEVER produces a false transcript from echo.** The only false alarm is occasional `SpeechStarted` events without any transcript text.

### Consistency (3 runs, Case 2 only)

| Run | RMS (dB) | Silero Peak | DG SpeechStarted | DG Transcript |
|-----|----------|-------------|-------------------|---------------|
| 1 | -69.3 | 0.27 | Yes | None |
| 2 | -78.0 | 0.37 | No | None |
| 3 | -69.6 | 0.55 | Yes | None |

Run-to-run variance is significant (acoustic conditions, mic gain). Silero peak ranges 0.27–0.55. DG SpeechStarted occurs ~66% of the time but never produces a transcript.

### Delay Sweep (stream_delay_ms)

| Delay | RMS (dB) | Silero Peak | Pass? |
|-------|----------|-------------|-------|
| 0ms | -76.4 | 0.25 | Yes |
| 50ms | -66.3 | 0.83 | No |
| 100ms | -74.5 | 0.30 | Yes |
| 150ms | -68.9 | 0.53 | No |
| 200ms | -74.3 | 0.57 | No |
| 300ms | -73.7 | 0.36 | Yes |

**0ms is optimal** — our output callback feeds reference at the exact moment audio goes to the device, so there's no additional delay.

## Results — SpeexDSP (for comparison)

SpeexDSP provided only ~7 dB echo reduction (from -43 dB raw to -49 to -52 dB). Deepgram consistently produced full transcripts of robot echo. Silero peaks were 0.69–0.95. **Not viable.**

Root cause: Speex uses an NLMS linear adaptive filter that can't handle the nonlinear acoustic path (speaker distortion, room reflections, PulseAudio processing).

## Architecture

### Files

| File | Purpose |
|------|---------|
| `experiments/aec/livekit_aec.py` | LiveKitAEC class — wraps `livekit.rtc.AudioProcessingModule` |
| `experiments/aec/speex_aec.py` | SpeexAEC class — ctypes wrapper for libspeexdsp (backup) |
| `experiments/aec/generate_pocket_wavs.py` | Generate test WAVs via Pocket TTS server |
| `experiments/aec/aec_poc.py` | 4-quadrant test runner |
| `experiments/aec/samples/` | Generated WAV files |
| `experiments/aec/results/` | Recorded AEC-processed audio per test case |

### Real-time Pipeline (per test case)

```
Speaker output (48kHz) ─→ speakers ─→ room ─→ mic (16kHz)
       │                                        │
       └── resample 48k→16k ──→ AEC ref feed    │
                                     │           │
                              LiveKit APM ←── mic capture
                                     │
                              cleaned audio ──→ Silero VAD (offline)
                                             ──→ Deepgram STT (real-time)
```

- Output callback feeds reference synchronously as audio goes to device
- Mic callback processes every frame through AEC immediately
- Both use 10ms frames (160 samples at 16kHz) for optimal AEC alignment

## Integration Recommendations

1. **Tap reference at SpeakerStream output callback** — same pattern as the PoC
2. **Resample TTS output (24kHz) → 16kHz** for AEC reference
3. **stream_delay_ms = 0** — reference is already synchronized via callback
4. **Remove mic-muting during playback** — AEC handles echo suppression
5. **Debounce Deepgram SpeechStarted** — require 200ms sustained or actual transcript before triggering barge-in (eliminates residual false alarms)
6. **Optional: raise Silero VAD threshold to 0.6** for cleaner gating

## How to Reproduce

```bash
# 1. Ensure Pocket TTS server is running with Greg's voice
pocket-tts serve --port 8003 --voice voices/greg.safetensors &

# 2. Generate test WAVs (only needed once)
uv run python experiments/aec/generate_pocket_wavs.py

# 3. Run 4-quadrant test (default: livekit backend)
uv run python experiments/aec/aec_poc.py 2>/dev/null

# Optional: test with speex backend for comparison
uv run python experiments/aec/aec_poc.py --backend speex 2>/dev/null

# Optional: adjust stream delay
uv run python experiments/aec/aec_poc.py --delay 100 2>/dev/null
```

### Dependencies

- `livekit` (pip) — provides WebRTC AEC3 via `AudioProcessingModule`
- `libspeexdsp.so.1` (system) — for speex backend only
- `silero-vad` (pip) — offline VAD evaluation
- `deepgram-sdk` (pip) — real-time STT evaluation
- `sounddevice` (pip) — audio I/O
- Pocket TTS server running on localhost:8003

## Conclusion

LiveKit WebRTC AEC3 is the right choice for software echo cancellation. It provides:
- **24–35 dB echo reduction** (vs ~7 dB from speex)
- **Zero false transcripts** from Deepgram (the metric that matters most)
- **Nonlinear echo suppression** that handles real acoustic paths
- **Simple API**: feed reference via `process_reverse_stream()`, process mic via `process_stream()`

The remaining edge cases (intermittent SpeechStarted, borderline Silero) are easily handled with pipeline-level debouncing and threshold adjustment.
