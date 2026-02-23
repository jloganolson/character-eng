# TTS Backend Benchmark Results

**Date:** 2026-02-22
**GPU:** NVIDIA GeForce RTX 4090 (24 GB)
**Ref audio:** GLaDOS voice clone (`/home/logan/Desktop/lmtts/ref/GLaDOS/data/0.wav`)
**Runs per sentence:** 3 (averages shown, warmup run excluded)

## Summary Table

| Backend | Sentence | Avg TTFA (ms) | Avg Total (ms) | Avg Audio (ms) | Avg RTF |
|---------|----------|---------------|----------------|----------------|---------|
| **Kokoro** | short (7w) | 51 | 51 | 2,149 | 0.023 |
| **Kokoro** | medium (35w) | 86 | 87 | 11,171 | 0.008 |
| **Kokoro** | long (66w) | 157 | 157 | 22,381 | 0.007 |
| **Pocket-TTS** | short (7w) | 74 | 700 | 2,787 | 0.251 |
| **Pocket-TTS** | medium (35w) | 92 | 2,680 | 10,573 | 0.253 |
| **Pocket-TTS** | long (66w) | 114 | 4,901 | 19,187 | 0.255 |
| **KaniTTS** | short (7w) | 301 | 591 | 3,520 | 0.168 |
| **KaniTTS** | medium (35w) | 301 | 1,460 | 12,373 | 0.118 |
| **KaniTTS** | long (66w) | 301 | 2,487 | 24,400 | 0.102 |
| **MioTTS** | short (7w) | 293 | 293 | 2,560 | 0.115 |
| **MioTTS** | medium (35w) | 645 | 1,640 | 14,227 | 0.115 |
| **MioTTS** | long (66w) | 899 | 2,919 | 25,227 | 0.116 |
| **KittenTTS** | short (7w) | 527 | 527 | 2,450 | 0.215 |
| **KittenTTS** | medium (35w) | 2,256 | 2,256 | 11,275 | 0.200 |
| **KittenTTS** | long (66w) | 4,081 | 4,081 | 21,375 | 0.191 |
| **Chatterbox** | short (7w) | 711 | 712 | 2,340 | 0.305 |
| **Chatterbox** | medium (35w) | 983 | 2,776 | 10,393 | 0.267 |
| **Chatterbox** | long (66w) | 1,609 | 5,108 | 20,920 | 0.244 |

## Key Metrics

- **TTFA** -- Time to first audio (latency before audio starts playing)
- **Total** -- Total wall-clock generation time
- **Audio** -- Duration of generated audio
- **RTF** -- Real-time factor (generation time / audio duration; lower = faster than realtime)

## Streaming Analysis

### TTFA by text length (the streaming test)

| Backend | Short TTFA | Medium TTFA | Long TTFA | Streaming type |
|---------|-----------|-------------|-----------|----------------|
| **Kokoro** | 51ms | 86ms | 157ms | One-shot ONNX (so fast it doesn't matter) |
| **Pocket-TTS** | 74ms | 92ms | 114ms | True autoregressive streaming (causal transformer) |
| **KaniTTS** | 301ms | 301ms | 301ms | True SSE streaming (flat) |
| **MioTTS** | 293ms | 645ms | 899ms | Sentence pipelining |
| **KittenTTS** | 527ms | 2,256ms | 4,081ms | One-shot ONNX (no streaming) |
| **Chatterbox** | 711ms | 983ms | 1,609ms | Sentence pipelining |

**Kokoro** is the TTFA winner -- the 82M ONNX model is so fast that TTFA = total time. Even on the long sentence, it finishes generating all audio in 157ms. No streaming needed when inference is 143x realtime.

**Pocket-TTS** has true autoregressive streaming via chunked HTTP WAV -- TTFA is nearly flat (74-114ms) while total time scales linearly. The causal transformer generates audio frame-by-frame, and each frame streams out as it's decoded.

**KaniTTS** has true SSE streaming -- TTFA is dead flat at ~300ms regardless of input length.

**KittenTTS** has no streaming and is the slowest. TTFA = Total because all audio arrives at once. The larger ONNX model (15-80M) generates at ~5x realtime but still takes 4 seconds for long text.

### Speed ranking (RTF on long text)

| Rank | Backend | RTF (long) | Speed vs realtime |
|------|---------|-----------|-------------------|
| 1 | Kokoro | 0.007 | **143x** realtime |
| 2 | KaniTTS | 0.102 | 10x realtime |
| 3 | MioTTS | 0.116 | 8.6x realtime |
| 4 | KittenTTS | 0.191 | 5.2x realtime |
| 5 | Chatterbox | 0.244 | 4.1x realtime |
| 6 | Pocket-TTS | 0.255 | 3.9x realtime |

### Setup Complexity

| Backend | Architecture | GPU VRAM | Voice Cloning |
|---------|-------------|----------|---------------|
| Kokoro | External server (FastAPI + ONNX) | ~1 GB | No (54 presets) |
| Pocket-TTS | External server (FastAPI + causal transformer) | CPU-only (100M) | Yes (ref audio) |
| KaniTTS | External server (FastAPI + NeMo) | ~4 GB | No |
| KittenTTS | External server (FastAPI + ONNX) | ~0.5 GB | No (8 voices) |
| Chatterbox | In-process (350M params) | ~1-2 GB | Yes (ref audio) |
| MioTTS | Two servers (vLLM LLM + FastAPI codec) | ~7 GB | No (presets) |

### Recommendation

- **Lowest latency:** Kokoro -- 51ms TTFA (short), 157ms (long). So fast streaming is irrelevant.
- **Best streaming:** Pocket-TTS -- true causal streaming, flat ~80ms TTFA, with voice cloning
- **Fastest RTF:** Kokoro at 0.007 (143x realtime)
- **Simplest setup:** Kokoro -- lightweight ONNX model, single server, 54 preset voices
- **Best for voice cloning:** Pocket-TTS -- only streaming model that supports ref audio
- **Best overall:** Kokoro for preset voices, Pocket-TTS for voice cloning

## Server Ports

| Backend | Port | Server Location |
|---------|------|----------------|
| KaniTTS | 8000 | `/home/logan/Projects/kani-tts-2-openai-server/` |
| MioTTS | 8001 (API) + 8100 (vLLM) | `/home/logan/Projects/MioTTS-Inference/` |
| Pocket-TTS | 8003 | `uv run pocket-tts serve --port 8003` |
| KittenTTS | 8005 | `/home/logan/Projects/Kitten-TTS-Server/` |
| Kokoro | 8880 | `/home/logan/Projects/Kokoro-FastAPI/` |

## Notes

- Benchmark includes a warmup run (excluded from metrics) to eliminate cold-start effects
- Kokoro and KittenTTS are one-shot (TTFA = Total) because ONNX inference completes before any chunked delivery
- Pocket-TTS runs on CPU by default; GPU provides no speedup for this 100M model
- MioTTS requires `--max-text-length 500` on the vLLM server
