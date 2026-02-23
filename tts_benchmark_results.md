# TTS Backend Benchmark Results

**Date:** 2026-02-22
**GPU:** NVIDIA GeForce RTX 4090 (24 GB)
**Ref audio:** GLaDOS voice clone (`/home/logan/Desktop/lmtts/ref/GLaDOS/data/0.wav`)
**Runs per sentence:** 3 (averages shown, warmup run excluded)

## Summary Table

| Backend | Sentence | Avg TTFA (ms) | Avg Total (ms) | Avg Audio (ms) | Avg RTF |
|---------|----------|---------------|----------------|----------------|---------|
| **Chatterbox** | short (7w) | 711 | 712 | 2,340 | 0.305 |
| **Chatterbox** | medium (35w) | 983 | 2,776 | 10,393 | 0.267 |
| **Chatterbox** | long (66w) | 1,609 | 5,108 | 20,920 | 0.244 |
| **KaniTTS** | short (7w) | 301 | 591 | 3,520 | 0.168 |
| **KaniTTS** | medium (35w) | 301 | 1,460 | 12,373 | 0.118 |
| **KaniTTS** | long (66w) | 301 | 2,487 | 24,400 | 0.102 |
| **MioTTS** | short (7w) | 293 | 293 | 2,560 | 0.115 |
| **MioTTS** | medium (35w) | 645 | 1,640 | 14,227 | 0.115 |
| **MioTTS** | long (66w) | 899 | 2,919 | 25,227 | 0.116 |

## Key Metrics

- **TTFA** — Time to first audio (latency before audio starts playing)
- **Total** — Total wall-clock generation time
- **Audio** — Duration of generated audio
- **RTF** — Real-time factor (generation time / audio duration; lower = faster than realtime)

## Streaming Analysis

### TTFA by text length (the streaming test)

| Backend | Short TTFA | Medium TTFA | Long TTFA | Streaming type |
|---------|-----------|-------------|-----------|----------------|
| **KaniTTS** | 301ms | 301ms | 301ms | True SSE streaming (flat) |
| **MioTTS** | 293ms | 645ms | 899ms | Sentence pipelining |
| **Chatterbox** | 711ms | 983ms | 1,609ms | Sentence pipelining |

**KaniTTS** has true token-level streaming via SSE — TTFA is dead flat at ~300ms regardless of input length. The server generates audio chunks as it decodes tokens and streams them immediately.

**MioTTS** and **Chatterbox** use sentence-level pipelining: text is split at sentence boundaries, the first sentence is generated and delivered immediately, then subsequent sentences are generated sequentially. TTFA = first-sentence generation time, which scales with first-sentence length but is much faster than waiting for the full text.

### Before vs after streaming (TTFA improvement)

| Backend | Old short | Old long | New short | New long | Long improvement |
|---------|----------|---------|----------|---------|-----------------|
| Chatterbox | 981ms | 4,458ms | 711ms | 1,609ms | **2.8x faster** |
| KaniTTS | 1,442ms* | 315ms | 301ms | 301ms | flat (was hidden by warmup) |
| MioTTS | 430ms | 2,360ms | 293ms | 899ms | **2.6x faster** |

*KaniTTS old short was inflated by server warmup on first request (now excluded via warmup run).

### Speed (RTF)

| Backend | Avg RTF (long) | Notes |
|---------|---------------|-------|
| KaniTTS | 0.102 | Fastest. ~10x realtime |
| MioTTS | 0.116 | ~8.6x realtime |
| Chatterbox | 0.244 | ~4.1x realtime |

### Setup Complexity

| Backend | Architecture | GPU VRAM |
|---------|-------------|----------|
| Chatterbox | In-process (350M params) | ~1-2 GB |
| KaniTTS | External server (FastAPI + NeMo) | ~4 GB |
| MioTTS | Two servers (vLLM LLM + FastAPI codec) | ~7 GB |

### Recommendation

- **Lowest latency:** KaniTTS — true streaming, flat 300ms TTFA at any text length
- **Fastest generation:** KaniTTS at 0.102 RTF
- **Simplest setup:** Chatterbox — no server, `pip install`, still 4.1x realtime
- **Best overall:** KaniTTS — streaming + fastest + single server

## Implementation Details

### True streaming (KaniTTS)
The KaniTTS server generates speech tokens autoregressively and converts each chunk to audio via a codec in real-time. Audio chunks are sent as base64-encoded PCM in SSE events (`speech.audio.delta`). The client streams the HTTP response and delivers audio as each SSE event arrives.

### Sentence pipelining (Chatterbox, MioTTS)
Neither model supports token-level streaming (Chatterbox needs all tokens before vocoding; MioTTS codec needs all tokens for decoding). Instead, text is split at sentence boundaries (`re.split(r"(?<=[.!?])\s+", text)`), and each sentence is generated independently. The first sentence's audio is delivered as soon as it's ready, while subsequent sentences are generated and delivered sequentially.

## Notes

- Benchmark includes a warmup run (excluded from metrics) to eliminate cold-start effects
- KaniTTS server: `/home/logan/Projects/kani-tts-2-openai-server/` on port 8000
- MioTTS servers: vLLM on port 8100, API on port 8001 with `--max-text-length 500`
- MioTTS server: `/home/logan/Projects/MioTTS-Inference/`
