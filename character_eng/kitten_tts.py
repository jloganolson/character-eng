"""KittenTTS — HTTP client for Kitten-TTS-Server (ONNX, 8 fixed voices).

Same 4-method interface: send_text(), flush(), wait_for_done(), close().
send_text() buffers, flush() triggers HTTP POST to /v1/audio/speech.
Receives complete WAV response, extracts PCM, delivers in ~100ms chunks.
24kHz native — no resampling needed.
"""

from __future__ import annotations

import io
import sys
import threading
import wave
from typing import Callable

import numpy as np


class KittenTTS:
    """HTTP client for Kitten-TTS-Server.

    Connects to a running KittenTTS server. Receives WAV responses,
    extracts PCM, and delivers in chunks via on_audio callback.
    """

    TARGET_RATE = 24000
    CHUNK_SAMPLES = 2400  # ~100ms at 24kHz

    def __init__(
        self,
        on_audio: Callable[[bytes], None],
        server_url: str = "http://localhost:8005",
        voice: str = "expr-voice-2-m",
    ):
        self._on_audio = on_audio
        self._server_url = server_url.rstrip("/")
        self._voice = voice
        self._buffer: list[str] = []
        self._cancelled = threading.Event()
        self._generation_done = threading.Event()
        self._gen_thread: threading.Thread | None = None

    def send_text(self, text: str):
        """Buffer a text chunk."""
        self._buffer.append(text)

    def flush(self):
        """Join buffered text and start background generation via HTTP POST."""
        text = "".join(self._buffer).strip()
        self._buffer.clear()
        if not text:
            self._generation_done.set()
            return

        self._cancelled.clear()
        self._generation_done.clear()
        self._gen_thread = threading.Thread(
            target=self._generate, args=(text,), daemon=True
        )
        self._gen_thread.start()

    def _generate(self, text: str):
        """Background thread: POST to KittenTTS server, decode WAV, deliver audio."""
        import requests

        try:
            url = f"{self._server_url}/v1/audio/speech"
            payload = {
                "model": "kitten-tts",
                "input": text,
                "voice": self._voice,
                "response_format": "wav",
            }

            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()

            if self._cancelled.is_set():
                return

            # Extract PCM from WAV response
            wav_data = io.BytesIO(resp.content)
            with wave.open(wav_data, "rb") as wf:
                source_rate = wf.getframerate()
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                raw_pcm = wf.readframes(wf.getnframes())

            # Convert to mono int16 if needed
            if sampwidth != 2:
                samples = np.frombuffer(raw_pcm, dtype=f"<i{sampwidth}")
                scale = 32767 / (2 ** (sampwidth * 8 - 1) - 1)
                raw_pcm = (samples * scale).clip(-32768, 32767).astype(np.int16).tobytes()

            if n_channels > 1:
                samples = np.frombuffer(raw_pcm, dtype=np.int16)
                raw_pcm = samples[::n_channels].copy().tobytes()

            # Resample if needed (should be 24kHz native)
            if source_rate != self.TARGET_RATE:
                raw_pcm = self._resample(raw_pcm, source_rate)

            # Deliver in chunks
            pcm_array = np.frombuffer(raw_pcm, dtype=np.int16)
            for i in range(0, len(pcm_array), self.CHUNK_SAMPLES):
                if self._cancelled.is_set():
                    return
                chunk = pcm_array[i : i + self.CHUNK_SAMPLES]
                self._on_audio(chunk.tobytes())

        except Exception as e:
            sys.stderr.write(f"[KittenTTS] Generation error: {e}\n")
            sys.stderr.flush()
        finally:
            self._generation_done.set()

    def _resample(self, pcm: bytes, source_rate: int) -> bytes:
        """Resample int16 PCM from source_rate to TARGET_RATE."""
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
        old_len = len(samples)
        if old_len == 0:
            return pcm
        new_len = round(old_len * self.TARGET_RATE / source_rate)
        old_idx = np.arange(old_len)
        new_idx = np.linspace(0, old_len - 1, new_len)
        resampled = np.interp(new_idx, old_idx, samples).astype(np.int16)
        return resampled.tobytes()

    def wait_for_done(self, timeout: float = 60.0) -> bool:
        """Block until generation finishes. Returns True if done, False on timeout."""
        return self._generation_done.wait(timeout=timeout)

    def close(self):
        """Cancel current generation, clear buffer, join thread."""
        self._cancelled.set()
        self._generation_done.set()
        self._buffer.clear()
        if self._gen_thread is not None:
            self._gen_thread.join(timeout=5)
            self._gen_thread = None
