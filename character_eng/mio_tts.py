"""MioTTS — HTTP client for MioTTS inference server.

Same 4-method interface: send_text(), flush(), wait_for_done(), close().
send_text() buffers, flush() triggers HTTP POST.
Receives JSON with base64-encoded WAV, decodes and extracts PCM,
resamples to 24kHz, delivers in ~100ms chunks via on_audio callback.

Sentence-level pipelining: splits text into sentences at flush() time,
POSTs the first sentence, delivers its audio immediately, then POSTs
subsequent sentences. TTFA ≈ first-sentence round-trip time regardless
of total text length.
"""

from __future__ import annotations

import base64
import io
import re
import sys
import threading
import wave
from typing import Callable

import numpy as np


class MioTTS:
    """HTTP client for MioTTS inference server.

    Connects to a running MioTTS server. Receives JSON responses with
    base64-encoded WAV audio, decodes and extracts PCM, resamples to
    24kHz, and delivers in chunks.
    """

    TARGET_RATE = 24000
    CHUNK_SAMPLES = 2400  # ~100ms at 24kHz

    def __init__(
        self,
        on_audio: Callable[[bytes], None],
        server_url: str = "http://localhost:8001",
        voice_preset: str = "en_female",
    ):
        self._on_audio = on_audio
        self._server_url = server_url.rstrip("/")
        self._voice_preset = voice_preset
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

    def _resample(self, pcm: bytes, source_rate: int) -> bytes:
        """Resample int16 PCM from source_rate to TARGET_RATE via linear interpolation."""
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
        old_len = len(samples)
        if old_len == 0:
            return pcm
        new_len = round(old_len * self.TARGET_RATE / source_rate)
        old_idx = np.arange(old_len)
        new_idx = np.linspace(0, old_len - 1, new_len)
        resampled = np.interp(new_idx, old_idx, samples).astype(np.int16)
        return resampled.tobytes()

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences for pipelined generation."""
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p for p in parts if p.strip()]

    def _post_and_deliver(self, text: str) -> bool:
        """POST one text segment to the server, decode WAV, deliver audio.

        Returns False if cancelled.
        """
        import requests

        url = f"{self._server_url}/v1/tts"
        payload = {
            "text": text,
            "reference": {
                "type": "preset",
                "preset_id": self._voice_preset,
            },
            "output": {
                "format": "wav",
            },
        }

        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()

        if self._cancelled.is_set():
            return False

        # Response is JSON with base64-encoded WAV when format=base64,
        # or raw WAV when format=wav (Content-Type: audio/wav)
        content_type = resp.headers.get("content-type", "")
        if "application/json" in content_type:
            data = resp.json()
            audio_b64 = data.get("audio", "")
            if not audio_b64:
                sys.stderr.write("[MioTTS] No audio in JSON response\n")
                sys.stderr.flush()
                return True
            wav_bytes = base64.b64decode(audio_b64)
        else:
            wav_bytes = resp.content

        # Extract PCM from WAV
        wav_data = io.BytesIO(wav_bytes)
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

        # Resample to target rate
        if source_rate != self.TARGET_RATE:
            raw_pcm = self._resample(raw_pcm, source_rate)

        # Deliver in chunks
        pcm_array = np.frombuffer(raw_pcm, dtype=np.int16)
        for i in range(0, len(pcm_array), self.CHUNK_SAMPLES):
            if self._cancelled.is_set():
                return False
            chunk = pcm_array[i : i + self.CHUNK_SAMPLES]
            self._on_audio(chunk.tobytes())

        return True

    def _generate(self, text: str):
        """Background thread: split into sentences, POST and deliver each.

        Sentence-level pipelining: first sentence audio is delivered as soon
        as the server responds, so TTFA ≈ first-sentence round-trip time
        regardless of total text length.
        """
        try:
            sentences = self._split_sentences(text)

            for sentence in sentences:
                if self._cancelled.is_set():
                    return
                if not self._post_and_deliver(sentence):
                    return

        except Exception as e:
            sys.stderr.write(f"[MioTTS] Generation error: {e}\n")
            sys.stderr.flush()
        finally:
            self._generation_done.set()

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
