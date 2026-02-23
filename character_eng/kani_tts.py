"""KaniTTS-2 — SSE streaming HTTP client for KaniTTS OpenAI-compatible server.

Same 4-method interface: send_text(), flush(), wait_for_done(), close().
send_text() buffers, flush() triggers HTTP POST with SSE streaming.
Resamples 22050Hz → 24kHz for SpeakerStream compatibility.
"""

from __future__ import annotations

import base64
import sys
import threading
from typing import Callable

import numpy as np


class KaniTTS:
    """HTTP client for KaniTTS-2 OpenAI-compatible TTS server.

    Connects to a running KaniTTS server via SSE streaming POST.
    Audio arrives as base64-encoded PCM chunks in SSE events.
    """

    SOURCE_RATE = 22050
    TARGET_RATE = 24000

    def __init__(
        self,
        on_audio: Callable[[bytes], None],
        server_url: str = "http://localhost:8000",
        speaker_id: str = "",
    ):
        self._on_audio = on_audio
        self._server_url = server_url.rstrip("/")
        self._speaker_id = speaker_id
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

    def _resample(self, pcm: bytes) -> bytes:
        """Resample int16 PCM from SOURCE_RATE to TARGET_RATE via linear interpolation."""
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
        old_len = len(samples)
        if old_len == 0:
            return pcm
        new_len = round(old_len * self.TARGET_RATE / self.SOURCE_RATE)
        old_idx = np.arange(old_len)
        new_idx = np.linspace(0, old_len - 1, new_len)
        resampled = np.interp(new_idx, old_idx, samples).astype(np.int16)
        return resampled.tobytes()

    def _generate(self, text: str):
        """Background thread: POST to KaniTTS server, parse SSE, deliver audio."""
        import requests

        try:
            url = f"{self._server_url}/v1/audio/speech"
            payload = {
                "input": text,
                "response_format": "pcm",
                "stream_format": "sse",
            }
            if self._speaker_id:
                payload["voice"] = self._speaker_id

            resp = requests.post(url, json=payload, stream=True, timeout=30)
            resp.raise_for_status()

            for line in resp.iter_lines(decode_unicode=True):
                if self._cancelled.is_set():
                    break
                if not line or not line.startswith("data:"):
                    continue

                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    break

                # Parse SSE JSON event
                import json
                try:
                    event = json.loads(data_str)
                except (json.JSONDecodeError, ValueError):
                    continue

                # Check for done event
                event_type = event.get("type", "")
                if event_type == "speech.audio.done":
                    break

                # Extract audio delta — base64-encoded PCM
                audio_b64 = event.get("audio", "") or event.get("delta", "")
                if not audio_b64:
                    continue

                pcm = base64.b64decode(audio_b64)
                pcm = self._resample(pcm)
                self._on_audio(pcm)

        except Exception as e:
            sys.stderr.write(f"[KaniTTS] Generation error: {e}\n")
            sys.stderr.flush()
        finally:
            self._generation_done.set()

    def wait_for_done(self, timeout: float = 30.0) -> bool:
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
