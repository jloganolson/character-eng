"""Pocket-TTS — streaming HTTP client for Pocket-TTS server (causal transformer).

Same 4-method interface: send_text(), flush(), wait_for_done(), close().
send_text() buffers, flush() triggers HTTP POST to /tts.
Receives streaming WAV response via chunked transfer encoding.
True autoregressive streaming — TTFA should be flat regardless of text length.
24kHz/16-bit/mono native — no resampling needed.
"""

from __future__ import annotations

import struct
import sys
import threading
from typing import Callable


class PocketTTS:
    """Streaming HTTP client for Pocket-TTS server.

    Connects to a running Pocket-TTS server. Receives streaming WAV
    via chunked HTTP transfer, skips WAV header, delivers raw int16 PCM
    chunks as they arrive for minimum latency.
    """

    TARGET_RATE = 24000
    WAV_HEADER_SIZE = 44  # Standard WAV header

    def __init__(
        self,
        on_audio: Callable[[bytes], None],
        server_url: str = "http://localhost:8003",
        voice: str = "alba",
        ref_audio: str = "",
    ):
        self._on_audio = on_audio
        self._server_url = server_url.rstrip("/")
        self._voice = voice
        self._ref_audio = ref_audio
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
        """Background thread: POST to Pocket-TTS, stream WAV response, deliver PCM."""
        import requests

        try:
            url = f"{self._server_url}/tts"
            # Pocket-TTS uses multipart/form-data
            data = {"text": text}
            files = None
            if self._ref_audio:
                files = {"voice_wav": open(self._ref_audio, "rb")}
            elif self._voice:
                data["voice_url"] = self._voice

            resp = requests.post(url, data=data, files=files, stream=True, timeout=120)
            resp.raise_for_status()

            if self._cancelled.is_set():
                return

            # Stream the response — first bytes are WAV header, rest is PCM
            header_buf = b""
            header_parsed = False

            for chunk in resp.iter_content(chunk_size=4800):  # ~100ms at 24kHz 16-bit
                if self._cancelled.is_set():
                    return

                if not header_parsed:
                    header_buf += chunk
                    if len(header_buf) >= self.WAV_HEADER_SIZE:
                        # Parse WAV header to verify format
                        if header_buf[:4] == b"RIFF" and header_buf[8:12] == b"WAVE":
                            # Find 'data' subchunk
                            pos = 12
                            while pos < len(header_buf) - 8:
                                subchunk_id = header_buf[pos:pos + 4]
                                subchunk_size = struct.unpack_from("<I", header_buf, pos + 4)[0]
                                if subchunk_id == b"data":
                                    data_start = pos + 8
                                    header_parsed = True
                                    # Deliver any PCM after the data header
                                    pcm = header_buf[data_start:]
                                    if pcm:
                                        self._on_audio(pcm)
                                    break
                                pos += 8 + subchunk_size
                            else:
                                # data subchunk not found yet, keep buffering
                                continue
                        else:
                            # Not a valid WAV — deliver raw as PCM anyway
                            header_parsed = True
                            self._on_audio(header_buf)
                else:
                    self._on_audio(chunk)

        except Exception as e:
            sys.stderr.write(f"[PocketTTS] Generation error: {e}\n")
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
