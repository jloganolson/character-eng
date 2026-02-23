"""Chatterbox-Turbo TTS — in-process zero-shot voice cloning.

Same 4-method interface: send_text(), flush(), wait_for_done(), close().
Like local_tts.py: send_text() buffers, flush() triggers generation.
Model is loaded once (singleton) and persists across utterances.

Sentence-level pipelining: splits text into sentences at flush() time,
generates the first sentence, delivers its audio immediately, then
generates subsequent sentences. TTFA ≈ first-sentence generation time
regardless of total text length.

24kHz output — no resampling needed for SpeakerStream.
"""

from __future__ import annotations

import re
import sys
import threading
from typing import Callable

import numpy as np

# Module-level singleton — loaded once, persists across utterances
_model_singleton = None
_model_lock = threading.Lock()


def load_model(device: str = "cuda:0"):
    """Load ChatterboxTurboTTS model. Thread-safe singleton.

    Call at startup for eager loading. Second call returns cached result.
    """
    global _model_singleton
    with _model_lock:
        if _model_singleton is not None:
            return _model_singleton

        # Patch perth watermarker if native backend unavailable
        import perth
        if perth.PerthImplicitWatermarker is None:
            perth.PerthImplicitWatermarker = perth.DummyWatermarker

        from chatterbox.tts import ChatterboxTTS as _ChatterboxModel

        sys.stderr.write(f"[ChatterboxTTS] Loading model on {device}...\n")
        sys.stderr.flush()

        model = _ChatterboxModel.from_pretrained(device=device)

        _model_singleton = model
        sys.stderr.write("[ChatterboxTTS] Ready.\n")
        sys.stderr.flush()
        return _model_singleton


class ChatterboxTTS:
    """In-process TTS via Chatterbox-Turbo with same interface as ElevenLabsTTS.

    send_text(chunk) buffers text. flush() joins buffer and starts background
    generation. Audio chunks are delivered via on_audio(pcm_bytes) callback
    (int16 PCM at 24kHz).
    """

    SAMPLE_RATE = 24000
    CHUNK_SAMPLES = 2400  # ~100ms at 24kHz

    def __init__(
        self,
        on_audio: Callable[[bytes], None],
        ref_audio_path: str = "",
        device: str = "cuda:0",
    ):
        self._on_audio = on_audio
        self._ref_audio_path = ref_audio_path
        self._device = device
        self._buffer: list[str] = []
        self._cancelled = threading.Event()
        self._generation_done = threading.Event()
        self._gen_thread: threading.Thread | None = None

    def send_text(self, text: str):
        """Buffer a text chunk."""
        self._buffer.append(text)

    def flush(self):
        """Join buffered text and start background generation."""
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

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences for pipelined generation."""
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p for p in parts if p.strip()]

    def _deliver_wav(self, wav) -> bool:
        """Convert wav tensor to PCM and deliver in chunks. Returns False if cancelled."""
        if hasattr(wav, "cpu"):
            samples = wav.squeeze().cpu().numpy().astype(np.float32)
        else:
            samples = np.asarray(wav, dtype=np.float32).squeeze()

        pcm_all = (
            (samples * 32767)
            .clip(-32768, 32767)
            .astype(np.int16)
        )

        for i in range(0, len(pcm_all), self.CHUNK_SAMPLES):
            if self._cancelled.is_set():
                return False
            chunk = pcm_all[i : i + self.CHUNK_SAMPLES]
            self._on_audio(chunk.tobytes())
        return True

    def _generate(self, text: str):
        """Background thread: split into sentences, generate and deliver each.

        Sentence-level pipelining: first sentence audio is delivered as soon
        as it's ready, so TTFA ≈ first-sentence generation time regardless
        of total text length.
        """
        try:
            model = load_model(self._device)
            sentences = self._split_sentences(text)

            for sentence in sentences:
                if self._cancelled.is_set():
                    return

                wav = model.generate(
                    sentence, audio_prompt_path=self._ref_audio_path
                )

                if not self._deliver_wav(wav):
                    return
        except Exception as e:
            sys.stderr.write(f"[ChatterboxTTS] Generation error: {e}\n")
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
