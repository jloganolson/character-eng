"""Local TTS via Qwen3-TTS — drop-in replacement for ElevenLabsTTS.

Same 4-method interface: send_text(), flush(), wait_for_done(), close().
Key difference: ElevenLabs accepts streaming text chunks; Qwen3-TTS needs
complete text per generation. send_text() buffers, flush() triggers generation.

Model is loaded once (singleton) and persists across utterances.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Callable

import numpy as np

# Add thirdparty/ to sys.path so vendored qwen_tts is importable
_THIRDPARTY = str(Path(__file__).resolve().parent.parent / "thirdparty")
if _THIRDPARTY not in sys.path:
    sys.path.insert(0, _THIRDPARTY)

# Module-level singleton — loaded once, persists across utterances
_model_singleton: dict | None = None
_model_lock = threading.Lock()


def _ensure_model(model_id: str, device: str, ref_audio_path: str) -> dict:
    """Load model + prompt on first call, return cached dict on subsequent calls."""
    global _model_singleton
    with _model_lock:
        if _model_singleton is not None:
            return _model_singleton

        import torch
        from qwen_tts import Qwen3TTSModel

        sys.stderr.write(f"[LocalTTS] Loading {model_id} on {device}...\n")
        sys.stderr.flush()

        model = Qwen3TTSModel.from_pretrained(
            model_id,
            device_map=device,
            dtype=torch.bfloat16,
        )

        # Pre-compute voice clone prompt from reference audio
        prompt_items = model.create_voice_clone_prompt(
            ref_audio=ref_audio_path,
            x_vector_only_mode=True,
        )
        prompt = prompt_items[0]

        _model_singleton = {"model": model, "prompt": prompt}
        sys.stderr.write("[LocalTTS] Model loaded.\n")
        sys.stderr.flush()
        return _model_singleton


class LocalTTS:
    """Local GPU TTS with same interface as ElevenLabsTTS.

    send_text(chunk) buffers text. flush() joins buffer and starts background
    generation. Audio chunks are delivered via on_audio(pcm_bytes) callback
    (int16 PCM at the model's native sample rate).
    """

    def __init__(
        self,
        on_audio: Callable[[bytes], None],
        ref_audio_path: str = "",
        model_id: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        device: str = "cuda:0",
    ):
        self._on_audio = on_audio
        self._ref_audio_path = ref_audio_path
        self._model_id = model_id
        self._device = device
        self._buffer: list[str] = []
        self._cancelled = threading.Event()
        self._generation_done = threading.Event()
        self._gen_thread: threading.Thread | None = None

    def send_text(self, text: str):
        """Buffer a text chunk. Model is lazy-loaded on first call."""
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

    def _generate(self, text: str):
        """Background thread: run model, stream PCM chunks via on_audio."""
        try:
            singleton = _ensure_model(
                self._model_id, self._device, self._ref_audio_path
            )
            model = singleton["model"]
            prompt = singleton["prompt"]

            streamer = model.generate_voice_clone_stream(
                text=text,
                voice_clone_prompt=[prompt],
            )

            for wav_chunk, sample_rate in streamer:
                if self._cancelled.is_set():
                    break
                # Convert float32 waveform → int16 PCM bytes
                pcm = (
                    (wav_chunk * 32767)
                    .clip(-32768, 32767)
                    .astype(np.int16)
                    .tobytes()
                )
                self._on_audio(pcm)
        except Exception as e:
            sys.stderr.write(f"[LocalTTS] Generation error: {e}\n")
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
