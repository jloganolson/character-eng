"""Tests for character_eng.local_tts."""

import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np

from character_eng.local_tts import LocalTTS, _ensure_model


def test_send_text_buffers():
    """send_text() should accumulate text in the buffer."""
    tts = LocalTTS(on_audio=MagicMock())
    tts.send_text("Hello ")
    tts.send_text("world")
    assert tts._buffer == ["Hello ", "world"]


def test_flush_clears_buffer():
    """flush() should clear the buffer after joining text."""
    tts = LocalTTS(on_audio=MagicMock())
    tts.send_text("Hello")
    # Patch _generate to prevent actual model loading
    with patch.object(tts, "_generate"):
        tts.flush()
    assert tts._buffer == []


def test_flush_empty_buffer_sets_done():
    """flush() with empty buffer should set _generation_done immediately."""
    tts = LocalTTS(on_audio=MagicMock())
    tts.flush()
    assert tts._generation_done.is_set()


def test_flush_whitespace_only_sets_done():
    """flush() with whitespace-only buffer should set _generation_done immediately."""
    tts = LocalTTS(on_audio=MagicMock())
    tts.send_text("   ")
    tts.flush()
    assert tts._generation_done.is_set()


def test_close_sets_cancelled():
    """close() should set _cancelled event."""
    tts = LocalTTS(on_audio=MagicMock())
    tts.close()
    assert tts._cancelled.is_set()


def test_close_sets_generation_done():
    """close() should set _generation_done to unblock waiters."""
    tts = LocalTTS(on_audio=MagicMock())
    tts.close()
    assert tts._generation_done.is_set()


def test_close_clears_buffer():
    """close() should clear any buffered text."""
    tts = LocalTTS(on_audio=MagicMock())
    tts.send_text("some text")
    tts.close()
    assert tts._buffer == []


def test_wait_for_done_returns_on_set():
    """wait_for_done should return True when _generation_done is set."""
    tts = LocalTTS(on_audio=MagicMock())
    tts._generation_done.set()
    assert tts.wait_for_done(timeout=0.05) is True


def test_wait_for_done_timeout():
    """wait_for_done should return False on timeout."""
    tts = LocalTTS(on_audio=MagicMock())
    assert tts.wait_for_done(timeout=0.05) is False


def test_generate_converts_float32_to_int16_pcm():
    """_generate should convert float32 waveform chunks to int16 PCM bytes."""
    on_audio = MagicMock()
    tts = LocalTTS(on_audio=on_audio, ref_audio_path="/fake/ref.wav")

    # Create a mock streamer that yields one chunk
    wav_chunk = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
    mock_streamer = iter([(wav_chunk, 24000)])

    mock_model = MagicMock()
    mock_model.generate_voice_clone_stream.return_value = mock_streamer
    mock_prompt = MagicMock()

    mock_singleton = {"model": mock_model, "prompt": mock_prompt}

    with patch("character_eng.local_tts._ensure_model", return_value=mock_singleton):
        tts._generate("Hello world")

    # Should have called on_audio once with int16 PCM bytes
    on_audio.assert_called_once()
    pcm = on_audio.call_args[0][0]

    # Verify it's valid int16 PCM
    samples = np.frombuffer(pcm, dtype=np.int16)
    assert len(samples) == 5
    assert samples[0] == 0       # 0.0 * 32767
    assert samples[1] == 16383   # 0.5 * 32767 ≈ 16383
    assert samples[3] == 32767   # 1.0 * 32767 clamped
    assert samples[4] == -32767  # -1.0 * 32767 clamped to [-32768, 32767]


def test_generate_respects_cancelled():
    """_generate should stop iterating when _cancelled is set."""
    on_audio = MagicMock()
    tts = LocalTTS(on_audio=on_audio, ref_audio_path="/fake/ref.wav")
    tts._cancelled.set()  # Pre-cancel

    wav_chunk = np.array([0.0, 0.5], dtype=np.float32)
    mock_streamer = iter([(wav_chunk, 24000), (wav_chunk, 24000)])

    mock_model = MagicMock()
    mock_model.generate_voice_clone_stream.return_value = mock_streamer
    mock_prompt = MagicMock()

    mock_singleton = {"model": mock_model, "prompt": mock_prompt}

    with patch("character_eng.local_tts._ensure_model", return_value=mock_singleton):
        tts._generate("Hello")

    # Should not have delivered any audio (cancelled before first chunk)
    on_audio.assert_not_called()


def test_generate_sets_done_on_completion():
    """_generate should set _generation_done when finished."""
    tts = LocalTTS(on_audio=MagicMock(), ref_audio_path="/fake/ref.wav")

    mock_model = MagicMock()
    mock_model.generate_voice_clone_stream.return_value = iter([])
    mock_singleton = {"model": mock_model, "prompt": MagicMock()}

    with patch("character_eng.local_tts._ensure_model", return_value=mock_singleton):
        tts._generate("Hello")

    assert tts._generation_done.is_set()


def test_generate_sets_done_on_error():
    """_generate should set _generation_done even if an error occurs."""
    tts = LocalTTS(on_audio=MagicMock(), ref_audio_path="/fake/ref.wav")

    with patch("character_eng.local_tts._ensure_model", side_effect=RuntimeError("fail")):
        tts._generate("Hello")

    assert tts._generation_done.is_set()


def test_flush_starts_generation_thread():
    """flush() with text should start a background thread."""
    tts = LocalTTS(on_audio=MagicMock(), ref_audio_path="/fake/ref.wav")
    tts.send_text("Hello world")

    # Patch _generate to be a no-op
    with patch.object(tts, "_generate"):
        tts.flush()
        assert tts._gen_thread is not None
