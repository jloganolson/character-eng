"""Tests for character_eng.chatterbox_tts."""

import threading
from unittest.mock import MagicMock, patch

import numpy as np

from character_eng.chatterbox_tts import ChatterboxTTS, load_model


def test_send_text_buffers():
    """send_text() should accumulate text in the buffer."""
    tts = ChatterboxTTS(on_audio=MagicMock())
    tts.send_text("Hello ")
    tts.send_text("world")
    assert tts._buffer == ["Hello ", "world"]


def test_flush_clears_buffer():
    """flush() should clear the buffer after joining text."""
    tts = ChatterboxTTS(on_audio=MagicMock())
    tts.send_text("Hello")
    with patch.object(tts, "_generate"):
        tts.flush()
    assert tts._buffer == []


def test_flush_empty_buffer_sets_done():
    """flush() with empty buffer should set _generation_done immediately."""
    tts = ChatterboxTTS(on_audio=MagicMock())
    tts.flush()
    assert tts._generation_done.is_set()


def test_flush_whitespace_only_sets_done():
    """flush() with whitespace-only buffer should set _generation_done immediately."""
    tts = ChatterboxTTS(on_audio=MagicMock())
    tts.send_text("   ")
    tts.flush()
    assert tts._generation_done.is_set()


def test_close_sets_cancelled():
    """close() should set _cancelled event."""
    tts = ChatterboxTTS(on_audio=MagicMock())
    tts.close()
    assert tts._cancelled.is_set()


def test_close_sets_generation_done():
    """close() should set _generation_done to unblock waiters."""
    tts = ChatterboxTTS(on_audio=MagicMock())
    tts.close()
    assert tts._generation_done.is_set()


def test_close_clears_buffer():
    """close() should clear any buffered text."""
    tts = ChatterboxTTS(on_audio=MagicMock())
    tts.send_text("some text")
    tts.close()
    assert tts._buffer == []


def test_wait_for_done_returns_on_set():
    """wait_for_done should return True when _generation_done is set."""
    tts = ChatterboxTTS(on_audio=MagicMock())
    tts._generation_done.set()
    assert tts.wait_for_done(timeout=0.05) is True


def test_wait_for_done_timeout():
    """wait_for_done should return False on timeout."""
    tts = ChatterboxTTS(on_audio=MagicMock())
    assert tts.wait_for_done(timeout=0.05) is False


def test_generate_converts_tensor_to_int16_pcm():
    """_generate should convert model output tensor to int16 PCM chunks."""
    on_audio = MagicMock()
    tts = ChatterboxTTS(on_audio=on_audio, ref_audio_path="/fake/ref.wav")

    # Create a mock tensor (simulates torch tensor with .cpu().numpy())
    samples = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
    mock_tensor = MagicMock()
    mock_tensor.squeeze.return_value.cpu.return_value.numpy.return_value = samples

    mock_model = MagicMock()
    mock_model.generate.return_value = mock_tensor

    with patch("character_eng.chatterbox_tts.load_model", return_value=mock_model):
        tts._generate("Hello world")

    # Should have called on_audio with int16 PCM bytes
    on_audio.assert_called_once()
    pcm = on_audio.call_args[0][0]

    result = np.frombuffer(pcm, dtype=np.int16)
    assert len(result) == 5
    assert result[0] == 0       # 0.0 * 32767
    assert result[1] == 16383   # 0.5 * 32767
    assert result[3] == 32767   # 1.0 clamped
    assert result[4] == -32767  # -1.0 clamped


def test_generate_delivers_chunked_audio():
    """_generate should deliver audio in ~100ms chunks."""
    received = []
    tts = ChatterboxTTS(on_audio=lambda pcm: received.append(pcm), ref_audio_path="/fake/ref.wav")

    # 5000 samples → should split into 2400 + 2400 + 200 = 3 chunks
    samples = np.zeros(5000, dtype=np.float32)
    mock_tensor = MagicMock()
    mock_tensor.squeeze.return_value.cpu.return_value.numpy.return_value = samples

    mock_model = MagicMock()
    mock_model.generate.return_value = mock_tensor

    with patch("character_eng.chatterbox_tts.load_model", return_value=mock_model):
        tts._generate("Hello world")

    assert len(received) == 3
    # First two chunks: 2400 samples * 2 bytes = 4800 bytes
    assert len(received[0]) == 4800
    assert len(received[1]) == 4800
    # Last chunk: 200 samples * 2 bytes = 400 bytes
    assert len(received[2]) == 400


def test_generate_respects_cancelled():
    """_generate should stop delivering chunks when _cancelled is set."""
    on_audio = MagicMock()
    tts = ChatterboxTTS(on_audio=on_audio, ref_audio_path="/fake/ref.wav")
    tts._cancelled.set()  # Pre-cancel

    samples = np.zeros(5000, dtype=np.float32)
    mock_tensor = MagicMock()
    mock_tensor.squeeze.return_value.cpu.return_value.numpy.return_value = samples

    mock_model = MagicMock()
    mock_model.generate.return_value = mock_tensor

    with patch("character_eng.chatterbox_tts.load_model", return_value=mock_model):
        tts._generate("Hello")

    on_audio.assert_not_called()


def test_generate_sets_done_on_completion():
    """_generate should set _generation_done when finished."""
    tts = ChatterboxTTS(on_audio=MagicMock(), ref_audio_path="/fake/ref.wav")

    samples = np.array([], dtype=np.float32)
    mock_tensor = MagicMock()
    mock_tensor.squeeze.return_value.cpu.return_value.numpy.return_value = samples

    mock_model = MagicMock()
    mock_model.generate.return_value = mock_tensor

    with patch("character_eng.chatterbox_tts.load_model", return_value=mock_model):
        tts._generate("Hello")

    assert tts._generation_done.is_set()


def test_generate_sets_done_on_error():
    """_generate should set _generation_done even if an error occurs."""
    tts = ChatterboxTTS(on_audio=MagicMock(), ref_audio_path="/fake/ref.wav")

    with patch("character_eng.chatterbox_tts.load_model", side_effect=RuntimeError("fail")):
        tts._generate("Hello")

    assert tts._generation_done.is_set()


def test_flush_starts_generation_thread():
    """flush() with text should start a background thread."""
    tts = ChatterboxTTS(on_audio=MagicMock(), ref_audio_path="/fake/ref.wav")
    tts.send_text("Hello world")

    with patch.object(tts, "_generate"):
        tts.flush()
        assert tts._gen_thread is not None
