"""Tests for character_eng.mio_tts."""

import io
import threading
import wave
from unittest.mock import MagicMock, patch

import numpy as np

from character_eng.mio_tts import MioTTS


def _make_wav(samples: np.ndarray, rate: int = 44100, channels: int = 1) -> bytes:
    """Create a WAV file in memory from int16 samples."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # int16
        wf.setframerate(rate)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


def test_send_text_buffers():
    """send_text() should accumulate text in the buffer."""
    tts = MioTTS(on_audio=MagicMock())
    tts.send_text("Hello ")
    tts.send_text("world")
    assert tts._buffer == ["Hello ", "world"]


def test_flush_clears_buffer():
    """flush() should clear the buffer after joining text."""
    tts = MioTTS(on_audio=MagicMock())
    tts.send_text("Hello")
    with patch.object(tts, "_generate"):
        tts.flush()
    assert tts._buffer == []


def test_flush_empty_buffer_sets_done():
    """flush() with empty buffer should set _generation_done immediately."""
    tts = MioTTS(on_audio=MagicMock())
    tts.flush()
    assert tts._generation_done.is_set()


def test_flush_whitespace_only_sets_done():
    """flush() with whitespace-only buffer should set _generation_done immediately."""
    tts = MioTTS(on_audio=MagicMock())
    tts.send_text("   ")
    tts.flush()
    assert tts._generation_done.is_set()


def test_close_sets_cancelled():
    """close() should set _cancelled event."""
    tts = MioTTS(on_audio=MagicMock())
    tts.close()
    assert tts._cancelled.is_set()


def test_close_sets_generation_done():
    """close() should set _generation_done to unblock waiters."""
    tts = MioTTS(on_audio=MagicMock())
    tts.close()
    assert tts._generation_done.is_set()


def test_close_clears_buffer():
    """close() should clear any buffered text."""
    tts = MioTTS(on_audio=MagicMock())
    tts.send_text("some text")
    tts.close()
    assert tts._buffer == []


def test_wait_for_done_returns_on_set():
    """wait_for_done should return True when _generation_done is set."""
    tts = MioTTS(on_audio=MagicMock())
    tts._generation_done.set()
    assert tts.wait_for_done(timeout=0.05) is True


def test_wait_for_done_timeout():
    """wait_for_done should return False on timeout."""
    tts = MioTTS(on_audio=MagicMock())
    assert tts.wait_for_done(timeout=0.05) is False


def test_resample_ratio():
    """_resample should convert 44100Hz to 24000Hz PCM."""
    tts = MioTTS(on_audio=MagicMock())

    # 44100 samples at source rate = 1 second
    source_samples = np.zeros(44100, dtype=np.int16)
    result = tts._resample(source_samples.tobytes(), 44100)
    result_samples = np.frombuffer(result, dtype=np.int16)

    # Should be 24000 samples (1 second at target rate)
    assert len(result_samples) == 24000


def test_resample_empty():
    """_resample with empty input should return empty."""
    tts = MioTTS(on_audio=MagicMock())
    result = tts._resample(b"", 44100)
    assert result == b""


def test_generate_extracts_wav_and_delivers_chunks():
    """_generate should extract WAV, resample, and deliver chunked PCM."""
    received = []
    tts = MioTTS(on_audio=lambda pcm: received.append(pcm))

    # Create a WAV with 5000 samples at 44100Hz
    source_samples = np.zeros(5000, dtype=np.int16)
    wav_bytes = _make_wav(source_samples, rate=44100)

    mock_response = MagicMock()
    mock_response.content = wav_bytes
    mock_response.raise_for_status = MagicMock()

    with patch("requests.post", return_value=mock_response):
        tts._generate("Hello world")

    # Should have resampled 5000 @ 44100 → ~2721 @ 24000
    total_samples = sum(len(np.frombuffer(c, dtype=np.int16)) for c in received)
    expected = round(5000 * 24000 / 44100)
    assert total_samples == expected

    # Should be chunked (2400 samples per chunk)
    assert len(received) >= 2  # 2721 → 2 chunks


def test_generate_respects_cancelled():
    """_generate should stop delivering chunks when _cancelled is set."""
    on_audio = MagicMock()
    tts = MioTTS(on_audio=on_audio)
    tts._cancelled.set()

    source_samples = np.zeros(5000, dtype=np.int16)
    wav_bytes = _make_wav(source_samples)

    mock_response = MagicMock()
    mock_response.content = wav_bytes
    mock_response.raise_for_status = MagicMock()

    with patch("requests.post", return_value=mock_response):
        tts._generate("Hello")

    on_audio.assert_not_called()


def test_generate_sets_done_on_completion():
    """_generate should set _generation_done when finished."""
    tts = MioTTS(on_audio=MagicMock())

    source_samples = np.zeros(100, dtype=np.int16)
    wav_bytes = _make_wav(source_samples)

    mock_response = MagicMock()
    mock_response.content = wav_bytes
    mock_response.raise_for_status = MagicMock()

    with patch("requests.post", return_value=mock_response):
        tts._generate("Hello")

    assert tts._generation_done.is_set()


def test_generate_sets_done_on_error():
    """_generate should set _generation_done even if an error occurs."""
    tts = MioTTS(on_audio=MagicMock())

    with patch("requests.post", side_effect=RuntimeError("connection refused")):
        tts._generate("Hello")

    assert tts._generation_done.is_set()


def test_flush_starts_generation_thread():
    """flush() with text should start a background thread."""
    tts = MioTTS(on_audio=MagicMock())
    tts.send_text("Hello world")

    with patch.object(tts, "_generate"):
        tts.flush()
        assert tts._gen_thread is not None


def test_server_url_stripped():
    """Server URL should have trailing slash stripped."""
    tts = MioTTS(on_audio=MagicMock(), server_url="http://localhost:8200/")
    assert tts._server_url == "http://localhost:8200"


def test_generate_handles_24khz_wav_no_resample():
    """_generate should skip resampling if WAV is already at 24kHz."""
    received = []
    tts = MioTTS(on_audio=lambda pcm: received.append(pcm))

    # WAV at target rate (24kHz) — no resampling needed
    source_samples = np.arange(100, dtype=np.int16)
    wav_bytes = _make_wav(source_samples, rate=24000)

    mock_response = MagicMock()
    mock_response.content = wav_bytes
    mock_response.raise_for_status = MagicMock()

    with patch("requests.post", return_value=mock_response):
        tts._generate("Hello")

    total_samples = sum(len(np.frombuffer(c, dtype=np.int16)) for c in received)
    assert total_samples == 100  # No resampling, same count
