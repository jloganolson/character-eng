"""Tests for character_eng.kani_tts."""

import base64
import json
import threading
from unittest.mock import MagicMock, patch

import numpy as np

from character_eng.kani_tts import KaniTTS


def test_send_text_buffers():
    """send_text() should accumulate text in the buffer."""
    tts = KaniTTS(on_audio=MagicMock())
    tts.send_text("Hello ")
    tts.send_text("world")
    assert tts._buffer == ["Hello ", "world"]


def test_flush_clears_buffer():
    """flush() should clear the buffer after joining text."""
    tts = KaniTTS(on_audio=MagicMock())
    tts.send_text("Hello")
    with patch.object(tts, "_generate"):
        tts.flush()
    assert tts._buffer == []


def test_flush_empty_buffer_sets_done():
    """flush() with empty buffer should set _generation_done immediately."""
    tts = KaniTTS(on_audio=MagicMock())
    tts.flush()
    assert tts._generation_done.is_set()


def test_flush_whitespace_only_sets_done():
    """flush() with whitespace-only buffer should set _generation_done immediately."""
    tts = KaniTTS(on_audio=MagicMock())
    tts.send_text("   ")
    tts.flush()
    assert tts._generation_done.is_set()


def test_close_sets_cancelled():
    """close() should set _cancelled event."""
    tts = KaniTTS(on_audio=MagicMock())
    tts.close()
    assert tts._cancelled.is_set()


def test_close_sets_generation_done():
    """close() should set _generation_done to unblock waiters."""
    tts = KaniTTS(on_audio=MagicMock())
    tts.close()
    assert tts._generation_done.is_set()


def test_close_clears_buffer():
    """close() should clear any buffered text."""
    tts = KaniTTS(on_audio=MagicMock())
    tts.send_text("some text")
    tts.close()
    assert tts._buffer == []


def test_wait_for_done_returns_on_set():
    """wait_for_done should return True when _generation_done is set."""
    tts = KaniTTS(on_audio=MagicMock())
    tts._generation_done.set()
    assert tts.wait_for_done(timeout=0.05) is True


def test_wait_for_done_timeout():
    """wait_for_done should return False on timeout."""
    tts = KaniTTS(on_audio=MagicMock())
    assert tts.wait_for_done(timeout=0.05) is False


def test_resample_ratio():
    """_resample should convert 22050Hz to 24000Hz PCM."""
    tts = KaniTTS(on_audio=MagicMock())

    # 22050 samples at source rate = 1 second
    source_samples = np.zeros(22050, dtype=np.int16)
    result = tts._resample(source_samples.tobytes())
    result_samples = np.frombuffer(result, dtype=np.int16)

    # Should be ~24000 samples (1 second at target rate)
    assert len(result_samples) == 24000


def test_resample_empty():
    """_resample with empty input should return empty."""
    tts = KaniTTS(on_audio=MagicMock())
    result = tts._resample(b"")
    assert result == b""


def test_generate_parses_sse_and_delivers_audio():
    """_generate should parse SSE events and deliver resampled PCM."""
    received = []
    tts = KaniTTS(on_audio=lambda pcm: received.append(pcm))

    # Create mock SSE response with base64-encoded PCM
    source_pcm = np.array([100, 200, 300], dtype=np.int16).tobytes()
    b64_audio = base64.b64encode(source_pcm).decode()

    sse_lines = [
        f'data:{{"delta": "{b64_audio}"}}',
        "data:[DONE]",
    ]

    mock_response = MagicMock()
    mock_response.iter_lines.return_value = iter(sse_lines)
    mock_response.raise_for_status = MagicMock()

    with patch("requests.post", return_value=mock_response):
        tts._generate("Hello world")

    # Should have received resampled audio
    assert len(received) == 1
    result = np.frombuffer(received[0], dtype=np.int16)
    # 3 samples at 22050Hz → round(3 * 24000/22050) = 3 samples at 24kHz
    expected_len = round(3 * 24000 / 22050)
    assert len(result) == expected_len


def test_generate_respects_cancelled():
    """_generate should stop on cancelled event."""
    on_audio = MagicMock()
    tts = KaniTTS(on_audio=on_audio)
    tts._cancelled.set()

    source_pcm = np.array([100], dtype=np.int16).tobytes()
    b64_audio = base64.b64encode(source_pcm).decode()

    sse_lines = [
        f'data:{{"delta": "{b64_audio}"}}',
        "data:[DONE]",
    ]

    mock_response = MagicMock()
    mock_response.iter_lines.return_value = iter(sse_lines)
    mock_response.raise_for_status = MagicMock()

    with patch("requests.post", return_value=mock_response):
        tts._generate("Hello")

    on_audio.assert_not_called()


def test_generate_sets_done_on_completion():
    """_generate should set _generation_done when finished."""
    tts = KaniTTS(on_audio=MagicMock())

    mock_response = MagicMock()
    mock_response.iter_lines.return_value = iter(["data:[DONE]"])
    mock_response.raise_for_status = MagicMock()

    with patch("requests.post", return_value=mock_response):
        tts._generate("Hello")

    assert tts._generation_done.is_set()


def test_generate_sets_done_on_error():
    """_generate should set _generation_done even if an error occurs."""
    tts = KaniTTS(on_audio=MagicMock())

    with patch("requests.post", side_effect=RuntimeError("connection refused")):
        tts._generate("Hello")

    assert tts._generation_done.is_set()


def test_flush_starts_generation_thread():
    """flush() with text should start a background thread."""
    tts = KaniTTS(on_audio=MagicMock())
    tts.send_text("Hello world")

    with patch.object(tts, "_generate"):
        tts.flush()
        assert tts._gen_thread is not None


def test_generate_skips_non_audio_events():
    """_generate should skip SSE events without audio data."""
    received = []
    tts = KaniTTS(on_audio=lambda pcm: received.append(pcm))

    sse_lines = [
        'data:{"status": "processing"}',  # no delta/audio field
        "",  # empty line
        "not-a-data-line",  # not SSE
        "data:[DONE]",
    ]

    mock_response = MagicMock()
    mock_response.iter_lines.return_value = iter(sse_lines)
    mock_response.raise_for_status = MagicMock()

    with patch("requests.post", return_value=mock_response):
        tts._generate("Hello")

    assert len(received) == 0


def test_server_url_stripped():
    """Server URL should have trailing slash stripped."""
    tts = KaniTTS(on_audio=MagicMock(), server_url="http://localhost:8880/")
    assert tts._server_url == "http://localhost:8880"
