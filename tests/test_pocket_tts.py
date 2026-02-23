"""Tests for character_eng.pocket_tts."""

import struct
import threading
import time
from unittest.mock import MagicMock, patch, mock_open

from character_eng.pocket_tts import PocketTTS


def test_send_text_buffers():
    """send_text() should accumulate text in the buffer."""
    tts = PocketTTS(on_audio=MagicMock())
    tts.send_text("Hello ")
    tts.send_text("world")
    assert tts._buffer == ["Hello ", "world"]


def test_flush_clears_buffer():
    """flush() should clear the buffer after joining text."""
    tts = PocketTTS(on_audio=MagicMock())
    tts.send_text("Hello")
    with patch.object(tts, "_generate"):
        tts.flush()
    assert tts._buffer == []


def test_flush_empty_buffer_sets_done():
    """flush() with empty buffer should set _generation_done immediately."""
    tts = PocketTTS(on_audio=MagicMock())
    tts.flush()
    assert tts._generation_done.is_set()


def test_flush_whitespace_only_sets_done():
    """flush() with whitespace-only buffer should set _generation_done immediately."""
    tts = PocketTTS(on_audio=MagicMock())
    tts.send_text("   ")
    tts.flush()
    assert tts._generation_done.is_set()


def test_close_sets_cancelled():
    """close() should set _cancelled event."""
    tts = PocketTTS(on_audio=MagicMock())
    tts.close()
    assert tts._cancelled.is_set()


def test_close_sets_generation_done():
    """close() should set _generation_done to unblock waiters."""
    tts = PocketTTS(on_audio=MagicMock())
    tts.close()
    assert tts._generation_done.is_set()


def test_close_clears_buffer():
    """close() should clear any buffered text."""
    tts = PocketTTS(on_audio=MagicMock())
    tts.send_text("some text")
    tts.close()
    assert tts._buffer == []


def test_wait_for_done_returns_on_set():
    """wait_for_done should return True when _generation_done is set."""
    tts = PocketTTS(on_audio=MagicMock())
    tts._generation_done.set()
    assert tts.wait_for_done(timeout=0.05) is True


def test_wait_for_done_timeout():
    """wait_for_done should return False on timeout."""
    tts = PocketTTS(on_audio=MagicMock())
    assert tts.wait_for_done(timeout=0.05) is False


def test_server_url_trailing_slash_stripped():
    """Server URL should have trailing slash stripped."""
    tts = PocketTTS(on_audio=MagicMock(), server_url="http://localhost:8003/")
    assert tts._server_url == "http://localhost:8003"


def test_server_url_default():
    """Default server URL should be localhost:8003."""
    tts = PocketTTS(on_audio=MagicMock())
    assert tts._server_url == "http://localhost:8003"


def test_ref_audio_stored():
    """ref_audio should be stored for voice cloning."""
    tts = PocketTTS(on_audio=MagicMock(), ref_audio="/path/to/voice.wav")
    assert tts._ref_audio == "/path/to/voice.wav"


def test_flush_starts_generation_thread():
    """flush() with text should start a background thread."""
    tts = PocketTTS(on_audio=MagicMock())
    tts.send_text("Hello world")

    with patch.object(tts, "_generate"):
        tts.flush()
        assert tts._gen_thread is not None


def test_generate_sets_done_on_error():
    """_generate should set _generation_done even if an error occurs."""
    tts = PocketTTS(on_audio=MagicMock())

    with patch("character_eng.pocket_tts.PocketTTS._generate.__module__", create=True):
        # Mock requests to raise an error
        mock_requests = MagicMock()
        mock_requests.post.side_effect = RuntimeError("connection refused")
        with patch.dict("sys.modules", {"requests": mock_requests}):
            tts._generate("Hello")

    assert tts._generation_done.is_set()


def test_generate_respects_cancelled():
    """_generate should check cancelled before streaming response."""
    on_audio = MagicMock()
    tts = PocketTTS(on_audio=on_audio)
    tts._cancelled.set()

    mock_requests = MagicMock()
    mock_resp = MagicMock()
    mock_resp.iter_content.return_value = [b"\x00" * 100]
    mock_requests.post.return_value = mock_resp
    with patch.dict("sys.modules", {"requests": mock_requests}):
        tts._generate("Hello")

    on_audio.assert_not_called()


def test_generate_delivers_pcm_after_wav_header():
    """_generate should skip WAV header and deliver PCM data."""
    on_audio = MagicMock()
    tts = PocketTTS(on_audio=on_audio)

    # Build a minimal WAV header (44 bytes) + some PCM data
    wav_header = b"RIFF"
    wav_header += struct.pack("<I", 36 + 100)  # file size - 8
    wav_header += b"WAVE"
    wav_header += b"fmt "
    wav_header += struct.pack("<I", 16)  # fmt chunk size
    wav_header += struct.pack("<HHIIHH", 1, 1, 24000, 48000, 2, 16)  # PCM format
    wav_header += b"data"
    wav_header += struct.pack("<I", 100)  # data size
    pcm_data = b"\x42" * 100

    mock_requests = MagicMock()
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    # Return header + data as one chunk, then more data
    mock_resp.iter_content.return_value = [wav_header + pcm_data, b"\x43" * 50]
    mock_requests.post.return_value = mock_resp

    with patch.dict("sys.modules", {"requests": mock_requests}):
        tts._generate("Hello")

    # Should have delivered PCM (skipping WAV header)
    assert on_audio.call_count == 2
    # First call: PCM after header
    assert on_audio.call_args_list[0][0][0] == pcm_data
    # Second call: raw PCM chunk
    assert on_audio.call_args_list[1][0][0] == b"\x43" * 50
