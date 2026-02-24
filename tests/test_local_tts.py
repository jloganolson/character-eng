"""Tests for character_eng.local_tts."""

from unittest.mock import MagicMock, patch

import numpy as np

from character_eng.local_tts import LocalTTS


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


def test_flush_starts_generation_thread():
    """flush() with text should start a background thread."""
    tts = LocalTTS(on_audio=MagicMock(), ref_audio_path="/fake/ref.wav")
    tts.send_text("Hello world")

    with patch.object(tts, "_generate"):
        tts.flush()
        assert tts._gen_thread is not None


def test_generate_converts_float32_to_int16_pcm():
    """_generate should convert float32 waveform chunks to int16 PCM bytes."""
    on_audio = MagicMock()
    tts = LocalTTS(on_audio=on_audio, ref_audio_path="/fake/ref.wav")

    wav_chunk = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
    mock_streamer = iter([(wav_chunk, 24000)])

    mock_model = MagicMock()
    mock_model.generate_voice_clone_stream.return_value = mock_streamer
    mock_prompt = MagicMock()

    mock_singleton = {"model": mock_model, "prompt": mock_prompt}

    with patch("character_eng.local_tts.load_model", return_value=mock_singleton):
        tts._generate("Hello world")

    on_audio.assert_called_once()
    pcm = on_audio.call_args[0][0]

    samples = np.frombuffer(pcm, dtype=np.int16)
    assert len(samples) == 5
    assert samples[0] == 0       # 0.0 * 32767
    assert samples[1] == 16383   # 0.5 * 32767
    assert samples[3] == 32767   # 1.0 * 32767 clamped
    assert samples[4] == -32767  # -1.0 * 32767 clamped


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

    with patch("character_eng.local_tts.load_model", return_value=mock_singleton):
        tts._generate("Hello")

    on_audio.assert_not_called()


def test_generate_sets_done_on_completion():
    """_generate should set _generation_done when finished."""
    tts = LocalTTS(on_audio=MagicMock(), ref_audio_path="/fake/ref.wav")

    mock_model = MagicMock()
    mock_model.generate_voice_clone_stream.return_value = iter([])
    mock_singleton = {"model": mock_model, "prompt": MagicMock()}

    with patch("character_eng.local_tts.load_model", return_value=mock_singleton):
        tts._generate("Hello")

    assert tts._generation_done.is_set()


def test_generate_sets_done_on_error():
    """_generate should set _generation_done even if an error occurs."""
    tts = LocalTTS(on_audio=MagicMock(), ref_audio_path="/fake/ref.wav")

    with patch("character_eng.local_tts.load_model", side_effect=RuntimeError("fail")):
        tts._generate("Hello")

    assert tts._generation_done.is_set()
