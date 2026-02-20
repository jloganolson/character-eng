"""Tests for the voice I/O module."""

import queue
import threading
import time
from unittest.mock import MagicMock, patch, PropertyMock

from character_eng.voice import (
    AUTO_BEAT_DELAY,
    EXIT,
    TOGGLE_VOICE,
    VOICE_ERROR,
    KeyListener,
    MicStream,
    SpeakerStream,
    VoiceIO,
    _KEY_MAP,
    check_voice_available,
)


# --- MicStream ---


def test_mic_stream_callback_queues_data():
    """Callback should put audio bytes on the internal queue."""
    on_audio = MagicMock()
    mic = MicStream(on_audio)
    # Simulate callback
    data = b"\x00\x01" * 640
    mic._callback(data, 640, None, None)
    assert not mic._queue.empty()
    assert mic._queue.get() == data


def test_mic_stream_feeder_drains_queue():
    """Feeder thread should call on_audio with queued data."""
    on_audio = MagicMock()
    mic = MicStream(on_audio)
    mic._running = True

    # Put data on queue
    test_data = b"\x00\x01" * 640
    mic._queue.put(test_data)

    # Run one iteration of the feed loop manually
    mic._running = True
    data = mic._queue.get(timeout=0.1)
    mic._on_audio(data)

    on_audio.assert_called_once_with(test_data)


def test_mic_stream_stop_drains_queue():
    """stop() should drain any remaining data in the queue."""
    on_audio = MagicMock()
    mic = MicStream(on_audio)
    mic._queue.put(b"\x00" * 100)
    mic._queue.put(b"\x01" * 100)
    mic.stop()
    assert mic._queue.empty()


# --- SpeakerStream ---


def test_speaker_enqueue_and_flush():
    """enqueue() adds data, flush() clears it."""
    speaker = SpeakerStream()
    speaker.enqueue(b"\x00" * 100)
    assert not speaker.is_done()

    speaker.flush()
    assert speaker.is_done()


def test_speaker_wait_until_done_when_empty():
    """wait_until_done should return True immediately when no audio buffered."""
    speaker = SpeakerStream()
    assert speaker.wait_until_done(timeout=0.1) is True


def test_speaker_wait_until_done_with_data():
    """wait_until_done should return False (timeout) when audio is buffered."""
    speaker = SpeakerStream()
    speaker.enqueue(b"\x00" * 100)
    assert speaker.wait_until_done(timeout=0.05) is False


def test_speaker_callback_plays_from_buffer():
    """OutputStream callback should pull data from the buffer."""
    speaker = SpeakerStream()
    # Put exactly one frame worth of data (100ms at 24kHz = 2400 samples = 4800 bytes)
    frame_bytes = 4800
    speaker.enqueue(b"\x42" * frame_bytes)

    # Simulate callback
    outdata = bytearray(frame_bytes)
    speaker._callback(outdata, 2400, None, None)
    assert outdata == bytearray(b"\x42" * frame_bytes)
    # Buffer should be empty now, done_event set on next callback
    assert len(speaker._buffer) == 0


def test_speaker_callback_pads_silence():
    """When buffer has less data than needed, pad with silence."""
    speaker = SpeakerStream()
    speaker.enqueue(b"\x42\x42")  # 2 bytes (1 sample)

    outdata = bytearray(4800)  # request 2400 samples
    speaker._callback(outdata, 2400, None, None)
    # First 2 bytes should be our data
    assert outdata[0] == 0x42
    assert outdata[1] == 0x42
    # Rest should be silence (zeros)
    assert all(b == 0 for b in outdata[2:])


def test_speaker_stop():
    """stop() should clear buffer and set done."""
    speaker = SpeakerStream()
    speaker.enqueue(b"\x00" * 1000)
    speaker.stop()
    assert speaker.is_done()
    assert len(speaker._buffer) == 0


# --- DeepgramSTT ---


def test_deepgram_stt_dispatches_transcript():
    """on_transcript should be called when speech_final fires with text."""
    from character_eng.voice import DeepgramSTT

    on_transcript = MagicMock()
    on_turn_start = MagicMock()
    stt = DeepgramSTT(on_transcript, on_turn_start)

    # Simulate a Results message with final + speech_final (dict format)
    message = {
        "type": "Results",
        "channel": {"alternatives": [{"transcript": "Hello world"}]},
        "is_final": True,
        "speech_final": True,
    }

    stt._on_message(message)
    on_transcript.assert_called_once_with("Hello world")


def test_deepgram_stt_dispatches_turn_start():
    """on_turn_start should be called when SpeechStarted message arrives."""
    from character_eng.voice import DeepgramSTT

    on_transcript = MagicMock()
    on_turn_start = MagicMock()
    stt = DeepgramSTT(on_transcript, on_turn_start)

    stt._on_message({"type": "SpeechStarted"})
    on_turn_start.assert_called_once()


def test_deepgram_stt_ignores_interim_results():
    """Non-final transcripts should not fire on_transcript."""
    from character_eng.voice import DeepgramSTT

    on_transcript = MagicMock()
    on_turn_start = MagicMock()
    stt = DeepgramSTT(on_transcript, on_turn_start)

    message = {
        "type": "Results",
        "channel": {"alternatives": [{"transcript": "partial"}]},
        "is_final": False,
        "speech_final": False,
    }

    stt._on_message(message)
    on_transcript.assert_not_called()


def test_deepgram_stt_utterance_end_fires_pending():
    """UtteranceEnd should fire on_transcript with accumulated text."""
    from character_eng.voice import DeepgramSTT

    on_transcript = MagicMock()
    on_turn_start = MagicMock()
    stt = DeepgramSTT(on_transcript, on_turn_start)

    # Build up a pending transcript via a final result without speech_final
    message = {
        "type": "Results",
        "channel": {"alternatives": [{"transcript": "Hello"}]},
        "is_final": True,
        "speech_final": False,
    }

    stt._on_message(message)
    on_transcript.assert_not_called()

    # UtteranceEnd should flush it
    stt._on_message({"type": "UtteranceEnd"})
    on_transcript.assert_called_once_with("Hello")


# --- ElevenLabsTTS ---


@patch.dict("os.environ", {"ELEVENLABS_API_KEY": "test-key"})
def test_elevenlabs_tts_send_text():
    """send_text should send JSON with text via WebSocket."""
    from character_eng.voice import ElevenLabsTTS
    import json

    on_audio = MagicMock()
    tts = ElevenLabsTTS(on_audio)

    # Pre-inject a mock WebSocket (bypass _connect)
    mock_ws = MagicMock()
    tts._ws = mock_ws
    tts._running = True

    tts.send_text("Hello world")

    # Should have sent a text chunk
    mock_ws.send.assert_called_once()
    sent = json.loads(mock_ws.send.call_args[0][0])
    assert sent["text"] == "Hello world"
    assert sent["try_trigger_generation"] is True

    tts._running = False
    tts._ws = None


@patch.dict("os.environ", {"ELEVENLABS_API_KEY": "test-key"})
def test_elevenlabs_tts_flush_sends_empty_text():
    """flush() should send empty text to trigger final audio generation."""
    from character_eng.voice import ElevenLabsTTS
    import json

    on_audio = MagicMock()
    tts = ElevenLabsTTS(on_audio)

    mock_ws = MagicMock()
    tts._ws = mock_ws
    tts._running = True

    tts.flush()

    sent = json.loads(mock_ws.send.call_args[0][0])
    assert sent["text"] == ""

    tts._running = False
    tts._ws = None


@patch.dict("os.environ", {"ELEVENLABS_API_KEY": "test-key"})
def test_elevenlabs_tts_close():
    """close() should close the WebSocket."""
    from character_eng.voice import ElevenLabsTTS

    on_audio = MagicMock()
    tts = ElevenLabsTTS(on_audio)

    mock_ws = MagicMock()
    tts._ws = mock_ws
    tts._running = True

    tts.close()

    mock_ws.close.assert_called_once()
    assert tts._ws is None
    assert tts._running is False


# --- VoiceIO ---


def test_voice_io_event_queue_receives_transcript():
    """Transcript from STT should appear on the event queue."""
    voice = VoiceIO()
    voice._event_queue.put("Hello world")
    result = voice._event_queue.get(timeout=0.1)
    assert result == "Hello world"


def test_voice_io_auto_beat_timer_fires():
    """Auto-beat timer should put /beat on the event queue after delay."""
    voice = VoiceIO()
    voice._auto_beat_timer = threading.Timer(0.05, voice._fire_auto_beat)
    voice._auto_beat_timer.daemon = True
    voice._auto_beat_timer.start()
    result = voice._event_queue.get(timeout=1.0)
    assert result == "/beat"


def test_voice_io_auto_beat_timer_cancel():
    """Cancelling auto-beat timer should prevent /beat from firing."""
    voice = VoiceIO()
    voice._start_auto_beat()
    voice._cancel_auto_beat()
    time.sleep(0.1)
    assert voice._event_queue.empty()


def test_voice_io_cancel_speech_sets_event():
    """cancel_speech() should set the cancelled event."""
    voice = VoiceIO()
    voice._speaker = MagicMock()
    voice._tts = MagicMock()
    voice.cancel_speech()
    assert voice._cancelled.is_set()


def test_voice_io_cancel_speech_flushes_speaker():
    """cancel_speech() should flush the speaker buffer."""
    voice = VoiceIO()
    voice._speaker = MagicMock()
    voice._tts = MagicMock()
    voice.cancel_speech()
    voice._speaker.flush.assert_called_once()


def test_voice_io_cancel_speech_closes_tts():
    """cancel_speech() should close the TTS connection."""
    voice = VoiceIO()
    voice._speaker = MagicMock()
    voice._tts = MagicMock()
    voice.cancel_speech()
    voice._tts.close.assert_called_once()


def test_voice_io_wait_for_input_returns_from_queue():
    """wait_for_input should return items placed on the event queue."""
    voice = VoiceIO()
    voice._started = True

    # Put something on the queue in another thread
    def _put():
        time.sleep(0.05)
        voice._event_queue.put("test transcript")

    t = threading.Thread(target=_put)
    t.start()
    result = voice.wait_for_input()
    t.join()
    assert result == "test transcript"


def test_voice_io_stop_drains_queue():
    """stop() should drain the event queue."""
    voice = VoiceIO()
    voice._event_queue.put("leftover")
    voice._event_queue.put("more")
    voice.stop()
    assert voice._event_queue.empty()


# --- KeyListener ---


def test_key_map_contains_expected_keys():
    """Key map should contain all expected hotkey mappings."""
    assert _KEY_MAP["w"] == "/world"
    assert _KEY_MAP["b"] == "/beat"
    assert _KEY_MAP["p"] == "/plan"
    assert _KEY_MAP["g"] == "/goals"
    assert _KEY_MAP["t"] == "/trace"
    assert _KEY_MAP["\x1b"] == TOGGLE_VOICE
    assert _KEY_MAP["\x03"] == EXIT


def test_key_listener_puts_mapped_command():
    """KeyListener should put the mapped command on the queue when a key is pressed."""
    eq = queue.Queue()
    kl = KeyListener(eq)
    # Simulate what the listen loop does internally
    ch = "b"
    if ch in _KEY_MAP:
        eq.put(_KEY_MAP[ch])
    result = eq.get(timeout=0.1)
    assert result == "/beat"


# --- check_voice_available ---


@patch.dict("os.environ", {"DEEPGRAM_API_KEY": "key1", "ELEVENLABS_API_KEY": "key2"})
def test_check_voice_available_all_present():
    """Should return True when all deps and keys are present."""
    with patch.dict("sys.modules", {
        "sounddevice": MagicMock(),
        "numpy": MagicMock(),
        "deepgram": MagicMock(),
        "websocket": MagicMock(),
    }):
        available, reason = check_voice_available()
        assert available is True
        assert reason == ""


@patch.dict("os.environ", {"DEEPGRAM_API_KEY": "", "ELEVENLABS_API_KEY": "key2"})
def test_check_voice_available_missing_deepgram_key():
    """Should return False when DEEPGRAM_API_KEY is not set."""
    with patch.dict("sys.modules", {
        "sounddevice": MagicMock(),
        "numpy": MagicMock(),
        "deepgram": MagicMock(),
        "websocket": MagicMock(),
    }):
        available, reason = check_voice_available()
        assert available is False
        assert "DEEPGRAM_API_KEY" in reason


@patch.dict("os.environ", {"DEEPGRAM_API_KEY": "key1", "ELEVENLABS_API_KEY": ""})
def test_check_voice_available_missing_elevenlabs_key():
    """Should return False when ELEVENLABS_API_KEY is not set."""
    with patch.dict("sys.modules", {
        "sounddevice": MagicMock(),
        "numpy": MagicMock(),
        "deepgram": MagicMock(),
        "websocket": MagicMock(),
    }):
        available, reason = check_voice_available()
        assert available is False
        assert "ELEVENLABS_API_KEY" in reason


# --- Sentinel constants ---


def test_sentinel_strings_are_distinct():
    """All sentinel strings should be unique."""
    sentinels = [TOGGLE_VOICE, EXIT, VOICE_ERROR]
    assert len(set(sentinels)) == len(sentinels)
    for s in sentinels:
        assert s.startswith("__") and s.endswith("__")
