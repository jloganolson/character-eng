"""Tests for the voice I/O module."""

import struct
import time
import threading
from unittest.mock import MagicMock, patch

from character_eng.bridge import BridgeServer
from character_eng.browser_voice import BrowserVoiceIO
from character_eng.voice import (
    AUTO_BEAT_DELAY,
    DeepgramSTT,
    MicStream,
    SpeakerStream,
    VoiceIO,
    check_voice_available,
    resolve_device,
)


# --- MicStream ---


def test_mic_stream_stop_drains_queue():
    """stop() should drain any remaining data in the queue."""
    on_audio = MagicMock()
    mic = MicStream(on_audio)
    mic._queue.put(b"\x00" * 100)
    mic._queue.put(b"\x01" * 100)
    mic.stop()
    assert mic._queue.empty()


def test_mic_stream_multichannel_downmix():
    """Callback should downmix multi-channel input to mono (channel 0)."""
    on_audio = MagicMock()
    mic = MicStream(on_audio)
    mic._actual_channels = 2

    # 2 frames of stereo int16: [L0, R0, L1, R1]
    stereo_data = struct.pack("<4h", 100, 200, 300, 400)
    mic._callback(stereo_data, 2, None, None)

    mono_data = mic._queue.get(timeout=0.1)
    samples = struct.unpack(f"<{len(mono_data) // 2}h", mono_data)
    # Should be channel 0 only: [100, 300]
    assert samples == (100, 300)


# --- SpeakerStream ---


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


def test_speaker_upsample_doubles_samples():
    """_upsample should duplicate each sample for 2x ratio."""
    speaker = SpeakerStream()
    speaker._actual_rate = 48000
    # Two int16 samples: [0x0100, 0x0200] in little-endian
    pcm = b"\x00\x01\x00\x02"
    result = speaker._upsample(pcm)
    # Each sample doubled: [0x0100, 0x0100, 0x0200, 0x0200]
    assert result == b"\x00\x01\x00\x01\x00\x02\x00\x02"


def test_speaker_upsample_non_integer_ratio():
    """_upsample should interpolate for non-integer ratios like 44100/24000."""
    speaker = SpeakerStream()
    speaker._actual_rate = 44100
    # 4 samples at 24kHz
    pcm = struct.pack("<4h", 0, 1000, 2000, 3000)
    result = speaker._upsample(pcm)
    # 44100/24000 * 4 ≈ 7.35 → 7 samples
    n_samples = len(result) // 2
    assert n_samples == round(4 * 44100 / 24000)


def test_speaker_enqueue_resamples_when_enabled():
    """enqueue() should upsample when _resample is True."""
    speaker = SpeakerStream()
    speaker._resample = True
    speaker._actual_rate = 48000
    pcm = b"\x00\x01\x00\x02"  # 2 samples
    speaker.enqueue(pcm)
    # Buffer should have 4 samples (8 bytes) after 2x upsample
    assert len(speaker._buffer) == 8


def test_speaker_bytes_enqueued_tracks_enqueue():
    """_bytes_enqueued should increment on enqueue."""
    speaker = SpeakerStream()
    assert speaker._bytes_enqueued == 0
    speaker.enqueue(b"\x00" * 100)
    assert speaker._bytes_enqueued == 100
    speaker.enqueue(b"\x00" * 200)
    assert speaker._bytes_enqueued == 300


def test_speaker_bytes_played_tracks_callback():
    """_bytes_played should increment when callback consumes from buffer."""
    speaker = SpeakerStream()
    speaker.enqueue(b"\x42" * 4800)
    outdata = bytearray(4800)
    speaker._callback(outdata, 2400, None, None)
    assert speaker._bytes_played == 4800


def test_speaker_playback_ratio():
    """playback_ratio should reflect fraction of enqueued audio that was played."""
    speaker = SpeakerStream()
    speaker.enqueue(b"\x42" * 4800)
    assert speaker.playback_ratio == 0.0
    # Play half
    outdata = bytearray(2400)
    speaker._callback(outdata, 1200, None, None)
    assert abs(speaker.playback_ratio - 0.5) < 0.01


def test_speaker_reset_counters():
    """reset_counters should zero both played and enqueued."""
    speaker = SpeakerStream()
    speaker.enqueue(b"\x00" * 100)
    outdata = bytearray(100)
    speaker._callback(outdata, 50, None, None)
    assert speaker._bytes_enqueued > 0
    assert speaker._bytes_played > 0
    speaker.reset_counters()
    assert speaker._bytes_enqueued == 0
    assert speaker._bytes_played == 0


def test_speaker_flush_preserves_counters():
    """flush() should NOT reset byte counters (needed for barge-in ratio)."""
    speaker = SpeakerStream()
    speaker.enqueue(b"\x00" * 100)
    outdata = bytearray(50)
    speaker._callback(outdata, 25, None, None)
    played_before = speaker._bytes_played
    enqueued_before = speaker._bytes_enqueued
    speaker.flush()
    assert speaker._bytes_played == played_before
    assert speaker._bytes_enqueued == enqueued_before


def test_speaker_on_output_callback():
    """on_output should be called with played PCM bytes after each callback."""
    played_chunks = []
    speaker = SpeakerStream(on_output=played_chunks.append)
    frame_bytes = 4800
    speaker.enqueue(b"\x42" * frame_bytes)

    outdata = bytearray(frame_bytes)
    speaker._callback(outdata, 2400, None, None)
    assert len(played_chunks) == 1
    assert played_chunks[0] == bytes(outdata)


def test_speaker_on_output_not_called_when_none():
    """No error when on_output is None (default)."""
    speaker = SpeakerStream()
    speaker.enqueue(b"\x42" * 4800)
    outdata = bytearray(4800)
    speaker._callback(outdata, 2400, None, None)  # should not raise


# --- DeepgramSTT ---


def test_deepgram_stt_dispatches_transcript():
    """on_transcript should be called when speech_final fires with text."""
    from character_eng.voice import DeepgramSTT

    on_transcript = MagicMock()
    on_turn_start = MagicMock()
    stt = DeepgramSTT(on_transcript, on_turn_start)

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


def test_deepgram_stt_send_audio_failure_sets_error_and_clears_ws_open():
    on_transcript = MagicMock()
    on_turn_start = MagicMock()
    stt = DeepgramSTT(on_transcript, on_turn_start)
    socket = MagicMock()
    socket.send_media.side_effect = RuntimeError("socket closed")
    stt._socket = socket
    stt._ws_open.set()

    stt.send_audio(b"\x00" * 32)

    status = stt.status_snapshot()
    assert status["ws_open"] is False
    assert "socket closed" in status["error"]


def test_deepgram_stt_on_error_clears_ws_open():
    on_transcript = MagicMock()
    on_turn_start = MagicMock()
    stt = DeepgramSTT(on_transcript, on_turn_start)
    stt._ws_open.set()

    stt._on_error(RuntimeError("deepgram lost connection"))

    status = stt.status_snapshot()
    assert status["ws_open"] is False
    assert "deepgram lost connection" in status["error"]


# --- ElevenLabsTTS ---


@patch.dict("os.environ", {"ELEVENLABS_API_KEY": "test-key"})
def test_elevenlabs_tts_send_text():
    """send_text should send JSON with text via WebSocket."""
    from character_eng.voice import ElevenLabsTTS
    import json

    on_audio = MagicMock()
    tts = ElevenLabsTTS(on_audio)

    mock_ws = MagicMock()
    tts._ws = mock_ws
    tts._running = True

    tts.send_text("Hello world")

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


@patch.dict("os.environ", {"ELEVENLABS_API_KEY": "test-key"})
def test_elevenlabs_tts_recv_loop_sets_generation_done_on_is_final():
    """_recv_loop should set _generation_done when isFinal message arrives."""
    from character_eng.voice import ElevenLabsTTS
    import json

    on_audio = MagicMock()
    tts = ElevenLabsTTS(on_audio)

    mock_ws = MagicMock()
    mock_ws.recv.side_effect = [
        json.dumps({"audio": "AAAA", "isFinal": False}),
        json.dumps({"isFinal": True}),
    ]
    tts._ws = mock_ws
    tts._running = True

    assert not tts._generation_done.is_set()
    tts._recv_loop()
    assert tts._generation_done.is_set()
    assert on_audio.call_count == 1


# --- VoiceIO ---


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


def test_voice_io_barge_in_guard_when_speaking_no_aec():
    """_on_turn_start should cancel speech when speaking (legacy, no AEC)."""
    voice = VoiceIO(aec=False)
    voice._speaker = MagicMock()
    voice._tts = MagicMock()
    voice._is_speaking = True

    voice._on_turn_start()
    assert voice._cancelled.is_set()


def test_voice_io_turn_start_ignored_during_playback_with_aec():
    """_on_turn_start should NOT cancel speech when AEC is active (too sensitive)."""
    voice = VoiceIO(aec=True)
    voice._aec = MagicMock()
    voice._speaker = MagicMock()
    voice._tts = MagicMock()
    voice._is_speaking = True

    voice._on_turn_start()
    assert not voice._cancelled.is_set()


def test_voice_io_transcript_triggers_barge_in_with_aec():
    """_on_transcript should cancel speech when AEC is active and speaking."""
    voice = VoiceIO(aec=True)
    voice._aec = MagicMock()
    voice._speaker = MagicMock()
    voice._tts = MagicMock()
    voice._is_speaking = True

    voice._on_transcript("hello")
    assert voice._cancelled.is_set()
    assert voice._event_queue.get(timeout=0.1) == "hello"


def test_voice_io_transcript_ignores_exact_echo_with_aec():
    events = []
    voice = VoiceIO(aec=True, trace_hook=lambda event_type, data: events.append((event_type, data)))
    voice._aec = MagicMock()
    voice._speaker = MagicMock()
    voice._tts = MagicMock()
    voice._is_speaking = True
    voice.begin_assistant_turn("Do you have water?")

    voice._on_transcript("do you have water")

    assert not voice._cancelled.is_set()
    assert voice._event_queue.empty()
    assert events == [(
        "user_transcript_final",
        {
            "text": "do you have water",
            "echo_candidate": True,
            "echo_disposition": "ignored",
            "echo_similarity": 1.0,
            "echo_reference": "active",
            "echo_reference_excerpt": "Do you have water?",
        },
    )]


def test_voice_io_transcript_ignores_fuzzy_echo_with_aec():
    voice = VoiceIO(aec=True)
    voice._aec = MagicMock()
    voice._speaker = MagicMock()
    voice._tts = MagicMock()
    voice._is_speaking = True
    voice.begin_assistant_turn("Nice scarf, man.")

    voice._on_transcript("nice scarf man")

    assert not voice._cancelled.is_set()
    assert voice._event_queue.empty()


def test_voice_io_transcript_no_barge_in_when_not_speaking():
    """_on_transcript should just queue text when not speaking."""
    voice = VoiceIO(aec=True)
    voice._aec = MagicMock()
    voice._is_speaking = False

    voice._on_transcript("hello")
    assert not voice._cancelled.is_set()
    assert voice._event_queue.get(timeout=0.1) == "hello"


def test_voice_io_barge_in_guard_when_not_speaking():
    """_on_turn_start should NOT call cancel_speech when _is_speaking is False."""
    voice = VoiceIO()
    voice._speaker = MagicMock()
    voice._tts = MagicMock()
    voice._is_speaking = False

    voice._on_turn_start()
    assert not voice._cancelled.is_set()


def test_voice_io_mic_muted_while_speaking_no_aec():
    """_on_mic_audio should not forward to STT when speaking and AEC is off."""
    voice = VoiceIO(aec=False)
    voice._stt = MagicMock()
    voice._is_speaking = True

    voice._on_mic_audio(b"\x00" * 100)
    voice._stt.send_audio.assert_not_called()


def test_voice_io_mic_active_when_not_speaking_no_aec():
    """_on_mic_audio should forward to STT when not speaking and AEC is off."""
    voice = VoiceIO(aec=False)
    voice._stt = MagicMock()
    voice._is_speaking = False

    voice._on_mic_audio(b"\x00" * 100)
    voice._stt.send_audio.assert_called_once_with(b"\x00" * 100)


def test_voice_io_mic_routes_through_aec():
    """_on_mic_audio should process through AEC when AEC is active."""
    voice = VoiceIO(aec=True)
    voice._stt = MagicMock()
    voice._is_speaking = True

    mock_aec = MagicMock()
    mock_aec.process_capture.return_value = b"\x01" * 50
    voice._aec = mock_aec

    voice._on_mic_audio(b"\x00" * 100)
    mock_aec.process_capture.assert_called_once_with(b"\x00" * 100)
    voice._stt.send_audio.assert_called_once_with(b"\x01" * 50)


def test_voice_io_cancel_speech_sets_barged_in():
    """cancel_speech() should set _barged_in."""
    voice = VoiceIO()
    voice._speaker = MagicMock()
    voice._tts = MagicMock()
    voice.cancel_speech()
    assert voice._barged_in is True


def test_voice_io_speak_clears_barged_in():
    """speak() should clear _barged_in at the start."""
    voice = VoiceIO()
    voice._speaker = MagicMock()
    voice._speaker.is_done.return_value = True
    voice._tts = MagicMock()
    voice._tts.wait_for_done.return_value = True
    voice._barged_in = True

    voice.speak(iter([]))
    assert voice._barged_in is False


def test_voice_io_speak_text_clears_barged_in():
    """speak_text() should clear _barged_in at the start."""
    voice = VoiceIO()
    voice._speaker = MagicMock()
    voice._speaker.is_done.return_value = True
    voice._tts = MagicMock()
    voice._tts.wait_for_done.return_value = True
    voice._barged_in = True

    voice.speak_text("hello")
    assert voice._barged_in is False


def test_voice_io_start_latency_filler_traces_phrase():
    events = []
    voice = VoiceIO(trace_hook=lambda event_type, data: events.append((event_type, data)))
    voice._speaker = MagicMock()
    voice._speaker.wait_for_audio.return_value = False
    voice._filler_bank = MagicMock()
    choice = MagicMock()
    choice.phrase = "Hmm."
    choice.sentiment = "curious"
    voice._filler_bank.pick.return_value = choice
    voice._filler_bank.clip_for_choice.return_value = b"\x00\x01"

    voice.start_latency_filler("hello", "curious")
    assert voice._filler_thread is not None
    voice._filler_thread.join(timeout=1)

    assert events == [("assistant_filler", {"phrase": "Hmm.", "sentiment": "curious"})]
    voice._speaker.enqueue.assert_called_once_with(b"\x00\x01")


def test_voice_io_speak_text_traces_live_timing():
    events = []
    voice = VoiceIO(trace_hook=lambda event_type, data: events.append((event_type, data)))
    voice._speaker = MagicMock()
    voice._speaker.is_done.return_value = True
    voice._speaker.wait_for_audio.return_value = True
    voice._tts = MagicMock()
    voice._tts.wait_for_done.return_value = True

    voice.speak_text("hello")

    names = [name for name, _ in events]
    assert "assistant_tts_live_start" in names
    assert "assistant_tts_live_synth_done" in names
    assert "assistant_tts_live_first_audio" in names
    assert "assistant_tts_live_done" in names


def test_voice_io_speaker_playback_ratio_delegates():
    """speaker_playback_ratio should delegate to speaker.playback_ratio."""
    voice = VoiceIO()
    mock_speaker = MagicMock()
    mock_speaker.playback_ratio = 0.75
    voice._speaker = mock_speaker
    assert voice.speaker_playback_ratio == 0.75


def test_voice_io_speaker_playback_ratio_no_speaker():
    """speaker_playback_ratio should return 1.0 when no speaker."""
    voice = VoiceIO()
    voice._speaker = None
    assert voice.speaker_playback_ratio == 1.0


def test_voice_io_inject_input_audio_uses_live_mic_path():
    voice = VoiceIO()
    seen = []
    voice._input_audio_hook = seen.append
    voice._stt = MagicMock()

    voice.inject_input_audio(b"\x00\x01" * 4)

    assert seen == [b"\x00\x01" * 4]
    voice._stt.send_audio.assert_called_once_with(b"\x00\x01" * 4)


def test_browser_voice_io_stt_stays_idle_until_browser_audio(monkeypatch):
    started = []
    sent_audio = []

    class StubTTS:
        def __init__(self, on_audio, voice_id=""):
            self.on_audio = on_audio

        def close(self):
            return

    class StubSTT:
        def __init__(self, on_transcript, on_turn_start):
            self._error = ""
            self._thread_alive = True
            self._ws_open = True

        def start(self):
            started.append("start")

        def send_audio(self, data: bytes):
            sent_audio.append(data)

        def status_snapshot(self):
            return {
                "thread_alive": self._thread_alive,
                "ws_open": self._ws_open,
                "error": self._error,
                "speaking": False,
                "pending_transcript": False,
            }

        def stop(self):
            return

    monkeypatch.setattr("character_eng.browser_voice.ElevenLabsTTS", StubTTS)
    monkeypatch.setattr("character_eng.browser_voice.DeepgramSTT", StubSTT)

    bridge = BridgeServer(port=0)
    voice = BrowserVoiceIO(bridge=bridge)
    voice.start()
    try:
        idle = voice.status_snapshot()
        assert started == []
        assert idle["browser_bridge"] is True
        assert idle["mic_ready"] is False
        assert idle["stt"] == {"state": "off", "detail": "Browser mic idle."}

        bridge.set_client_state({"mic": True})
        voice.inject_input_audio(b"\x00\x01" * 4)

        active = voice.status_snapshot()
        assert started == ["start"]
        assert sent_audio == [b"\x00\x01" * 4]
        assert active["mic_ready"] is True
        assert active["stt"]["state"] == "ready"
    finally:
        voice.stop()


def test_browser_voice_io_hides_stt_timeout_when_browser_mic_is_off(monkeypatch):
    class StubTTS:
        def __init__(self, on_audio, voice_id=""):
            self.on_audio = on_audio

        def close(self):
            return

    class StubSTT:
        def __init__(self, on_transcript, on_turn_start):
            return

        def start(self):
            return

        def send_audio(self, data: bytes):
            return

        def status_snapshot(self):
            return {
                "thread_alive": False,
                "ws_open": False,
                "error": "Deepgram timeout",
                "speaking": False,
                "pending_transcript": False,
            }

        def stop(self):
            return

    monkeypatch.setattr("character_eng.browser_voice.ElevenLabsTTS", StubTTS)
    monkeypatch.setattr("character_eng.browser_voice.DeepgramSTT", StubSTT)

    bridge = BridgeServer(port=0)
    voice = BrowserVoiceIO(bridge=bridge)
    voice.start()
    try:
        voice._stt = StubSTT(None, None)
        bridge.set_client_state({"mic": False})
        snapshot = voice.status_snapshot()
        assert snapshot["mic_ready"] is False
        assert snapshot["stt"] == {"state": "off", "detail": "Browser mic idle."}
    finally:
        voice.stop()


def test_browser_voice_io_uses_preloaded_pocket_voice(monkeypatch):
    captured = {}

    class StubPocketTTS:
        def __init__(self, on_audio, server_url="", voice="", ref_audio=""):
            captured["server_url"] = server_url
            captured["voice"] = voice
            captured["ref_audio"] = ref_audio

        def close(self):
            return

    bridge = BridgeServer(port=0)
    voice = BrowserVoiceIO(bridge=bridge, tts_backend="pocket", pocket_voice="voices/greg.safetensors")
    monkeypatch.setattr(voice, "_ensure_pocket_server", lambda server_url: None)
    monkeypatch.setattr("character_eng.browser_voice.PocketTTS", StubPocketTTS, raising=False)
    with patch.dict("sys.modules", {"character_eng.pocket_tts": MagicMock(PocketTTS=StubPocketTTS)}):
        voice.start()
    try:
        assert captured["server_url"] == "http://localhost:8003"
        assert captured["voice"] == ""
        assert captured["ref_audio"] == ""
    finally:
        voice.stop()


def test_voice_io_turn_start_cancels_auto_beat_when_not_speaking():
    """_on_turn_start should cancel auto-beat timer even when _is_speaking is False."""
    voice = VoiceIO()
    voice._is_speaking = False
    voice._start_auto_beat()
    assert voice._auto_beat_timer is not None

    voice._on_turn_start()
    assert voice._auto_beat_timer is None
    assert not voice._cancelled.is_set()  # cancel_speech NOT called


def test_voice_io_turn_start_ignores_duplicate_vad_after_recent_transcript():
    voice = VoiceIO()
    voice._speaker = MagicMock()
    voice._tts = MagicMock()
    voice._is_speaking = True
    voice._last_transcript_final_at = 10.0

    with patch("character_eng.voice.time.time", return_value=10.2):
        voice._on_turn_start()

    assert voice._last_speech_started_at is None
    assert not voice._cancelled.is_set()


def test_voice_io_consume_input_timing_captures_latest_speech_window():
    voice = VoiceIO(trace_hook=MagicMock())
    with patch("character_eng.voice.time.time", side_effect=[10.0, 10.42]):
        voice._on_turn_start()
        voice._on_transcript("hello there")

    payload = voice.consume_input_timing("hello there")

    assert payload["input_source"] == "voice"
    assert payload["speech_started_ts"] == 10.0
    assert payload["transcript_final_ts"] == 10.42
    assert payload["stt_ms"] == 420
    assert payload["stt_text"] == "hello there"
    assert payload["stt_text_match"] is True
    assert voice.consume_input_timing() == {}


def test_voice_io_trace_hook_reports_stt_timing():
    events = []
    voice = VoiceIO(trace_hook=lambda event_type, data: events.append((event_type, data)))
    voice._stt = MagicMock()
    voice._stt._last_is_final_at = 10.35

    with patch("character_eng.voice.time.time", side_effect=[10.0, 10.42]):
        voice._on_turn_start()
        voice._on_transcript("hello there")

    assert events == [
        ("user_speech_started", {"speech_started_ts": 10.0}),
        ("user_transcript_final", {
            "text": "hello there",
            "transcript_final_ts": 10.42,
            "speech_started_ts": 10.0,
            "stt_ms": 420,
            "speech_ended_ts": 10.35,
            "endpointing_ms": 70,
        }),
    ]


def test_voice_io_merge_deferred_input_prefix():
    voice = VoiceIO(trace_hook=MagicMock())
    voice.defer_next_user_turn("well hm")
    assert voice.merge_deferred_input("i don't think so") == "well hm i don't think so"
    assert voice.merge_deferred_input("next thing") == "next thing"


def test_voice_io_prune_stale_transcripts_removes_covered_fragments_only():
    voice = VoiceIO(trace_hook=MagicMock())
    voice._event_queue.put("i don't think so")
    voice._event_queue.put("/beat")
    voice._event_queue.put("something else")

    drained = voice.prune_stale_transcripts("well hm i don't think so")

    assert drained == 1
    assert voice._event_queue.get(timeout=0.1) == "/beat"
    assert voice._event_queue.get(timeout=0.1) == "something else"
    assert voice._event_queue.empty()


def test_voice_io_turn_start_cancels_before_first_audio():
    voice = VoiceIO(aec=True)
    voice._started = True
    voice._is_speaking = True
    voice._speaker = MagicMock()
    voice._speaker._audio_started.is_set.return_value = False
    voice.cancel_speech = MagicMock()

    voice._on_turn_start()

    voice.cancel_speech.assert_called_once()


def test_voice_io_recent_echo_is_ignored_just_after_playback():
    voice = VoiceIO(aec=True)
    voice._aec = MagicMock()
    voice.begin_assistant_turn("Let me think about that.")
    with patch("character_eng.voice.time.time", return_value=10.0):
        voice.finish_assistant_turn()
    voice._is_speaking = False

    with patch("character_eng.voice.time.time", return_value=10.5):
        voice._on_transcript("let me think about that")

    assert voice._event_queue.empty()


# --- check_voice_available ---


@patch.dict("os.environ", {"DEEPGRAM_API_KEY": "key1", "ELEVENLABS_API_KEY": "key2"})
def test_check_voice_available_all_present():
    """Should return True when all keys are present (elevenlabs backend)."""
    available, reason = check_voice_available()
    assert available is True
    assert reason == ""


@patch.dict("os.environ", {"DEEPGRAM_API_KEY": "", "ELEVENLABS_API_KEY": "key2"})
def test_check_voice_available_missing_deepgram_key():
    """Should return False when DEEPGRAM_API_KEY is not set."""
    available, reason = check_voice_available()
    assert available is False
    assert "DEEPGRAM_API_KEY" in reason


@patch.dict("os.environ", {"DEEPGRAM_API_KEY": "key1", "ELEVENLABS_API_KEY": ""})
def test_check_voice_available_missing_elevenlabs_key():
    """Should return False when ELEVENLABS_API_KEY is not set."""
    available, reason = check_voice_available()
    assert available is False
    assert "ELEVENLABS_API_KEY" in reason


@patch.dict("os.environ", {"DEEPGRAM_API_KEY": "key1"})
def test_check_voice_available_local_backend():
    """Should return True for local backend when torch/transformers are present."""
    with patch.dict("sys.modules", {
        "torch": MagicMock(),
        "transformers": MagicMock(),
    }):
        available, reason = check_voice_available(tts_backend="local")
        assert available is True
        assert reason == ""


@patch.dict("os.environ", {"DEEPGRAM_API_KEY": "key1"})
def test_check_voice_available_local_missing_torch():
    """Should return False for local backend when torch is missing."""
    with patch.dict("sys.modules", {
        "torch": None,
        "transformers": MagicMock(),
    }):
        available, reason = check_voice_available(tts_backend="local")
        assert available is False
        assert "torch" in reason


@patch.dict("os.environ", {"DEEPGRAM_API_KEY": "key1"})
def test_check_voice_available_qwen_backend():
    """Should return True for qwen backend (alias for local) when torch/transformers are present."""
    with patch.dict("sys.modules", {
        "torch": MagicMock(),
        "transformers": MagicMock(),
    }):
        available, reason = check_voice_available(tts_backend="qwen")
        assert available is True
        assert reason == ""


@patch.dict("os.environ", {"DEEPGRAM_API_KEY": "key1"})
def test_check_voice_available_pocket_backend():
    """Should return True for pocket backend (all deps are main dependencies)."""
    available, reason = check_voice_available(tts_backend="pocket")
    assert available is True
    assert reason == ""


# --- resolve_device ---


def test_resolve_device_none_returns_none():
    """None spec → None (use system default)."""
    assert resolve_device(None) is None


def test_resolve_device_int_passthrough():
    """Integer spec → returned as-is."""
    assert resolve_device(5, "input") == 5
    assert resolve_device(0, "output") == 0


def test_resolve_device_string_match():
    """String spec matches device by name substring."""
    fake_devices = [
        {"name": "Audioengine 2+", "max_input_channels": 0, "max_output_channels": 2},
        {"name": "reSpeaker XVF3800 4-Mic Array", "max_input_channels": 2, "max_output_channels": 2},
        {"name": "HyperX Quadcast", "max_input_channels": 2, "max_output_channels": 0},
    ]
    with patch("sounddevice.query_devices", return_value=fake_devices):
        assert resolve_device("reSpeaker", "input") == 1
        assert resolve_device("reSpeaker", "output") == 1
        assert resolve_device("audioengine", "output") == 0
        assert resolve_device("Quadcast", "input") == 2


def test_resolve_device_string_no_match():
    """String spec with no match raises RuntimeError."""
    fake_devices = [
        {"name": "Some Device", "max_input_channels": 2, "max_output_channels": 2},
    ]
    with patch("sounddevice.query_devices", return_value=fake_devices):
        import pytest
        with pytest.raises(RuntimeError, match="No output device matching"):
            resolve_device("nonexistent", "output")


def test_resolve_device_string_wrong_kind():
    """String spec matching a device without the right I/O kind raises RuntimeError."""
    fake_devices = [
        {"name": "Output Only", "max_input_channels": 0, "max_output_channels": 2},
    ]
    with patch("sounddevice.query_devices", return_value=fake_devices):
        import pytest
        with pytest.raises(RuntimeError, match="No input device matching"):
            resolve_device("Output Only", "input")
