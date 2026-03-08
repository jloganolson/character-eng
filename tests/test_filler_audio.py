"""Tests for the filler-audio helper."""

from unittest.mock import MagicMock, patch

from character_eng.filler_audio import FILLER_LIBRARY, FillerBank


def test_infer_sentiment_prefers_expression():
    sentiment = FillerBank.infer_sentiment(response_text="Okay?", expression="curious")
    assert sentiment == "curious"


def test_infer_sentiment_uses_text_when_expression_missing():
    assert FillerBank.infer_sentiment(response_text="Wait!") == "urgent"
    assert FillerBank.infer_sentiment(response_text="Do you want that?") == "curious"
    assert FillerBank.infer_sentiment(response_text="No, not really.") == "skeptical"


def test_pick_rotates_phrases_for_sentiment():
    bank = FillerBank("http://localhost:8003")
    phrases = [bank.pick("neutral").phrase for _ in range(4)]
    assert phrases[:3] == list(FILLER_LIBRARY["neutral"])
    assert phrases[3] == FILLER_LIBRARY["neutral"][0]


def test_clip_for_caches_synthesized_audio():
    fake_tts = MagicMock()
    fake_tts.wait_for_done.return_value = True

    def fake_ctor(*args, **kwargs):
        on_audio = kwargs["on_audio"]
        on_audio(b"\x01\x02")
        return fake_tts

    bank = FillerBank("http://localhost:8003")
    with patch("character_eng.filler_audio.PocketTTS", side_effect=fake_ctor) as ctor:
        clip_a = bank.clip_for("neutral")
        bank._indexes["neutral"] = 0
        clip_b = bank.clip_for("neutral")

    assert clip_a == b"\x01\x02"
    assert clip_b == b"\x01\x02"
    assert ctor.call_count == 1


def test_clip_for_uses_default_voice_when_none_configured():
    fake_tts = MagicMock()
    fake_tts.wait_for_done.return_value = True

    def fake_ctor(*args, **kwargs):
        on_audio = kwargs["on_audio"]
        on_audio(b"\x01\x02")
        assert kwargs["voice"] == "alba"
        return fake_tts

    bank = FillerBank("http://localhost:8003")
    with patch("character_eng.filler_audio.PocketTTS", side_effect=fake_ctor):
        assert bank.clip_for("neutral") == b"\x01\x02"


def test_clip_for_does_not_send_local_voice_path_as_voice_url():
    fake_tts = MagicMock()
    fake_tts.wait_for_done.return_value = True

    def fake_ctor(*args, **kwargs):
        on_audio = kwargs["on_audio"]
        on_audio(b"\x03\x04")
        assert kwargs["voice"] == ""
        return fake_tts

    bank = FillerBank("http://localhost:8003", voice="voices/greg.safetensors")
    with patch("character_eng.filler_audio.PocketTTS", side_effect=fake_ctor):
        assert bank.clip_for("neutral") == b"\x03\x04"


def test_clip_for_preserves_remote_voice_url():
    fake_tts = MagicMock()
    fake_tts.wait_for_done.return_value = True

    def fake_ctor(*args, **kwargs):
        on_audio = kwargs["on_audio"]
        on_audio(b"\x05\x06")
        assert kwargs["voice"] == "https://example.com/greg.safetensors"
        return fake_tts

    bank = FillerBank("http://localhost:8003", voice="https://example.com/greg.safetensors")
    with patch("character_eng.filler_audio.PocketTTS", side_effect=fake_ctor):
        assert bank.clip_for("neutral") == b"\x05\x06"
