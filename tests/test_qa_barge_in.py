import json
import wave
from pathlib import Path

from character_eng.qa_barge_in import evaluate_barge_in_snippet, iter_barge_in_snippets


def _write_wav(path: Path, sample_rate: int = 16000, frames: int = 1600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * frames)


def test_iter_barge_in_snippets_loads_manifest_and_tags(tmp_path):
    snippet_dir = tmp_path / "moments" / "barge-snippet"
    _write_wav(snippet_dir / "media" / "audio" / "user_input_clip.wav")
    _write_wav(snippet_dir / "media" / "audio" / "assistant_output_clip.wav", sample_rate=24000)
    (snippet_dir / "manifest.json").write_text(json.dumps({
        "type": "snippet",
        "title": "false barge in",
        "session_id": "sess-barge",
        "capture_mode": "barge_in_turn",
        "tags": ["barge-in", "false-barge-in"],
        "bundle": {
            "barge_in": {
                "transcript_text": "oh hey man",
                "reason": "barge_in",
                "speech_started_s": 0.2,
                "transcript_final_s": 0.5,
                "assistant_first_audio_s": 0.1,
            },
        },
    }), encoding="utf-8")

    snippets = list(iter_barge_in_snippets(tmp_path, required_tags={"barge-in"}))
    assert len(snippets) == 1
    assert snippets[0].title == "false barge in"
    assert snippets[0].expected_cancel is False
    assert snippets[0].assistant_audio_path is not None


def test_evaluate_barge_in_snippet_differs_for_aec_vs_legacy(tmp_path):
    snippet_dir = tmp_path / "moments" / "barge-snippet"
    _write_wav(snippet_dir / "media" / "audio" / "user_input_clip.wav")
    _write_wav(snippet_dir / "media" / "audio" / "assistant_output_clip.wav", sample_rate=24000)
    (snippet_dir / "manifest.json").write_text(json.dumps({
        "type": "snippet",
        "title": "speech started during playback",
        "session_id": "sess-barge",
        "capture_mode": "barge_in_turn",
        "tags": ["barge-in", "false-barge-in"],
        "bundle": {
            "barge_in": {
                "transcript_text": "",
                "reason": "barge_in",
                "speech_started_s": 0.2,
                "transcript_final_s": None,
                "assistant_first_audio_s": 0.1,
            },
        },
    }), encoding="utf-8")

    snippet = next(iter_barge_in_snippets(tmp_path))
    legacy = evaluate_barge_in_snippet(snippet, aec=False)
    aec = evaluate_barge_in_snippet(snippet, aec=True)

    assert legacy.cancelled is True
    assert aec.cancelled is False
