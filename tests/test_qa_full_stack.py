"""Focused tests for the full-stack QA helpers and report output."""

from pathlib import Path

from character_eng.gemini_media import SceneEvaluation
from character_eng.qa_full_stack import (
    LIVE_SCENE_FALLBACK,
    SCENE_SPECS,
    TurnRecord,
    _ground_evaluation,
    _ground_live_evaluation,
    _scene_event_for_eval,
    _write_report,
)


def test_ground_evaluation_rejects_off_scene_departure_summary_and_user_line():
    spec = SCENE_SPECS[2]
    summary, facts, goal, user_line = _ground_evaluation(
        spec,
        "On a quiet street, a disembodied robot head sells water and advice.",
        [
            "The setting is a tree-lined sidewalk in front of brick apartment buildings.",
            "The vendor is a stationary robot head, not a human.",
            "The stand offers water and advice instead of lemonade.",
        ],
        "A cinematic shot of a wooden stand where Greg sells water and advice.",
        "Do you have enough processing power to fix my life for fifty cents?",
    )

    assert summary == spec["fallback_summary"]
    assert facts == spec["fallback_facts"]
    assert goal == spec["fallback_goal"]
    assert user_line == spec["fallback_user_line"]


def test_ground_evaluation_accepts_grounded_linger_scene():
    spec = SCENE_SPECS[1]
    summary, facts, goal, user_line = _ground_evaluation(
        spec,
        "The hooded visitor stands by Greg's table, reading a flyer and holding a paper cup.",
        [
            "Greg's stand still has a sign, cups, water, and flyers.",
            "The hooded visitor is reading a flyer.",
            "The visitor is holding a paper cup near the stand.",
        ],
        "Keep the exchange on the flyer and the advice stand.",
        "What's on that flyer?",
    )

    assert summary.startswith("The hooded visitor")
    assert facts[0].startswith("Greg's stand")
    assert goal == "Keep the exchange on the flyer and the advice stand."
    assert user_line == "What's on that flyer?"


def test_scene_event_falls_back_when_required_terms_are_missing():
    spec = SCENE_SPECS[2]
    event_text = _scene_event_for_eval(
        spec,
        "A robot head sits at a stand on a quiet residential street.",
    )
    assert event_text == spec["fallback_event"]


def test_write_report_includes_dashboard_trace(tmp_path: Path):
    json_path = tmp_path / "report.json"
    html_path = tmp_path / "report.html"
    turn = TurnRecord(
        scene_name="approach",
        image_prompt="test prompt",
        visual_summary="A hooded passerby slows down at Greg's stand.",
        scripted_user_line="What's the pitch?",
        stt_transcript="What's the pitch?",
        assistant_response="Free water. Free advice. Interested?",
        assistant_audio_ms=820,
        response_ttft_ms=210,
        response_total_ms=980,
        trace_events=[
            {
                "type": "response_done",
                "timestamp": 0.0,
                "seq": 1,
                "data": {"ttft_ms": 210, "total_ms": 980},
            }
        ],
    )

    _write_report(
        [turn],
        json_path,
        html_path,
        session_events=[
            {
                "type": "vision_poll",
                "timestamp": 0.0,
                "seq": 1,
                "data": {"faces": 0, "persons": 1, "objects": 0},
            },
            {
                "type": "response_done",
                "timestamp": 0.5,
                "seq": 2,
                "data": {"ttft_ms": 210, "total_ms": 980},
            },
        ],
    )

    body = html_path.read_text()
    assert "Thread Lanes" in body
    assert "Interwoven Streams" in body
    assert "Raw Chronology" in body
    assert "Time Zoom" in body
    assert "Row Scale" in body
    assert "Reset View" in body
    assert "+0.000s" in body
    assert "vision_poll" in body


def test_ground_live_evaluation_falls_back_for_stand_drift():
    plan = _ground_live_evaluation(
        SceneEvaluation(
            summary="Greg is visible at a lemonade stand on the sidewalk.",
            world_facts=[
                "Greg is a robot head on a folding table",
                "There are flyers at the stand",
            ],
            user_line="How much is the lemonade?",
            followup_line="Can I get a coupon?",
            visual_goal="Sell more flyers by the sidewalk stand",
        )
    )

    assert plan.summary == LIVE_SCENE_FALLBACK["summary"]
    assert plan.world_facts[0] == LIVE_SCENE_FALLBACK["world_facts"][0]
    assert plan.user_lines[0] == LIVE_SCENE_FALLBACK["user_line"]
    assert plan.user_lines[1] == LIVE_SCENE_FALLBACK["followup_line"]
