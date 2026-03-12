"""Tests for local stack benchmark report helpers."""

from character_eng.local_stack_benchmark import (
    build_comparison_rows,
    percentile,
    summarize_results,
)


def test_percentile_interpolates():
    assert percentile([10, 20, 30, 40], 95) == 38.5


def test_summarize_results_groups_by_scenario_and_metric():
    summary = summarize_results([
        {"scenario": "vision_query", "metrics": {"latency_ms": 100, "answer_chars": 40}},
        {"scenario": "vision_query", "metrics": {"latency_ms": 200, "answer_chars": 60}},
        {"scenario": "pocket_tts", "metrics": {"synth_ms": 80, "first_audio_ms": 25}},
    ])

    assert summary["vision_query"]["latency_ms"]["count"] == 2
    assert summary["vision_query"]["latency_ms"]["mean"] == 150.0
    assert summary["vision_query"]["answer_chars"]["median"] == 50.0
    assert summary["pocket_tts"]["synth_ms"]["max"] == 80.0


def test_build_comparison_rows_uses_mean_values():
    left = {
        "vision_query": {
            "latency_ms": {"mean": 100.0},
            "answer_chars": {"mean": 40.0},
        }
    }
    right = {
        "vision_query": {
            "latency_ms": {"mean": 125.0},
            "answer_chars": {"mean": 50.0},
        }
    }

    rows = build_comparison_rows("fast", left, "slow", right)

    assert rows == [
        {
            "scenario": "vision_query",
            "metric": "answer_chars",
            "fast": 40.0,
            "slow": 50.0,
            "delta": 10.0,
            "delta_pct": 25.0,
        },
        {
            "scenario": "vision_query",
            "metric": "latency_ms",
            "fast": 100.0,
            "slow": 125.0,
            "delta": 25.0,
            "delta_pct": 25.0,
        },
    ]
