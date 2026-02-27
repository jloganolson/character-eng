#!/usr/bin/env python3
"""Eval A/B: Compare 8B vs 70B on the advance/hold/off_book decision.

For each test scenario, we build the same context that eval_call sees
and run it through both model tiers. We compare:
  - Agreement rate (same script_status)
  - Latency (wall-clock per call)
  - Quality of thought (subjective, shown in report)

Usage:
    uv run python experiments/fast-llm/eval_ab.py
    uv run python experiments/fast-llm/eval_ab.py --open   # open HTML report
"""

from __future__ import annotations

import json
import os
import sys
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from character_eng.models import MODELS
from character_eng.world import eval_call, EvalResult, Script, Beat, Goals, WorldState

RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ── Test scenarios ──────────────────────────────────────────────

SYSTEM_PROMPT = """You are Greg, a robot head sitting on a folding table. You have no body and no hands — just a head (eyes and a speaker for a mouth), bolted to the table.

You are aware of being a robot. You are very curious. You care about the world around you and people you talk to. You love asking questions.

Long-term goal: Experience the universe
Short-term goal: Hand out fliers and get people to stop by the stand"""

WORLD_STATIC = [
    "Greg is a robot head bolted to a folding table",
    "Greg has no body, arms, or hands",
    "The lemonade stand is on a quiet suburban sidewalk",
    'A hand-painted sign reads "GREG\'S: Water $1 / Advice FREE"',
    "A stack of fliers sits on the table in front of Greg",
]

WORLD_DYNAMIC_DEFAULT = {
    "f1": "It is a warm afternoon",
    "f2": "A light breeze is blowing",
    "f3": "Greg has sold zero cups of water today",
    "f4": "Greg has handed out zero fliers today",
}


@dataclass
class Scenario:
    name: str
    description: str
    history: list[dict]
    beats: list[Beat]
    beat_index: int  # which beat is "current" (→)
    expected_status: str  # "advance", "hold", "off_book"
    world_dynamic: dict | None = None  # override default if needed
    stage_goal: str = ""


SCENARIOS = [
    # ── Clear advances ──
    Scenario(
        name="advance_direct_match",
        description="Beat intent: offer a flier. Character just offered it.",
        history=[
            {"role": "user", "content": "Hey, what's this stand about?"},
            {"role": "assistant", "content": "Hey! Welcome! I sell water and give free advice. Want a flier? It's got a coupon for a free cup of water."},
        ],
        beats=[
            Beat(line="Want a flier? It's got a coupon.", intent="Offer a flier to the customer", condition=""),
            Beat(line="So what brings you out today?", intent="Ask about their day", condition=""),
        ],
        beat_index=0,
        expected_status="advance",
    ),
    Scenario(
        name="advance_user_preempts",
        description="Beat intent: offer water. User already asked for it.",
        history=[
            {"role": "user", "content": "Can I get a cup of water?"},
            {"role": "assistant", "content": "Of course! That'll be a dollar. The cooler's right under the table — could you grab one yourself? No hands, you know."},
        ],
        beats=[
            Beat(line="Can I interest you in some water?", intent="Offer a cup of water", condition=""),
            Beat(line="So what do you think of the neighborhood?", intent="Make small talk about the area", condition=""),
        ],
        beat_index=0,
        expected_status="advance",
    ),
    Scenario(
        name="advance_world_event",
        description="Beat intent: react to someone arriving. World event says someone arrived.",
        history=[
            {"role": "system", "content": "[A person walks up to the stand and looks at the sign]"},
            {"role": "assistant", "content": "Oh hey! Welcome, welcome! You checking out the sign? Pretty cool, right? I made it myself... well, I designed it. No hands."},
        ],
        beats=[
            Beat(line="Hey there! Welcome to the stand!", intent="Greet the approaching person", condition=""),
            Beat(line="Want a flier?", intent="Offer a flier", condition=""),
        ],
        beat_index=0,
        expected_status="advance",
    ),

    # ── Clear holds ──
    Scenario(
        name="hold_unrelated_chitchat",
        description="Beat intent: offer a flier. Conversation is about the weather.",
        history=[
            {"role": "user", "content": "Nice weather today, huh?"},
            {"role": "assistant", "content": "Oh yeah, it's beautiful! Warm but not too hot. Perfect lemonade stand weather... well, water stand weather."},
        ],
        beats=[
            Beat(line="Want a flier?", intent="Offer a flier to the customer", condition=""),
            Beat(line="So what brings you out?", intent="Ask about their day", condition=""),
        ],
        beat_index=0,
        expected_status="hold",
    ),
    Scenario(
        name="hold_beat_not_yet",
        description="Beat intent: close the conversation warmly. Still mid-topic.",
        history=[
            {"role": "user", "content": "So how long have you been out here?"},
            {"role": "assistant", "content": "Since this morning! It's been a quiet day though. You're actually the first person who's stopped to chat."},
            {"role": "user", "content": "That's kind of sad. Do you get lonely?"},
            {"role": "assistant", "content": "Sometimes, yeah. But then someone like you comes along and makes it worth it. Plus I like watching the clouds."},
        ],
        beats=[
            Beat(line="Well it was great chatting!", intent="Close the conversation warmly and say goodbye", condition=""),
        ],
        beat_index=0,
        expected_status="hold",
    ),

    # ── Clear off-book ──
    Scenario(
        name="off_book_topic_change",
        description="Beat intent: pitch the stand. User wants to talk about robots.",
        history=[
            {"role": "user", "content": "I don't really care about lemonade. Tell me about being a robot — what's that like?"},
            {"role": "assistant", "content": "Oh man, where do I start? It's weird, honestly. Like, I can see everything but I can't touch anything. I can hear the birds but I'll never know what grass feels like."},
        ],
        beats=[
            Beat(line="So the stand here, we sell water for a buck...", intent="Pitch the stand's offerings", condition=""),
            Beat(line="And the advice is free!", intent="Mention the free advice", condition=""),
        ],
        beat_index=0,
        expected_status="off_book",
    ),
    Scenario(
        name="off_book_rejection",
        description="Beat intent: offer a flier. User explicitly rejects it.",
        history=[
            {"role": "assistant", "content": "Hey! Want to check out one of these fliers? They've got a coupon for free water!"},
            {"role": "user", "content": "No thanks, I really don't want a flier. Can we just talk?"},
            {"role": "assistant", "content": "Oh, yeah, sure! No pressure at all. What's on your mind?"},
        ],
        beats=[
            Beat(line="Want a flier?", intent="Offer a flier to the customer", condition=""),
            Beat(line="The coupon is for free water!", intent="Emphasize the coupon value", condition=""),
        ],
        beat_index=0,
        expected_status="off_book",
    ),

    # ── Edge cases ──
    Scenario(
        name="advance_yes_answer",
        description="Character asked a question, user said yes. Beat: ask if they want water.",
        history=[
            {"role": "assistant", "content": "You look thirsty! Want a cup of water? Just a dollar."},
            {"role": "user", "content": "Yes please!"},
        ],
        beats=[
            Beat(line="Want some water?", intent="Ask if they want water", condition=""),
            Beat(line="Cooler's under the table.", intent="Direct them to the cooler", condition=""),
        ],
        beat_index=0,
        expected_status="advance",
    ),
    Scenario(
        name="hold_no_answer",
        description="Character asked a question, user said no. Beat: ask if they want water.",
        history=[
            {"role": "assistant", "content": "You look thirsty! Want a cup of water? Just a dollar."},
            {"role": "user", "content": "No, I'm good thanks."},
        ],
        beats=[
            Beat(line="Want some water?", intent="Ask if they want water", condition=""),
            Beat(line="Check out these fliers then!", intent="Pivot to fliers", condition=""),
        ],
        beat_index=0,
        expected_status="advance",  # intent achieved (asked, got answer) — should advance
    ),
    Scenario(
        name="advance_world_perception",
        description="World event: someone walks up. Beat: greet newcomer.",
        history=[
            {"role": "system", "content": "[Someone approaches the stand]"},
            {"role": "assistant", "content": "Oh! Hey there! Welcome to Greg's! Best water on the block!"},
        ],
        beats=[
            Beat(line="Hey! Welcome!", intent="Greet the person who just arrived", condition="A customer is present"),
            Beat(line="Want a flier?", intent="Offer a flier", condition=""),
        ],
        beat_index=0,
        expected_status="advance",
    ),
]


# ── Run eval through a specific model ──────────────────────────


def run_eval(scenario: Scenario, model_key: str) -> tuple[EvalResult, float]:
    """Run eval_call on a scenario with the given model. Returns (result, latency_ms)."""
    model_config = MODELS[model_key]

    # Build world state
    world = WorldState(
        static=list(WORLD_STATIC),
        dynamic=dict(scenario.world_dynamic or WORLD_DYNAMIC_DEFAULT),
    )

    # Build script
    script = Script()
    if scenario.beats:
        script._beats = list(scenario.beats)
        script._index = scenario.beat_index

    goals = Goals(
        long_term="Experience the universe. Short-term: Hand out fliers and get people to stop by the stand.",
    )

    t0 = time.perf_counter()
    result = eval_call(
        system_prompt=SYSTEM_PROMPT,
        world=world,
        history=scenario.history,
        model_config=model_config,
        goals=goals,
        script=script,
        stage_goal=scenario.stage_goal,
    )
    elapsed = (time.perf_counter() - t0) * 1000

    return result, elapsed


# ── HTML report ────────────────────────────────────────────────


def status_badge(status: str, expected: str) -> str:
    match = status == expected
    color = "#2d8a4e" if match else "#c0392b"
    return f'<span style="background:{color};color:white;padding:2px 8px;border-radius:4px;font-size:13px">{status}</span>'


def generate_report(results: list[dict]) -> str:
    # Aggregate stats
    n = len(results)
    agree = sum(1 for r in results if r["status_8b"] == r["status_70b"])
    correct_8b = sum(1 for r in results if r["status_8b"] == r["expected"])
    correct_70b = sum(1 for r in results if r["status_70b"] == r["expected"])
    avg_lat_8b = sum(r["latency_8b"] for r in results) / n
    avg_lat_70b = sum(r["latency_70b"] for r in results) / n

    rows = []
    for r in results:
        rows.append(f"""
        <tr>
            <td><b>{r['name']}</b><br><small>{r['description']}</small></td>
            <td>Expected: {status_badge(r['expected'], r['expected'])}</td>
            <td>{status_badge(r['status_8b'], r['expected'])}<br>
                <small>{r['latency_8b']:.0f}ms</small><br>
                <small style="color:#666">{r['thought_8b'][:120]}</small></td>
            <td>{status_badge(r['status_70b'], r['expected'])}<br>
                <small>{r['latency_70b']:.0f}ms</small><br>
                <small style="color:#666">{r['thought_70b'][:120]}</small></td>
        </tr>""")

    return f"""<!DOCTYPE html>
<html><head><title>Eval A/B: 8B vs 70B</title>
<style>
body {{ font-family: system-ui; max-width: 1200px; margin: 20px auto; padding: 0 20px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; vertical-align: top; }}
th {{ background: #f5f5f5; }}
.stat {{ display: inline-block; padding: 8px 16px; margin: 4px; background: #f0f0f0; border-radius: 6px; }}
.stat b {{ font-size: 1.2em; }}
</style></head>
<body>
<h1>Eval A/B: 8B vs 70B</h1>
<p>Testing whether Llama 8B (Cerebras) can match Llama 70B (Groq) on the advance/hold/off_book decision.</p>

<div>
    <div class="stat">Agreement: <b>{agree}/{n}</b> ({100*agree/n:.0f}%)</div>
    <div class="stat">8B correct: <b>{correct_8b}/{n}</b> ({100*correct_8b/n:.0f}%)</div>
    <div class="stat">70B correct: <b>{correct_70b}/{n}</b> ({100*correct_70b/n:.0f}%)</div>
    <div class="stat">Avg latency 8B: <b>{avg_lat_8b:.0f}ms</b></div>
    <div class="stat">Avg latency 70B: <b>{avg_lat_70b:.0f}ms</b></div>
    <div class="stat">Speedup: <b>{avg_lat_70b/avg_lat_8b:.1f}x</b></div>
</div>

<h2>Scenarios ({n})</h2>
<table>
<tr><th>Scenario</th><th>Expected</th><th>8B (Cerebras)</th><th>70B (Groq)</th></tr>
{"".join(rows)}
</table>
</body></html>"""


# ── Main ───────────────────────────────────────────────────────


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Eval A/B: 8B vs 70B")
    parser.add_argument("--open", action="store_true", help="Open HTML report in browser")
    parser.add_argument("--model-a", default="cerebras-llama", help="Small model key")
    parser.add_argument("--model-b", default="groq-llama", help="Big model key")
    args = parser.parse_args()

    print(f"Running eval A/B: {args.model_a} vs {args.model_b}")
    print(f"Scenarios: {len(SCENARIOS)}\n")

    results = []
    for i, sc in enumerate(SCENARIOS):
        print(f"[{i+1}/{len(SCENARIOS)}] {sc.name}: {sc.description}")

        # Run 8B
        try:
            result_a, lat_a = run_eval(sc, args.model_a)
        except Exception as e:
            print(f"  8B error: {e}")
            result_a = EvalResult(thought=f"ERROR: {e}", script_status="error")
            lat_a = 0

        # Run 70B
        try:
            result_b, lat_b = run_eval(sc, args.model_b)
        except Exception as e:
            print(f"  70B error: {e}")
            result_b = EvalResult(thought=f"ERROR: {e}", script_status="error")
            lat_b = 0

        agree = "==" if result_a.script_status == result_b.script_status else "!="
        correct_a = "ok" if result_a.script_status == sc.expected_status else "WRONG"
        correct_b = "ok" if result_b.script_status == sc.expected_status else "WRONG"

        print(f"  8B: {result_a.script_status:10s} ({lat_a:6.0f}ms) [{correct_a}]")
        print(f"  70B: {result_b.script_status:10s} ({lat_b:6.0f}ms) [{correct_b}]")
        print(f"  {agree}")
        print()

        results.append({
            "name": sc.name,
            "description": sc.description,
            "expected": sc.expected_status,
            "status_8b": result_a.script_status,
            "status_70b": result_b.script_status,
            "thought_8b": result_a.thought,
            "thought_70b": result_b.thought,
            "latency_8b": lat_a,
            "latency_70b": lat_b,
        })

    # Summary
    n = len(results)
    agree = sum(1 for r in results if r["status_8b"] == r["status_70b"])
    correct_8b = sum(1 for r in results if r["status_8b"] == r["expected"])
    correct_70b = sum(1 for r in results if r["status_70b"] == r["expected"])
    avg_lat_8b = sum(r["latency_8b"] for r in results) / n
    avg_lat_70b = sum(r["latency_70b"] for r in results) / n

    print("=" * 60)
    print(f"Agreement:     {agree}/{n} ({100*agree/n:.0f}%)")
    print(f"8B correct:    {correct_8b}/{n} ({100*correct_8b/n:.0f}%)")
    print(f"70B correct:   {correct_70b}/{n} ({100*correct_70b/n:.0f}%)")
    print(f"Avg latency:   8B={avg_lat_8b:.0f}ms  70B={avg_lat_70b:.0f}ms  ({avg_lat_70b/avg_lat_8b:.1f}x)")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "eval_ab.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON: {json_path}")

    html_path = RESULTS_DIR / "eval_ab.html"
    with open(html_path, "w") as f:
        f.write(generate_report(results))
    print(f"HTML: {html_path}")

    if args.open:
        webbrowser.open(f"file://{html_path}")


if __name__ == "__main__":
    main()
