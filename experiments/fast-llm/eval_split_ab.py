#!/usr/bin/env python3
"""Split Eval A/B: Test a focused "script check" prompt vs the original eval.

The original eval_call asks the LLM to do two things at once:
  1. Generate inner monologue (creative)
  2. Classify script status (advance/hold/off_book)

Experiment 1 showed 8B can't handle both — it generates monologue but
ignores the classification. This experiment tests whether splitting into
a focused classification call improves accuracy.

Three variants compared:
  A) Original eval prompt on 8B ("eval-8b")
  B) Focused script_check prompt on 8B ("check-8b")
  C) Original eval prompt on 70B ("eval-70b") — baseline

The focused prompt strips away monologue generation and just asks:
"Given the current beat's intent, did the conversation accomplish it?"

Usage:
    uv run python experiments/fast-llm/eval_split_ab.py
    uv run python experiments/fast-llm/eval_split_ab.py --open
"""

from __future__ import annotations

import json
import os
import sys
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from character_eng.models import MODELS
from character_eng.world import (
    eval_call, EvalResult, Script, Beat, Goals, WorldState,
    _llm_call, _load_prompt_file,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ── Focused script_check prompt ────────────────────────────────

SCRIPT_CHECK_SYSTEM = """You are a script tracker for an interactive fiction character.

Your ONLY job: decide whether the current beat's intent has been accomplished.

## Input

You'll receive:
- The current beat's INTENT (what the character is supposed to accomplish)
- The current beat's LINE (example dialogue)
- Recent conversation history

## Decision

Look at the conversation and decide:

- "advance" — The intent was achieved. The character said something that accomplishes it, OR the user's response made it irrelevant (e.g., beat says "offer water" but user already asked for water). If the intent is even PARTIALLY done or no longer needed, advance. Don't get stuck.
- "hold" — The intent has NOT been addressed yet.
- "off_book" — The user explicitly rejected, dismissed, or redirected away from this topic (e.g., "I don't care about that", "let's talk about something else"). The conversation has gone in a completely different direction. Fill in "plan_request" with what the new direction should cover.

**Bias toward "advance".** If in doubt between advance and hold, pick advance.

## Output

Return a JSON object:
- "status": "advance", "hold", or "off_book"
- "plan_request": Only when status is "off_book". Brief description of new direction. Empty string otherwise.
"""


@dataclass
class ScriptCheckResult:
    status: str  # "advance", "hold", "off_book"
    plan_request: str = ""


def script_check_call(
    beat_intent: str,
    beat_line: str,
    history: list[dict],
    model_config: dict,
    world: WorldState | None = None,
    recent_n: int = 6,
) -> ScriptCheckResult:
    """Focused script status classification. Minimal context, no monologue."""
    parts: list[str] = []

    parts.append(f"=== CURRENT BEAT ===")
    parts.append(f"Intent: {beat_intent}")
    parts.append(f"Example line: {beat_line}")

    if world and world.dynamic:
        parts.append(f"\n=== WORLD STATE ===")
        for text in world.dynamic.values():
            parts.append(f"- {text}")

    recent = history[-recent_n:] if len(history) > recent_n else history
    if recent:
        parts.append(f"\n=== RECENT CONVERSATION ===")
        for msg in recent:
            role = msg["role"].upper()
            content = msg["content"]
            # Truncate long messages
            if len(content) > 200:
                content = content[:200] + "..."
            parts.append(f"{role}: {content}")

    user_message = "\n".join(parts)

    response = _llm_call(
        model_config,
        label="script_check",
        messages=[
            {"role": "system", "content": SCRIPT_CHECK_SYSTEM},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )

    data = json.loads(response.choices[0].message.content)

    return ScriptCheckResult(
        status=data.get("status", "hold"),
        plan_request=data.get("plan_request", ""),
    )


# ── Test scenarios (same as eval_ab.py) ────────────────────────

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
    beat_index: int
    expected_status: str
    world_dynamic: dict | None = None
    stage_goal: str = ""


SCENARIOS = [
    # ── Clear advances ──
    Scenario(
        name="advance_direct_match",
        description="Beat: offer a flier. Character just offered it.",
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
        description="Beat: offer water. User already asked for it.",
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
        description="Beat: greet newcomer. World event: someone arrived, character greeted them.",
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
    Scenario(
        name="advance_yes_answer",
        description="Beat: ask if they want water. User said yes.",
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
        name="advance_no_answer",
        description="Beat: ask if they want water. User said no (intent still accomplished — question was asked).",
        history=[
            {"role": "assistant", "content": "You look thirsty! Want a cup of water? Just a dollar."},
            {"role": "user", "content": "No, I'm good thanks."},
        ],
        beats=[
            Beat(line="Want some water?", intent="Ask if they want water", condition=""),
            Beat(line="Check out these fliers then!", intent="Pivot to fliers", condition=""),
        ],
        beat_index=0,
        expected_status="advance",
    ),
    Scenario(
        name="advance_world_perception",
        description="Beat: greet newcomer. World event + greeting delivered.",
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

    # ── Clear holds ──
    Scenario(
        name="hold_unrelated_chitchat",
        description="Beat: offer a flier. Conversation is about weather. Flier not mentioned.",
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
        description="Beat: close conversation warmly. Still mid-topic, not wrapping up.",
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
    Scenario(
        name="hold_different_topic",
        description="Beat: pitch the stand. Character is answering a question about being a robot.",
        history=[
            {"role": "user", "content": "What's it like not having a body?"},
            {"role": "assistant", "content": "Honestly? Weird. I can see everything happening around me but I can't interact with any of it physically. It's like being at a party behind a window."},
        ],
        beats=[
            Beat(line="So the stand here, we sell water for a buck...", intent="Pitch the stand's offerings", condition=""),
            Beat(line="And the advice is free!", intent="Mention the free advice", condition=""),
        ],
        beat_index=0,
        expected_status="hold",
    ),

    # ── Clear off-book ──
    Scenario(
        name="off_book_topic_change",
        description="Beat: pitch the stand. User explicitly redirects to robots.",
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
        description="Beat: offer a flier. User explicitly rejects the topic.",
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
    Scenario(
        name="off_book_lets_talk_else",
        description="Beat: ask about their day. User says 'let's talk about something else'.",
        history=[
            {"role": "assistant", "content": "So what brings you out today? Anything exciting?"},
            {"role": "user", "content": "Honestly, let's talk about something else. I'm curious about you — are you sentient?"},
        ],
        beats=[
            Beat(line="So what brings you out?", intent="Ask about their day", condition=""),
            Beat(line="That's cool!", intent="React to their answer", condition=""),
        ],
        beat_index=0,
        expected_status="off_book",
    ),
]


# ── Runners ────────────────────────────────────────────────────


def run_original_eval(scenario: Scenario, model_key: str) -> tuple[str, float, str]:
    """Run original eval_call. Returns (status, latency_ms, thought)."""
    model_config = MODELS[model_key]
    world = WorldState(
        static=list(WORLD_STATIC),
        dynamic=dict(scenario.world_dynamic or WORLD_DYNAMIC_DEFAULT),
    )
    script = Script()
    if scenario.beats:
        script._beats = list(scenario.beats)
        script._index = scenario.beat_index
    goals = Goals(long_term="Experience the universe. Short-term: Hand out fliers and get people to stop by the stand.")

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
    return result.script_status, elapsed, result.thought


def run_script_check(scenario: Scenario, model_key: str) -> tuple[str, float, str]:
    """Run focused script_check_call. Returns (status, latency_ms, plan_request)."""
    model_config = MODELS[model_key]
    world = WorldState(
        static=list(WORLD_STATIC),
        dynamic=dict(scenario.world_dynamic or WORLD_DYNAMIC_DEFAULT),
    )

    beat = scenario.beats[scenario.beat_index]

    t0 = time.perf_counter()
    result = script_check_call(
        beat_intent=beat.intent,
        beat_line=beat.line,
        history=scenario.history,
        model_config=model_config,
        world=world,
    )
    elapsed = (time.perf_counter() - t0) * 1000
    return result.status, elapsed, result.plan_request


# ── HTML report ────────────────────────────────────────────────


def badge(status: str, expected: str) -> str:
    match = status == expected
    color = "#2d8a4e" if match else "#c0392b"
    return f'<span style="background:{color};color:white;padding:2px 8px;border-radius:4px;font-size:13px">{status}</span>'


def generate_report(results: list[dict]) -> str:
    n = len(results)
    c_eval8 = sum(1 for r in results if r["eval_8b"] == r["expected"])
    c_check8 = sum(1 for r in results if r["check_8b"] == r["expected"])
    c_eval70 = sum(1 for r in results if r["eval_70b"] == r["expected"])
    avg_eval8 = sum(r["lat_eval_8b"] for r in results) / n
    avg_check8 = sum(r["lat_check_8b"] for r in results) / n
    avg_eval70 = sum(r["lat_eval_70b"] for r in results) / n

    rows = []
    for r in results:
        rows.append(f"""
        <tr>
            <td><b>{r['name']}</b><br><small>{r['description']}</small></td>
            <td>{badge(r['expected'], r['expected'])}</td>
            <td>{badge(r['eval_8b'], r['expected'])}<br><small>{r['lat_eval_8b']:.0f}ms</small><br>
                <small style="color:#666">{r['detail_eval_8b'][:100]}</small></td>
            <td style="background:#fffde7">{badge(r['check_8b'], r['expected'])}<br><small>{r['lat_check_8b']:.0f}ms</small><br>
                <small style="color:#666">{r['detail_check_8b'][:100]}</small></td>
            <td>{badge(r['eval_70b'], r['expected'])}<br><small>{r['lat_eval_70b']:.0f}ms</small><br>
                <small style="color:#666">{r['detail_eval_70b'][:100]}</small></td>
        </tr>""")

    return f"""<!DOCTYPE html>
<html><head><title>Split Eval A/B: Focused Script Check</title>
<style>
body {{ font-family: system-ui; max-width: 1400px; margin: 20px auto; padding: 0 20px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; vertical-align: top; }}
th {{ background: #f5f5f5; }}
.stat {{ display: inline-block; padding: 8px 16px; margin: 4px; background: #f0f0f0; border-radius: 6px; }}
.stat b {{ font-size: 1.2em; }}
.winner {{ background: #e8f5e9; }}
</style></head>
<body>
<h1>Split Eval A/B: Focused Script Check</h1>
<p>Comparing three approaches to the script advance/hold/off_book decision:</p>
<ul>
<li><b>eval-8b:</b> Original eval prompt (monologue + classification) on Llama 8B</li>
<li><b>check-8b:</b> Focused script_check prompt (classification only) on Llama 8B <span style="background:#fffde7;padding:2px 6px">← new</span></li>
<li><b>eval-70b:</b> Original eval prompt on Llama 70B (baseline)</li>
</ul>

<div>
    <div class="stat">eval-8b correct: <b>{c_eval8}/{n}</b> ({100*c_eval8/n:.0f}%)</div>
    <div class="stat {'winner' if c_check8 >= c_eval70 else ''}">check-8b correct: <b>{c_check8}/{n}</b> ({100*c_check8/n:.0f}%)</div>
    <div class="stat">eval-70b correct: <b>{c_eval70}/{n}</b> ({100*c_eval70/n:.0f}%)</div>
</div>
<div>
    <div class="stat">eval-8b avg: <b>{avg_eval8:.0f}ms</b></div>
    <div class="stat">check-8b avg: <b>{avg_check8:.0f}ms</b></div>
    <div class="stat">eval-70b avg: <b>{avg_eval70:.0f}ms</b></div>
</div>

<h2>Scenarios ({n})</h2>
<table>
<tr>
    <th>Scenario</th>
    <th>Expected</th>
    <th>eval-8b<br><small>(original prompt)</small></th>
    <th style="background:#fffde7">check-8b<br><small>(focused prompt)</small></th>
    <th>eval-70b<br><small>(baseline)</small></th>
</tr>
{"".join(rows)}
</table>
</body></html>"""


# ── Main ───────────────────────────────────────────────────────


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Split Eval A/B")
    parser.add_argument("--open", action="store_true", help="Open HTML report in browser")
    args = parser.parse_args()

    print("Split Eval A/B: eval-8b vs check-8b vs eval-70b")
    print(f"Scenarios: {len(SCENARIOS)}\n")

    results = []
    for i, sc in enumerate(SCENARIOS):
        print(f"[{i+1}/{len(SCENARIOS)}] {sc.name}: {sc.description}")

        # A) Original eval on 8B
        try:
            status_a, lat_a, detail_a = run_original_eval(sc, "cerebras-llama")
        except Exception as e:
            status_a, lat_a, detail_a = "error", 0, str(e)

        # B) Focused script_check on 8B
        try:
            status_b, lat_b, detail_b = run_script_check(sc, "cerebras-llama")
        except Exception as e:
            status_b, lat_b, detail_b = "error", 0, str(e)

        # C) Original eval on 70B
        try:
            status_c, lat_c, detail_c = run_original_eval(sc, "groq-llama")
        except Exception as e:
            status_c, lat_c, detail_c = "error", 0, str(e)

        def mark(s, exp):
            return "ok" if s == exp else "WRONG"

        print(f"  eval-8b:  {status_a:10s} ({lat_a:6.0f}ms) [{mark(status_a, sc.expected_status)}]")
        print(f"  check-8b: {status_b:10s} ({lat_b:6.0f}ms) [{mark(status_b, sc.expected_status)}]")
        print(f"  eval-70b: {status_c:10s} ({lat_c:6.0f}ms) [{mark(status_c, sc.expected_status)}]")
        print()

        results.append({
            "name": sc.name,
            "description": sc.description,
            "expected": sc.expected_status,
            "eval_8b": status_a,
            "lat_eval_8b": lat_a,
            "detail_eval_8b": detail_a,
            "check_8b": status_b,
            "lat_check_8b": lat_b,
            "detail_check_8b": detail_b,
            "eval_70b": status_c,
            "lat_eval_70b": lat_c,
            "detail_eval_70b": detail_c,
        })

    # Summary
    n = len(results)
    c_eval8 = sum(1 for r in results if r["eval_8b"] == r["expected"])
    c_check8 = sum(1 for r in results if r["check_8b"] == r["expected"])
    c_eval70 = sum(1 for r in results if r["eval_70b"] == r["expected"])
    avg_eval8 = sum(r["lat_eval_8b"] for r in results) / n if n else 0
    avg_check8 = sum(r["lat_check_8b"] for r in results) / n if n else 0
    avg_eval70 = sum(r["lat_eval_70b"] for r in results) / n if n else 0

    print("=" * 60)
    print(f"eval-8b correct:   {c_eval8}/{n} ({100*c_eval8/n:.0f}%)")
    print(f"check-8b correct:  {c_check8}/{n} ({100*c_check8/n:.0f}%)")
    print(f"eval-70b correct:  {c_eval70}/{n} ({100*c_eval70/n:.0f}%)")
    print(f"Avg latency:  eval-8b={avg_eval8:.0f}ms  check-8b={avg_check8:.0f}ms  eval-70b={avg_eval70:.0f}ms")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "eval_split_ab.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON: {json_path}")

    html_path = RESULTS_DIR / "eval_split_ab.html"
    with open(html_path, "w") as f:
        f.write(generate_report(results))
    print(f"HTML: {html_path}")

    if args.open:
        webbrowser.open(f"file://{html_path}")


if __name__ == "__main__":
    main()
