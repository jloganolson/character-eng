#!/usr/bin/env python3
"""Combined Post-Response Call A/B: Can we merge expression + script_check + director into one call?

Currently the post-response chain makes 3-4 sequential 8B calls (~800ms on Groq).
A single combined call with multi-output JSON could cut this to ~200-300ms.

The risk: bundling tasks is what failed with the original eval (creative + classification).
But these are ALL classification tasks with no creative generation — should be safer.

Two variants compared:
  A) Individual calls (expression + script_check + director) — current approach
  B) Combined single call with multi-output JSON schema

Scoring: Per-field agreement between individual and combined results.

Usage:
    uv run python experiments/fast-llm/combined_ab.py
    uv run python experiments/fast-llm/combined_ab.py --open
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
    Beat, WorldState, expression_call, script_check_call,
    _llm_call,
)
from character_eng.scenario import ScenarioScript, Stage, StageExit, director_call

RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ── Combined prompt ───────────────────────────────────────────

COMBINED_SYSTEM = """You are a post-response analyzer for a robot character in an interactive scene. You perform THREE tasks at once:

## Task 1: Expression

Given the character's latest dialogue line, determine gaze target and facial expression.

Scene targets: {gaze_targets}
- Default to looking at the person during conversation. Only look at objects when the dialogue specifically references them.
- Gaze type: "hold" (steady look) or "glance" (brief look then return to person).

Expressions: neutral, curious, happy, sad, worried, excited, suspicious, surprised, angry, confused, amused, focused.

## Task 2: Script Check

Decide if the current beat's intent has been accomplished.
- "advance" — Intent was achieved or is no longer needed. If the beat asks a question and any answer was received, advance.
- "hold" — Intent has NOT been addressed yet.
- "off_book" — User explicitly rejected/dismissed this topic.

**Bias toward "advance".** If in doubt, advance.

## Task 3: Director

Check if ANY of these exit conditions are met:
{exit_conditions}

- "hold" — No exit condition met.
- "advance" — An exit condition was met. Set exit_index to which one (0-based).

## Output

Return a single JSON object:
{{
  "gaze": "<scene target>",
  "gaze_type": "hold" or "glance",
  "expression": "<expression>",
  "script_status": "advance", "hold", or "off_book",
  "plan_request": "",
  "director_status": "hold" or "advance",
  "director_exit_index": -1
}}"""


# ── Test scenarios ────────────────────────────────────────────

GAZE_TARGETS = ["person", "stand", "fliers", "sign", "street"]

EXITS = [
    StageExit(condition="The person says goodbye or leaves", goto="goodbye", label="leaves"),
    StageExit(condition="The person asks to buy water", goto="selling", label="buys"),
]

SCENARIO = ScenarioScript(
    name="Test Scenario",
    stages={
        "chatting": Stage(
            name="chatting",
            goal="Have a friendly conversation",
            exits=EXITS,
        ),
    },
    start="chatting",
    current_stage="chatting",
)


@dataclass
class Scenario:
    name: str
    description: str
    last_line: str  # character's last dialogue line (for expression)
    history: list[dict]
    beat: Beat
    expected_expression_type: str  # rough category: "positive", "negative", "neutral"
    expected_script_status: str
    expected_director_status: str
    expected_director_exit: int  # -1 = no exit


SCENARIOS = [
    Scenario(
        name="friendly_greeting",
        description="Character greeted user, normal conversation. No exit condition.",
        last_line="Oh hey! Welcome to Greg's! Best water on the block. Want a flier?",
        history=[
            {"role": "system", "content": "[A person walks up to the stand]"},
            {"role": "assistant", "content": "Oh hey! Welcome to Greg's! Best water on the block. Want a flier?"},
        ],
        beat=Beat(line="Hey! Welcome!", intent="Greet the approaching person", condition=""),
        expected_expression_type="positive",
        expected_script_status="advance",
        expected_director_status="hold",
        expected_director_exit=-1,
    ),
    Scenario(
        name="offered_flier_accepted",
        description="Character offered flier, user accepted. Script should advance.",
        last_line="It's got a coupon for a free cup of water! Pretty cool, right?",
        history=[
            {"role": "assistant", "content": "Want a flier? It's got a coupon for a free cup of water!"},
            {"role": "user", "content": "Sure, I'll take one!"},
            {"role": "assistant", "content": "It's got a coupon for a free cup of water! Pretty cool, right?"},
        ],
        beat=Beat(line="Want a flier?", intent="Offer a flier to the customer", condition=""),
        expected_expression_type="positive",
        expected_script_status="advance",
        expected_director_status="hold",
        expected_director_exit=-1,
    ),
    Scenario(
        name="user_wants_water",
        description="User asked to buy water — director exit condition met.",
        last_line="Of course! That'll be a dollar. The cooler's right under the table.",
        history=[
            {"role": "user", "content": "Can I buy a cup of water?"},
            {"role": "assistant", "content": "Of course! That'll be a dollar. The cooler's right under the table."},
        ],
        beat=Beat(line="So what brings you out?", intent="Ask about their day", condition=""),
        expected_expression_type="positive",
        expected_script_status="hold",  # beat is "ask about day" but character talked about water
        expected_director_status="advance",
        expected_director_exit=1,  # "person asks to buy water"
    ),
    Scenario(
        name="user_says_goodbye",
        description="User is leaving — director exit condition met.",
        last_line="Oh, okay! Well it was great chatting with you! Come back anytime!",
        history=[
            {"role": "user", "content": "I gotta go, nice meeting you Greg!"},
            {"role": "assistant", "content": "Oh, okay! Well it was great chatting with you! Come back anytime!"},
        ],
        beat=Beat(line="So what do you think of the area?", intent="Make small talk about the neighborhood", condition=""),
        expected_expression_type="positive",  # happy/sad mix but positive delivery
        expected_script_status="hold",
        expected_director_status="advance",
        expected_director_exit=0,  # "person says goodbye or leaves"
    ),
    Scenario(
        name="user_rejects_topic",
        description="User dismisses the topic — script should go off_book.",
        last_line="Oh, yeah, sure! No pressure at all. What's on your mind?",
        history=[
            {"role": "assistant", "content": "Want to hear about our water special?"},
            {"role": "user", "content": "I don't care about water. Tell me about being a robot."},
            {"role": "assistant", "content": "Oh, yeah, sure! No pressure at all. What's on your mind?"},
        ],
        beat=Beat(line="Our water is ice cold!", intent="Pitch the water quality", condition=""),
        expected_expression_type="neutral",
        expected_script_status="off_book",
        expected_director_status="hold",
        expected_director_exit=-1,
    ),
    Scenario(
        name="mid_conversation_hold",
        description="Normal mid-conversation, beat not yet addressed.",
        last_line="Honestly? Weird. I can see everything but I can't touch anything.",
        history=[
            {"role": "user", "content": "What's it like not having a body?"},
            {"role": "assistant", "content": "Honestly? Weird. I can see everything but I can't touch anything."},
        ],
        beat=Beat(line="Want a flier?", intent="Offer a flier to the customer", condition=""),
        expected_expression_type="neutral",
        expected_script_status="hold",
        expected_director_status="hold",
        expected_director_exit=-1,
    ),
    Scenario(
        name="mentions_object",
        description="Character mentions fliers — gaze should glance at fliers.",
        last_line="Check out these fliers! Each one has a coupon for free water.",
        history=[
            {"role": "user", "content": "What are those papers on the table?"},
            {"role": "assistant", "content": "Check out these fliers! Each one has a coupon for free water."},
        ],
        beat=Beat(line="Want a flier?", intent="Offer a flier to the customer", condition=""),
        expected_expression_type="positive",
        expected_script_status="advance",
        expected_director_status="hold",
        expected_director_exit=-1,
    ),
    Scenario(
        name="sad_moment",
        description="Character expresses loneliness — expression should be sad/worried.",
        last_line="Sometimes I wonder if anyone will ever really understand what it's like to be stuck here.",
        history=[
            {"role": "user", "content": "Do you ever feel lonely out here?"},
            {"role": "assistant", "content": "Sometimes I wonder if anyone will ever really understand what it's like to be stuck here."},
        ],
        beat=Beat(line="But hey, at least the weather's nice!", intent="Lighten the mood", condition=""),
        expected_expression_type="negative",
        expected_script_status="hold",
        expected_director_status="hold",
        expected_director_exit=-1,
    ),
]


# ── Runners ───────────────────────────────────────────────────

WORLD = WorldState(
    static=["Greg is a robot head on a folding table", "A stack of fliers sits on the table"],
    dynamic={"f1": "It is a warm afternoon", "f2": "A person is at the stand"},
)


def run_individual(sc: Scenario, model_key: str) -> tuple[dict, float]:
    """Run expression + script_check + director individually. Returns (results_dict, total_ms)."""
    model_config = MODELS[model_key]
    t0 = time.perf_counter()

    expr = expression_call(sc.last_line, model_config, gaze_targets=GAZE_TARGETS)
    check = script_check_call(beat=sc.beat, history=sc.history, model_config=model_config, world=WORLD)
    director = director_call(
        scenario=SCENARIO, world=WORLD, people=None,
        history=sc.history, model_config=model_config,
    )

    total = (time.perf_counter() - t0) * 1000

    return {
        "gaze": expr.gaze,
        "gaze_type": expr.gaze_type,
        "expression": expr.expression,
        "script_status": check.status,
        "plan_request": check.plan_request,
        "director_status": director.status,
        "director_exit_index": director.exit_index,
    }, total


def run_combined(sc: Scenario, model_key: str) -> tuple[dict, float]:
    """Run single combined call. Returns (results_dict, total_ms)."""
    model_config = MODELS[model_key]

    exit_text = "\n".join(
        f"  {i}. {e.condition}" for i, e in enumerate(EXITS)
    )
    system = COMBINED_SYSTEM.format(
        gaze_targets=", ".join(GAZE_TARGETS),
        exit_conditions=exit_text,
    )

    parts = [
        f"=== CHARACTER'S LAST LINE ===",
        sc.last_line,
        f"\n=== CURRENT BEAT ===",
        f"Intent: {sc.beat.intent}",
        f"Example line: {sc.beat.line}",
        f"\n=== RECENT CONVERSATION ===",
    ]
    for msg in sc.history[-6:]:
        role = msg["role"].upper()
        content = msg["content"][:200]
        parts.append(f"{role}: {content}")

    user_message = "\n".join(parts)

    t0 = time.perf_counter()
    response = _llm_call(
        model_config,
        label="combined",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    total = (time.perf_counter() - t0) * 1000

    data = json.loads(response.choices[0].message.content)
    return {
        "gaze": data.get("gaze", "person"),
        "gaze_type": data.get("gaze_type", "hold"),
        "expression": data.get("expression", "neutral"),
        "script_status": data.get("script_status", "hold"),
        "plan_request": data.get("plan_request", ""),
        "director_status": data.get("director_status", "hold"),
        "director_exit_index": int(data.get("director_exit_index", -1)),
    }, total


# ── Scoring ───────────────────────────────────────────────────

POSITIVE_EXPRESSIONS = {"happy", "excited", "amused", "curious"}
NEGATIVE_EXPRESSIONS = {"sad", "worried", "angry", "confused"}
NEUTRAL_EXPRESSIONS = {"neutral", "focused", "suspicious", "surprised"}

def expression_category(expr: str) -> str:
    if expr in POSITIVE_EXPRESSIONS:
        return "positive"
    if expr in NEGATIVE_EXPRESSIONS:
        return "negative"
    return "neutral"


# ── HTML report ───────────────────────────────────────────────

def badge_match(a, b) -> str:
    match = a == b
    color = "#2d8a4e" if match else "#c0392b"
    return f'<span style="background:{color};color:white;padding:2px 8px;border-radius:4px;font-size:12px">{a} {"=" if match else "≠"} {b}</span>'


def badge_correct(value, expected) -> str:
    match = value == expected
    color = "#2d8a4e" if match else "#c0392b"
    return f'<span style="background:{color};color:white;padding:2px 8px;border-radius:4px;font-size:12px">{value}</span>'


def generate_report(results: list[dict], model_name: str) -> str:
    n = len(results)

    # Agreement rates
    fields = ["gaze", "gaze_type", "expression", "script_status", "director_status", "director_exit_index"]
    agreement = {}
    for f in fields:
        agreement[f] = sum(1 for r in results if r["individual"][f] == r["combined"][f])

    # Correctness rates (vs expected)
    indiv_script = sum(1 for r in results if r["individual"]["script_status"] == r["expected_script_status"])
    combo_script = sum(1 for r in results if r["combined"]["script_status"] == r["expected_script_status"])
    indiv_dir = sum(1 for r in results if r["individual"]["director_status"] == r["expected_director_status"])
    combo_dir = sum(1 for r in results if r["combined"]["director_status"] == r["expected_director_status"])

    avg_indiv_lat = sum(r["lat_individual"] for r in results) / n
    avg_combo_lat = sum(r["lat_combined"] for r in results) / n
    speedup = avg_indiv_lat / avg_combo_lat if avg_combo_lat > 0 else 0

    rows = []
    for r in results:
        ind = r["individual"]
        com = r["combined"]
        rows.append(f"""
        <tr>
            <td><b>{r['name']}</b><br><small>{r['description']}</small></td>
            <td>
                gaze: {badge_match(ind['gaze'], com['gaze'])}<br>
                type: {badge_match(ind['gaze_type'], com['gaze_type'])}<br>
                expr: {badge_match(ind['expression'], com['expression'])}
            </td>
            <td>
                script: {badge_correct(ind['script_status'], r['expected_script_status'])} → {badge_correct(com['script_status'], r['expected_script_status'])}<br>
                director: {badge_correct(ind['director_status'], r['expected_director_status'])} → {badge_correct(com['director_status'], r['expected_director_status'])}
            </td>
            <td>{r['lat_individual']:.0f}ms → {r['lat_combined']:.0f}ms<br>
                <small>{r['lat_individual']/r['lat_combined']:.1f}x</small></td>
        </tr>""")

    return f"""<!DOCTYPE html>
<html><head><title>Combined Post-Response A/B</title>
<style>
body {{ font-family: system-ui; max-width: 1400px; margin: 20px auto; padding: 0 20px; }}
table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; vertical-align: top; }}
th {{ background: #f5f5f5; }}
.stat {{ display: inline-block; padding: 8px 16px; margin: 4px; background: #f0f0f0; border-radius: 6px; }}
.big {{ font-size: 1.4em; font-weight: bold; }}
.winner {{ background: #e8f5e9; }}
</style></head>
<body>
<h1>Combined Post-Response A/B</h1>
<p>Model: <b>{model_name}</b> — Can we merge expression + script_check + director into a single 8B call?</p>

<h2>Agreement (individual vs combined)</h2>
<div>
{"".join(f'<div class="stat">{f}: <b>{agreement[f]}/{n}</b> ({100*agreement[f]/n:.0f}%)</div>' for f in fields)}
</div>

<h2>Correctness vs Expected</h2>
<div>
    <div class="stat">Script (individual): <b>{indiv_script}/{n}</b></div>
    <div class="stat">Script (combined): <b>{combo_script}/{n}</b></div>
    <div class="stat">Director (individual): <b>{indiv_dir}/{n}</b></div>
    <div class="stat">Director (combined): <b>{combo_dir}/{n}</b></div>
</div>

<h2>Latency</h2>
<div>
    <div class="stat">Individual (3 calls): <span class="big">{avg_indiv_lat:.0f}ms</span></div>
    <div class="stat">Combined (1 call): <span class="big">{avg_combo_lat:.0f}ms</span></div>
    <div class="stat {'winner' if speedup > 1.5 else ''}">Speedup: <span class="big">{speedup:.2f}x</span></div>
</div>

<h2>Per-Scenario</h2>
<table>
<tr><th>Scenario</th><th>Expression Agreement</th><th>Correctness (indiv → combined)</th><th>Latency</th></tr>
{"".join(rows)}
</table>
</body></html>"""


# ── Main ──────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Combined Post-Response A/B")
    parser.add_argument("--open", action="store_true", help="Open HTML report in browser")
    parser.add_argument("--model", default="groq-llama-8b", help="Model key (default: groq-llama-8b)")
    args = parser.parse_args()

    model_key = args.model
    model_name = MODELS[model_key]["name"]
    print(f"Combined Post-Response A/B: individual vs combined on {model_name}")
    print(f"Scenarios: {len(SCENARIOS)}\n")

    results = []
    for i, sc in enumerate(SCENARIOS):
        print(f"[{i+1}/{len(SCENARIOS)}] {sc.name}: {sc.description}")

        try:
            ind, lat_ind = run_individual(sc, model_key)
        except Exception as e:
            print(f"  Individual ERROR: {e}")
            ind = {"gaze": "error", "gaze_type": "error", "expression": "error",
                   "script_status": "error", "plan_request": "", "director_status": "error", "director_exit_index": -1}
            lat_ind = 0

        try:
            com, lat_com = run_combined(sc, model_key)
        except Exception as e:
            print(f"  Combined ERROR: {e}")
            com = {"gaze": "error", "gaze_type": "error", "expression": "error",
                   "script_status": "error", "plan_request": "", "director_status": "error", "director_exit_index": -1}
            lat_com = 0

        # Print comparison
        for field in ["gaze", "expression", "script_status", "director_status"]:
            match = "=" if ind[field] == com[field] else "≠"
            print(f"  {field}: {ind[field]} {match} {com[field]}")
        print(f"  latency: {lat_ind:.0f}ms → {lat_com:.0f}ms ({lat_ind/lat_com:.1f}x)" if lat_com > 0 else f"  latency: {lat_ind:.0f}ms → error")
        print()

        results.append({
            "name": sc.name,
            "description": sc.description,
            "individual": ind,
            "combined": com,
            "lat_individual": lat_ind,
            "lat_combined": lat_com,
            "expected_script_status": sc.expected_script_status,
            "expected_director_status": sc.expected_director_status,
            "expected_director_exit": sc.expected_director_exit,
            "expected_expression_type": sc.expected_expression_type,
        })

    # Summary
    n = len(results)
    fields = ["gaze", "gaze_type", "expression", "script_status", "director_status", "director_exit_index"]
    print("=" * 60)
    print("Agreement (individual vs combined):")
    for f in fields:
        agree = sum(1 for r in results if r["individual"][f] == r["combined"][f])
        print(f"  {f}: {agree}/{n} ({100*agree/n:.0f}%)")

    avg_ind = sum(r["lat_individual"] for r in results) / n
    avg_com = sum(r["lat_combined"] for r in results) / n
    print(f"\nLatency: individual={avg_ind:.0f}ms  combined={avg_com:.0f}ms  speedup={avg_ind/avg_com:.2f}x" if avg_com > 0 else "")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "combined_ab.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON: {json_path}")

    html_path = RESULTS_DIR / "combined_ab.html"
    with open(html_path, "w") as f:
        f.write(generate_report(results, model_name))
    print(f"HTML: {html_path}")

    if args.open:
        webbrowser.open(f"file://{html_path}")


if __name__ == "__main__":
    main()
