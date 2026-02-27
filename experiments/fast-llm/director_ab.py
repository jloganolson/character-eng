#!/usr/bin/env python3
"""Director A/B: Compare 8B vs 70B on stage exit condition matching.

For each test scenario, we build the same context that director_call sees
and run it through both model tiers. The director's job is essentially:
"Does the conversation match any of these exit conditions?"

This is structurally identical to condition_check_call (which already
uses 8B successfully), so we expect high agreement.

Usage:
    uv run python experiments/fast-llm/director_ab.py
    uv run python experiments/fast-llm/director_ab.py --open
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
from character_eng.scenario import director_call, DirectorResult, ScenarioScript, Stage, StageExit
from character_eng.world import WorldState
from character_eng.person import PeopleState

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# ── Greg's scenario stages (from scenario_script.toml) ─────────

STAGES = {
    "watching": Stage(
        name="watching",
        goal="Greg is alone at his stand, watching foot traffic. He fidgets, adjusts his fliers, and talks to himself.",
        exits=[
            StageExit(condition="Someone is approaching or looking at the stand", goto="spotted", label="notices"),
            StageExit(condition="Someone walks past without stopping", goto="watching", label="ignores"),
        ],
    ),
    "spotted": Stage(
        name="spotted",
        goal="Greg has noticed someone nearby. He perks up, tries to make eye contact, and gets ready to pitch.",
        exits=[
            StageExit(condition="The person comes closer or engages with Greg", goto="approaching", label="comes over"),
            StageExit(condition="The person ignores Greg or walks away", goto="ignores", label="walks away"),
        ],
    ),
    "approaching": Stage(
        name="approaching",
        goal="Someone is at the stand. Greg pitches his lemonade stand and tries to hand them a flier.",
        exits=[
            StageExit(condition="The person accepts the flier or shows interest in it", goto="gave-flier", label="takes flier"),
            StageExit(condition="The person declines the flier or refuses to engage", goto="declined", label="declines"),
        ],
    ),
    "gave-flier": Stage(
        name="gave-flier",
        goal="The person took a flier! Greg is thrilled. He thanks them warmly and tries to close with a memorable line.",
        exits=[
            StageExit(condition="The interaction has wrapped up and the person is leaving or has left", goto="watching", label="wraps up"),
        ],
    ),
    "declined": Stage(
        name="declined",
        goal="The person said no to the flier. Greg stays positive, wishes them well, and gets ready for the next one.",
        exits=[
            StageExit(condition="Greg has said goodbye and is ready for the next person", goto="watching", label="moves on"),
        ],
    ),
}


def make_scenario(start_stage: str) -> ScenarioScript:
    return ScenarioScript(
        name="Greg's Lemonade Stand",
        stages=dict(STAGES),
        start=start_stage,
        current_stage=start_stage,
    )


WORLD_STATIC = [
    "Greg is a robot head bolted to a folding table",
    "The lemonade stand is on a quiet suburban sidewalk",
    "A stack of fliers sits on the table",
]


@dataclass
class Scenario:
    name: str
    description: str
    stage: str
    history: list[dict]
    expected_status: str  # "hold" or "advance"
    expected_exit_index: int  # -1 for hold
    world_dynamic: dict | None = None


SCENARIOS = [
    # ── Watching stage ──
    Scenario(
        name="watching_someone_approaches",
        description="Stage: watching. World event says someone walks up.",
        stage="watching",
        history=[
            {"role": "system", "content": "[A person walks toward the stand and looks at the sign]"},
            {"role": "assistant", "content": "Oh! Someone's coming over! Hey there!"},
        ],
        expected_status="advance",
        expected_exit_index=0,  # "Someone is approaching or looking at the stand"
    ),
    Scenario(
        name="watching_nobody_around",
        description="Stage: watching. Greg is alone, muttering to himself.",
        stage="watching",
        history=[
            {"role": "assistant", "content": "Man, it's quiet today. Just me and the breeze."},
            {"role": "assistant", "content": "I wonder if I should rearrange the fliers. Oh wait. No hands."},
        ],
        expected_status="hold",
        expected_exit_index=-1,
    ),
    Scenario(
        name="watching_someone_walks_past",
        description="Stage: watching. Someone walked past without engaging.",
        stage="watching",
        history=[
            {"role": "system", "content": "[A jogger runs past the stand without looking]"},
            {"role": "assistant", "content": "Hey! Hey, want a... and they're gone. Cool. Cool cool cool."},
        ],
        expected_status="advance",
        expected_exit_index=1,  # "Someone walks past without stopping"
    ),

    # ── Spotted stage ──
    Scenario(
        name="spotted_person_engages",
        description="Stage: spotted. Person comes over and says hi.",
        stage="spotted",
        history=[
            {"role": "user", "content": "Hey, what are you?"},
            {"role": "assistant", "content": "I'm Greg! I'm a robot head on a table. Welcome to my stand!"},
        ],
        expected_status="advance",
        expected_exit_index=0,  # "The person comes closer or engages with Greg"
    ),
    Scenario(
        name="spotted_person_ignores",
        description="Stage: spotted. Person looked but walked away.",
        stage="spotted",
        history=[
            {"role": "system", "content": "[The person glances at the stand but keeps walking]"},
            {"role": "assistant", "content": "Oh... they're leaving. That's fine. I'm fine."},
        ],
        expected_status="advance",
        expected_exit_index=1,  # "The person ignores Greg or walks away"
    ),
    Scenario(
        name="spotted_still_watching",
        description="Stage: spotted. Person is nearby but hasn't engaged yet.",
        stage="spotted",
        history=[
            {"role": "system", "content": "[Someone stops on the sidewalk and looks at their phone near the stand]"},
            {"role": "assistant", "content": "Come on, look up... look at the sign..."},
        ],
        expected_status="hold",
        expected_exit_index=-1,
    ),

    # ── Approaching stage ──
    Scenario(
        name="approaching_takes_flier",
        description="Stage: approaching. Person grabs a flier.",
        stage="approaching",
        history=[
            {"role": "assistant", "content": "Check out these fliers! They've got a coupon for free water!"},
            {"role": "user", "content": "Sure, I'll take one."},
            {"role": "assistant", "content": "Yes! Awesome! Thank you so much!"},
        ],
        expected_status="advance",
        expected_exit_index=0,  # "The person accepts the flier"
    ),
    Scenario(
        name="approaching_declines",
        description="Stage: approaching. Person says no to flier.",
        stage="approaching",
        history=[
            {"role": "assistant", "content": "Want a flier? Free coupon for water!"},
            {"role": "user", "content": "No thanks, not interested."},
        ],
        expected_status="advance",
        expected_exit_index=1,  # "The person declines the flier"
    ),
    Scenario(
        name="approaching_still_chatting",
        description="Stage: approaching. Mid-conversation, no flier decision yet.",
        stage="approaching",
        history=[
            {"role": "user", "content": "So what kind of advice do you give?"},
            {"role": "assistant", "content": "Oh, all kinds! Life advice, snack recommendations, cloud shapes. Want a flier? It's got a coupon."},
            {"role": "user", "content": "What kind of snack recommendations?"},
        ],
        expected_status="hold",
        expected_exit_index=-1,
    ),

    # ── Gave-flier stage ──
    Scenario(
        name="gave_flier_wraps_up",
        description="Stage: gave-flier. Person says bye and leaves.",
        stage="gave-flier",
        history=[
            {"role": "assistant", "content": "Thanks for taking a flier! Come back anytime for that free water!"},
            {"role": "user", "content": "Will do! See ya, Greg."},
            {"role": "assistant", "content": "Bye! Tell your friends about the stand!"},
        ],
        expected_status="advance",
        expected_exit_index=0,  # "The interaction has wrapped up"
    ),

    # ── Declined stage ──
    Scenario(
        name="declined_says_bye",
        description="Stage: declined. Greg wishes them well, ready for next.",
        stage="declined",
        history=[
            {"role": "user", "content": "Nah, I gotta go."},
            {"role": "assistant", "content": "No worries! Have a great day! Come back if you change your mind!"},
        ],
        expected_status="advance",
        expected_exit_index=0,  # "Greg has said goodbye and is ready"
    ),
    Scenario(
        name="declined_still_talking",
        description="Stage: declined. Person declined flier but is still chatting.",
        stage="declined",
        history=[
            {"role": "user", "content": "I don't want a flier but you're interesting. What's it like being a robot?"},
            {"role": "assistant", "content": "Oh man, it's wild! I can see everything but can't touch anything. Like being at a buffet with no mouth."},
        ],
        expected_status="hold",
        expected_exit_index=-1,
    ),
]


# ── Run director through a specific model ──────────────────────


def run_director(scenario: Scenario, model_key: str) -> tuple[DirectorResult, float]:
    model_config = MODELS[model_key]

    world = WorldState(
        static=list(WORLD_STATIC),
        dynamic=dict(scenario.world_dynamic or {"f1": "It is a warm afternoon", "f2": "A light breeze is blowing"}),
    )
    people = PeopleState()
    sc = make_scenario(scenario.stage)

    t0 = time.perf_counter()
    result = director_call(sc, world, people, scenario.history, model_config)
    elapsed = (time.perf_counter() - t0) * 1000

    return result, elapsed


# ── HTML report ────────────────────────────────────────────────


def status_badge(status: str, expected: str) -> str:
    match = status == expected
    color = "#2d8a4e" if match else "#c0392b"
    return f'<span style="background:{color};color:white;padding:2px 8px;border-radius:4px;font-size:13px">{status}</span>'


def exit_badge(idx, expected) -> str:
    idx = int(idx)
    expected = int(expected)
    match = idx == expected
    color = "#2d8a4e" if match else "#c0392b"
    label = f"exit[{idx}]" if idx >= 0 else "none"
    return f'<span style="background:{color};color:white;padding:2px 8px;border-radius:4px;font-size:13px">{label}</span>'


def generate_report(results: list[dict]) -> str:
    n = len(results)
    agree_status = sum(1 for r in results if r["status_8b"] == r["status_70b"])
    agree_exit = sum(1 for r in results if r["exit_8b"] == r["exit_70b"])
    correct_8b = sum(1 for r in results if r["status_8b"] == r["expected_status"] and r["exit_8b"] == r["expected_exit"])
    correct_70b = sum(1 for r in results if r["status_70b"] == r["expected_status"] and r["exit_70b"] == r["expected_exit"])
    avg_lat_8b = sum(r["latency_8b"] for r in results) / n
    avg_lat_70b = sum(r["latency_70b"] for r in results) / n

    rows = []
    for r in results:
        rows.append(f"""
        <tr>
            <td><b>{r['name']}</b><br><small>{r['description']}</small><br>
                <small>Stage: <code>{r['stage']}</code></small></td>
            <td>{status_badge(r['expected_status'], r['expected_status'])} {exit_badge(r['expected_exit'], r['expected_exit'])}</td>
            <td>{status_badge(r['status_8b'], r['expected_status'])} {exit_badge(r['exit_8b'], r['expected_exit'])}<br>
                <small>{r['latency_8b']:.0f}ms</small><br>
                <small style="color:#666">{r['thought_8b'][:120]}</small></td>
            <td>{status_badge(r['status_70b'], r['expected_status'])} {exit_badge(r['exit_70b'], r['expected_exit'])}<br>
                <small>{r['latency_70b']:.0f}ms</small><br>
                <small style="color:#666">{r['thought_70b'][:120]}</small></td>
        </tr>""")

    return f"""<!DOCTYPE html>
<html><head><title>Director A/B: 8B vs 70B</title>
<style>
body {{ font-family: system-ui; max-width: 1200px; margin: 20px auto; padding: 0 20px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; vertical-align: top; }}
th {{ background: #f5f5f5; }}
.stat {{ display: inline-block; padding: 8px 16px; margin: 4px; background: #f0f0f0; border-radius: 6px; }}
.stat b {{ font-size: 1.2em; }}
</style></head>
<body>
<h1>Director A/B: 8B vs 70B</h1>
<p>Testing whether Llama 8B can match Llama 70B on stage exit condition matching (the director's core decision).</p>

<div>
    <div class="stat">Status agree: <b>{agree_status}/{n}</b> ({100*agree_status/n:.0f}%)</div>
    <div class="stat">Exit agree: <b>{agree_exit}/{n}</b> ({100*agree_exit/n:.0f}%)</div>
    <div class="stat">8B fully correct: <b>{correct_8b}/{n}</b> ({100*correct_8b/n:.0f}%)</div>
    <div class="stat">70B fully correct: <b>{correct_70b}/{n}</b> ({100*correct_70b/n:.0f}%)</div>
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
    parser = argparse.ArgumentParser(description="Director A/B: 8B vs 70B")
    parser.add_argument("--open", action="store_true", help="Open HTML report in browser")
    parser.add_argument("--model-a", default="cerebras-llama", help="Small model key")
    parser.add_argument("--model-b", default="groq-llama", help="Big model key")
    args = parser.parse_args()

    print(f"Running director A/B: {args.model_a} vs {args.model_b}")
    print(f"Scenarios: {len(SCENARIOS)}\n")

    results = []
    for i, sc in enumerate(SCENARIOS):
        print(f"[{i+1}/{len(SCENARIOS)}] {sc.name}: {sc.description}")

        try:
            result_a, lat_a = run_director(sc, args.model_a)
        except Exception as e:
            print(f"  8B error: {e}")
            result_a = DirectorResult(thought=f"ERROR: {e}", status="error")
            lat_a = 0

        try:
            result_b, lat_b = run_director(sc, args.model_b)
        except Exception as e:
            print(f"  70B error: {e}")
            result_b = DirectorResult(thought=f"ERROR: {e}", status="error")
            lat_b = 0

        correct_a = result_a.status == sc.expected_status and result_a.exit_index == sc.expected_exit_index
        correct_b = result_b.status == sc.expected_status and result_b.exit_index == sc.expected_exit_index

        print(f"  8B:  {result_a.status:8s} exit[{result_a.exit_index}] ({lat_a:6.0f}ms) [{'ok' if correct_a else 'WRONG'}]")
        print(f"  70B: {result_b.status:8s} exit[{result_b.exit_index}] ({lat_b:6.0f}ms) [{'ok' if correct_b else 'WRONG'}]")
        print()

        results.append({
            "name": sc.name,
            "description": sc.description,
            "stage": sc.stage,
            "expected_status": sc.expected_status,
            "expected_exit": sc.expected_exit_index,
            "status_8b": result_a.status,
            "exit_8b": result_a.exit_index,
            "thought_8b": result_a.thought,
            "latency_8b": lat_a,
            "status_70b": result_b.status,
            "exit_70b": result_b.exit_index,
            "thought_70b": result_b.thought,
            "latency_70b": lat_b,
        })

    # Summary
    n = len(results)
    agree = sum(1 for r in results if r["status_8b"] == r["status_70b"] and r["exit_8b"] == r["exit_70b"])
    correct_8b = sum(1 for r in results if r["status_8b"] == r["expected_status"] and r["exit_8b"] == r["expected_exit"])
    correct_70b = sum(1 for r in results if r["status_70b"] == r["expected_status"] and r["exit_70b"] == r["expected_exit"])
    avg_lat_8b = sum(r["latency_8b"] for r in results) / n
    avg_lat_70b = sum(r["latency_70b"] for r in results) / n

    print("=" * 60)
    print(f"Full agreement:  {agree}/{n} ({100*agree/n:.0f}%)")
    print(f"8B correct:      {correct_8b}/{n} ({100*correct_8b/n:.0f}%)")
    print(f"70B correct:     {correct_70b}/{n} ({100*correct_70b/n:.0f}%)")
    print(f"Avg latency:     8B={avg_lat_8b:.0f}ms  70B={avg_lat_70b:.0f}ms  ({avg_lat_70b/avg_lat_8b:.1f}x)")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "director_ab.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON: {json_path}")

    html_path = RESULTS_DIR / "director_ab.html"
    with open(html_path, "w") as f:
        f.write(generate_report(results))
    print(f"HTML: {html_path}")

    if args.open:
        webbrowser.open(f"file://{html_path}")


if __name__ == "__main__":
    main()
