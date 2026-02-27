#!/usr/bin/env python3
"""Single-Beat Planning A/B: Can 8B generate one good beat at a time?

Currently plan_call generates 1-6 beats with 70B and takes 1-5s. If 8B can
generate one good beat after each advance, we eliminate the background plan
thread entirely — the character always has exactly one beat ahead.

Two approaches compared:
  A) Multi-beat planning on 70B (current) — take first beat as reference
  B) Single-beat planning on 8B (new) — generate just one beat per call

Quality is judged on:
  - Is the line in-character? (subjective, scored by 70B judge)
  - Does the intent advance the goal?
  - Is the condition sensible?

Usage:
    uv run python experiments/fast-llm/plan_single_ab.py
    uv run python experiments/fast-llm/plan_single_ab.py --open
"""

from __future__ import annotations

import json
import os
import sys
import time
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from character_eng.models import MODELS
from character_eng.world import (
    Beat, WorldState, Goals, PlanResult, plan_call,
    _llm_call, _load_prompt_file,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ── Single-beat prompt ────────────────────────────────────────

SINGLE_BEAT_SYSTEM = """You are a conversation planner for a character in an interactive fiction.

Generate EXACTLY ONE conversation beat — the next thing the character should say or do.

## Rules

- Read the WORLD STATE and CONVERSATION carefully. Plan for the character's CURRENT situation.
- Write dialogue in the character's authentic voice — match their personality and speech patterns.
- The intent should be a short, achievable conversational move (one clause).
- The line should be 1-2 sentences in the character's voice.
- Only set a condition if the beat genuinely requires something to be true first. Most beats need no condition.

## Stage goal

If a CURRENT STAGE GOAL is present, the beat should advance that goal naturally.

## People

If a PEOPLE PRESENT section is present, the beat should acknowledge and interact with them.

## Output format

Return a JSON object with exactly this key:
- "beats": An array with EXACTLY ONE object containing:
  - "line": The dialogue line in the character's voice.
  - "intent": Brief description of what this beat accomplishes.
  - "condition": Precondition (empty string if none needed)."""


# ── Test scenarios ────────────────────────────────────────────

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

GOALS = Goals(long_term="Experience the universe. Short-term: Hand out fliers and get people to stop by the stand.")


@dataclass
class Scenario:
    name: str
    description: str
    history: list[dict]
    world_dynamic: dict[str, str]
    stage_goal: str
    plan_request: str = ""
    quality_criteria: str = ""  # what makes a good beat here


SCENARIOS = [
    Scenario(
        name="opening_alone",
        description="No conversation yet, character alone. Should plan idle/environmental beat.",
        history=[],
        world_dynamic={
            "f1": "It is a warm afternoon",
            "f2": "A light breeze is blowing",
            "f3": "No one is around",
        },
        stage_goal="Watch for potential customers and try to attract attention",
        quality_criteria="Beat should be idle behavior — muttering, reacting to environment, or trying to attract passersby. Should NOT greet someone who isn't there.",
    ),
    Scenario(
        name="after_greeting",
        description="User just said hello. Should plan engagement beat.",
        history=[
            {"role": "user", "content": "Hey there! What's this stand about?"},
            {"role": "assistant", "content": "Hey! Welcome to Greg's! I sell water for a dollar and give advice for free. What brings you out today?"},
        ],
        world_dynamic={
            "f1": "It is a warm afternoon",
            "f2": "A person is at the stand",
        },
        stage_goal="Engage with the customer and hand out fliers",
        quality_criteria="Beat should continue the conversation — offer something, ask a question, or share something interesting. Should reference the customer being there.",
    ),
    Scenario(
        name="after_question",
        description="User asked about water. Should plan answer/offer beat.",
        history=[
            {"role": "user", "content": "How much is the water?"},
            {"role": "assistant", "content": "Just a dollar! And it's ice cold. The cooler's right under the table."},
            {"role": "user", "content": "Is it just water or do you have other drinks?"},
        ],
        world_dynamic={
            "f1": "It is a warm afternoon",
            "f2": "A person is at the stand asking about drinks",
        },
        stage_goal="Engage with the customer and hand out fliers",
        quality_criteria="Beat should answer the drinks question or pivot naturally. Should stay on topic about the stand's offerings.",
    ),
    Scenario(
        name="after_world_event",
        description="A customer is approaching. Should plan greeting beat.",
        history=[
            {"role": "system", "content": "[A person walks up to the stand and looks at the sign]"},
        ],
        world_dynamic={
            "f1": "It is a warm afternoon",
            "f2": "A person is approaching the stand",
        },
        stage_goal="Greet potential customers and offer fliers",
        quality_criteria="Beat should greet the newcomer warmly. Should reference them looking at the sign or approaching.",
    ),
    Scenario(
        name="after_off_book",
        description="User changed topic to robots. Should plan beat following new direction.",
        history=[
            {"role": "assistant", "content": "Want to check out our fliers?"},
            {"role": "user", "content": "I don't care about fliers. What's it like being a robot?"},
            {"role": "assistant", "content": "Oh man, that's a great question! It's honestly pretty weird."},
        ],
        world_dynamic={
            "f1": "It is a warm afternoon",
            "f2": "A person is at the stand",
        },
        stage_goal="Engage with the customer and hand out fliers",
        plan_request="The user wants to talk about being a robot and what it's like. Follow their interest.",
        quality_criteria="Beat should follow the robot topic — share an experience, ask the user a question about being human, etc. Should NOT try to push fliers.",
    ),
    Scenario(
        name="mid_script_continuation",
        description="Two beats done, conversation flowing. Should plan natural continuation.",
        history=[
            {"role": "assistant", "content": "Hey! Welcome to Greg's!"},
            {"role": "user", "content": "Thanks! What do you sell?"},
            {"role": "assistant", "content": "Water for a dollar, and advice is free! Want a flier?"},
            {"role": "user", "content": "Sure, I'll take a flier."},
            {"role": "assistant", "content": "Awesome! It's got a coupon for a free cup of water on there."},
        ],
        world_dynamic={
            "f1": "It is a warm afternoon",
            "f2": "A person is at the stand with a flier",
            "f3": "Greg has handed out one flier",
        },
        stage_goal="Engage with the customer and keep them at the stand",
        quality_criteria="Beat should continue naturally — ask a question, make small talk, or offer water. Conversation is flowing well.",
    ),
]


# ── Runners ───────────────────────────────────────────────────

def run_multi_beat(sc: Scenario, model_key: str) -> tuple[Beat | None, float, list[Beat]]:
    """Run 70B multi-beat plan. Returns (first_beat, latency_ms, all_beats)."""
    model_config = MODELS[model_key]
    world = WorldState(static=list(WORLD_STATIC), dynamic=dict(sc.world_dynamic))

    t0 = time.perf_counter()
    result = plan_call(
        system_prompt=SYSTEM_PROMPT,
        world=world,
        history=sc.history,
        goals=GOALS,
        plan_request=sc.plan_request,
        plan_model_config=model_config,
        stage_goal=sc.stage_goal,
    )
    elapsed = (time.perf_counter() - t0) * 1000

    beats = result.beats if result.beats else []
    first = beats[0] if beats else None
    return first, elapsed, beats


def run_single_beat(sc: Scenario, model_key: str) -> tuple[Beat | None, float]:
    """Run 8B single-beat plan. Returns (beat, latency_ms)."""
    model_config = MODELS[model_key]
    world = WorldState(static=list(WORLD_STATIC), dynamic=dict(sc.world_dynamic))

    context_parts = [
        "=== CHARACTER SYSTEM PROMPT ===",
        SYSTEM_PROMPT,
        "\n=== WORLD STATE ===",
        world.render(),
        f"\n=== CHARACTER GOALS ===",
        GOALS.render(),
    ]
    if sc.stage_goal:
        context_parts.append(f"\n=== CURRENT STAGE GOAL ===")
        context_parts.append(sc.stage_goal)

    recent = sc.history[-6:] if len(sc.history) > 6 else sc.history
    if recent:
        context_parts.append("\n=== RECENT CONVERSATION ===")
        for msg in recent:
            role = msg["role"].upper()
            context_parts.append(f"{role}: {msg['content']}")

    if sc.plan_request:
        context_parts.append(f"\n=== PLAN REQUEST ===")
        context_parts.append(sc.plan_request)

    user_message = "\n".join(context_parts)

    t0 = time.perf_counter()
    response = _llm_call(
        model_config,
        label="plan_single",
        messages=[
            {"role": "system", "content": SINGLE_BEAT_SYSTEM},
            {"role": "user", "content": user_message},
        ],
        temperature=0.7,
        response_format={"type": "json_object"},
    )
    elapsed = (time.perf_counter() - t0) * 1000

    data = json.loads(response.choices[0].message.content)
    beats = data.get("beats", [])
    if beats:
        b = beats[0]
        return Beat(line=b["line"], intent=b["intent"], condition=b.get("condition", "")), elapsed
    return None, elapsed


# ── LLM judge ────────────────────────────────────────────────

JUDGE_SYSTEM = """You are a quality judge for interactive fiction dialogue beats.

Given a character description, situation, and a candidate dialogue beat, rate the beat on three criteria:

1. **voice** (0-10): Does the line sound like this character? Match their personality, vocabulary, and speaking style.
2. **relevance** (0-10): Does the beat make sense given the current situation and goal?
3. **quality** (0-10): Is the line engaging, natural, and well-written?

Return a JSON object: {"voice": <0-10>, "relevance": <0-10>, "quality": <0-10>, "notes": "<brief explanation>"}"""


def judge_beat(beat: Beat, sc: Scenario) -> dict:
    """Use 70B to judge beat quality. Returns {"voice": int, "relevance": int, "quality": int, "notes": str}."""
    model_config = MODELS["groq-llama"]
    context = f"""Character: Greg, a curious robot head on a folding table at a water stand. No body or hands. Loves asking questions.
Situation: {sc.description}
Stage goal: {sc.stage_goal}
Quality criteria: {sc.quality_criteria}

Beat to judge:
  Line: "{beat.line}"
  Intent: {beat.intent}
  Condition: {beat.condition or "(none)"}"""

    try:
        response = _llm_call(
            model_config,
            label="judge",
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": context},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"voice": 0, "relevance": 0, "quality": 0, "notes": f"Judge error: {e}"}


# ── HTML report ───────────────────────────────────────────────

def score_color(score: int) -> str:
    if score >= 7:
        return "#2d8a4e"
    if score >= 4:
        return "#f39c12"
    return "#c0392b"


def score_badge(score: int, label: str) -> str:
    return f'<span style="background:{score_color(score)};color:white;padding:2px 6px;border-radius:4px;font-size:12px">{label}:{score}</span>'


def generate_report(results: list[dict]) -> str:
    n = len(results)

    # Averages
    multi_voice = sum(r["multi_judge"]["voice"] for r in results) / n
    multi_rel = sum(r["multi_judge"]["relevance"] for r in results) / n
    multi_qual = sum(r["multi_judge"]["quality"] for r in results) / n
    single_voice = sum(r["single_judge"]["voice"] for r in results) / n
    single_rel = sum(r["single_judge"]["relevance"] for r in results) / n
    single_qual = sum(r["single_judge"]["quality"] for r in results) / n
    avg_multi_lat = sum(r["lat_multi"] for r in results) / n
    avg_single_lat = sum(r["lat_single"] for r in results) / n

    rows = []
    for r in results:
        mj = r["multi_judge"]
        sj = r["single_judge"]
        mb = r.get("multi_beat")
        sb = r.get("single_beat")
        rows.append(f"""
        <tr>
            <td><b>{r['name']}</b><br><small>{r['description']}</small><br><small style="color:#888">{r['quality_criteria']}</small></td>
            <td>
                <div style="background:#f5f5f5;padding:8px;border-radius:4px;margin-bottom:4px">
                    <b>"{mb['line'] if mb else 'N/A'}"</b><br>
                    <small>Intent: {mb['intent'] if mb else 'N/A'}</small>
                </div>
                {score_badge(mj['voice'], 'voice')} {score_badge(mj['relevance'], 'rel')} {score_badge(mj['quality'], 'qual')}<br>
                <small style="color:#666">{mj.get('notes', '')}</small><br>
                <small>{r['lat_multi']:.0f}ms | {r.get('n_beats_multi', 0)} beats total</small>
            </td>
            <td>
                <div style="background:#fffde7;padding:8px;border-radius:4px;margin-bottom:4px">
                    <b>"{sb['line'] if sb else 'N/A'}"</b><br>
                    <small>Intent: {sb['intent'] if sb else 'N/A'}</small>
                </div>
                {score_badge(sj['voice'], 'voice')} {score_badge(sj['relevance'], 'rel')} {score_badge(sj['quality'], 'qual')}<br>
                <small style="color:#666">{sj.get('notes', '')}</small><br>
                <small>{r['lat_single']:.0f}ms</small>
            </td>
        </tr>""")

    return f"""<!DOCTYPE html>
<html><head><title>Single-Beat Planning A/B</title>
<style>
body {{ font-family: system-ui; max-width: 1400px; margin: 20px auto; padding: 0 20px; }}
table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; vertical-align: top; }}
th {{ background: #f5f5f5; }}
.stat {{ display: inline-block; padding: 8px 16px; margin: 4px; background: #f0f0f0; border-radius: 6px; }}
.winner {{ background: #e8f5e9; }}
</style></head>
<body>
<h1>Single-Beat Planning A/B</h1>
<p>Can 8B generate one good beat at a time, replacing 70B multi-beat planning?</p>
<p>Quality judged by Llama 70B on voice (in-character?), relevance (fits situation?), quality (engaging?).</p>

<h2>Average Scores</h2>
<table>
<tr><th></th><th>Voice</th><th>Relevance</th><th>Quality</th><th>Avg Latency</th></tr>
<tr>
    <td><b>Multi-beat 70B</b> (first beat)</td>
    <td>{multi_voice:.1f}/10</td><td>{multi_rel:.1f}/10</td><td>{multi_qual:.1f}/10</td>
    <td>{avg_multi_lat:.0f}ms</td>
</tr>
<tr style="background:#fffde7">
    <td><b>Single-beat 8B</b></td>
    <td>{single_voice:.1f}/10</td><td>{single_rel:.1f}/10</td><td>{single_qual:.1f}/10</td>
    <td>{avg_single_lat:.0f}ms</td>
</tr>
</table>

<h2>Per-Scenario ({n})</h2>
<table>
<tr><th>Scenario</th><th>Multi-beat 70B (first beat)</th><th style="background:#fffde7">Single-beat 8B</th></tr>
{"".join(rows)}
</table>
</body></html>"""


# ── Main ──────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Single-Beat Planning A/B")
    parser.add_argument("--open", action="store_true", help="Open HTML report in browser")
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM judge (faster, no quality scores)")
    args = parser.parse_args()

    print(f"Single-Beat Planning A/B: 70B multi vs 8B single")
    print(f"Scenarios: {len(SCENARIOS)}\n")

    results = []
    for i, sc in enumerate(SCENARIOS):
        print(f"[{i+1}/{len(SCENARIOS)}] {sc.name}: {sc.description}")

        # A) Multi-beat 70B
        try:
            multi_beat, lat_multi, all_beats = run_multi_beat(sc, "groq-llama")
            print(f"  70B multi: \"{multi_beat.line[:60]}...\" [{multi_beat.intent}] ({lat_multi:.0f}ms, {len(all_beats)} beats)")
        except Exception as e:
            print(f"  70B multi: ERROR — {e}")
            multi_beat, lat_multi, all_beats = None, 0, []

        # B) Single-beat 8B
        try:
            single_beat, lat_single = run_single_beat(sc, "groq-llama-8b")
            if single_beat:
                print(f"  8B single: \"{single_beat.line[:60]}...\" [{single_beat.intent}] ({lat_single:.0f}ms)")
            else:
                print(f"  8B single: No beat returned ({lat_single:.0f}ms)")
        except Exception as e:
            print(f"  8B single: ERROR — {e}")
            single_beat, lat_single = None, 0

        # Judge
        if not args.no_judge:
            multi_judge = judge_beat(multi_beat, sc) if multi_beat else {"voice": 0, "relevance": 0, "quality": 0, "notes": "No beat"}
            single_judge = judge_beat(single_beat, sc) if single_beat else {"voice": 0, "relevance": 0, "quality": 0, "notes": "No beat"}
            print(f"  Judge 70B: voice={multi_judge['voice']} rel={multi_judge['relevance']} qual={multi_judge['quality']}")
            print(f"  Judge 8B:  voice={single_judge['voice']} rel={single_judge['relevance']} qual={single_judge['quality']}")
        else:
            multi_judge = {"voice": 5, "relevance": 5, "quality": 5, "notes": "Judging skipped"}
            single_judge = {"voice": 5, "relevance": 5, "quality": 5, "notes": "Judging skipped"}

        print()
        results.append({
            "name": sc.name,
            "description": sc.description,
            "quality_criteria": sc.quality_criteria,
            "multi_beat": {"line": multi_beat.line, "intent": multi_beat.intent, "condition": multi_beat.condition} if multi_beat else None,
            "single_beat": {"line": single_beat.line, "intent": single_beat.intent, "condition": single_beat.condition} if single_beat else None,
            "lat_multi": lat_multi,
            "lat_single": lat_single,
            "n_beats_multi": len(all_beats),
            "multi_judge": multi_judge,
            "single_judge": single_judge,
        })

    # Summary
    n = len(results)
    print("=" * 60)
    for label, key in [("70B multi", "multi_judge"), ("8B single", "single_judge")]:
        avg_v = sum(r[key]["voice"] for r in results) / n
        avg_r = sum(r[key]["relevance"] for r in results) / n
        avg_q = sum(r[key]["quality"] for r in results) / n
        print(f"{label}: voice={avg_v:.1f} rel={avg_r:.1f} qual={avg_q:.1f}")

    lat_m = sum(r["lat_multi"] for r in results) / n
    lat_s = sum(r["lat_single"] for r in results) / n
    print(f"Latency: multi={lat_m:.0f}ms  single={lat_s:.0f}ms  speedup={lat_m/lat_s:.1f}x" if lat_s > 0 else "")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "plan_single_ab.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON: {json_path}")

    html_path = RESULTS_DIR / "plan_single_ab.html"
    with open(html_path, "w") as f:
        f.write(generate_report(results))
    print(f"HTML: {html_path}")

    if args.open:
        webbrowser.open(f"file://{html_path}")


if __name__ == "__main__":
    main()
