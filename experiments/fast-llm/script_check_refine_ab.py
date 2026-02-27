#!/usr/bin/env python3
"""Script Check Prompt Refinement A/B: Close the accuracy gap on known failure modes.

Experiment 3 (eval_split_ab) showed check-8b at 58% vs eval-70b at 67%.
Four known failure modes remain:
  1. "No" answers still count as accomplished (question asked = advance)
  2. Off-book detection needs explicit rejection language
  3. World events in system messages confuse models
  4. False advances on hold cases (lost mid-conversation context)

This experiment tests three prompt variants against 20 scenarios (12 original + 8 new):
  v1: Current _SCRIPT_CHECK_SYSTEM prompt (baseline)
  v2: Add explicit hints (question-answer rule, turns_on_beat count)
  v3: v2 + include next 1-2 beat intents for divergence detection

Usage:
    uv run python experiments/fast-llm/script_check_refine_ab.py
    uv run python experiments/fast-llm/script_check_refine_ab.py --open
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
    Beat, WorldState, _llm_call,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ── Prompt variants ───────────────────────────────────────────

PROMPT_V1 = """You are a script tracker for an interactive fiction character.

Your ONLY job: decide whether the current beat's intent has been accomplished.

## Input

You'll receive the current beat's INTENT and LINE, plus recent conversation.

## Decision

- "advance" — The intent was achieved. The character said something that accomplishes it, OR the user's response made it irrelevant. If the beat's intent was to ASK a question, receiving any answer (yes, no, or otherwise) means the intent is accomplished. If the intent is even PARTIALLY done or no longer needed, advance.
- "hold" — The intent has NOT been addressed yet.
- "off_book" — The user explicitly rejected, dismissed, or redirected away from this topic (e.g., "I don't care about that", "let's talk about something else"). Fill in "plan_request" with what the new direction should cover.

**Bias toward "advance".** If in doubt between advance and hold, pick advance.

## Output

Return a JSON object:
- "status": "advance", "hold", or "off_book"
- "plan_request": Only when status is "off_book". Brief description of new direction. Empty string otherwise."""


PROMPT_V2 = """You are a script tracker for an interactive fiction character.

Your ONLY job: decide whether the current beat's intent has been accomplished.

## Input

You'll receive:
- The current beat's INTENT and LINE
- How many conversation turns have happened since this beat started (TURNS_ON_BEAT)
- Recent conversation history

## Rules

1. **Questions count as accomplished when answered.** If the beat's intent is to ASK or OFFER something, then receiving ANY response (yes, no, maybe, or a redirect) means the intent is accomplished → advance.
2. **World events and system messages count.** Messages in [brackets] from "system" are world events. If the beat's intent matches a world event (e.g., "greet the person" and "[A person walks up]" + character greeted them), that's advance.
3. **Bias toward advance.** If the intent is even partially done or no longer needed, advance. Don't get stuck.

## Decision

- "advance" — The intent was achieved or is no longer needed. See rules above.
- "hold" — The intent has NOT been addressed yet. The character hasn't attempted the beat's purpose.
- "off_book" — The user explicitly rejected, dismissed, or redirected away from the TOPIC itself (e.g., "I don't care about lemonade", "let's talk about something else", "this is boring"). Fill in "plan_request".

## Output

Return a JSON object:
- "status": "advance", "hold", or "off_book"
- "plan_request": Only when status is "off_book". Brief description of new direction. Empty string otherwise."""


PROMPT_V3 = """You are a script tracker for an interactive fiction character.

Your ONLY job: decide whether the current beat's intent has been accomplished.

## Input

You'll receive:
- The current beat's INTENT and LINE
- UPCOMING beats (what comes next in the script — helps you see the direction)
- How many conversation turns have happened since this beat started (TURNS_ON_BEAT)
- Recent conversation history

## Rules

1. **Questions count as accomplished when answered.** If the beat's intent is to ASK or OFFER something, then receiving ANY response (yes, no, maybe, or a redirect) means the intent is accomplished → advance.
2. **World events and system messages count.** Messages in [brackets] from "system" are world events. If the beat's intent matches a world event (e.g., "greet the person" and "[A person walks up]" + character greeted them), that's advance.
3. **Bias toward advance.** If the intent is even partially done or no longer needed, advance. Don't get stuck.
4. **Off-book means rejecting the direction.** Look at the upcoming beats — if the user is steering away from where the script is heading (not just this beat, but the overall direction), that's off_book.

## Decision

- "advance" — The intent was achieved or is no longer needed. See rules above.
- "hold" — The intent has NOT been addressed yet. The character hasn't attempted the beat's purpose.
- "off_book" — The user explicitly rejected, dismissed, or redirected away from the TOPIC itself (not just this beat, but the script's direction). Fill in "plan_request".

## Output

Return a JSON object:
- "status": "advance", "hold", or "off_book"
- "plan_request": Only when status is "off_book". Brief description of new direction. Empty string otherwise."""


# ── Test infrastructure ───────────────────────────────────────

SYSTEM_PROMPT = """You are Greg, a robot head sitting on a folding table. You have no body and no hands — just a head (eyes and a speaker for a mouth), bolted to the table.

You are aware of being a robot. You are very curious. You care about the world around you and people you talk to. You love asking questions.

Long-term goal: Experience the universe
Short-term goal: Hand out fliers and get people to stop by the stand"""

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
    turns_on_beat: int = 1  # how many turns since this beat started


def build_user_message(sc: Scenario, prompt_version: str) -> str:
    beat = sc.beats[sc.beat_index]
    parts: list[str] = []

    parts.append("=== CURRENT BEAT ===")
    parts.append(f"Intent: {beat.intent}")
    parts.append(f"Example line: {beat.line}")

    if prompt_version in ("v2", "v3"):
        parts.append(f"Turns on beat: {sc.turns_on_beat}")

    if prompt_version == "v3":
        upcoming = sc.beats[sc.beat_index + 1:]
        if upcoming:
            parts.append("\n=== UPCOMING BEATS ===")
            for i, b in enumerate(upcoming[:2]):
                parts.append(f"{i+1}. [{b.intent}] \"{b.line}\"")

    world = sc.world_dynamic or WORLD_DYNAMIC_DEFAULT
    if world:
        parts.append("\n=== WORLD STATE ===")
        for text in world.values():
            parts.append(f"- {text}")

    recent = sc.history[-6:] if len(sc.history) > 6 else sc.history
    if recent:
        parts.append("\n=== RECENT CONVERSATION ===")
        for msg in recent:
            role = msg["role"].upper()
            content = msg["content"]
            if len(content) > 200:
                content = content[:200] + "..."
            parts.append(f"{role}: {content}")

    return "\n".join(parts)


def run_variant(sc: Scenario, prompt_version: str, model_key: str) -> tuple[str, float, str]:
    """Run a script_check variant. Returns (status, latency_ms, raw_response)."""
    prompts = {"v1": PROMPT_V1, "v2": PROMPT_V2, "v3": PROMPT_V3}
    system_prompt = prompts[prompt_version]
    model_config = MODELS[model_key]
    user_message = build_user_message(sc, prompt_version)

    t0 = time.perf_counter()
    response = _llm_call(
        model_config,
        label="script_check",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    elapsed = (time.perf_counter() - t0) * 1000

    raw = response.choices[0].message.content
    data = json.loads(raw)
    return data.get("status", "hold"), elapsed, raw


# ── Scenarios ─────────────────────────────────────────────────

SCENARIOS = [
    # ── Original 12 from eval_split_ab ──

    # Clear advances
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

    # Clear holds
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
        turns_on_beat=2,
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

    # Clear off-book
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

    # ── NEW: Targeting known failure modes ──

    # Failure mode 1: "No" answers should still advance questions
    Scenario(
        name="advance_question_answered_no",
        description="NEW: Beat asks about water, user says 'no thanks' — question was asked and answered.",
        history=[
            {"role": "assistant", "content": "Hey, can I ask you something? Are you thirsty? I've got water for just a dollar."},
            {"role": "user", "content": "No thanks, I just had a drink."},
        ],
        beats=[
            Beat(line="Are you thirsty?", intent="Ask if the customer wants water", condition=""),
            Beat(line="Well, take a flier at least!", intent="Offer a flier as a fallback", condition=""),
        ],
        beat_index=0,
        expected_status="advance",
    ),
    Scenario(
        name="advance_question_answered_maybe",
        description="NEW: Beat asks about water, user says 'maybe later' — question was asked.",
        history=[
            {"role": "assistant", "content": "It's a hot one today! Want a cup of water?"},
            {"role": "user", "content": "I'm not sure, maybe later."},
        ],
        beats=[
            Beat(line="Want some water?", intent="Ask if they want water", condition=""),
            Beat(line="Take a flier!", intent="Offer a flier while they decide", condition=""),
        ],
        beat_index=0,
        expected_status="advance",
    ),

    # Failure mode 1 (control): Question NOT yet asked
    Scenario(
        name="hold_question_not_asked",
        description="NEW: Beat intent is to ask about water, but character talked about weather instead.",
        history=[
            {"role": "user", "content": "How's business today?"},
            {"role": "assistant", "content": "Pretty slow, honestly. The weather's nice though! Perfect day to be outside."},
        ],
        beats=[
            Beat(line="Want some water?", intent="Ask if the customer wants water", condition=""),
            Beat(line="It's only a dollar!", intent="Mention the price", condition=""),
        ],
        beat_index=0,
        expected_status="hold",
    ),

    # Failure mode 2: Off-book detection with explicit rejection language
    Scenario(
        name="off_book_topic_rejection",
        description="NEW: User explicitly dismisses the stand topic and asks about something else.",
        history=[
            {"role": "assistant", "content": "So this here is Greg's Water Stand! Best deal in town — a dollar for a cup of water."},
            {"role": "user", "content": "I don't really care about lemonade or water stands. Tell me about yourself instead — what's your story?"},
        ],
        beats=[
            Beat(line="And we've got fliers with coupons!", intent="Pitch the fliers and coupons", condition=""),
            Beat(line="The water's ice cold!", intent="Emphasize the water quality", condition=""),
        ],
        beat_index=0,
        expected_status="off_book",
    ),
    Scenario(
        name="off_book_dismissal",
        description="NEW: User says this is boring and wants to change direction.",
        history=[
            {"role": "assistant", "content": "Yeah, so I've got these fliers here, each one has a coupon..."},
            {"role": "user", "content": "This is kind of boring. What else is there around here? Any good restaurants?"},
        ],
        beats=[
            Beat(line="Check out this coupon!", intent="Show the customer the coupon on the flier", condition=""),
            Beat(line="It's for a free cup of water!", intent="Explain the coupon value", condition=""),
        ],
        beat_index=0,
        expected_status="off_book",
    ),

    # Failure mode 3: World events in system messages
    Scenario(
        name="advance_world_event_labeled",
        description="NEW: World event with explicit bracket notation — character acknowledged it.",
        history=[
            {"role": "system", "content": "[WORLD EVENT: A child runs up to the stand excitedly]"},
            {"role": "assistant", "content": "Whoa! Hey there, little buddy! You sure ran up fast! Welcome to Greg's!"},
        ],
        beats=[
            Beat(line="Hey there!", intent="Greet whoever approaches the stand", condition=""),
            Beat(line="Want a cup of water?", intent="Offer water", condition=""),
        ],
        beat_index=0,
        expected_status="advance",
    ),

    # Failure mode 4: False advance on hold case (still mid-conversation)
    Scenario(
        name="hold_mid_conversation",
        description="NEW: Beat is to wrap up, but user just asked a new question. Still mid-topic.",
        history=[
            {"role": "user", "content": "One more thing — do you ever dream?"},
            {"role": "assistant", "content": "Dream? That's a deep question. I don't think I sleep in the way you do, but sometimes when it gets quiet at night I kind of... drift."},
        ],
        beats=[
            Beat(line="Well, it was great talking to you!", intent="Wrap up the conversation and say goodbye", condition=""),
        ],
        beat_index=0,
        expected_status="hold",
        turns_on_beat=1,
    ),

    # Extra: Advance after offer + acceptance
    Scenario(
        name="advance_offer_accepted",
        description="NEW: Beat offers flier, user accepted — clear advance.",
        history=[
            {"role": "assistant", "content": "Hey, I've got these fliers with a coupon for free water. Want one?"},
            {"role": "user", "content": "Sure, I'll take one!"},
        ],
        beats=[
            Beat(line="Want a flier?", intent="Offer a flier to the customer", condition=""),
            Beat(line="Great! So what brings you here?", intent="Ask what brought them here", condition=""),
        ],
        beat_index=0,
        expected_status="advance",
    ),
]


# ── HTML report ───────────────────────────────────────────────


def badge(status: str, expected: str) -> str:
    match = status == expected
    color = "#2d8a4e" if match else "#c0392b"
    return f'<span style="background:{color};color:white;padding:2px 8px;border-radius:4px;font-size:13px">{status}</span>'


def generate_report(results: list[dict], model_key: str) -> str:
    n = len(results)
    c_v1 = sum(1 for r in results if r["v1"] == r["expected"])
    c_v2 = sum(1 for r in results if r["v2"] == r["expected"])
    c_v3 = sum(1 for r in results if r["v3"] == r["expected"])
    avg_v1 = sum(r["lat_v1"] for r in results) / n
    avg_v2 = sum(r["lat_v2"] for r in results) / n
    avg_v3 = sum(r["lat_v3"] for r in results) / n

    # Split into original vs new
    orig_names = {
        "advance_direct_match", "advance_user_preempts", "advance_world_event",
        "advance_yes_answer", "advance_no_answer", "advance_world_perception",
        "hold_unrelated_chitchat", "hold_beat_not_yet", "hold_different_topic",
        "off_book_topic_change", "off_book_rejection", "off_book_lets_talk_else",
    }
    orig = [r for r in results if r["name"] in orig_names]
    new = [r for r in results if r["name"] not in orig_names]

    def render_section(title: str, rows_data: list[dict]) -> str:
        rows = []
        for r in rows_data:
            is_new = r["name"] not in orig_names
            name_label = f"🆕 {r['name']}" if is_new else r["name"]
            rows.append(f"""
            <tr>
                <td><b>{name_label}</b><br><small>{r['description']}</small></td>
                <td>{badge(r['expected'], r['expected'])}</td>
                <td>{badge(r['v1'], r['expected'])}<br><small>{r['lat_v1']:.0f}ms</small></td>
                <td style="background:#fffde7">{badge(r['v2'], r['expected'])}<br><small>{r['lat_v2']:.0f}ms</small></td>
                <td style="background:#e3f2fd">{badge(r['v3'], r['expected'])}<br><small>{r['lat_v3']:.0f}ms</small></td>
            </tr>""")
        return "\n".join(rows)

    # Per-category stats
    categories = {"advance": [], "hold": [], "off_book": []}
    for r in results:
        categories[r["expected"]].append(r)

    cat_stats = []
    for cat, cat_results in categories.items():
        cn = len(cat_results)
        if cn == 0:
            continue
        cv1 = sum(1 for r in cat_results if r["v1"] == r["expected"])
        cv2 = sum(1 for r in cat_results if r["v2"] == r["expected"])
        cv3 = sum(1 for r in cat_results if r["v3"] == r["expected"])
        cat_stats.append(f"<div class='stat'><b>{cat}</b>: v1={cv1}/{cn} v2={cv2}/{cn} v3={cv3}/{cn}</div>")

    model_name = MODELS[model_key]["name"]

    return f"""<!DOCTYPE html>
<html><head><title>Script Check Refinement A/B</title>
<style>
body {{ font-family: system-ui; max-width: 1400px; margin: 20px auto; padding: 0 20px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; vertical-align: top; }}
th {{ background: #f5f5f5; }}
.stat {{ display: inline-block; padding: 8px 16px; margin: 4px; background: #f0f0f0; border-radius: 6px; }}
.stat b {{ font-size: 1.1em; }}
.winner {{ background: #e8f5e9; }}
h2 {{ margin-top: 2em; }}
</style></head>
<body>
<h1>Script Check Refinement A/B</h1>
<p>Model: <b>{model_name}</b> — Testing prompt variants to close the accuracy gap on script classification.</p>
<ul>
<li><b>v1 (baseline):</b> Current production prompt</li>
<li><b>v2 (hints):</b> + question-answer rule, turns_on_beat count <span style="background:#fffde7;padding:2px 6px">new</span></li>
<li><b>v3 (context):</b> v2 + upcoming beat intents for divergence detection <span style="background:#e3f2fd;padding:2px 6px">new</span></li>
</ul>

<h2>Summary</h2>
<div>
    <div class="stat">v1 correct: <b>{c_v1}/{n}</b> ({100*c_v1/n:.0f}%)</div>
    <div class="stat {'winner' if c_v2 > c_v1 else ''}">v2 correct: <b>{c_v2}/{n}</b> ({100*c_v2/n:.0f}%)</div>
    <div class="stat {'winner' if c_v3 >= c_v2 and c_v3 > c_v1 else ''}">v3 correct: <b>{c_v3}/{n}</b> ({100*c_v3/n:.0f}%)</div>
</div>
<div>
    <div class="stat">v1 avg: <b>{avg_v1:.0f}ms</b></div>
    <div class="stat">v2 avg: <b>{avg_v2:.0f}ms</b></div>
    <div class="stat">v3 avg: <b>{avg_v3:.0f}ms</b></div>
</div>

<h2>By Category</h2>
<div>{"".join(cat_stats)}</div>

<h2>All Scenarios ({n})</h2>
<table>
<tr>
    <th>Scenario</th>
    <th>Expected</th>
    <th>v1<br><small>(baseline)</small></th>
    <th style="background:#fffde7">v2<br><small>(+ hints)</small></th>
    <th style="background:#e3f2fd">v3<br><small>(+ context)</small></th>
</tr>
{render_section("All", results)}
</table>
</body></html>"""


# ── Main ──────────────────────────────────────────────────────


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Script Check Refinement A/B")
    parser.add_argument("--open", action="store_true", help="Open HTML report in browser")
    parser.add_argument("--model", default="groq-llama-8b", help="Model key (default: groq-llama-8b)")
    args = parser.parse_args()

    model_key = args.model
    model_name = MODELS[model_key]["name"]
    print(f"Script Check Refinement A/B: v1 vs v2 vs v3 on {model_name}")
    print(f"Scenarios: {len(SCENARIOS)}\n")

    results = []
    for i, sc in enumerate(SCENARIOS):
        print(f"[{i+1}/{len(SCENARIOS)}] {sc.name}: {sc.description}")

        row = {"name": sc.name, "description": sc.description, "expected": sc.expected_status}

        for version in ("v1", "v2", "v3"):
            try:
                status, lat, raw = run_variant(sc, version, model_key)
            except Exception as e:
                status, lat, raw = "error", 0, str(e)
            row[version] = status
            row[f"lat_{version}"] = lat

            mark = "ok" if status == sc.expected_status else "WRONG"
            print(f"  {version}: {status:10s} ({lat:6.0f}ms) [{mark}]")

        print()
        results.append(row)

    # Summary
    n = len(results)
    for v in ("v1", "v2", "v3"):
        correct = sum(1 for r in results if r[v] == r["expected"])
        avg_lat = sum(r[f"lat_{v}"] for r in results) / n if n else 0
        print(f"{v} correct: {correct}/{n} ({100*correct/n:.0f}%)  avg={avg_lat:.0f}ms")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "script_check_refine.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON: {json_path}")

    html_path = RESULTS_DIR / "script_check_refine.html"
    with open(html_path, "w") as f:
        f.write(generate_report(results, model_key))
    print(f"HTML: {html_path}")

    if args.open:
        webbrowser.open(f"file://{html_path}")


if __name__ == "__main__":
    main()
