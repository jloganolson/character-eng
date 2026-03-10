"""Vision synthesis — distill raw visual data into perception events via 8B LLM."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from character_eng.creative import load_prompt_asset
from character_eng.vision.context import VisualContext

PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"


@dataclass
class SynthesisResult:
    events: list[str] = field(default_factory=list)
    gaze_candidates: list[str] = field(default_factory=list)
    thought: str = ""
    signals: list[str] = field(default_factory=list)


def _load_prompt() -> str:
    return load_prompt_asset("vision_synthesis", prompts_dir=PROMPTS_DIR, default_filename="vision_synthesis.txt")


def vision_synthesis_call(
    visual_context: VisualContext,
    world,
    people,
    beat,
    stage_goal: str,
    model_config: dict,
    previous_synthesis: SynthesisResult | None = None,
) -> SynthesisResult:
    """Distill raw visual state into 0-3 perception events.

    Uses 8B via Groq, JSON output, ~350ms.
    """
    from character_eng.world import _llm_call

    visual_text = visual_context.render_for_prompt()
    if not visual_text:
        return SynthesisResult()

    system_prompt = _load_prompt()

    user_parts = [f"## Current Visual State\n{visual_text}"]
    if world:
        user_parts.append(f"## World State\n{world.render()}")
    if people:
        user_parts.append(f"## People\n{people.render()}")
    if beat:
        user_parts.append(f"## Current Beat\nIntent: {beat.intent}")
    if stage_goal:
        user_parts.append(f"## Stage Goal\n{stage_goal}")
    if previous_synthesis and previous_synthesis.events:
        user_parts.append(f"## Previous Events (avoid repeating)\n" + "\n".join(previous_synthesis.events))

    user_msg = "\n\n".join(user_parts)

    completion = _llm_call(
        model_config,
        label="vision_synthesis",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.3,
        response_format={"type": "json_object"},
    )

    try:
        data = json.loads(completion.choices[0].message.content)
    except json.JSONDecodeError:
        return SynthesisResult()

    return SynthesisResult(
        events=data.get("events", [])[:3],
        gaze_candidates=data.get("gaze_candidates", []),
        thought=data.get("thought", ""),
        signals=[str(item).strip() for item in data.get("signals", []) if str(item).strip()],
    )
