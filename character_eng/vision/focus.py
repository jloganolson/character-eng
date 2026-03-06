"""Visual focus planner — generate ephemeral VLM questions and SAM targets."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"


@dataclass
class VisualFocusResult:
    constant_questions: list[str] = field(default_factory=list)
    ephemeral_questions: list[str] = field(default_factory=list)
    constant_sam_targets: list[str] = field(default_factory=list)
    ephemeral_sam_targets: list[str] = field(default_factory=list)


def _load_prompt() -> str:
    return (PROMPTS_DIR / "visual_focus.txt").read_text()


def visual_focus_call(
    beat,
    stage_goal: str,
    thought: str,
    world,
    people,
    model_config: dict,
) -> VisualFocusResult:
    """Generate constant + ephemeral visual questions and SAM targets.

    Fires on beat change, stage change, script completion.
    Uses 8B via Groq, JSON output.
    """
    from character_eng.world import _llm_call

    system_prompt = _load_prompt()

    user_parts = []
    if beat:
        user_parts.append(f"Current beat intent: {beat.intent}")
    if stage_goal:
        user_parts.append(f"Stage goal: {stage_goal}")
    if thought:
        user_parts.append(f"Latest thought: {thought}")
    if world:
        user_parts.append(f"World state:\n{world.render()}")
    if people:
        user_parts.append(f"People:\n{people.render()}")

    if not user_parts:
        return VisualFocusResult(
            constant_questions=["How many people are visible?"],
            constant_sam_targets=["person"],
        )

    user_msg = "\n\n".join(user_parts)

    completion = _llm_call(
        model_config,
        label="visual_focus",
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
        return VisualFocusResult(
            constant_questions=["How many people are visible?"],
            constant_sam_targets=["person"],
        )

    return VisualFocusResult(
        constant_questions=data.get("constant_questions", ["How many people are visible?"])[:3],
        ephemeral_questions=data.get("ephemeral_questions", [])[:2],
        constant_sam_targets=data.get("constant_sam_targets", ["person"])[:2],
        ephemeral_sam_targets=data.get("ephemeral_sam_targets", [])[:2],
    )
