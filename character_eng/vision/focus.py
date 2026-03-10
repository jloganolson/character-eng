"""Visual focus planner — generate ephemeral VLM questions and SAM targets."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from character_eng.creative import load_prompt_asset

PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"
CORE_CONSTANT_QUESTIONS = [
    "What stands out most about the nearest person's appearance, clothing, or accessories?",
    "What is the nearest person doing with their body, hands, or gaze?",
    "What facial expression, mood, or energy does the nearest person seem to have?",
]
CORE_CONSTANT_SAM_TARGETS = ["person"]


@dataclass
class VisualFocusResult:
    constant_questions: list[str] = field(default_factory=list)
    ephemeral_questions: list[str] = field(default_factory=list)
    constant_sam_targets: list[str] = field(default_factory=list)
    ephemeral_sam_targets: list[str] = field(default_factory=list)


def _load_prompt() -> str:
    return load_prompt_asset("visual_focus", prompts_dir=PROMPTS_DIR, default_filename="visual_focus.txt")


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
            constant_questions=list(CORE_CONSTANT_QUESTIONS),
            constant_sam_targets=list(CORE_CONSTANT_SAM_TARGETS),
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
            constant_questions=list(CORE_CONSTANT_QUESTIONS),
            constant_sam_targets=list(CORE_CONSTANT_SAM_TARGETS),
        )

    return VisualFocusResult(
        constant_questions=list(CORE_CONSTANT_QUESTIONS),
        ephemeral_questions=data.get("ephemeral_questions", [])[:2],
        constant_sam_targets=list(CORE_CONSTANT_SAM_TARGETS),
        ephemeral_sam_targets=data.get("ephemeral_sam_targets", [])[:2],
    )
