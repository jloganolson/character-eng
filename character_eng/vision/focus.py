"""Visual focus planner — generate ephemeral VLM questions and SAM targets."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from character_eng.creative import load_prompt_asset

PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"
DEFAULT_CONSTANT_QUESTIONS = [
    "What stands out most about the nearest person's appearance, clothing, or accessories?",
    "What is the nearest person doing with their body, hands, or gaze?",
    "What facial expression, mood, or energy does the nearest person seem to have?",
]
DEFAULT_CONSTANT_SAM_TARGETS = ["person", "rude gesture"]
CORE_CONSTANT_QUESTIONS = DEFAULT_CONSTANT_QUESTIONS
CORE_CONSTANT_SAM_TARGETS = DEFAULT_CONSTANT_SAM_TARGETS


@dataclass
class VisualFocusResult:
    constant_questions: list[str] = field(default_factory=list)
    ephemeral_questions: list[str] = field(default_factory=list)
    constant_sam_targets: list[str] = field(default_factory=list)
    ephemeral_sam_targets: list[str] = field(default_factory=list)


def _load_prompt() -> str:
    return load_prompt_asset("visual_focus", prompts_dir=PROMPTS_DIR, default_filename="visual_focus.txt")


def _load_line_asset(key: str, default_filename: str, fallback: list[str]) -> list[str]:
    try:
        text = load_prompt_asset(key, prompts_dir=PROMPTS_DIR, default_filename=default_filename)
    except FileNotFoundError:
        return list(fallback)
    items = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    return items or list(fallback)


def visual_focus_call(
    beat,
    stage_goal: str,
    thought: str,
    world,
    people,
    model_config: dict,
    scenario=None,
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

    authored_constant_questions = _load_line_asset(
        "vision_constant_questions",
        "vision_constant_questions.txt",
        DEFAULT_CONSTANT_QUESTIONS,
    )
    authored_constant_sam_targets = _load_line_asset(
        "vision_constant_sam_targets",
        "vision_constant_sam_targets.txt",
        DEFAULT_CONSTANT_SAM_TARGETS,
    )
    if scenario is not None:
        requirements = scenario.active_visual_requirements()
        for question in requirements.constant_questions:
            if question not in authored_constant_questions:
                authored_constant_questions.append(question)
        for target in requirements.constant_sam_targets:
            if target not in authored_constant_sam_targets:
                authored_constant_sam_targets.append(target)

    if not user_parts:
        return VisualFocusResult(
            constant_questions=authored_constant_questions,
            constant_sam_targets=authored_constant_sam_targets,
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
            constant_questions=authored_constant_questions,
            constant_sam_targets=authored_constant_sam_targets,
        )

    return VisualFocusResult(
        constant_questions=authored_constant_questions,
        ephemeral_questions=data.get("ephemeral_questions", [])[:2],
        constant_sam_targets=authored_constant_sam_targets,
        ephemeral_sam_targets=data.get("ephemeral_sam_targets", [])[:2],
    )
