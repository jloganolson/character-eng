"""Visual focus planner — generate ephemeral VLM questions and SAM targets."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from character_eng.creative import load_prompt_asset
from character_eng.vision.vlm import VLMTaskSpec, generic_task_specs

PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"
DEFAULT_CONSTANT_QUESTIONS = [
    "Describe the visible room, objects, and any state changes that matter right now.",
    "Describe the nearest visible person's appearance, clothing, and accessories.",
    "Describe what the nearest visible person is doing right now with their body, hands, and nearby objects.",
]
DEFAULT_CONSTANT_SAM_TARGETS = ["person", "rude gesture"]
CORE_CONSTANT_QUESTIONS = DEFAULT_CONSTANT_QUESTIONS
CORE_CONSTANT_SAM_TARGETS = DEFAULT_CONSTANT_SAM_TARGETS
CORE_CONSTANT_VLM_SPECS = [
    VLMTaskSpec(
        task_id="world_state",
        label="World State",
        question=DEFAULT_CONSTANT_QUESTIONS[0],
        target="scene",
        cadence_s=3.0,
        interpret_as="world_state",
        system_role="system",
    ),
    VLMTaskSpec(
        task_id="person_description_static",
        label="Person Description Static",
        question=DEFAULT_CONSTANT_QUESTIONS[1],
        target="nearest_person",
        cadence_s=6.0,
        interpret_as="person_description_static",
        system_role="system",
    ),
    VLMTaskSpec(
        task_id="person_description_dynamic",
        label="Person Description Dynamic",
        question=DEFAULT_CONSTANT_QUESTIONS[2],
        target="nearest_person",
        cadence_s=2.0,
        interpret_as="person_description_dynamic",
        system_role="system",
    ),
]


@dataclass
class VisualFocusResult:
    constant_questions: list[str] = field(default_factory=list)
    ephemeral_questions: list[str] = field(default_factory=list)
    constant_sam_targets: list[str] = field(default_factory=list)
    ephemeral_sam_targets: list[str] = field(default_factory=list)
    constant_vlm_specs: list[VLMTaskSpec] = field(default_factory=list)
    ephemeral_vlm_specs: list[VLMTaskSpec] = field(default_factory=list)


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


def _clean_model_output_list(
    values,
    *,
    limit: int,
    dict_keys: tuple[str, ...],
) -> list[str]:
    """Normalize loosely-structured LLM JSON arrays into distinct strings."""
    if not isinstance(values, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        candidate = ""
        if isinstance(value, str):
            candidate = value.strip()
        elif isinstance(value, dict):
            for key in dict_keys:
                raw = value.get(key)
                if isinstance(raw, str) and raw.strip():
                    candidate = raw.strip()
                    break
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        result.append(candidate)
        if len(result) >= limit:
            break
    return result


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

    authored_constant_specs = [VLMTaskSpec.from_payload(spec.to_payload()) for spec in CORE_CONSTANT_VLM_SPECS]
    authored_constant_specs.extend(generic_task_specs(
        [
            item
            for item in authored_constant_questions
            if item not in CORE_CONSTANT_QUESTIONS
        ],
        prefix="constant",
        cadence_s=3.0,
    ))

    if not user_parts:
        return VisualFocusResult(
            constant_questions=authored_constant_questions,
            constant_sam_targets=authored_constant_sam_targets,
            constant_vlm_specs=authored_constant_specs,
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
            constant_vlm_specs=authored_constant_specs,
        )

    ephemeral_questions = _clean_model_output_list(
        data.get("ephemeral_questions", []),
        limit=2,
        dict_keys=("question", "text", "label", "name", "value"),
    )
    ephemeral_specs = generic_task_specs(ephemeral_questions, prefix="ephemeral", cadence_s=2.0)
    ephemeral_sam_targets = _clean_model_output_list(
        data.get("ephemeral_sam_targets", []),
        limit=2,
        dict_keys=("target", "label", "name", "text", "value"),
    )

    return VisualFocusResult(
        constant_questions=authored_constant_questions,
        ephemeral_questions=ephemeral_questions,
        constant_sam_targets=authored_constant_sam_targets,
        ephemeral_sam_targets=ephemeral_sam_targets,
        constant_vlm_specs=authored_constant_specs,
        ephemeral_vlm_specs=ephemeral_specs,
    )
