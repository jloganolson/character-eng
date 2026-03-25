"""Visual focus planner — deterministic authored/manual VLM questions and SAM targets."""

from __future__ import annotations

from dataclasses import dataclass, field
from character_eng.vision.vlm import VLMTaskSpec, generic_task_specs

CORE_CONSTANT_QUESTIONS = [
    "Describe the visible room, objects, and any state changes that matter right now.",
    "Describe the nearest visible person's appearance, clothing, and accessories.",
    "Describe what the nearest visible person is doing right now with their body, posture, and orientation. Avoid inferring held objects unless they are visually obvious.",
]
CORE_CONSTANT_SAM_TARGETS = ["person"]
CORE_CONSTANT_VLM_SPECS = [
    VLMTaskSpec(
        task_id="world_state",
        label="World State",
        question=CORE_CONSTANT_QUESTIONS[0],
        target="scene",
        cadence_s=3.0,
        interpret_as="world_state",
        system_role="system_core",
    ),
    VLMTaskSpec(
        task_id="person_description_static",
        label="Person Description Static",
        question=CORE_CONSTANT_QUESTIONS[1],
        target="nearest_person",
        cadence_s=6.0,
        interpret_as="person_description_static",
        system_role="system_core",
    ),
    VLMTaskSpec(
        task_id="person_description_dynamic",
        label="Person Description Dynamic",
        question=CORE_CONSTANT_QUESTIONS[2],
        target="nearest_person",
        cadence_s=2.0,
        interpret_as="person_description_dynamic",
        system_role="system_core",
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
    scenario_questions: list[str] = field(default_factory=list)
    stage_questions: list[str] = field(default_factory=list)
    manual_questions: list[str] = field(default_factory=list)
    scenario_sam_targets: list[str] = field(default_factory=list)
    stage_sam_targets: list[str] = field(default_factory=list)
    manual_sam_targets: list[str] = field(default_factory=list)


def _dedupe_strings(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = str(value or "").strip()
        if not cleaned:
            continue
        normalized = cleaned.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(cleaned)
    return result


def _clone_specs(specs: list[VLMTaskSpec]) -> list[VLMTaskSpec]:
    return [VLMTaskSpec.from_payload(spec.to_payload()) for spec in specs]


def _specs_from_strings(
    questions: list[str],
    *,
    prefix: str,
    cadence_s: float,
    system_role: str,
) -> list[VLMTaskSpec]:
    specs = generic_task_specs(questions, prefix=prefix, cadence_s=cadence_s)
    for spec in specs:
        spec.system_role = system_role
    return specs


def _manual_specs(items: list[dict | str]) -> list[VLMTaskSpec]:
    specs: list[VLMTaskSpec] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(items, start=1):
        spec = VLMTaskSpec.from_payload(item, default_id=f"manual_{index}")
        if not spec.question:
            continue
        base_task_id = spec.task_id
        suffix = 2
        while spec.task_id in seen_ids:
            spec.task_id = f"{base_task_id}_{suffix}"
            suffix += 1
        spec.system_role = spec.system_role or "manual"
        if spec.cadence_s <= 0:
            spec.cadence_s = 3.0
        seen_ids.add(spec.task_id)
        specs.append(spec)
    return specs


def _scenario_requirement_lists(scenario) -> tuple[list[str], list[str], list[str], list[str]]:
    scenario_questions: list[str] = []
    stage_questions: list[str] = []
    scenario_sam_targets: list[str] = []
    stage_sam_targets: list[str] = []
    if scenario is None:
        return scenario_questions, stage_questions, scenario_sam_targets, stage_sam_targets

    scenario_requirements = getattr(scenario, "visual_requirements", None)
    if scenario_requirements is not None:
        scenario_questions = _dedupe_strings(list(getattr(scenario_requirements, "constant_questions", []) or []))
        scenario_sam_targets = _dedupe_strings(list(getattr(scenario_requirements, "constant_sam_targets", []) or []))

    active_stage = getattr(scenario, "active_stage", None)
    if active_stage is not None:
        stage_requirements = getattr(active_stage, "visual_requirements", None)
        if stage_requirements is not None:
            stage_questions = _dedupe_strings(list(getattr(stage_requirements, "constant_questions", []) or []))
            stage_sam_targets = _dedupe_strings(list(getattr(stage_requirements, "constant_sam_targets", []) or []))

    if not (scenario_questions or stage_questions or scenario_sam_targets or stage_sam_targets):
        fallback = getattr(scenario, "active_visual_requirements", None)
        if callable(fallback):
            combined = fallback()
            scenario_questions = _dedupe_strings(list(getattr(combined, "constant_questions", []) or []))
            scenario_sam_targets = _dedupe_strings(list(getattr(combined, "constant_sam_targets", []) or []))

    return scenario_questions, stage_questions, scenario_sam_targets, stage_sam_targets


def visual_focus_call(
    beat,
    stage_goal: str,
    thought: str,
    world,
    people,
    model_config: dict,
    scenario=None,
    manual_questions: list[dict | str] | None = None,
    manual_sam_targets: list[str] | None = None,
) -> VisualFocusResult:
    """Build deterministic visual focus from core system tasks, scenario/stage config, and manual overrides."""
    del beat, stage_goal, thought, world, people, model_config

    scenario_questions, stage_questions, scenario_sam_targets, stage_sam_targets = _scenario_requirement_lists(scenario)

    manual_specs = _manual_specs(list(manual_questions or []))
    manual_question_text = _dedupe_strings([spec.question for spec in manual_specs])
    manual_targets = _dedupe_strings(list(manual_sam_targets or []))

    authored_questions = _dedupe_strings([*CORE_CONSTANT_QUESTIONS, *scenario_questions, *stage_questions, *manual_question_text])
    authored_sam_targets = _dedupe_strings([*CORE_CONSTANT_SAM_TARGETS, *scenario_sam_targets, *stage_sam_targets, *manual_targets])

    constant_specs = _clone_specs(CORE_CONSTANT_VLM_SPECS)
    constant_specs.extend(_specs_from_strings(scenario_questions, prefix="scenario", cadence_s=3.0, system_role="scenario"))
    constant_specs.extend(_specs_from_strings(stage_questions, prefix="stage", cadence_s=3.0, system_role="stage"))
    constant_specs.extend(manual_specs)

    return VisualFocusResult(
        constant_questions=authored_questions,
        ephemeral_questions=[],
        constant_sam_targets=authored_sam_targets,
        ephemeral_sam_targets=[],
        constant_vlm_specs=constant_specs,
        ephemeral_vlm_specs=[],
        scenario_questions=scenario_questions,
        stage_questions=stage_questions,
        manual_questions=manual_question_text,
        scenario_sam_targets=scenario_sam_targets,
        stage_sam_targets=stage_sam_targets,
        manual_sam_targets=manual_targets,
    )
