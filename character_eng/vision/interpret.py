from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from character_eng.creative import load_prompt_asset
from character_eng.world import (
    PersonUpdate,
    WorldUpdate,
    _PERSON_FACT_ID_RE,
    _VALID_PRESENCE,
    _WORLD_FACT_ID_RE,
    _clean_id_list,
    _clean_string_list,
)

PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"


@dataclass
class VisionStateUpdateResult:
    update: WorldUpdate = field(default_factory=WorldUpdate)
    thought: str = ""
    summary: str = ""


def _load_prompt() -> str:
    return load_prompt_asset("vision_state_update", prompts_dir=PROMPTS_DIR, default_filename="vision_state_update.txt")


def vision_state_update_call(
    world,
    people,
    task_answers: list[dict],
    model_config: dict,
) -> VisionStateUpdateResult:
    from character_eng.world import _llm_call

    if not task_answers:
        return VisionStateUpdateResult()

    lines: list[str] = []
    if world is not None:
        if world.static:
            lines.append("Permanent facts:")
            for fact in world.static:
                lines.append(f"- {fact}")
            lines.append("")
        if world.dynamic:
            lines.append("Current dynamic facts:")
            lines.append(world.render_for_reconcile())
            lines.append("")
        if world.events:
            lines.append("Recent world events:")
            for event in world.events[-8:]:
                lines.append(f"- {event}")
            lines.append("")
    if people is not None:
        people_text = people.render_for_reconcile()
        if people_text:
            lines.append("Current people:")
            lines.append(people_text)
            lines.append("")

    lines.append("Changed vision task answers:")
    for item in task_answers:
        lines.append(f"- task_id: {item.get('task_id', '')}")
        lines.append(f"  label: {item.get('label', '')}")
        lines.append(f"  interpret_as: {item.get('interpret_as', 'general')}")
        lines.append(f"  target: {item.get('target', 'scene')}")
        if item.get("target_person_id"):
            lines.append(f"  target_person_id: {item.get('target_person_id')}")
        if item.get("target_identity"):
            lines.append(f"  target_identity: {item.get('target_identity')}")
        lines.append(f"  question: {item.get('question', '')}")
        lines.append(f"  answer: {item.get('answer', '')}")

    response = _llm_call(
        model_config,
        label="vision_state_update",
        messages=[
            {"role": "system", "content": _load_prompt()},
            {"role": "user", "content": "\n".join(lines)},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    data = json.loads(response.choices[0].message.content)

    person_updates: list[PersonUpdate] = []
    for raw in data.get("person_updates", []):
        if not isinstance(raw, dict) or "person_id" not in raw:
            continue
        set_presence = raw.get("set_presence")
        invalid_presence = None
        if set_presence not in _VALID_PRESENCE:
            invalid_presence = set_presence if isinstance(set_presence, str) and set_presence.strip() else None
            set_presence = None
        person_updates.append(PersonUpdate(
            person_id=str(raw.get("person_id", "")).strip(),
            remove_facts=_clean_id_list(raw.get("remove_facts", []), _PERSON_FACT_ID_RE),
            add_facts=_clean_string_list(raw.get("add_facts", []), fact_text=True),
            set_name=str(raw.get("set_name", "")).strip() or None,
            set_presence=set_presence,
            invalid_presence=invalid_presence,
        ))

    update = WorldUpdate(
        remove_facts=_clean_id_list(data.get("remove_facts", []), _WORLD_FACT_ID_RE),
        add_facts=_clean_string_list(data.get("add_facts", []), fact_text=True),
        events=_clean_string_list(data.get("events", [])),
        person_updates=person_updates,
    )
    return VisionStateUpdateResult(
        update=update,
        thought=str(data.get("thought", "")).strip(),
        summary=str(data.get("summary", "")).strip(),
    )
