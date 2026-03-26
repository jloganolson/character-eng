from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from character_eng.creative import load_prompt_asset
from character_eng.name_memory import has_explicit_name_evidence
from character_eng.state_fidelity import sanitize_state_fact_list
from character_eng.world import (
    PersonUpdate,
    WorldUpdate,
    _PERSON_FACT_ID_RE,
    _VALID_PRESENCE,
    _WORLD_FACT_ID_RE,
    _clean_optional_text,
    _clean_id_list,
    _clean_string_list,
    _fact_text_match,
)

PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"


@dataclass
class VisionStateUpdateResult:
    update: WorldUpdate = field(default_factory=WorldUpdate)
    thought: str = ""
    summary: str = ""


def _load_prompt() -> str:
    return load_prompt_asset("vision_state_update", prompts_dir=PROMPTS_DIR, default_filename="vision_state_update.txt")


def _looks_person_fact(text: str, *, target_identity: str = "") -> bool:
    cleaned = " ".join(str(text or "").strip().lower().split())
    if not cleaned:
        return False
    if target_identity and cleaned.startswith(target_identity.strip().lower()):
        return True
    return cleaned.startswith((
        "person ",
        "the person ",
        "the person is",
        "the visible person",
        "visible person",
        "his ",
        "her ",
        "their ",
        "he ",
        "she ",
        "they ",
    ))


def _normalize_person_fact_updates(update: WorldUpdate, task_answers: list[dict]) -> WorldUpdate:
    target_ids = {
        str(item.get("target_person_id", "")).strip()
        for item in task_answers
        if str(item.get("target_person_id", "")).strip()
    }
    target_identity = next((
        str(item.get("target_identity", "")).strip()
        for item in task_answers
        if str(item.get("target_identity", "")).strip()
    ), "")

    person_updates_by_id = {item.person_id: item for item in update.person_updates}
    normalized_world_facts: list[str] = []

    for fact in update.add_facts:
        if any(_fact_text_match(fact, existing) for item in update.person_updates for existing in item.add_facts):
            continue
        if len(target_ids) == 1 and _looks_person_fact(fact, target_identity=target_identity):
            target_id = next(iter(target_ids))
            person_update = person_updates_by_id.get(target_id)
            if person_update is None:
                is_dynamic = any(
                    str(item.get("interpret_as", "")).strip() == "person_description_dynamic"
                    and str(item.get("target_person_id", "")).strip() == target_id
                    for item in task_answers
                )
                person_update = PersonUpdate(
                    person_id=target_id,
                    fact_scope="ephemeral" if is_dynamic else "static",
                )
                update.person_updates.append(person_update)
                person_updates_by_id[target_id] = person_update
            if not any(_fact_text_match(fact, existing) for existing in person_update.add_facts):
                person_update.add_facts.append(fact)
            person_update.fact_scope = person_update.fact_scope or "static"
            continue
        normalized_world_facts.append(fact)

    update.add_facts = normalized_world_facts
    update.person_updates = [
        item
        for item in update.person_updates
        if item.remove_facts or item.add_facts or item.set_name is not None or item.set_presence is not None
    ]
    return update


def _sanitize_vision_fact_updates(update: WorldUpdate) -> WorldUpdate:
    update.add_facts = sanitize_state_fact_list(update.add_facts)

    sanitized_person_updates: list[PersonUpdate] = []
    for person_update in update.person_updates:
        person_update.add_facts = sanitize_state_fact_list(person_update.add_facts)
        if person_update.remove_facts or person_update.add_facts or person_update.set_name is not None or person_update.set_presence is not None:
            sanitized_person_updates.append(person_update)
    update.person_updates = sanitized_person_updates
    return update


def _strip_unsupported_vision_names(update: WorldUpdate, task_answers: list[dict], people) -> WorldUpdate:
    answer_texts = [str(item.get("answer", "")).strip() for item in task_answers]
    people_by_id = {} if people is None else dict(getattr(people, "people", {}) or {})
    for person_update in update.person_updates:
        if person_update.set_name is None:
            continue
        candidate = str(person_update.set_name).strip()
        if not candidate:
            person_update.set_name = None
            continue
        existing = people_by_id.get(person_update.person_id)
        existing_name = str(getattr(existing, "name", "") or "").strip()
        if existing_name and existing_name.lower() == candidate.lower():
            continue
        if not has_explicit_name_evidence(candidate, answer_texts):
            person_update.set_name = None
    update.person_updates = [
        item
        for item in update.person_updates
        if item.remove_facts or item.add_facts or item.set_name is not None or item.set_presence is not None
    ]
    return update


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
            fact_scope=_clean_optional_text(raw.get("fact_scope")),
            set_name=_clean_optional_text(raw.get("set_name")),
            set_presence=set_presence,
            invalid_presence=invalid_presence,
        ))

    update = WorldUpdate(
        remove_facts=_clean_id_list(data.get("remove_facts", []), _WORLD_FACT_ID_RE),
        add_facts=_clean_string_list(data.get("add_facts", []), fact_text=True),
        events=_clean_string_list(data.get("events", [])),
        person_updates=person_updates,
    )
    update = _normalize_person_fact_updates(update, task_answers)
    update = _sanitize_vision_fact_updates(update)
    update = _strip_unsupported_vision_names(update, task_answers, people)
    return VisionStateUpdateResult(
        update=update,
        thought=str(data.get("thought", "")).strip(),
        summary=str(data.get("summary", "")).strip(),
    )
