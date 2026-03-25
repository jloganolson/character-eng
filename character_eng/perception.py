from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from character_eng.person import PersonUpdate
from character_eng.world import (
    WorldUpdate,
    _clean_optional_text,
    format_narrator_message,
    format_pending_narrator,
)

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
CHARACTERS_DIR = PROMPTS_DIR / "characters"


@dataclass
class PerceptionEvent:
    description: str
    source: str = "manual"  # "manual", "sim", "persona"
    timestamp: float = field(default_factory=time.time)
    trace: dict = field(default_factory=dict)
    kind: str = "free_text"
    payload: dict = field(default_factory=dict)

def _append_world_event(world, text: str, limit: int = 20) -> None:
    if world is None:
        return
    cleaned = (text or "").strip()
    if not cleaned:
        return
    if not world.events or world.events[-1] != cleaned:
        world.events.append(cleaned)
        if len(world.events) > limit:
            world.events[:] = world.events[-limit:]


def _apply_person_presence(event: PerceptionEvent, people, world) -> None:
    if people is None:
        return
    payload = event.payload or {}
    identity = str(payload.get("identity") or event.trace.get("identity") or "").strip()
    person_id = str(payload.get("person_id") or "").strip()
    presence = str(payload.get("presence") or "").strip() or "present"

    person = None
    if person_id and person_id in people.people:
        person = people.people[person_id]
    elif identity:
        resolved_id = people.get_or_create(identity)
        person = people.people[resolved_id]
    if person is None:
        person = people.add_person(name=identity or None, presence=presence)

    if identity and not person.name:
        person.name = identity
    person.presence = presence

    history_line = f"visual {payload.get('change', 'update')}: {event.description}"
    if not person.history or person.history[-1] != history_line:
        person.history.append(history_line)
    _append_world_event(world, event.description)


def _payload_world_update(event: PerceptionEvent) -> WorldUpdate:
    payload = event.payload or {}
    person_updates = []
    for raw in payload.get("person_updates", []) or []:
        if not isinstance(raw, dict):
            continue
        person_updates.append(PersonUpdate(
            person_id=str(raw.get("person_id", "")).strip(),
            remove_facts=list(raw.get("remove_facts", []) or []),
            add_facts=list(raw.get("add_facts", []) or []),
            fact_scope=_clean_optional_text(raw.get("fact_scope")),
            set_name=_clean_optional_text(raw.get("set_name")),
            set_presence=_clean_optional_text(raw.get("set_presence")),
        ))
    return WorldUpdate(
        remove_facts=list(payload.get("remove_facts", []) or []),
        add_facts=list(payload.get("add_facts", []) or []),
        events=list(payload.get("events", []) or []),
        person_updates=person_updates,
    )


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _is_generic_presence_claim(event: PerceptionEvent) -> bool:
    payload = event.payload or {}
    signals = {
        str(item or "").strip().lower()
        for item in (payload.get("signals", []) or [])
        if str(item or "").strip()
    }
    if signals and signals.issubset({"person_visible", "person_departed"}):
        return True

    normalized = _normalize_text(event.description)
    return normalized in {
        "a person appeared in view",
        "a person has appeared in view",
        "a person entered the room",
        "a person has entered the room",
        "a person is standing nearby",
        "a person is standing in the room",
        "a person is now visible",
        "a person is now visible in the room",
        "a person is no longer visible",
    }


def _filtered_world_update(event: PerceptionEvent) -> WorldUpdate:
    update = _payload_world_update(event)
    update.events = [item for item in update.events if not _is_generic_presence_claim(PerceptionEvent(description=item, payload=event.payload, kind="visual_claim"))]
    return update


def process_perception(event: PerceptionEvent, people, world) -> tuple[None, str]:
    """Apply direct perception updates when possible, else queue for reconcile."""
    if event.kind == "person_presence":
        _apply_person_presence(event, people, world)
    elif event.kind == "visual_claim" and _is_generic_presence_claim(event):
        return (None, "")
    elif event.kind == "vision_state_update":
        update = _filtered_world_update(event)
        if not bool((event.payload or {}).get("already_applied")):
            if world is not None:
                world.apply_update(update)
            if people is not None and update.person_updates:
                people.apply_updates(update.person_updates)
        narrator_msg = format_pending_narrator(event.description or format_narrator_message(update))
        return (None, narrator_msg)
    else:
        world.add_pending(event.description)
    narrator_msg = format_pending_narrator(event.description)
    return (None, narrator_msg)


@dataclass
class SimEvent:
    time_offset: float  # seconds from start
    description: str


@dataclass
class SimScript:
    events: list[SimEvent] = field(default_factory=list)
    name: str = ""


def load_sim_script(character: str, sim_name: str) -> SimScript:
    """Load a sim script from prompts/characters/{name}/sims/{sim_name}.sim.txt.

    Format: time_offset | description per line.
    Lines starting with # are comments. Blank lines are skipped.
    Quoted descriptions (starting with ") are treated as dialogue.
    """
    path = CHARACTERS_DIR / character / "sims" / f"{sim_name}.sim.txt"
    text = path.read_text()

    events: list[SimEvent] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "|" not in stripped:
            continue
        offset_str, desc = stripped.split("|", 1)
        try:
            offset = float(offset_str.strip())
        except ValueError:
            continue
        events.append(SimEvent(time_offset=offset, description=desc.strip()))

    return SimScript(events=events, name=sim_name)
