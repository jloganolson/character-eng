from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from character_eng.world import format_pending_narrator

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


def process_perception(event: PerceptionEvent, people, world) -> tuple[None, str]:
    """Apply direct perception updates when possible, else queue for reconcile."""
    if event.kind == "person_presence":
        _apply_person_presence(event, people, world)
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
