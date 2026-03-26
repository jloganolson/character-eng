#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from character_eng.name_memory import extract_self_identified_name


def _load_events(path: Path) -> list[dict]:
    events: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        events.append(json.loads(line))
    return events


def _people_names(payload: dict) -> dict[str, str]:
    result: dict[str, str] = {}
    for person in (payload.get("people", []) or []):
        person_id = str(person.get("id") or person.get("person_id") or "").strip()
        if not person_id:
            continue
        result[person_id] = str(person.get("name") or "").strip()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Print a name/state timeline from an exported archive.")
    parser.add_argument("archive", type=Path, help="Path to an archive directory with events.jsonl")
    args = parser.parse_args()

    events_path = args.archive / "events.jsonl"
    if not events_path.exists():
        raise SystemExit(f"missing events.jsonl: {events_path}")

    events = _load_events(events_path)
    last_people: dict[str, str] = {}

    for index, event in enumerate(events, start=1):
        event_type = str(event.get("type") or "")
        data = dict(event.get("data") or {})
        seq = event.get("seq", index)

        if event_type == "user_transcript_final":
            text = str(data.get("text") or "").strip()
            extracted = extract_self_identified_name(text)
            print(f"{seq:>4} user_transcript_final: {text}")
            if extracted:
                print(f"      self-identified-name: {extracted}")
            continue

        if event_type == "response_done":
            text = str(data.get("full_text") or data.get("response") or "").strip()
            if text:
                print(f"{seq:>4} response_done: {text}")
            continue

        if event_type in {"vision_state_update", "reconcile"}:
            updates = list(data.get("person_updates") or [])
            named = [item for item in updates if str(item.get("set_name") or "").strip()]
            for item in named:
                print(f"{seq:>4} {event_type}: set_name {item.get('person_id')} -> {item.get('set_name')}")
            continue

        if event_type == "people_state":
            current = _people_names(data)
            changed = []
            for person_id, name in current.items():
                if last_people.get(person_id, "") != name:
                    changed.append((person_id, name))
            for person_id, name in changed:
                print(f"{seq:>4} people_state: {person_id} name -> {name or '(unset)'}")
            last_people = current

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
