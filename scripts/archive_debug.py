#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from character_eng.name_memory import extract_self_identified_name

HISTORY_ROOT = ROOT / "history"
SEARCH_BUCKETS = ("moments", "pinned", "sessions")


def resolve_archive(ref: str) -> Path:
    candidate = Path(ref).expanduser()
    if candidate.exists():
        return candidate.resolve()
    for bucket in SEARCH_BUCKETS:
        path = HISTORY_ROOT / bucket / ref
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(f"archive not found: {ref}")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_events(path: Path) -> list[dict]:
    events: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        events.append(json.loads(line))
    return events


def people_names(payload: dict) -> dict[str, str]:
    result: dict[str, str] = {}
    for person in (payload.get("people", []) or []):
        person_id = str(person.get("id") or person.get("person_id") or "").strip()
        if not person_id:
            continue
        result[person_id] = str(person.get("name") or "").strip()
    return result


def print_dialogue_timeline(events: list[dict]) -> None:
    print("Dialogue timeline:")
    printed = False
    for event in events:
        event_type = str(event.get("type") or "")
        data = dict(event.get("data") or {})
        seq = event.get("seq", "?")
        if event_type == "user_transcript_final":
            text = str(data.get("text") or "").strip()
            print(f"  {seq}: user  {text}")
            name = extract_self_identified_name(text)
            if name:
                print(f"      self-identification: {name}")
            printed = True
        elif event_type == "response_done":
            text = str(data.get("full_text") or data.get("response") or "").strip()
            if text:
                print(f"  {seq}: greg  {text}")
                printed = True
    if not printed:
        print("  (no dialogue events)")


def print_name_timeline(events: list[dict]) -> None:
    print("Name/state timeline:")
    printed = False
    last_people: dict[str, str] = {}
    for event in events:
        event_type = str(event.get("type") or "")
        data = dict(event.get("data") or {})
        seq = event.get("seq", "?")

        if event_type in {"vision_state_update", "reconcile"}:
            for update in (data.get("person_updates") or []):
                name = str(update.get("set_name") or "").strip()
                if name:
                    print(f"  {seq}: {event_type} set_name {update.get('person_id')} -> {name}")
                    printed = True
            continue

        if event_type == "people_state":
            current = people_names(data)
            for person_id, name in current.items():
                if last_people.get(person_id, "") != name:
                    label = name or "(unset)"
                    print(f"  {seq}: people_state {person_id} -> {label}")
                    printed = True
            last_people = current
    if not printed:
        print("  (no name transitions)")


def print_event_counts(events: list[dict]) -> None:
    counts = Counter(str(event.get("type") or "<missing>") for event in events)
    print("Event counts:")
    for event_type, count in counts.most_common(12):
        print(f"  {count:>3} {event_type}")


def print_recommendation(issue: str) -> None:
    lowered = issue.lower()
    print("Recommended next pass:")
    if any(token in lowered for token in ("name", "naming", "called", "who are you", "state")):
        print("  - Check whether name first went wrong in vision, reconcile, or transcript promotion.")
        print("  - Convert the earliest bad state mutation into a vision or transcript fixture.")
        print("  - Add a deterministic guard before touching prompts.")
        return
    if any(token in lowered for token in ("stt", "transcript", "heard", "speech", "audio")):
        print("  - Verify transcript text first, then inspect audio only if the text looks suspect.")
        print("  - If transcript is right but behavior is wrong, move immediately to prompt/state analysis.")
        return
    if any(token in lowered for token in ("vision", "see", "saw", "camera", "visual")):
        print("  - Compare raw VLM answers, vision_state_update, and resulting people/world state.")
        print("  - Pull media frames only if raw answers are ambiguous or clearly implausible.")
        return
    print("  - Start with transcript/reply/state timeline.")
    print("  - Identify the earliest incorrect event and turn that into an offline fixture.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Quick archive-first triage for exported sessions.")
    parser.add_argument("archive", help="Archive id, session id, or absolute path")
    parser.add_argument("issue", nargs="?", default="", help="Short issue summary")
    args = parser.parse_args()

    archive_dir = resolve_archive(args.archive)
    manifest = load_json(archive_dir / "manifest.json")
    events = load_events(archive_dir / "events.jsonl")

    print(f"Archive: {archive_dir}")
    print(f"Title:   {manifest.get('title', archive_dir.name)}")
    print(f"Type:    {manifest.get('type', '')}")
    print(f"Issue:   {args.issue or '(none provided)'}")
    print()

    print_event_counts(events)
    print()
    print_dialogue_timeline(events)
    print()
    print_name_timeline(events)
    print()
    print_recommendation(args.issue or "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
