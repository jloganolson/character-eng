from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from character_eng.name_memory import has_explicit_name_evidence
from character_eng.state_fidelity import fact_is_speculative
_IGNORE_CAPITALIZED = {
    "A", "An", "The", "Person", "People", "Visitor", "World", "State",
}


@dataclass(frozen=True)
class ArchiveIssue:
    seq: int
    event_type: str
    reason: str
    detail: str


def load_archive_events(path: Path) -> list[dict]:
    events: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        events.append(json.loads(line))
    return events
def _summary_subject_candidate(text: str) -> str:
    match = re.search(r"\bfor\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b", str(text or ""))
    if not match:
        return ""
    candidate = match.group(1).strip()
    if candidate in _IGNORE_CAPITALIZED:
        return ""
    return candidate


def _vision_task_answers(event: dict) -> list[dict]:
    return list((event.get("data") or {}).get("task_answers") or [])


def _vision_answer_texts(event: dict) -> list[str]:
    return [str(item.get("answer", "")).strip() for item in _vision_task_answers(event)]


def find_archive_state_issues(events: list[dict]) -> list[ArchiveIssue]:
    issues: list[ArchiveIssue] = []
    for event in events:
        event_type = str(event.get("type") or "")
        data = dict(event.get("data") or {})
        seq = int(event.get("seq") or 0)

        if event_type == "vision_state_update":
            answers = _vision_answer_texts(event)
            for update in data.get("person_updates") or []:
                name = str(update.get("set_name") or "").strip()
                if name and not has_explicit_name_evidence(name, answers):
                    issues.append(ArchiveIssue(
                        seq=seq,
                        event_type=event_type,
                        reason="unsupported_set_name",
                        detail=f"{update.get('person_id')} -> {name}",
                    ))
                for fact in update.get("add_facts") or []:
                    if fact_is_speculative(fact):
                        issues.append(ArchiveIssue(
                            seq=seq,
                            event_type=event_type,
                            reason="speculative_person_fact",
                            detail=str(fact),
                        ))
            for fact in data.get("add_facts") or []:
                if fact_is_speculative(fact):
                    issues.append(ArchiveIssue(
                        seq=seq,
                        event_type=event_type,
                        reason="speculative_world_fact",
                        detail=str(fact),
                    ))
            summary = str(data.get("summary") or "").strip()
            if summary:
                candidate = _summary_subject_candidate(summary)
                identities = {
                    str(item.get("target_identity") or "").strip()
                    for item in _vision_task_answers(event)
                    if str(item.get("target_identity") or "").strip()
                }
                if candidate and identities and candidate not in identities:
                    issues.append(ArchiveIssue(
                        seq=seq,
                        event_type=event_type,
                        reason="summary_name_mismatch",
                        detail=summary,
                    ))

        if event_type == "reconcile":
            for fact in data.get("add_facts") or []:
                if fact_is_speculative(fact):
                    issues.append(ArchiveIssue(
                        seq=seq,
                        event_type=event_type,
                        reason="speculative_reconcile_fact",
                        detail=str(fact),
                    ))
    return issues
