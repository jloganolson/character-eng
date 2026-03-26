from __future__ import annotations

import re

_NAME_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")
_SELF_IDENTIFY_PATTERNS = (
    re.compile(r"\bmy name is\s+(?P<name>[A-Za-z][A-Za-z'\-\s]{0,40})", re.IGNORECASE),
    re.compile(r"\bi am\s+(?P<name>[A-Za-z][A-Za-z'\-\s]{0,40})", re.IGNORECASE),
    re.compile(r"\bi'm\s+(?P<name>[A-Za-z][A-Za-z'\-\s]{0,40})", re.IGNORECASE),
    re.compile(r"\bcall me\s+(?P<name>[A-Za-z][A-Za-z'\-\s]{0,40})", re.IGNORECASE),
)
_EXPLICIT_NAME_EVIDENCE_TEMPLATES = (
    r"\bmy name is {name}\b",
    r"\bname is {name}\b",
    r"\bnamed {name}\b",
    r"\bcalled {name}\b",
    r"\bname tag (?:says|said|reads|reading)\s+{name}\b",
    r"\bbadge (?:says|said|reads|reading)\s+{name}\b",
)
_DISALLOWED_SINGLE_TOKEN_NAMES = {
    "a", "an", "and", "are", "back", "be", "because", "busy", "but", "done",
    "fine", "good", "great", "here", "hungry", "i", "im", "it", "its", "just",
    "late", "man", "me", "mine", "my", "no", "not", "nothing", "okay", "ok",
    "ready", "right", "same", "so", "something", "sure", "that", "the", "there",
    "they", "this", "tired", "uh", "um", "we", "well", "yeah", "yep", "you",
}


def normalize_person_name(text: str) -> str | None:
    tokens = _NAME_TOKEN_RE.findall(str(text or "").strip())
    if not tokens:
        return None
    tokens = tokens[:3]
    lowered = [token.lower() for token in tokens]
    if len(tokens) == 1 and lowered[0] in _DISALLOWED_SINGLE_TOKEN_NAMES:
        return None
    return " ".join(token[:1].upper() + token[1:] for token in tokens)


def extract_self_identified_name(text: str) -> str | None:
    source = str(text or "").strip()
    if not source:
        return None
    for pattern in _SELF_IDENTIFY_PATTERNS:
        match = pattern.search(source)
        if not match:
            continue
        candidate = normalize_person_name(match.group("name"))
        if candidate:
            return candidate
    return None


def has_explicit_name_evidence(name: str, texts: list[str] | tuple[str, ...]) -> bool:
    candidate = normalize_person_name(name)
    if not candidate:
        return False
    escaped = re.escape(candidate)
    patterns = [
        re.compile(template.format(name=escaped), re.IGNORECASE)
        for template in _EXPLICIT_NAME_EVIDENCE_TEMPLATES
    ]
    for text in texts:
        source = str(text or "").strip()
        if not source:
            continue
        if any(pattern.search(source) for pattern in patterns):
            return True
    return False


def apply_transcript_name_to_people(people, transcript: str) -> tuple[str | None, str | None]:
    if people is None:
        return (None, None)
    candidate = extract_self_identified_name(transcript)
    if not candidate:
        return (None, None)

    present = people.present_people()
    if not present:
        return (None, None)

    target = next((person for person in present if (person.name or "").strip().lower() == candidate.lower()), None)
    if target is None and len(present) == 1:
        target = present[0]
    if target is None:
        target = next((person for person in present if not (person.name or "").strip()), None)
    if target is None:
        return (None, None)

    prior = str(target.name or "").strip()
    target.remember_alias(candidate)
    target.name = candidate
    if prior == candidate:
        return (None, candidate)
    return (target.person_id, candidate)
