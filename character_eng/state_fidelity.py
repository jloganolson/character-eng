from __future__ import annotations

import re

_SPECULATION_PATTERNS = (
    re.compile(r"\bperhaps\b", re.IGNORECASE),
    re.compile(r"\bpossibly\b", re.IGNORECASE),
    re.compile(r"\bmaybe\b", re.IGNORECASE),
    re.compile(r"\bmight\b", re.IGNORECASE),
    re.compile(r"\bcould be\b", re.IGNORECASE),
    re.compile(r"\bsuggest(?:ing|s)?\b", re.IGNORECASE),
    re.compile(r"\bappears to\b", re.IGNORECASE),
    re.compile(r"\bseems to\b", re.IGNORECASE),
    re.compile(r"\bestimated to be\b", re.IGNORECASE),
    re.compile(r"\bmay indicate\b", re.IGNORECASE),
)
_GROUNDED_REWRITE_PATTERNS = (
    (re.compile(r"\bappears to have\b", re.IGNORECASE), "has"),
    (re.compile(r"\bseems to have\b", re.IGNORECASE), "has"),
    (
        re.compile(
            r"\b(?:appears|seems) to be (?=(?:standing|seated|sitting|holding|wearing|drinking|looking|facing|leaning|turned|oriented)\b)",
            re.IGNORECASE,
        ),
        "is ",
    ),
)
_SPECULATIVE_CLAUSE_PATTERNS = (
    re.compile(r",\s*estimated to be [^,]+,", re.IGNORECASE),
    re.compile(r",\s*suggesting\b.*$", re.IGNORECASE),
    re.compile(r",\s*indicating\b.*$", re.IGNORECASE),
    re.compile(r",\s*possibly\b.*$", re.IGNORECASE),
    re.compile(r",\s*perhaps\b.*$", re.IGNORECASE),
    re.compile(r",\s*maybe\b.*$", re.IGNORECASE),
    re.compile(r",\s*but it can also be interpreted as\b.*$", re.IGNORECASE),
    re.compile(r",\s*which suggests\b.*$", re.IGNORECASE),
    re.compile(r",\s*which may indicate\b.*$", re.IGNORECASE),
)
_DROP_IF_STILL_SPECULATIVE_PATTERNS = (
    re.compile(r"\bcould be\b", re.IGNORECASE),
    re.compile(r"\bmight be\b", re.IGNORECASE),
    re.compile(r"\bpossibly\b", re.IGNORECASE),
    re.compile(r"\bperhaps\b", re.IGNORECASE),
    re.compile(r"\bestimated to be\b", re.IGNORECASE),
    re.compile(r"\binterpreted as\b", re.IGNORECASE),
    re.compile(r"\bappears to\b", re.IGNORECASE),
    re.compile(r"\bseems to\b", re.IGNORECASE),
    re.compile(r"\bsuggests they are\b", re.IGNORECASE),
    re.compile(r"\bsuggested by\b", re.IGNORECASE),
)


def fact_is_speculative(text: str) -> bool:
    source = str(text or "").strip()
    if not source:
        return False
    return any(pattern.search(source) for pattern in _SPECULATION_PATTERNS)


def sanitize_state_fact_text(text: str) -> str:
    cleaned = " ".join(str(text or "").strip().split())
    if not cleaned:
        return ""
    for pattern, replacement in _GROUNDED_REWRITE_PATTERNS:
        cleaned = pattern.sub(replacement, cleaned)
    for pattern in _SPECULATIVE_CLAUSE_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    cleaned = re.sub(r"\s+,", ",", cleaned)
    cleaned = re.sub(r",\s*\.", ".", cleaned)
    cleaned = re.sub(r"\.\.+", ".", cleaned)
    cleaned = cleaned.strip(" ,;")
    if any(pattern.search(cleaned) for pattern in _DROP_IF_STILL_SPECULATIVE_PATTERNS):
        return ""
    return cleaned


def sanitize_state_fact_list(facts: list[str]) -> list[str]:
    sanitized: list[str] = []
    for fact in facts:
        cleaned = sanitize_state_fact_text(fact)
        if cleaned:
            sanitized.append(cleaned)
    return sanitized
