from __future__ import annotations

import re
from dataclasses import dataclass, field

from rich.panel import Panel

_FACT_ID_PREFIX_RE = re.compile(r"^(?:f\d+|p\d+f\d+)\.\s*")
_VALID_PRESENCE = {"approaching", "present", "leaving", "gone"}
_VALID_FACT_SCOPES = {"static", "ephemeral"}


def _normalized_fact_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", _FACT_ID_PREFIX_RE.sub("", str(text or "").strip()).lower()))


def _fact_text_match(a: str, b: str) -> bool:
    left = _normalized_fact_text(a)
    right = _normalized_fact_text(b)
    if not left or not right:
        return False
    return left == right or left in right or right in left


@dataclass
class PersonUpdate:
    person_id: str
    remove_facts: list[str] = field(default_factory=list)
    add_facts: list[str] = field(default_factory=list)
    fact_scope: str | None = None
    set_name: str | None = None
    set_presence: str | None = None
    invalid_presence: str | None = None


@dataclass
class Person:
    person_id: str
    name: str | None = None
    aliases: list[str] = field(default_factory=list)
    presence: str = "approaching"
    facts: dict[str, str] = field(default_factory=dict)
    fact_scopes: dict[str, str] = field(default_factory=dict)
    history: list[str] = field(default_factory=list)
    _next_fact_id: int = field(default=1, repr=False)

    def remember_alias(self, alias: str) -> bool:
        cleaned = str(alias or "").strip()
        if not cleaned:
            return False
        if not any(existing.lower() == cleaned.lower() for existing in self.aliases):
            self.aliases.append(cleaned)
            return True
        return False

    @property
    def display_name(self) -> str:
        if self.name and self.name.strip():
            return self.name
        return self.person_id

    def add_fact(self, text: str, *, scope: str = "static") -> str:
        """Assign next sequential fact ID scoped to this person, add fact, return the ID."""
        fid = f"{self.person_id}f{self._next_fact_id}"
        self._next_fact_id += 1
        self.facts[fid] = text
        self.fact_scopes[fid] = scope if scope in _VALID_FACT_SCOPES else "static"
        return fid

    def fact_scope(self, fid: str) -> str:
        return self.fact_scopes.get(fid, "static")

    def render(self) -> str:
        parts: list[str] = []
        display = self.display_name
        parts.append(f"{display} ({self.presence})")
        for fid, text in self.facts.items():
            scope = self.fact_scope(fid)
            prefix = "~ " if scope == "ephemeral" else ""
            parts.append(f"  - {prefix}{text}")
        return "\n".join(parts)

    def render_for_reconcile(self) -> str:
        lines: list[str] = []
        display = self.display_name
        lines.append(f"{self.person_id} [{display}] ({self.presence}):")
        for fid, text in self.facts.items():
            scope = self.fact_scope(fid)
            suffix = " [ephemeral]" if scope == "ephemeral" else ""
            lines.append(f"  {fid}. {text}{suffix}")
        return "\n".join(lines)


@dataclass
class PeopleState:
    people: dict[str, Person] = field(default_factory=dict)
    _next_id: int = field(default=1, repr=False)

    def add_person(self, name: str | None = None, presence: str = "approaching") -> Person:
        pid = f"p{self._next_id}"
        self._next_id += 1
        person = Person(person_id=pid, name=name, presence=presence)
        person.remember_alias(name or "")
        self.people[pid] = person
        return person

    def get_or_create(self, name: str) -> str:
        """Find person by name or create a new one. Returns person_id."""
        person = self.find_by_name(name)
        if person is not None:
            return person.person_id
        return self.add_person(name=name, presence="present").person_id

    def find_by_name(self, name: str) -> Person | None:
        cleaned = str(name or "").strip().lower()
        if not cleaned:
            return None
        for person in self.people.values():
            if person.name and person.name.lower() == cleaned:
                return person
            if any(alias.lower() == cleaned for alias in person.aliases):
                return person
        return None

    def present_people(self) -> list[Person]:
        return [p for p in self.people.values() if p.presence in ("approaching", "present")]

    def render(self) -> str:
        if not self.people:
            return ""
        parts: list[str] = []
        for person in self.people.values():
            if person.presence == "gone":
                continue
            parts.append(person.render())
        return "\n".join(parts)

    def render_for_reconcile(self) -> str:
        if not self.people:
            return ""
        parts: list[str] = []
        for person in self.people.values():
            if person.presence == "gone":
                continue
            parts.append(person.render_for_reconcile())
        return "\n".join(parts)

    def _find_matching_fact_id(self, person: Person, text: str) -> str | None:
        for fid, existing in person.facts.items():
            if _fact_text_match(text, existing):
                return fid
        return None

    def apply_updates(self, updates: list[PersonUpdate]) -> None:
        for update in updates:
            person = self.people.get(update.person_id)
            if person is None:
                continue
            for fid in update.remove_facts:
                person.facts.pop(fid, None)
                person.fact_scopes.pop(fid, None)
            scope = update.fact_scope if update.fact_scope in _VALID_FACT_SCOPES else "static"
            cleaned_facts: list[str] = []
            for text in update.add_facts:
                cleaned = _FACT_ID_PREFIX_RE.sub("", text.strip())
                if cleaned and not any(_fact_text_match(cleaned, existing) for existing in cleaned_facts):
                    cleaned_facts.append(cleaned)
            if scope == "ephemeral" and cleaned_facts:
                keep_ids = {
                    fid
                    for text in cleaned_facts
                    for fid in ([self._find_matching_fact_id(person, text)] if self._find_matching_fact_id(person, text) else [])
                }
                for fid in [fid for fid in list(person.facts) if person.fact_scope(fid) == "ephemeral" and fid not in keep_ids]:
                    person.facts.pop(fid, None)
                    person.fact_scopes.pop(fid, None)
            for cleaned in cleaned_facts:
                matched_id = self._find_matching_fact_id(person, cleaned)
                if matched_id is not None:
                    if scope == "static" and person.fact_scope(matched_id) != "static":
                        person.fact_scopes[matched_id] = "static"
                    continue
                person.add_fact(cleaned, scope=scope)
            if update.set_name is not None:
                person.remember_alias(update.set_name)
                person.name = update.set_name
            if update.set_presence in _VALID_PRESENCE:
                person.presence = update.set_presence
            elif update.invalid_presence:
                person.history.append(f"ignored invalid presence: {update.invalid_presence}")

    def show(self) -> Panel:
        lines: list[str] = []
        if not self.people:
            body = "[dim]No people tracked.[/dim]"
        else:
            for person in self.people.values():
                display = person.display_name
                lines.append(f"[bold]{display}[/bold] [dim]({person.person_id})[/dim] — {person.presence}")
                for fid, text in person.facts.items():
                    scope = person.fact_scope(fid)
                    suffix = " [ephemeral]" if scope == "ephemeral" else ""
                    lines.append(f"  [dim]{fid}.[/dim] {text}{suffix}")
                if person.history:
                    for h in person.history:
                        lines.append(f"  [dim italic]{h}[/dim italic]")
            body = "\n".join(lines)
        return Panel(body, title="People", border_style="blue")
