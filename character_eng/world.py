from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from rich.panel import Panel
from rich.text import Text

load_dotenv()

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
CHARACTERS_DIR = PROMPTS_DIR / "characters"

MODEL = "gemini-2.0-flash"


# --- Data classes ---


@dataclass
class WorldUpdate:
    remove_facts: list[int] = field(default_factory=list)
    add_facts: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)


@dataclass
class WorldState:
    static: list[str] = field(default_factory=list)
    dynamic: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines: list[str] = []
        if self.static:
            lines.append("Permanent facts:")
            for fact in self.static:
                lines.append(f"- {fact}")
        if self.dynamic:
            if lines:
                lines.append("")
            lines.append("Current state:")
            for fact in self.dynamic:
                lines.append(f"- {fact}")
        if self.events:
            if lines:
                lines.append("")
            lines.append("Recent events:")
            for event in self.events:
                lines.append(f"- {event}")
        return "\n".join(lines)

    def apply_update(self, update: WorldUpdate) -> None:
        # Remove facts by index (highest first so indices stay valid)
        for idx in sorted(update.remove_facts, reverse=True):
            if 0 <= idx < len(self.dynamic):
                del self.dynamic[idx]
        self.dynamic.extend(update.add_facts)
        self.events.extend(update.events)

    def show(self) -> Panel:
        lines: list[str] = []
        if self.static:
            lines.append("[bold]Permanent facts:[/bold]")
            for fact in self.static:
                lines.append(f"  - {fact}")
        if self.dynamic:
            if lines:
                lines.append("")
            lines.append("[bold]Current state:[/bold]")
            for i, fact in enumerate(self.dynamic):
                lines.append(f"  [dim]{i}.[/dim] {fact}")
        if self.events:
            if lines:
                lines.append("")
            lines.append("[bold]Event log:[/bold]")
            for event in self.events:
                lines.append(f"  - {event}")
        body = "\n".join(lines) if lines else "[dim]No world state loaded.[/dim]"
        return Panel(body, title="World State", border_style="yellow")


# --- File loading ---


def _read_lines(path: Path) -> list[str]:
    """Read a file as a list of non-empty stripped lines."""
    try:
        text = path.read_text()
    except FileNotFoundError:
        return []
    return [line.strip() for line in text.splitlines() if line.strip()]


def load_world_state(character: str) -> WorldState | None:
    char_dir = CHARACTERS_DIR / character
    static_path = char_dir / "world_static.txt"
    dynamic_path = char_dir / "world_dynamic.txt"

    if not static_path.exists() and not dynamic_path.exists():
        return None

    return WorldState(
        static=_read_lines(static_path),
        dynamic=_read_lines(dynamic_path),
    )


# --- Narrator formatting ---


def format_narrator_message(update: WorldUpdate) -> str:
    parts: list[str] = []
    parts.extend(update.events)
    for fact in update.add_facts:
        parts.append(fact)
    return "[" + " ".join(parts) + "]"


# --- Director LLM call ---

DIRECTOR_SYSTEM_PROMPT = """\
You are a world-state director for an interactive fiction engine.

Given the current world state and a description of a change, return a JSON object that updates the world state.

Rules:
- remove_facts: list of integer indices of dynamic facts to remove (because they are no longer true)
- add_facts: list of new dynamic fact strings to add (short, present-tense)
- events: list of past-tense event descriptions (what just happened)
- Only reference valid indices from the numbered dynamic facts list
- Keep facts concise — one clause each
- Events should be vivid but brief
"""

WORLD_UPDATE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "remove_facts": {
            "type": "ARRAY",
            "items": {"type": "INTEGER"},
        },
        "add_facts": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
        },
        "events": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
        },
    },
    "required": ["remove_facts", "add_facts", "events"],
}


def director_call(world: WorldState, change_description: str) -> WorldUpdate:
    """Ask the director LLM to produce a WorldUpdate from a change description."""
    # Build the user message with numbered dynamic facts
    lines: list[str] = []
    if world.static:
        lines.append("Permanent facts (cannot be changed):")
        for fact in world.static:
            lines.append(f"- {fact}")
        lines.append("")
    if world.dynamic:
        lines.append("Current dynamic facts:")
        for i, fact in enumerate(world.dynamic):
            lines.append(f"  {i}. {fact}")
        lines.append("")
    if world.events:
        lines.append("Recent events:")
        for event in world.events:
            lines.append(f"- {event}")
        lines.append("")
    lines.append(f"Change: {change_description}")

    user_message = "\n".join(lines)

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    response = client.models.generate_content(
        model=MODEL,
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=DIRECTOR_SYSTEM_PROMPT,
            temperature=0.2,
            response_mime_type="application/json",
            response_schema=WORLD_UPDATE_SCHEMA,
        ),
    )

    data = json.loads(response.text)
    return WorldUpdate(
        remove_facts=data.get("remove_facts", []),
        add_facts=data.get("add_facts", []),
        events=data.get("events", []),
    )
