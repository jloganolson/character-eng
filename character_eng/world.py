from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from rich.panel import Panel
from rich.text import Text

load_dotenv()

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
CHARACTERS_DIR = PROMPTS_DIR / "characters"

MODEL = "gemini-2.0-flash"
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


# --- Data classes ---


@dataclass
class WorldUpdate:
    remove_facts: list[int] = field(default_factory=list)
    add_facts: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)


@dataclass
class ThinkResult:
    thought: str
    action: str  # "no_change" | "talk" | "emote"
    action_detail: str


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


# --- OpenAI client ---


def _make_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ["GEMINI_API_KEY"],
        base_url=BASE_URL,
    )


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

You MUST return a JSON object with exactly these keys: "remove_facts", "add_facts", "events".
"""


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

    client = _make_client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": DIRECTOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )

    data = json.loads(response.choices[0].message.content)
    return WorldUpdate(
        remove_facts=data.get("remove_facts", []),
        add_facts=data.get("add_facts", []),
        events=data.get("events", []),
    )


# --- Think LLM call ---

THINK_SYSTEM_PROMPT = """\
You are the inner monologue of a character in an interactive fiction.

You have access to the character's system prompt, the current world state, and recent conversation history. Your job is to think about what the character would be feeling, noticing, or wanting to do right now.

Return a JSON object with exactly these keys:
- "thought": A 1-3 sentence inner monologue in first person, from the character's perspective.
- "action": One of "no_change", "talk", or "emote".
  - "no_change": The character has nothing to say or do right now.
  - "talk": The character wants to say something unprompted.
  - "emote": The character wants to perform a physical action or express an emotion.
- "action_detail": If action is "talk", what the character wants to say (a brief directive, not the full dialogue). If action is "emote", what the character does. Empty string if action is "no_change".
"""


def think_call(
    system_prompt: str,
    world: WorldState | None,
    history: list[dict],
    recent_n: int = 10,
) -> ThinkResult:
    """Ask the LLM to produce an inner monologue and optional action."""
    context_parts: list[str] = []

    context_parts.append("=== CHARACTER SYSTEM PROMPT ===")
    context_parts.append(system_prompt)

    if world:
        context_parts.append("\n=== WORLD STATE ===")
        context_parts.append(world.render())

    recent = history[-recent_n:] if len(history) > recent_n else history
    if recent:
        context_parts.append("\n=== RECENT CONVERSATION ===")
        for msg in recent:
            role = msg["role"].upper()
            context_parts.append(f"{role}: {msg['content']}")

    user_message = "\n".join(context_parts)

    client = _make_client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": THINK_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.7,
        response_format={"type": "json_object"},
    )

    data = json.loads(response.choices[0].message.content)
    return ThinkResult(
        thought=data.get("thought", ""),
        action=data.get("action", "no_change"),
        action_detail=data.get("action_detail", ""),
    )
