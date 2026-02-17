from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from rich.panel import Panel
from rich.text import Text

load_dotenv()

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
CHARACTERS_DIR = PROMPTS_DIR / "characters"



# --- Data classes ---


@dataclass
class WorldUpdate:
    remove_facts: list[int] = field(default_factory=list)
    add_facts: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)


@dataclass
class GoalUpdate:
    remove_goals: list[int] = field(default_factory=list)
    add_goals: list[str] = field(default_factory=list)


@dataclass
class ThinkResult:
    thought: str
    action: str  # "no_change" | "talk" | "emote"
    action_detail: str
    gaze: str  # what the character is looking at
    expression: str  # facial expression from fixed vocab
    goal_updates: GoalUpdate | None = None


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


@dataclass
class Goals:
    static: list[str] = field(default_factory=list)
    dynamic: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines: list[str] = []
        if self.static:
            lines.append("Core drives (permanent):")
            for drive in self.static:
                lines.append(f"- {drive}")
        if self.dynamic:
            if lines:
                lines.append("")
            lines.append("Current goals:")
            for i, goal in enumerate(self.dynamic):
                lines.append(f"  {i}. {goal}")
        return "\n".join(lines)

    def apply_update(self, update: GoalUpdate) -> None:
        for idx in sorted(update.remove_goals, reverse=True):
            if 0 <= idx < len(self.dynamic):
                del self.dynamic[idx]
        self.dynamic.extend(update.add_goals)

    def show(self) -> Panel:
        lines: list[str] = []
        if self.static:
            lines.append("[bold]Core drives:[/bold]")
            for drive in self.static:
                lines.append(f"  - {drive}")
        if self.dynamic:
            if lines:
                lines.append("")
            lines.append("[bold]Current goals:[/bold]")
            for i, goal in enumerate(self.dynamic):
                lines.append(f"  [dim]{i}.[/dim] {goal}")
        body = "\n".join(lines) if lines else "[dim]No goals loaded.[/dim]"
        return Panel(body, title="Character Goals", border_style="cyan")


# --- File loading ---


def _read_lines(path: Path) -> list[str]:
    """Read a file as a list of non-empty stripped lines."""
    try:
        text = path.read_text()
    except FileNotFoundError:
        return []
    return [line.strip() for line in text.splitlines() if line.strip()]


def _load_prompt_file(filename: str) -> str:
    """Read a system prompt from prompts/ directory."""
    return (PROMPTS_DIR / filename).read_text()


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


def load_goals(character: str) -> Goals | None:
    char_dir = CHARACTERS_DIR / character
    static_path = char_dir / "goals_static.txt"
    dynamic_path = char_dir / "goals_dynamic.txt"

    if not static_path.exists() and not dynamic_path.exists():
        return None

    return Goals(
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


def _make_client(model_config: dict) -> OpenAI:
    api_key = model_config.get("api_key") or os.environ[model_config["api_key_env"]]
    return OpenAI(
        api_key=api_key,
        base_url=model_config["base_url"],
    )


# --- Director LLM call ---


def director_call(world: WorldState, change_description: str, model_config: dict) -> WorldUpdate:
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

    client = _make_client(model_config)
    response = client.chat.completions.create(
        model=model_config["model"],
        messages=[
            {"role": "system", "content": _load_prompt_file("director_system.txt")},
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


def _strip_tag(value: str, tag: str) -> str:
    """Strip wrapping like '<gaze:orb>' down to just 'orb'."""
    m = re.match(rf"<{tag}:(.+?)>", value)
    return m.group(1) if m else value


# --- Think LLM call ---


def think_call(
    system_prompt: str,
    world: WorldState | None,
    history: list[dict],
    model_config: dict,
    goals: Goals | None = None,
    recent_n: int = 10,
) -> ThinkResult:
    """Ask the LLM to produce an inner monologue and optional action."""
    context_parts: list[str] = []

    context_parts.append("=== CHARACTER SYSTEM PROMPT ===")
    context_parts.append(system_prompt)

    if world:
        context_parts.append("\n=== WORLD STATE ===")
        context_parts.append(world.render())

    if goals:
        context_parts.append("\n=== CHARACTER GOALS ===")
        context_parts.append(goals.render())

    recent = history[-recent_n:] if len(history) > recent_n else history
    if recent:
        context_parts.append("\n=== RECENT CONVERSATION ===")
        for msg in recent:
            role = msg["role"].upper()
            context_parts.append(f"{role}: {msg['content']}")

    user_message = "\n".join(context_parts)

    client = _make_client(model_config)
    response = client.chat.completions.create(
        model=model_config["model"],
        messages=[
            {"role": "system", "content": _load_prompt_file("think_system.txt")},
            {"role": "user", "content": user_message},
        ],
        temperature=0.7,
        response_format={"type": "json_object"},
    )

    data = json.loads(response.choices[0].message.content)

    goal_updates = None
    if data.get("goal_updates"):
        gu = data["goal_updates"]
        goal_updates = GoalUpdate(
            remove_goals=gu.get("remove_goals", []),
            add_goals=gu.get("add_goals", []),
        )

    return ThinkResult(
        thought=data.get("thought", ""),
        action=data.get("action", "no_change"),
        action_detail=data.get("action_detail", ""),
        gaze=_strip_tag(data.get("gaze", ""), "gaze"),
        expression=_strip_tag(data.get("expression", "neutral"), "emote"),
        goal_updates=goal_updates,
    )
