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
    remove_facts: list[str] = field(default_factory=list)
    add_facts: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)


@dataclass
class Beat:
    line: str  # pre-rendered dialogue line
    intent: str  # what this beat is trying to accomplish
    gaze: str = ""  # where the character looks (e.g. "orb", "visitor")
    expression: str = ""  # facial expression (e.g. "curious", "excited")
    condition: str = ""  # natural language precondition (empty = auto-deliver)


@dataclass
class ConditionResult:
    met: bool
    idle: str = ""  # in-character idle line when not met
    gaze: str = ""
    expression: str = ""


@dataclass
class Script:
    beats: list[Beat] = field(default_factory=list)
    _index: int = field(default=0, repr=False)

    @property
    def current_beat(self) -> Beat | None:
        if 0 <= self._index < len(self.beats):
            return self.beats[self._index]
        return None

    def advance(self) -> None:
        self._index += 1

    def replace(self, beats: list[Beat]) -> None:
        self.beats = beats
        self._index = 0

    def is_empty(self) -> bool:
        return len(self.beats) == 0

    def render(self) -> str:
        if not self.beats:
            return ""
        lines: list[str] = []
        for i, beat in enumerate(self.beats):
            marker = "→" if i == self._index else " "
            extras = ""
            if beat.gaze or beat.expression:
                parts = []
                if beat.gaze:
                    parts.append(beat.gaze)
                if beat.expression:
                    parts.append(beat.expression)
                extras = f" ({', '.join(parts)})"
            cond = f" IF: {beat.condition}" if beat.condition else ""
            lines.append(f"{marker} {i}. [{beat.intent}] \"{beat.line}\"{extras}{cond}")
        return "\n".join(lines)

    def show(self) -> Panel:
        body = self.render() if self.beats else "[dim]No script loaded.[/dim]"
        return Panel(body, title="Script", border_style="magenta")


@dataclass
class EvalResult:
    thought: str
    gaze: str
    expression: str
    script_status: str  # "advance" | "hold" | "off_book" | "bootstrap"
    bootstrap_line: str = ""
    bootstrap_intent: str = ""
    plan_request: str = ""


@dataclass
class PlanResult:
    beats: list[Beat] = field(default_factory=list)


@dataclass
class WorldState:
    static: list[str] = field(default_factory=list)
    dynamic: dict[str, str] = field(default_factory=dict)
    events: list[str] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)
    _next_id: int = field(default=1, repr=False)

    def add_fact(self, text: str) -> str:
        """Assign next sequential ID, add fact, return the ID."""
        fid = f"f{self._next_id}"
        self._next_id += 1
        self.dynamic[fid] = text
        return fid

    def add_pending(self, text: str) -> None:
        """Append an unreconciled change description."""
        self.pending.append(text)

    def clear_pending(self) -> list[str]:
        """Pop and return all pending changes."""
        result = list(self.pending)
        self.pending.clear()
        return result

    def render_for_reconcile(self) -> str:
        """Show ID-tagged facts for reconciler context."""
        lines: list[str] = []
        for fid, text in self.dynamic.items():
            lines.append(f"{fid}. {text}")
        return "\n".join(lines)

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
            for text in self.dynamic.values():
                lines.append(f"- {text}")
        if self.pending:
            if lines:
                lines.append("")
            lines.append("Pending changes:")
            for change in self.pending:
                lines.append(f"- {change}")
        if self.events:
            if lines:
                lines.append("")
            lines.append("Recent events:")
            for event in self.events:
                lines.append(f"- {event}")
        return "\n".join(lines)

    def apply_update(self, update: WorldUpdate) -> None:
        for fid in update.remove_facts:
            self.dynamic.pop(fid, None)
        for text in update.add_facts:
            self.add_fact(text)
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
            for fid, text in self.dynamic.items():
                lines.append(f"  [dim]{fid}.[/dim] {text}")
        if self.pending:
            if lines:
                lines.append("")
            lines.append("[bold yellow]Pending changes:[/bold yellow]")
            for change in self.pending:
                lines.append(f"  [yellow]- {change}[/yellow]")
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
    long_term: str = ""

    def render(self) -> str:
        if self.long_term:
            return f"Long-term goal: {self.long_term}"
        return ""

    def show(self) -> Panel:
        if self.long_term:
            body = f"[bold]Long-term goal:[/bold] {self.long_term}"
        else:
            body = "[dim]No goals loaded.[/dim]"
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

    ws = WorldState(static=_read_lines(static_path))
    for line in _read_lines(dynamic_path):
        ws.add_fact(line)
    return ws


def load_goals(character: str) -> Goals | None:
    """Parse Long-term goal line from character.txt."""
    char_dir = CHARACTERS_DIR / character
    char_path = char_dir / "character.txt"
    try:
        text = char_path.read_text()
    except FileNotFoundError:
        return None

    long_term = ""
    for line in text.splitlines():
        stripped = line.strip()
        low = stripped.lower()
        if low.startswith("long-term goal:"):
            long_term = stripped.split(":", 1)[1].strip()

    if not long_term:
        return None

    return Goals(long_term=long_term)


# --- Narrator formatting ---


def format_narrator_message(update: WorldUpdate) -> str:
    parts: list[str] = []
    parts.extend(update.events)
    for fact in update.add_facts:
        parts.append(fact)
    return "[" + " ".join(parts) + "]"


def format_pending_narrator(change_text: str) -> str:
    """Bracket wrapper for fast-path narrator injection."""
    return f"[{change_text}]"


def load_beat_guide(intent: str, line: str) -> str:
    """Read beat_guide.txt and substitute {intent} and {line} placeholders."""
    template = _load_prompt_file("beat_guide.txt")
    return template.replace("{intent}", intent).replace("{line}", line)


# --- OpenAI client ---


def _make_client(model_config: dict) -> OpenAI:
    api_key = model_config.get("api_key") or os.environ[model_config["api_key_env"]]
    return OpenAI(
        api_key=api_key,
        base_url=model_config["base_url"],
    )


# --- Reconcile LLM call ---


def reconcile_call(world: WorldState, pending_changes: list[str], model_config: dict) -> WorldUpdate:
    """Ask the reconciler LLM to produce a WorldUpdate from pending change descriptions."""
    lines: list[str] = []
    if world.static:
        lines.append("Permanent facts (cannot be changed):")
        for fact in world.static:
            lines.append(f"- {fact}")
        lines.append("")
    if world.dynamic:
        lines.append("Current dynamic facts:")
        lines.append(world.render_for_reconcile())
        lines.append("")
    if world.events:
        lines.append("Recent events:")
        for event in world.events:
            lines.append(f"- {event}")
        lines.append("")
    lines.append("Pending changes to reconcile:")
    for change in pending_changes:
        lines.append(f"- {change}")

    user_message = "\n".join(lines)

    client = _make_client(model_config)
    response = client.chat.completions.create(
        model=model_config["model"],
        messages=[
            {"role": "system", "content": _load_prompt_file("reconcile_system.txt")},
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


# --- Eval LLM call ---


def eval_call(
    system_prompt: str,
    world: WorldState | None,
    history: list[dict],
    model_config: dict,
    goals: Goals | None = None,
    script: Script | None = None,
    recent_n: int = 10,
) -> EvalResult:
    """Ask the LLM to evaluate the character's state and script progress."""
    context_parts: list[str] = []

    context_parts.append("=== CHARACTER SYSTEM PROMPT ===")
    context_parts.append(system_prompt)

    if world:
        context_parts.append("\n=== WORLD STATE ===")
        context_parts.append(world.render())

    if goals:
        context_parts.append("\n=== CHARACTER GOALS ===")
        context_parts.append(goals.render())

    if script and not script.is_empty():
        context_parts.append("\n=== CURRENT SCRIPT ===")
        context_parts.append(script.render())

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
            {"role": "system", "content": _load_prompt_file("eval_system.txt")},
            {"role": "user", "content": user_message},
        ],
        temperature=0.7,
        response_format={"type": "json_object"},
    )

    data = json.loads(response.choices[0].message.content)

    return EvalResult(
        thought=data.get("thought", ""),
        gaze=_strip_tag(data.get("gaze", ""), "gaze"),
        expression=_strip_tag(data.get("expression", "neutral"), "emote"),
        script_status=data.get("script_status", "hold"),
        bootstrap_line=data.get("bootstrap_line", ""),
        bootstrap_intent=data.get("bootstrap_intent", ""),
        plan_request=data.get("plan_request", ""),
    )


# --- Condition check LLM call ---


def condition_check_call(
    condition: str,
    system_prompt: str,
    world: WorldState | None,
    history: list[dict],
    model_config: dict,
    recent_n: int = 10,
) -> ConditionResult:
    """Check if a beat's condition is met given world state and conversation."""
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

    context_parts.append(f"\n=== CONDITION TO CHECK ===")
    context_parts.append(condition)

    user_message = "\n".join(context_parts)

    client = _make_client(model_config)
    response = client.chat.completions.create(
        model=model_config["model"],
        messages=[
            {"role": "system", "content": _load_prompt_file("condition_system.txt")},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,
        response_format={"type": "json_object"},
    )

    data = json.loads(response.choices[0].message.content)

    return ConditionResult(
        met=bool(data.get("met", False)),
        idle=data.get("idle", ""),
        gaze=_strip_tag(data.get("gaze", ""), "gaze"),
        expression=_strip_tag(data.get("expression", "neutral"), "emote"),
    )


# --- Plan LLM call ---


def plan_call(
    system_prompt: str,
    world: WorldState | None,
    history: list[dict],
    goals: Goals | None = None,
    plan_request: str = "",
    plan_model_config: dict | None = None,
    recent_n: int = 10,
) -> PlanResult:
    """Ask the planner LLM to generate a multi-beat script."""
    if plan_model_config is None:
        return PlanResult()

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

    if plan_request:
        context_parts.append(f"\n=== PLAN REQUEST ===")
        context_parts.append(plan_request)

    user_message = "\n".join(context_parts)

    client = _make_client(plan_model_config)
    response = client.chat.completions.create(
        model=plan_model_config["model"],
        messages=[
            {"role": "system", "content": _load_prompt_file("plan_system.txt")},
            {"role": "user", "content": user_message},
        ],
        temperature=0.7,
        response_format={"type": "json_object"},
    )

    data = json.loads(response.choices[0].message.content)

    beats = []
    for b in data.get("beats", []):
        if isinstance(b, dict) and "line" in b and "intent" in b:
            beats.append(Beat(
                line=b["line"],
                intent=b["intent"],
                gaze=b.get("gaze", ""),
                expression=b.get("expression", ""),
                condition=b.get("condition", ""),
            ))

    return PlanResult(beats=beats)
