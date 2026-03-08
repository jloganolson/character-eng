from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from character_eng.models import MODELS, get_fallback_chain
from character_eng.utils import ts as _ts

_console = Console()

load_dotenv()

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
CHARACTERS_DIR = PROMPTS_DIR / "characters"
_FACT_ID_PREFIX_RE = re.compile(r"^(?:f\d+|p\d+f\d+)\.\s*")
_WORLD_FACT_ID_RE = re.compile(r"^f\d+$")
_PERSON_FACT_ID_RE = re.compile(r"^p\d+f\d+$")
_VALID_PRESENCE = {"approaching", "present", "leaving", "gone"}
_EXPLICIT_REDIRECT_PATTERNS = (
    re.compile(r"\bi don't care about\b"),
    re.compile(r"\bi do not care about\b"),
    re.compile(r"\bnot interested\b"),
    re.compile(r"\bdon't want to talk about\b"),
    re.compile(r"\bdo not want to talk about\b"),
    re.compile(r"\bcan we talk about something else\b"),
    re.compile(r"\blet'?s talk about something else\b"),
    re.compile(r"\bchange the subject\b"),
)
_TOPIC_REQUEST_RE = re.compile(r"\b(?:can we|let'?s|want to)\s+(?:talk|chat|speak)\s+about\s+([^?.!]+)")
_llm_trace_hook = None



# --- Data classes ---


@dataclass
class PersonUpdate:
    """Mirrors person.PersonUpdate for reconciler output parsing."""
    person_id: str
    remove_facts: list[str] = field(default_factory=list)
    add_facts: list[str] = field(default_factory=list)
    set_name: str | None = None
    set_presence: str | None = None


@dataclass
class WorldUpdate:
    remove_facts: list[str] = field(default_factory=list)
    add_facts: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    person_updates: list[PersonUpdate] = field(default_factory=list)


@dataclass
class Beat:
    line: str  # pre-rendered dialogue line
    intent: str  # what this beat is trying to accomplish
    condition: str = ""  # natural language precondition (empty = auto-deliver)


@dataclass
class ConditionResult:
    met: bool
    idle: str = ""  # in-character idle line when not met


@dataclass
class ScriptCheckResult:
    """Result from the focused script status classification (8B microservice)."""
    status: str  # "advance", "hold", or "off_book"
    plan_request: str = ""


@dataclass
class ThoughtResult:
    """Result from the inner monologue call (8B microservice)."""
    thought: str = ""


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
            cond = f" IF: {beat.condition}" if beat.condition else ""
            lines.append(f"{marker} {i}. [{beat.intent}] \"{beat.line}\"{cond}")
        return "\n".join(lines)

    def show(self) -> Panel:
        body = self.render() if self.beats else "[dim]No script loaded.[/dim]"
        return Panel(body, title="Script", border_style="magenta")


@dataclass
class EvalResult:
    thought: str
    script_status: str  # "advance" | "hold" | "off_book" | "bootstrap"
    bootstrap_line: str = ""
    bootstrap_intent: str = ""
    plan_request: str = ""


@dataclass
class ExpressionResult:
    gaze: str = ""
    gaze_type: str = "hold"  # "hold" (steady) or "glance" (brief look then return)
    expression: str = ""


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
        for text in _clean_string_list(update.add_facts, fact_text=True):
            cleaned = _clean_fact_text(text)
            if cleaned:
                self.add_fact(cleaned)
        self.events.extend(_clean_string_list(update.events))

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


def _clean_fact_text(text: str) -> str:
    """Strip accidental LLM-generated ID prefixes from newly added facts."""
    return _FACT_ID_PREFIX_RE.sub("", text.strip())


def _clean_string_list(values: list, *, fact_text: bool = False) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        cleaned = _clean_fact_text(value) if fact_text else value.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    return result


def _clean_id_list(values: list, pattern: re.Pattern[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        cleaned = value.strip()
        if not pattern.match(cleaned) or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    return result


def _detect_explicit_redirect(history: list[dict], beat: Beat) -> ScriptCheckResult | None:
    """Return off_book when the latest user turn explicitly rejects the current topic."""
    for msg in reversed(history):
        if msg.get("role") != "user":
            continue
        content = str(msg.get("content", "")).strip()
        lowered = content.lower()
        if not lowered:
            return None
        if not any(pattern.search(lowered) for pattern in _EXPLICIT_REDIRECT_PATTERNS):
            return None

        topic_match = _TOPIC_REQUEST_RE.search(lowered)
        if topic_match:
            requested_topic = topic_match.group(1).strip(" .")
            if requested_topic and requested_topic != "something else":
                plan_request = (
                    f"Drop the current beat about {beat.intent.lower()} and talk about "
                    f"{requested_topic} instead."
                )
                return ScriptCheckResult(status="off_book", plan_request=plan_request)

        return ScriptCheckResult(
            status="off_book",
            plan_request=f"Drop the current beat about {beat.intent.lower()} and follow the user's new topic.",
        )
    return None


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


def set_llm_trace_hook(hook):
    global _llm_trace_hook
    previous = _llm_trace_hook
    _llm_trace_hook = hook
    return previous


def _llm_call(model_config: dict, label: str = "", **create_kwargs):
    """Make an LLM call with automatic fallback on provider failure.

    Looks up the fallback chain for the given model_config and tries each
    provider in order. Only falls back on connection/server errors, not on
    content or auth errors.
    """
    # Find model key to get fallback chain
    chain = [model_config]
    for key, cfg in MODELS.items():
        if cfg is model_config:
            chain = get_fallback_chain(key)
            break

    tag = f" ({label})" if label else ""

    last_err = None
    for i, cfg in enumerate(chain):
        if i == 0:
            _console.print(f"[dim]  {_ts()} → {cfg['name']}{tag}[/dim]")
        else:
            _console.print(f"[yellow]  {_ts()} → fallback: {cfg['name']}{tag}[/yellow]")
        started_at = time.time()
        try:
            client = _make_client(cfg)
            response = client.chat.completions.create(
                model=cfg["model"], **create_kwargs
            )
            if _llm_trace_hook is not None:
                try:
                    _llm_trace_hook({
                        "label": label,
                        "provider": cfg["name"],
                        "model": cfg["model"],
                        "messages": [dict(message) for message in create_kwargs.get("messages", [])],
                        "temperature": create_kwargs.get("temperature"),
                        "response_format": create_kwargs.get("response_format"),
                        "started_at": started_at,
                        "finished_at": time.time(),
                        "output": response.choices[0].message.content if response.choices else "",
                    })
                except Exception:
                    pass
            return response
        except Exception as e:
            # Don't fallback on auth errors
            status = getattr(e, "status_code", None)
            if status and 400 <= status < 500:
                raise
            last_err = e
            _console.print(f"[red]  → {cfg['name']} failed: {e}[/red]")

    raise last_err


# --- Reconcile LLM call ---


def reconcile_call(world: WorldState, pending_changes: list[str], model_config: dict, people=None) -> WorldUpdate:
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
    if people is not None:
        people_text = people.render_for_reconcile()
        if people_text:
            lines.append("Current people:")
            lines.append(people_text)
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

    response = _llm_call(
        model_config,
        label="reconcile",
        messages=[
            {"role": "system", "content": _load_prompt_file("reconcile_system.txt")},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )

    data = json.loads(response.choices[0].message.content)

    # Parse person_updates if present
    person_updates = []
    for pu in data.get("person_updates", []):
        if isinstance(pu, dict) and "person_id" in pu:
            set_presence = pu.get("set_presence")
            if set_presence not in _VALID_PRESENCE:
                set_presence = None
            person_updates.append(PersonUpdate(
                person_id=pu["person_id"],
                remove_facts=_clean_id_list(pu.get("remove_facts", []), _PERSON_FACT_ID_RE),
                add_facts=_clean_string_list(pu.get("add_facts", []), fact_text=True),
                set_name=pu.get("set_name"),
                set_presence=set_presence,
            ))

    return WorldUpdate(
        remove_facts=_clean_id_list(data.get("remove_facts", []), _WORLD_FACT_ID_RE),
        add_facts=_clean_string_list(data.get("add_facts", []), fact_text=True),
        events=_clean_string_list(data.get("events", [])),
        person_updates=person_updates,
    )


# --- Eval LLM call ---


def eval_call(
    system_prompt: str,
    world: WorldState | None,
    history: list[dict],
    model_config: dict,
    goals: Goals | None = None,
    script: Script | None = None,
    recent_n: int = 10,
    people=None,
    stage_goal: str = "",
) -> EvalResult:
    """Ask the LLM to evaluate the character's state and script progress."""
    context_parts: list[str] = []

    context_parts.append("=== CHARACTER SYSTEM PROMPT ===")
    context_parts.append(system_prompt)

    if world:
        context_parts.append("\n=== WORLD STATE ===")
        context_parts.append(world.render())

    if people is not None:
        people_text = people.render()
        if people_text:
            context_parts.append("\n=== PEOPLE PRESENT ===")
            context_parts.append(people_text)

    if goals:
        context_parts.append("\n=== CHARACTER GOALS ===")
        context_parts.append(goals.render())

    if stage_goal:
        context_parts.append("\n=== CURRENT STAGE GOAL ===")
        context_parts.append(stage_goal)

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

    response = _llm_call(
        model_config,
        label="eval",
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
        script_status=data.get("script_status", "hold"),
        bootstrap_line=data.get("bootstrap_line", ""),
        bootstrap_intent=data.get("bootstrap_intent", ""),
        plan_request=data.get("plan_request", ""),
    )


# --- Script check LLM call (8B microservice) ---

_SCRIPT_CHECK_SYSTEM = """You are a script tracker for an interactive fiction character.

Your ONLY job: decide whether the current beat's intent has been accomplished.

## Input

You'll receive the current beat's INTENT and LINE, plus recent conversation.

## Decision

- "advance" — The intent was achieved. The character said something that accomplishes it, OR the user's response made it irrelevant. If the beat's intent was to ASK a question, receiving any answer (yes, no, or otherwise) means the intent is accomplished. If the intent is even PARTIALLY done or no longer needed, advance.
- "hold" — The intent has NOT been addressed yet.
- "off_book" — The user explicitly rejected, dismissed, or redirected away from this topic (e.g., "I don't care about that", "let's talk about something else"). Fill in "plan_request" with what the new direction should cover.

**Bias toward "advance".** If in doubt between advance and hold, pick advance.

## Output

Return a JSON object:
- "status": "advance", "hold", or "off_book"
- "plan_request": Only when status is "off_book". Brief description of new direction. Empty string otherwise."""


def script_check_call(
    beat: Beat,
    history: list[dict],
    model_config: dict,
    world: WorldState | None = None,
    recent_n: int = 6,
) -> ScriptCheckResult:
    """Focused script status classification. Minimal context, no monologue. 8B-friendly."""
    redirect = _detect_explicit_redirect(history, beat)
    if redirect is not None:
        return redirect

    parts: list[str] = []

    parts.append("=== CURRENT BEAT ===")
    parts.append(f"Intent: {beat.intent}")
    parts.append(f"Example line: {beat.line}")

    if world and world.dynamic:
        parts.append("\n=== WORLD STATE ===")
        for text in world.dynamic.values():
            parts.append(f"- {text}")

    recent = history[-recent_n:] if len(history) > recent_n else history
    if recent:
        parts.append("\n=== RECENT CONVERSATION ===")
        for msg in recent:
            role = msg["role"].upper()
            content = msg["content"]
            if len(content) > 200:
                content = content[:200] + "..."
            parts.append(f"{role}: {content}")

    user_message = "\n".join(parts)

    response = _llm_call(
        model_config,
        label="script_check",
        messages=[
            {"role": "system", "content": _SCRIPT_CHECK_SYSTEM},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )

    data = json.loads(response.choices[0].message.content)

    return ScriptCheckResult(
        status=data.get("status", "hold"),
        plan_request=data.get("plan_request", ""),
    )


# --- Thought LLM call (8B microservice) ---


def thought_call(
    system_prompt: str,
    history: list[dict],
    model_config: dict,
    recent_n: int = 6,
) -> ThoughtResult:
    """Generate a brief inner monologue for the character. 8B-friendly."""
    recent = history[-recent_n:] if len(history) > recent_n else history
    parts: list[str] = []
    if recent:
        for msg in recent:
            role = msg["role"].upper()
            content = msg["content"]
            if len(content) > 200:
                content = content[:200] + "..."
            parts.append(f"{role}: {content}")

    user_message = "\n".join(parts)

    response = _llm_call(
        model_config,
        label="thought",
        messages=[
            {"role": "system", "content": (
                "You are the inner voice of a character in an interactive fiction.\n\n"
                f"Character:\n{system_prompt[:500]}\n\n"
                "Given the recent conversation, write 1-2 sentences of what the character "
                "is thinking RIGHT NOW. First person. Be specific and emotional, not generic. "
                "React to what just happened.\n\n"
                'Return a JSON object: {"thought": "..."}'
            )},
            {"role": "user", "content": user_message},
        ],
        temperature=0.7,
        response_format={"type": "json_object"},
    )

    data = json.loads(response.choices[0].message.content)

    return ThoughtResult(thought=data.get("thought", ""))


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

    response = _llm_call(
        model_config,
        label="condition",
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
    people=None,
    stage_goal: str = "",
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

    if people is not None:
        people_text = people.render()
        if people_text:
            context_parts.append("\n=== PEOPLE PRESENT ===")
            context_parts.append(people_text)

    if goals:
        context_parts.append("\n=== CHARACTER GOALS ===")
        context_parts.append(goals.render())

    if stage_goal:
        context_parts.append("\n=== CURRENT STAGE GOAL ===")
        context_parts.append(stage_goal)

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

    response = _llm_call(
        plan_model_config,
        label="plan",
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
                condition=b.get("condition", ""),
            ))

    return PlanResult(beats=beats)


# --- Single-beat planning (8B) ---


_SINGLE_BEAT_SYSTEM = """You are a conversation planner for a character in an interactive fiction.

Generate EXACTLY ONE conversation beat — the next thing the character should say or do.

## Rules

- Read the WORLD STATE and CONVERSATION carefully. Plan for the character's CURRENT situation.
- Write dialogue in the character's authentic voice — match their personality and speech patterns.
- The intent should be a short, achievable conversational move (one clause).
- The line should be 1-2 sentences in the character's voice.
- Only set a condition if the beat genuinely requires something to be true first. Most beats need no condition.

## Stage goal

If a CURRENT STAGE GOAL is present, the beat should advance that goal naturally.

## People

If a PEOPLE PRESENT section is present, the beat should acknowledge and interact with them.

## Output format

Return a JSON object with exactly this key:
- "beats": An array with EXACTLY ONE object containing:
  - "line": The dialogue line in the character's voice.
  - "intent": Brief description of what this beat accomplishes.
  - "condition": Precondition (empty string if none needed)."""


def single_beat_call(
    system_prompt: str,
    world: WorldState | None,
    history: list[dict],
    goals: Goals | None = None,
    model_config: dict | None = None,
    plan_request: str = "",
    people=None,
    stage_goal: str = "",
    recent_n: int = 6,
) -> PlanResult:
    """Generate a single beat using 8B. Returns PlanResult with 0-1 beats."""
    if model_config is None:
        return PlanResult()

    context_parts: list[str] = []

    context_parts.append("=== CHARACTER SYSTEM PROMPT ===")
    context_parts.append(system_prompt)

    if world:
        context_parts.append("\n=== WORLD STATE ===")
        context_parts.append(world.render())

    if people is not None:
        people_text = people.render()
        if people_text:
            context_parts.append("\n=== PEOPLE PRESENT ===")
            context_parts.append(people_text)

    if goals:
        context_parts.append("\n=== CHARACTER GOALS ===")
        context_parts.append(goals.render())

    if stage_goal:
        context_parts.append("\n=== CURRENT STAGE GOAL ===")
        context_parts.append(stage_goal)

    recent = history[-recent_n:] if len(history) > recent_n else history
    if recent:
        context_parts.append("\n=== RECENT CONVERSATION ===")
        for msg in recent:
            role = msg["role"].upper()
            context_parts.append(f"{role}: {msg['content']}")

    if plan_request:
        context_parts.append("\n=== PLAN REQUEST ===")
        context_parts.append(plan_request)

    user_message = "\n".join(context_parts)

    response = _llm_call(
        model_config,
        label="plan_single",
        messages=[
            {"role": "system", "content": _SINGLE_BEAT_SYSTEM},
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
                condition=b.get("condition", ""),
            ))

    return PlanResult(beats=beats)


# --- Expression post-processing LLM call ---


DEFAULT_GAZE_TARGETS = ["person", "fliers", "stand"]


def expression_call(
    last_line: str,
    model_config: dict,
    gaze_targets: list[str] | None = None,
) -> ExpressionResult:
    """Derive gaze target and facial expression from the character's latest dialogue."""
    targets = gaze_targets or DEFAULT_GAZE_TARGETS
    system_prompt = _load_prompt_file("expression_system.txt").replace(
        "{gaze_targets}", ", ".join(targets)
    )

    response = _llm_call(
        model_config,
        label="expression",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": last_line},
        ],
        temperature=0.3,
        response_format={"type": "json_object"},
    )

    data = json.loads(response.choices[0].message.content)
    gaze_type = data.get("gaze_type", "hold")
    if gaze_type not in ("hold", "glance"):
        gaze_type = "hold"
    return ExpressionResult(
        gaze=data.get("gaze", ""),
        gaze_type=gaze_type,
        expression=data.get("expression", "neutral"),
    )
