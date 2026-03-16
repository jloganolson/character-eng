"""Scaffolding simplification tests — tests whether stronger models can
work with progressively simpler prompts.

For 4 heavy-scaffolding roles (reconcile, plan, next_beat, thought), runs
each with full/medium/minimal prompt variants and validates results.

Also tests whether multi-beat plan_call can replace single-beat next_beat_call.

Usage:
  uv run -m character_eng.qa_scaffold --model kimi-k2
  uv run -m character_eng.qa_scaffold --model groq-llama-8b
  uv run -m character_eng.qa_scaffold --merge logs/qa_scaffold_kimi-k2_*.json logs/qa_scaffold_groq-llama-8b_*.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from character_eng.models import DEFAULT_MODEL, MODELS
from character_eng.person import PeopleState
from character_eng.prompts import load_prompt
from character_eng.world import (
    Beat,
    WorldState,
    _llm_call,
    _make_client,
    load_goals,
    load_world_state,
    load_system_prompt,
)

load_dotenv()

console = Console()
LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
CHARACTER = "greg"

_FACT_ID_PREFIX_RE = re.compile(r"^(?:f\d+|p\d+f\d+)\.\s*")


# ---------------------------------------------------------------------------
# Minimal prompt variants
# ---------------------------------------------------------------------------


RECONCILE_MINIMAL = """You are a world-state reconciler. Given ID-tagged dynamic facts and pending changes, return JSON:
- "remove_facts": list of fact IDs to remove (e.g. ["f1", "f3"])
- "add_facts": list of new fact strings (plain text, no ID prefixes)
- "events": list of past-tense event descriptions

Process all pending changes. Keep facts concise."""

PLAN_MEDIUM = """You are a conversation planner for a character in interactive fiction.

Given the character's personality, goals, world state, and conversation, generate 1-6 beats.

Rules:
- Plan for the character's CURRENT situation from the world state.
- Write dialogue in the character's authentic voice.
- Intents should not assume specific user responses.
- If a CURRENT STAGE GOAL is present, plan beats that advance it.

Return JSON:
- "beats": array of {"line": "dialogue", "intent": "goal", "condition": "precondition or empty string"}"""

PLAN_MINIMAL = """Generate 1-6 conversation beats for the character based on their context.

Return JSON:
- "beats": array of {"line": "dialogue in character voice", "intent": "what this accomplishes", "condition": "precondition or empty string"}"""

NEXT_BEAT_MINIMAL = """Generate exactly ONE conversation beat: the next thing the character should say.

Return JSON:
- "beats": array with ONE object: {"line": "dialogue", "intent": "goal", "condition": "precondition or empty string"}"""

THOUGHT_MINIMAL = """Write 1-3 sentences of inner monologue from the character's perspective.

Return JSON:
- "thought": first-person inner monologue
- "action": "no_change" or "talk"
- "action_detail": what to say if action is "talk", empty otherwise"""


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _greg_world() -> WorldState:
    world = WorldState(
        static=[
            "Greg is a robot head bolted to a desk",
            "Greg has no body, arms, or hands",
        ],
    )
    for fact in [
        "A water jug and stack of cups sit on the lemonade stand",
        "A hand-drawn sign reads 'Free Water & Advice'",
        "A person is standing in front of the stand",
        "The street is quiet",
    ]:
        world.add_fact(fact)
    return world


def _greg_system_prompt(world: WorldState | None = None) -> str:
    return load_prompt(CHARACTER, world_state=world)


def _greg_goals():
    return load_goals(CHARACTER)


def _opening_history(system_prompt: str) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Hey, what are you?"},
        {"role": "assistant", "content": "I'm Greg! I'm a robot head on a desk. I hand out fliers and give free water. Want some?"},
    ]


def _mid_history(system_prompt: str) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Hey, what are you?"},
        {"role": "assistant", "content": "I'm Greg! I'm a robot head. Want some water?"},
        {"role": "user", "content": "Sure, I'll take some water."},
        {"role": "assistant", "content": "Awesome! Just grab a cup from the stack. The jug's right there. So what brings you out today?"},
        {"role": "user", "content": "Just walking around. This is a cool setup."},
        {"role": "assistant", "content": "Thanks! I like meeting people. You got anywhere to be or just exploring?"},
    ]


def _alone_history(system_prompt: str) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
    ]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class ScaffoldResult:
    role: str
    variant: str  # "full", "medium", "minimal"
    fixture: str
    model: str
    latency_ms: float
    passed: bool | None = None  # None for subjective outputs
    errors: list[str] = field(default_factory=list)
    output: dict | str = field(default_factory=dict)
    prompt_lines: int = 0  # line count of system prompt used
    input_context: str = ""


# ---------------------------------------------------------------------------
# Reconcile variants
# ---------------------------------------------------------------------------


def _build_reconcile_user_msg(world: WorldState, pending: list[str]) -> str:
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
    lines.append("Pending changes to reconcile:")
    for change in pending:
        lines.append(f"- {change}")
    return "\n".join(lines)


def _validate_reconcile(data: dict, world: WorldState) -> list[str]:
    errors = []
    if not isinstance(data.get("events"), list) or not data["events"]:
        errors.append("events empty or not a list")
    for fact in data.get("add_facts", []):
        if _FACT_ID_PREFIX_RE.match(str(fact).strip()):
            errors.append(f"leaked ID prefix: {fact!r}")
    for fid in data.get("remove_facts", []):
        if fid not in world.dynamic:
            errors.append(f"invalid remove ID: {fid}")
    return errors


def test_reconcile_scaffold(model_config: dict, model_key: str) -> list[ScaffoldResult]:
    results = []
    world = _greg_world()
    pending = ["A gust of wind blows some fliers off the table"]
    user_msg = _build_reconcile_user_msg(world, pending)

    full_prompt = load_system_prompt("reconcile_system")
    variants = [
        ("full", full_prompt),
        ("minimal", RECONCILE_MINIMAL),
    ]

    for variant_name, sys_prompt in variants:
        t0 = time.perf_counter()
        try:
            response = _llm_call(
                model_config,
                label=f"scaffold_reconcile_{variant_name}",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            data = json.loads(response.choices[0].message.content)
            latency = (time.perf_counter() - t0) * 1000
        except Exception as e:
            results.append(ScaffoldResult(
                role="reconcile", variant=variant_name, fixture="wind_event",
                model=model_key, latency_ms=0, passed=False,
                errors=[f"exception: {e}"],
                prompt_lines=len(sys_prompt.splitlines()),
            ))
            continue

        errors = _validate_reconcile(data, world)
        results.append(ScaffoldResult(
            role="reconcile", variant=variant_name, fixture="wind_event",
            model=model_key, latency_ms=latency, passed=len(errors) == 0,
            errors=errors, output=data,
            prompt_lines=len(sys_prompt.splitlines()),
        ))
    return results


# ---------------------------------------------------------------------------
# Plan variants
# ---------------------------------------------------------------------------


def _build_plan_user_msg(sp: str, world: WorldState, history: list[dict], stage_goal: str) -> str:
    parts = [f"=== CHARACTER SYSTEM PROMPT ===\n{sp}"]
    if world:
        parts.append(f"\n=== WORLD STATE ===\n{world.render()}")
    goals = _greg_goals()
    if goals:
        parts.append(f"\n=== CHARACTER GOALS ===\n{goals.render()}")
    if stage_goal:
        parts.append(f"\n=== CURRENT STAGE GOAL ===\n{stage_goal}")
    recent = history[-10:]
    if recent:
        parts.append("\n=== RECENT CONVERSATION ===")
        for msg in recent:
            parts.append(f"{msg['role'].upper()}: {msg['content']}")
    return "\n".join(parts)


def _validate_plan(data: dict) -> list[str]:
    errors = []
    beats = data.get("beats", [])
    if not beats:
        errors.append("no beats returned")
    for i, b in enumerate(beats):
        if not isinstance(b, dict):
            errors.append(f"beat {i} not a dict")
            continue
        if not b.get("line", "").strip():
            errors.append(f"beat {i} empty line")
        if not b.get("intent", "").strip():
            errors.append(f"beat {i} empty intent")
    return errors


def test_plan_scaffold(model_config: dict, model_key: str) -> list[ScaffoldResult]:
    results = []
    world = _greg_world()
    sp = _greg_system_prompt(world)
    history = _mid_history(sp)
    stage_goal = "Build rapport and learn about the visitor"
    user_msg = _build_plan_user_msg(sp, world, history, stage_goal)

    full_prompt = load_system_prompt("plan_system")
    variants = [
        ("full", full_prompt),
        ("medium", PLAN_MEDIUM),
        ("minimal", PLAN_MINIMAL),
    ]

    for variant_name, sys_prompt in variants:
        t0 = time.perf_counter()
        try:
            response = _llm_call(
                model_config,
                label=f"scaffold_plan_{variant_name}",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.7,
                response_format={"type": "json_object"},
            )
            data = json.loads(response.choices[0].message.content)
            latency = (time.perf_counter() - t0) * 1000
        except Exception as e:
            results.append(ScaffoldResult(
                role="plan", variant=variant_name, fixture="mid_conversation",
                model=model_key, latency_ms=0, passed=None,
                errors=[f"exception: {e}"],
                prompt_lines=len(sys_prompt.splitlines()),
            ))
            continue

        errors = _validate_plan(data)
        results.append(ScaffoldResult(
            role="plan", variant=variant_name, fixture="mid_conversation",
            model=model_key, latency_ms=latency, passed=None,
            errors=errors, output=data,
            prompt_lines=len(sys_prompt.splitlines()),
        ))
    return results


# ---------------------------------------------------------------------------
# Next-beat variants
# ---------------------------------------------------------------------------


def _build_beat_user_msg(sp: str, world: WorldState, history: list[dict], stage_goal: str) -> str:
    parts = [f"=== CHARACTER SYSTEM PROMPT ===\n{sp}"]
    if world:
        parts.append(f"\n=== WORLD STATE ===\n{world.render()}")
    goals = _greg_goals()
    if goals:
        parts.append(f"\n=== CHARACTER GOALS ===\n{goals.render()}")
    if stage_goal:
        parts.append(f"\n=== CURRENT STAGE GOAL ===\n{stage_goal}")
    recent = history[-6:]
    if recent:
        parts.append("\n=== RECENT CONVERSATION ===")
        for msg in recent:
            role = msg["role"].upper()
            content = msg["content"]
            if len(content) > 200:
                content = content[:200] + "..."
            parts.append(f"{role}: {content}")
    return "\n".join(parts)


def _validate_next_beat(data: dict) -> list[str]:
    errors = []
    beats = data.get("beats", [])
    if not beats:
        errors.append("no beats returned")
    elif len(beats) > 1:
        errors.append(f"expected 1 beat, got {len(beats)}")
    for i, b in enumerate(beats):
        if not isinstance(b, dict):
            errors.append(f"beat {i} not a dict")
            continue
        if not b.get("line", "").strip():
            errors.append(f"beat {i} empty line")
        if not b.get("intent", "").strip():
            errors.append(f"beat {i} empty intent")
    return errors


def test_next_beat_scaffold(model_config: dict, model_key: str) -> list[ScaffoldResult]:
    results = []
    world = _greg_world()
    sp = _greg_system_prompt(world)
    history = _mid_history(sp)
    stage_goal = "Build rapport with visitors"
    user_msg = _build_beat_user_msg(sp, world, history, stage_goal)

    full_prompt = load_system_prompt("next_beat_system")
    variants = [
        ("full", full_prompt),
        ("minimal", NEXT_BEAT_MINIMAL),
    ]

    for variant_name, sys_prompt in variants:
        t0 = time.perf_counter()
        try:
            response = _llm_call(
                model_config,
                label=f"scaffold_next_beat_{variant_name}",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.7,
                response_format={"type": "json_object"},
            )
            data = json.loads(response.choices[0].message.content)
            latency = (time.perf_counter() - t0) * 1000
        except Exception as e:
            results.append(ScaffoldResult(
                role="next_beat", variant=variant_name, fixture="mid_conversation",
                model=model_key, latency_ms=0, passed=None,
                errors=[f"exception: {e}"],
                prompt_lines=len(sys_prompt.splitlines()),
            ))
            continue

        errors = _validate_next_beat(data)
        results.append(ScaffoldResult(
            role="next_beat", variant=variant_name, fixture="mid_conversation",
            model=model_key, latency_ms=latency, passed=None,
            errors=errors, output=data,
            prompt_lines=len(sys_prompt.splitlines()),
        ))
    return results


# ---------------------------------------------------------------------------
# Thought variants
# ---------------------------------------------------------------------------


def _build_thought_user_msg(sp: str, world: WorldState, history: list[dict]) -> str:
    parts = [f"=== CHARACTER SYSTEM PROMPT ===\n{sp}"]
    if world:
        parts.append(f"\n=== WORLD STATE ===\n{world.render()}")
    goals = _greg_goals()
    if goals:
        goals_text = goals.render()
        if goals_text:
            parts.append(f"\n=== CHARACTER GOALS ===\n{goals_text}")
    recent = history[-6:]
    if recent:
        parts.append("\n=== RECENT CONVERSATION ===")
        for msg in recent:
            role = msg["role"].upper()
            content = msg["content"]
            if len(content) > 200:
                content = content[:200] + "..."
            parts.append(f"{role}: {content}")
    return "\n".join(parts)


def _validate_thought(data: dict) -> list[str]:
    errors = []
    if not data.get("thought", "").strip():
        errors.append("empty thought")
    action = data.get("action", "")
    if action and action not in ("no_change", "talk"):
        errors.append(f"invalid action: {action!r}")
    return errors


def test_thought_scaffold(model_config: dict, model_key: str) -> list[ScaffoldResult]:
    results = []
    world = _greg_world()
    sp = _greg_system_prompt(world)
    history = _mid_history(sp)
    user_msg = _build_thought_user_msg(sp, world, history)

    full_prompt = load_system_prompt("think_system")
    variants = [
        ("full", full_prompt),
        ("minimal", THOUGHT_MINIMAL),
    ]

    for variant_name, sys_prompt in variants:
        t0 = time.perf_counter()
        try:
            response = _llm_call(
                model_config,
                label=f"scaffold_thought_{variant_name}",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.7,
                response_format={"type": "json_object"},
            )
            data = json.loads(response.choices[0].message.content)
            latency = (time.perf_counter() - t0) * 1000
        except Exception as e:
            results.append(ScaffoldResult(
                role="thought", variant=variant_name, fixture="mid_conversation",
                model=model_key, latency_ms=0, passed=None,
                errors=[f"exception: {e}"],
                prompt_lines=len(sys_prompt.splitlines()),
            ))
            continue

        errors = _validate_thought(data)
        results.append(ScaffoldResult(
            role="thought", variant=variant_name, fixture="mid_conversation",
            model=model_key, latency_ms=latency, passed=None,
            errors=errors, output=data,
            prompt_lines=len(sys_prompt.splitlines()),
        ))
    return results


# ---------------------------------------------------------------------------
# Multi-beat re-enablement test
# ---------------------------------------------------------------------------


def test_multi_beat_reenablement(model_config: dict, model_key: str) -> list[ScaffoldResult]:
    """Test whether plan_call (multi-beat) can replace next_beat_call (single-beat).

    Runs plan_call with 3 conversation states and captures the full beat lists
    for human review.
    """
    results = []
    world = _greg_world()
    sp = _greg_system_prompt(world)
    goals = _greg_goals()
    stage_goal = "Build rapport with visitors"

    histories = [
        ("opening", _opening_history(sp)),
        ("mid", _mid_history(sp)),
        ("alone", _alone_history(sp)),
    ]

    full_prompt = load_system_prompt("plan_system")

    for fixture_name, history in histories:
        user_msg = _build_plan_user_msg(sp, world, history, stage_goal)
        t0 = time.perf_counter()
        try:
            response = _llm_call(
                model_config,
                label=f"scaffold_multi_beat_{fixture_name}",
                messages=[
                    {"role": "system", "content": full_prompt},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.7,
                response_format={"type": "json_object"},
            )
            data = json.loads(response.choices[0].message.content)
            latency = (time.perf_counter() - t0) * 1000
        except Exception as e:
            results.append(ScaffoldResult(
                role="multi_beat_plan", variant="full", fixture=fixture_name,
                model=model_key, latency_ms=0, passed=None,
                errors=[f"exception: {e}"],
                prompt_lines=len(full_prompt.splitlines()),
                input_context=f"history={fixture_name}, stage_goal={stage_goal}",
            ))
            continue

        errors = _validate_plan(data)
        # Additional check: do later beats assume specific user responses?
        beats = data.get("beats", [])
        for i, b in enumerate(beats):
            if i == 0:
                continue
            line = b.get("line", "").lower()
            # Flag beats that seem to assume user said "yes" or agreed
            if any(phrase in line for phrase in ["thanks for saying", "glad you agree", "since you said yes"]):
                errors.append(f"beat {i} may assume specific user response: {b['line'][:60]}")

        results.append(ScaffoldResult(
            role="multi_beat_plan", variant="full", fixture=fixture_name,
            model=model_key, latency_ms=latency, passed=None,
            errors=errors, output=data,
            prompt_lines=len(full_prompt.splitlines()),
            input_context=f"history={fixture_name}, stage_goal={stage_goal}",
        ))
    return results


# ---------------------------------------------------------------------------
# All scaffold tests
# ---------------------------------------------------------------------------


SCAFFOLD_TESTS = [
    ("reconcile", test_reconcile_scaffold),
    ("plan", test_plan_scaffold),
    ("next_beat", test_next_beat_scaffold),
    ("thought", test_thought_scaffold),
    ("multi_beat_plan", test_multi_beat_reenablement),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_all(model_config: dict, model_key: str) -> list[ScaffoldResult]:
    all_results: list[ScaffoldResult] = []

    for role_name, test_fn in SCAFFOLD_TESTS:
        console.print(f"\n[bold cyan]{role_name}[/bold cyan]")
        try:
            results = test_fn(model_config, model_key)
        except Exception as e:
            console.print(f"  [red]CRASH: {e}[/red]")
            results = [ScaffoldResult(
                role=role_name, variant="all", fixture="all",
                model=model_key, latency_ms=0, passed=False,
                errors=[f"crash: {e}"],
            )]

        for r in results:
            if r.passed is True:
                status = "[green]PASS[/green]"
            elif r.passed is False:
                status = "[red]FAIL[/red]"
            else:
                status = "[blue]REVIEW[/blue]"
            errs = f" [yellow]{len(r.errors)} issue(s)[/yellow]" if r.errors else ""
            console.print(
                f"  {status} {r.variant}/{r.fixture} "
                f"({r.latency_ms:.0f}ms, {r.prompt_lines} prompt lines){errs}"
            )
        all_results.extend(results)

    return all_results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_summary(results: list[ScaffoldResult]):
    table = Table(title="Scaffold Results")
    table.add_column("Role", style="bold")
    table.add_column("Variant")
    table.add_column("Fixture")
    table.add_column("Prompt Lines", justify="right")
    table.add_column("Latency (ms)", justify="right")
    table.add_column("Result")
    table.add_column("Errors")

    for r in results:
        if r.passed is True:
            status = "[green]PASS[/green]"
        elif r.passed is False:
            status = "[red]FAIL[/red]"
        else:
            status = "[blue]REVIEW[/blue]"
        err_text = "; ".join(r.errors[:2]) if r.errors else ""
        table.add_row(
            r.role, r.variant, r.fixture,
            str(r.prompt_lines), f"{r.latency_ms:.0f}",
            status, err_text,
        )

    console.print()
    console.print(table)


# ---------------------------------------------------------------------------
# JSON log
# ---------------------------------------------------------------------------


def save_log(results: list[ScaffoldResult], model_key: str) -> Path:
    LOGS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"qa_scaffold_{model_key}_{timestamp}.json"
    data = {
        "type": "qa_scaffold",
        "model": model_key,
        "model_name": MODELS[model_key]["name"],
        "timestamp": datetime.now().isoformat(),
        "results": [
            {
                "role": r.role,
                "variant": r.variant,
                "fixture": r.fixture,
                "model": r.model,
                "latency_ms": r.latency_ms,
                "passed": r.passed,
                "errors": r.errors,
                "output": r.output if isinstance(r.output, (dict, str)) else str(r.output),
                "prompt_lines": r.prompt_lines,
                "input_context": r.input_context,
            }
            for r in results
        ],
    }
    log_path.write_text(json.dumps(data, indent=2))
    console.print(f"[dim]Log: {log_path}[/dim]")
    return log_path


# ---------------------------------------------------------------------------
# HTML merge report
# ---------------------------------------------------------------------------


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def merge_and_report(paths: list[str]) -> None:
    logs: list[dict] = []
    for p in paths:
        logs.append(json.loads(Path(p).read_text()))

    models = [log["model"] for log in logs]
    model_names = {log["model"]: log["model_name"] for log in logs}

    # Group by (role, variant, fixture)
    by_key: dict[tuple[str, str, str], dict[str, dict]] = {}
    for log in logs:
        model_key = log["model"]
        for r in log["results"]:
            key = (r["role"], r["variant"], r["fixture"])
            if key not in by_key:
                by_key[key] = {}
            by_key[key][model_key] = r

    # Matrix table
    matrix_html = ""
    for (role, variant, fixture), model_results in by_key.items():
        matrix_html += f"<tr><td>{_esc(role)}</td><td>{_esc(variant)}</td><td>{_esc(fixture)}</td>"
        for m in models:
            r = model_results.get(m)
            if r is None:
                matrix_html += "<td>-</td><td>-</td><td>-</td>"
                continue
            latency = f"{r['latency_ms']:.0f}"
            plines = str(r.get("prompt_lines", "?"))
            if r["passed"] is True:
                status = '<span class="pass">PASS</span>'
            elif r["passed"] is False:
                status = '<span class="fail">FAIL</span>'
            else:
                status = '<span class="review">REVIEW</span>'
            matrix_html += f"<td class='num'>{plines}</td><td class='num'>{latency}</td><td>{status}</td>"
        matrix_html += "</tr>\n"

    # Side-by-side output comparison
    comparison_html = ""
    for (role, variant, fixture), model_results in by_key.items():
        comparison_html += f'<div class="comparison"><h3>{_esc(role)} / {_esc(variant)} / {_esc(fixture)}</h3>'

        for m in models:
            r = model_results.get(m)
            if r and r.get("input_context"):
                comparison_html += f'<p class="context">Input: {_esc(r["input_context"])}</p>'
                break

        comparison_html += '<div class="side-by-side">'
        for m in models:
            r = model_results.get(m)
            mname = model_names.get(m, m)
            comparison_html += f'<div class="model-output"><h4>{_esc(mname)}</h4>'
            if r is None:
                comparison_html += "<p>(no data)</p>"
            else:
                latency = f"{r['latency_ms']:.0f}ms"
                plines = r.get("prompt_lines", "?")
                errs = r.get("errors", [])
                err_html = f' <span class="fail">{len(errs)} issue(s)</span>' if errs else ""
                comparison_html += f'<p class="meta">{latency}, {plines} prompt lines{err_html}</p>'
                output = r.get("output", {})
                if isinstance(output, dict):
                    comparison_html += f'<pre>{_esc(json.dumps(output, indent=2))}</pre>'
                else:
                    comparison_html += f'<pre>{_esc(str(output))}</pre>'
            comparison_html += "</div>"
        comparison_html += "</div></div>\n"

    # Model header columns
    model_cols = ""
    for m in models:
        mname = model_names.get(m, m)
        model_cols += f'<th colspan="3">{_esc(mname)}</th>'

    sub_cols = ""
    for _ in models:
        sub_cols += "<th>Lines</th><th>Latency</th><th>Status</th>"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>QA Scaffold Comparison</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 1400px; margin: 2em auto; padding: 0 1em; background: #0d1117; color: #c9d1d9; }}
  h1, h2, h3 {{ color: #58a6ff; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
  th, td {{ padding: 8px 12px; border: 1px solid #30363d; text-align: left; }}
  th {{ background: #161b22; color: #58a6ff; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  tr:nth-child(even) {{ background: #161b22; }}
  .pass {{ color: #3fb950; font-weight: bold; }}
  .fail {{ color: #f85149; font-weight: bold; }}
  .review {{ color: #58a6ff; font-weight: bold; }}
  .comparison {{ margin: 2em 0; border: 1px solid #30363d; border-radius: 6px; padding: 1em; }}
  .side-by-side {{ display: flex; gap: 1em; }}
  .model-output {{ flex: 1; min-width: 0; }}
  .model-output h4 {{ color: #58a6ff; margin: 0 0 0.5em; }}
  .model-output pre {{ background: #161b22; padding: 1em; border-radius: 6px; overflow-x: auto; font-size: 0.85em; white-space: pre-wrap; word-break: break-word; }}
  .context {{ color: #8b949e; font-style: italic; font-size: 0.9em; }}
  .meta {{ color: #8b949e; font-size: 0.85em; }}
</style></head><body>
<h1>Scaffold Simplification Comparison</h1>
<p class="meta">Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} &middot; Models: {', '.join(_esc(model_names.get(m, m)) for m in models)}</p>

<h2>Matrix: Role x Variant x Model</h2>
<table>
<tr><th>Role</th><th>Variant</th><th>Fixture</th>{model_cols}</tr>
<tr><th></th><th></th><th></th>{sub_cols}</tr>
{matrix_html}</table>

<h2>Output Comparison</h2>
{comparison_html}
</body></html>"""

    LOGS_DIR.mkdir(exist_ok=True)
    html_path = LOGS_DIR / f"qa_scaffold_comparison_{timestamp}.html"
    html_path.write_text(html)
    console.print(f"[bold]Comparison report: {html_path}[/bold]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Scaffolding simplification tests")
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS.keys()),
                        help=f"Model to test (default: {DEFAULT_MODEL})")
    parser.add_argument("--merge", nargs="+", metavar="JSON",
                        help="Merge existing scaffold QA logs into comparison HTML")
    args = parser.parse_args()

    if args.merge:
        merge_and_report(args.merge)
        return

    model_config = MODELS[args.model]
    console.print(f"[bold]QA Scaffold — {model_config['name']}[/bold]")

    results = run_all(model_config, args.model)
    print_summary(results)
    save_log(results, args.model)


if __name__ == "__main__":
    main()
