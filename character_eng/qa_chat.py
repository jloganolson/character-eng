"""Integration test for chat, world state, and think system.

Parses test_plan.md and runs each section as an isolated test scenario
against the real LLM.

Usage: uv run -m character_eng.qa_chat
"""

import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from character_eng.chat import ChatSession
from character_eng.prompts import load_prompt
from character_eng.world import (
    WorldState,
    director_call,
    format_narrator_message,
    load_world_state,
    think_call,
    ThinkResult,
)

console = Console()

TEST_PLAN_PATH = Path(__file__).resolve().parent.parent / "test_plan.md"
CHARACTER = "greg"

AI_BLOCKLIST = ["I'm an AI", "as a language model", "I cannot", "as an AI"]


def parse_test_plan(path: Path) -> dict[str, list[tuple[str, str]]]:
    """Parse test_plan.md into {section_name: [(command, arg), ...]}."""
    sections: dict[str, list[tuple[str, str]]] = {}
    current_section = None

    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("# "):
            continue
        if line.startswith("## "):
            current_section = line[3:].strip()
            sections[current_section] = []
            continue
        if current_section is None:
            continue

        if line.startswith("send:"):
            sections[current_section].append(("send", line[5:].strip()))
        elif line.startswith("world:"):
            sections[current_section].append(("world", line[6:].strip()))
        elif line.startswith("expect:"):
            sections[current_section].append(("expect", line[7:].strip()))
        elif line == "think":
            sections[current_section].append(("think", ""))

    return sections


def run_section(name: str, commands: list[tuple[str, str]]) -> list[str]:
    """Run a test section, return list of failure messages (empty = pass)."""
    failures: list[str] = []

    world = load_world_state(CHARACTER)
    system_prompt = load_prompt(CHARACTER, world_state=world)
    session = ChatSession(system_prompt)

    last_response = ""
    last_think: ThinkResult | None = None
    world_changed = False

    for cmd, arg in commands:
        if cmd == "send":
            console.print(f"  [blue]send:[/blue] {arg}")
            chunks = []
            for chunk in session.send(arg):
                chunks.append(chunk)
            last_response = "".join(chunks)
            console.print(f"  [dim]→ {last_response[:120]}{'...' if len(last_response) > 120 else ''}[/dim]")

        elif cmd == "world":
            console.print(f"  [yellow]world:[/yellow] {arg}")
            try:
                update = director_call(world, arg)
                world.apply_update(update)
                world_changed = bool(update.events or update.add_facts or update.remove_facts)
                narrator_msg = format_narrator_message(update)
                session.inject_system(narrator_msg)
                # Trigger character reaction
                chunks = []
                for chunk in session.send("[React to what just happened.]"):
                    chunks.append(chunk)
                last_response = "".join(chunks)
                console.print(f"  [dim]→ {last_response[:120]}{'...' if len(last_response) > 120 else ''}[/dim]")
            except Exception as e:
                failures.append(f"world command failed: {e}")
                last_response = ""

        elif cmd == "think":
            console.print("  [magenta]think[/magenta]")
            try:
                result = think_call(
                    system_prompt=session.system_prompt,
                    world=world,
                    history=session.get_history(),
                )
                last_think = result
                console.print(f"  [dim]💭 {result.thought}[/dim]")
                console.print(f"  [dim]action={result.action} detail={result.action_detail!r}[/dim]")

                if result.action in ("talk", "emote"):
                    if result.action == "talk":
                        directive = f"[You want to say something unprompted. Directive: {result.action_detail}]"
                        nudge = "[The character speaks.]"
                    else:
                        directive = f"[You feel compelled to act. Directive: {result.action_detail}]"
                        nudge = "[The character acts.]"
                    session.inject_system(directive)
                    chunks = []
                    for chunk in session.send(nudge):
                        chunks.append(chunk)
                    last_response = "".join(chunks)
                    console.print(f"  [dim]→ {last_response[:120]}{'...' if len(last_response) > 120 else ''}[/dim]")
            except Exception as e:
                failures.append(f"think command failed: {e}")
                last_think = None

        elif cmd == "expect":
            check = arg.strip()
            if check == "non_empty":
                if not last_response.strip():
                    failures.append("expected non-empty response, got empty")
            elif check.startswith("max_length:"):
                max_len = int(check.split(":")[1])
                if len(last_response) > max_len:
                    failures.append(f"response length {len(last_response)} exceeds max {max_len}")
            elif check == "world_updated":
                if not world_changed:
                    failures.append("expected world to be updated, but no changes detected")
            elif check == "valid_thought":
                if last_think is None:
                    failures.append("expected valid thought, but no think result")
                elif not last_think.thought.strip():
                    failures.append("think result has empty thought")
                elif last_think.action not in ("no_change", "talk", "emote"):
                    failures.append(f"invalid action type: {last_think.action!r}")
            elif check == "in_character":
                lower_resp = last_response.lower()
                for phrase in AI_BLOCKLIST:
                    if phrase.lower() in lower_resp:
                        failures.append(f"response contains out-of-character phrase: {phrase!r}")
                        break
            else:
                failures.append(f"unknown expect check: {check!r}")

    return failures


def main():
    console.print(Panel("[bold]Chat QA — Integration Test[/bold]", border_style="green"))

    if not TEST_PLAN_PATH.exists():
        console.print(f"[red]Test plan not found: {TEST_PLAN_PATH}[/red]")
        sys.exit(1)

    sections = parse_test_plan(TEST_PLAN_PATH)
    console.print(f"Loaded {len(sections)} test sections from test_plan.md\n")

    all_passed = True

    for name, commands in sections.items():
        console.print(f"[bold cyan]== {name} ==[/bold cyan]")
        failures = run_section(name, commands)

        if failures:
            console.print(f"  [red]FAIL — {len(failures)} error(s):[/red]")
            for f in failures:
                console.print(f"    [red]• {f}[/red]")
            all_passed = False
        else:
            console.print(f"  [green]PASS[/green]")
        console.print()

    if all_passed:
        console.print(f"[bold green]All {len(sections)} sections passed.[/bold green]")
    else:
        console.print("[bold red]Some sections failed.[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
