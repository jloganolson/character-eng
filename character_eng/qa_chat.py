"""Integration test for chat, world state, and think system.

Parses test_plan.md and runs each section as an isolated test scenario
against the real LLM. Saves full JSON logs to logs/.

Usage: uv run -m character_eng.qa_chat [--model gemini|cerebras]
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from character_eng.chat import ChatSession
from character_eng.models import DEFAULT_MODEL, MODELS
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_PLAN_PATH = PROJECT_ROOT / "test_plan.md"
LOGS_DIR = PROJECT_ROOT / "logs"
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


def run_section(name: str, commands: list[tuple[str, str]], model_config: dict) -> tuple[list[str], list[dict]]:
    """Run a test section. Returns (failures, log_entries)."""
    failures: list[str] = []
    log: list[dict] = []

    world = load_world_state(CHARACTER)
    system_prompt = load_prompt(CHARACTER, world_state=world)
    session = ChatSession(system_prompt, model_config)

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
            log.append({"type": "send", "input": arg, "response": last_response})

        elif cmd == "world":
            console.print(f"  [yellow]world:[/yellow] {arg}")
            try:
                update = director_call(world, arg, model_config)
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
                log.append({
                    "type": "world",
                    "input": arg,
                    "update": {
                        "remove_facts": update.remove_facts,
                        "add_facts": update.add_facts,
                        "events": update.events,
                    },
                    "narrator": narrator_msg,
                    "response": last_response,
                })
            except Exception as e:
                failures.append(f"world command failed: {e}")
                log.append({"type": "world", "input": arg, "error": str(e)})
                last_response = ""

        elif cmd == "think":
            console.print("  [magenta]think[/magenta]")
            try:
                result = think_call(
                    system_prompt=session.system_prompt,
                    world=world,
                    history=session.get_history(),
                    model_config=model_config,
                )
                last_think = result
                console.print(f"  [dim]💭 {result.thought}[/dim]")
                console.print(f"  [dim]action={result.action} detail={result.action_detail!r}[/dim]")

                think_entry = {
                    "type": "think",
                    "thought": result.thought,
                    "action": result.action,
                    "action_detail": result.action_detail,
                }

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
                    think_entry["response"] = last_response

                log.append(think_entry)
            except Exception as e:
                failures.append(f"think command failed: {e}")
                log.append({"type": "think", "error": str(e)})
                last_think = None

        elif cmd == "expect":
            check = arg.strip()
            passed = True
            if check == "non_empty":
                if not last_response.strip():
                    failures.append("expected non-empty response, got empty")
                    passed = False
            elif check.startswith("max_length:"):
                max_len = int(check.split(":")[1])
                if len(last_response) > max_len:
                    failures.append(f"response length {len(last_response)} exceeds max {max_len}")
                    passed = False
            elif check == "world_updated":
                if not world_changed:
                    failures.append("expected world to be updated, but no changes detected")
                    passed = False
            elif check == "valid_thought":
                if last_think is None:
                    failures.append("expected valid thought, but no think result")
                    passed = False
                elif not last_think.thought.strip():
                    failures.append("think result has empty thought")
                    passed = False
                elif last_think.action not in ("no_change", "talk", "emote"):
                    failures.append(f"invalid action type: {last_think.action!r}")
                    passed = False
            elif check == "in_character":
                lower_resp = last_response.lower()
                for phrase in AI_BLOCKLIST:
                    if phrase.lower() in lower_resp:
                        failures.append(f"response contains out-of-character phrase: {phrase!r}")
                        passed = False
                        break
            else:
                failures.append(f"unknown expect check: {check!r}")
                passed = False
            log.append({"type": "expect", "check": check, "passed": passed})

    return failures, log


def main():
    parser = argparse.ArgumentParser(description="Chat QA integration test")
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=MODELS.keys(),
                        help=f"Model to test with (default: {DEFAULT_MODEL})")
    args = parser.parse_args()
    model_config = MODELS[args.model]

    console.print(Panel(f"[bold]Chat QA — Integration Test ({model_config['name']})[/bold]", border_style="green"))

    if not TEST_PLAN_PATH.exists():
        console.print(f"[red]Test plan not found: {TEST_PLAN_PATH}[/red]")
        sys.exit(1)

    sections = parse_test_plan(TEST_PLAN_PATH)
    console.print(f"Loaded {len(sections)} test sections from test_plan.md\n")

    all_passed = True
    full_log = {
        "test": "qa_chat",
        "model": args.model,
        "model_name": model_config["name"],
        "timestamp": datetime.now().isoformat(),
        "sections": {},
    }

    for name, commands in sections.items():
        console.print(f"[bold cyan]== {name} ==[/bold cyan]")
        failures, log = run_section(name, commands, model_config)

        section_result = "PASS" if not failures else "FAIL"
        full_log["sections"][name] = {
            "result": section_result,
            "failures": failures,
            "log": log,
        }

        if failures:
            console.print(f"  [red]FAIL — {len(failures)} error(s):[/red]")
            for f in failures:
                console.print(f"    [red]• {f}[/red]")
            all_passed = False
        else:
            console.print(f"  [green]PASS[/green]")
        console.print()

    full_log["all_passed"] = all_passed

    # Write JSON log
    LOGS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"qa_chat_{args.model}_{timestamp}.json"
    log_path.write_text(json.dumps(full_log, indent=2))

    if all_passed:
        console.print(f"[bold green]All {len(sections)} sections passed.[/bold green]")
    else:
        console.print("[bold red]Some sections failed.[/bold red]")

    console.print(f"[dim]Log: {log_path}[/dim]")

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
