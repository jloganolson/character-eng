"""Integration test for the world state reconciler.

Runs 4 sequential scenarios against the real LLM, validating schema,
fact IDs, and non-empty events. Updates are applied cumulatively.
Saves full JSON logs to logs/.

Usage: uv run -m character_eng.qa_world [--model gemini|cerebras]
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from character_eng.models import DEFAULT_MODEL, MODELS
from character_eng.world import WorldState, WorldUpdate, reconcile_call

console = Console()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"

SCENARIOS = [
    "A gust of wind blows the stack of fliers off the stand",
    "A dog wanders up to the stand and sniffs the water jug",
    "The water jug tips over and spills across the counter",
    "The person picks up the fliers and places them back on the stand",
]

_FACT_ID_PREFIX_RE = re.compile(r"^(?:f\d+|p\d+f\d+)\.\s*")


def validate_update(update: WorldUpdate, world: WorldState) -> list[str]:
    """Return a list of validation errors (empty if all good)."""
    errors = []

    if not isinstance(update.remove_facts, list):
        errors.append(f"remove_facts is not a list: {type(update.remove_facts)}")
    if not isinstance(update.add_facts, list):
        errors.append(f"add_facts is not a list: {type(update.add_facts)}")
    if not isinstance(update.events, list):
        errors.append(f"events is not a list: {type(update.events)}")

    # Check fact IDs are valid strings referencing existing dynamic facts
    for fid in update.remove_facts:
        if not isinstance(fid, str):
            errors.append(f"remove_facts contains non-string: {fid}")
        elif fid not in world.dynamic:
            errors.append(f"remove_facts ID {fid!r} not found in dynamic facts (valid: {list(world.dynamic.keys())})")

    # Events should not be empty
    if not update.events:
        errors.append("events list is empty — expected at least one event")

    # All strings should be non-empty
    for fact in update.add_facts:
        if not isinstance(fact, str) or not fact.strip():
            errors.append(f"add_facts contains empty/non-string: {fact!r}")
        elif _FACT_ID_PREFIX_RE.match(fact.strip()):
            errors.append(f"add_facts contains leaked fact ID prefix: {fact!r}")
    for event in update.events:
        if not isinstance(event, str) or not event.strip():
            errors.append(f"events contains empty/non-string: {event!r}")

    # New events should be new, not copies of already-logged history.
    stale_events = [event for event in update.events if event in world.events]
    if stale_events:
        errors.append(f"events repeated existing history: {stale_events!r}")

    if len(set(update.events)) != len(update.events):
        errors.append(f"events contains duplicates: {update.events!r}")

    return errors


def main():
    parser = argparse.ArgumentParser(description="World state reconciler QA integration test")
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=MODELS.keys(),
                        help=f"Model to test with (default: {DEFAULT_MODEL})")
    args = parser.parse_args()
    model_config = MODELS[args.model]

    console.print(Panel(f"[bold]World State Reconciler — QA Integration Test ({model_config['name']})[/bold]", border_style="green"))

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

    console.print(world.show())
    console.print()

    all_passed = True
    scenarios_log = []

    for i, scenario in enumerate(SCENARIOS):
        console.print(f"[bold cyan]Scenario {i + 1}:[/bold cyan] {scenario}")

        scenario_entry = {
            "scenario": scenario,
            "world_before": {"static": list(world.static), "dynamic": dict(world.dynamic), "events": list(world.events)},
        }

        try:
            update = reconcile_call(world, [scenario], model_config)
        except Exception as e:
            console.print(f"  [red]FAIL — reconcile_call raised: {e}[/red]")
            all_passed = False
            scenario_entry["error"] = str(e)
            scenario_entry["result"] = "FAIL"
            scenarios_log.append(scenario_entry)
            continue

        errors = validate_update(update, world)

        scenario_entry["update"] = {
            "remove_facts": update.remove_facts,
            "add_facts": update.add_facts,
            "events": update.events,
        }
        scenario_entry["validation_errors"] = errors
        scenario_entry["result"] = "PASS" if not errors else "FAIL"

        if errors:
            console.print(f"  [red]FAIL — {len(errors)} validation error(s):[/red]")
            for err in errors:
                console.print(f"    [red]• {err}[/red]")
            all_passed = False
        else:
            console.print(f"  [green]PASS[/green]")

        # Show the update
        console.print(f"  remove: {update.remove_facts}")
        console.print(f"  add:    {update.add_facts}")
        console.print(f"  events: {update.events}")

        # Apply cumulatively
        world.apply_update(update)
        scenario_entry["world_after"] = {"static": list(world.static), "dynamic": dict(world.dynamic), "events": list(world.events)}
        scenarios_log.append(scenario_entry)
        console.print()

    console.print(world.show())
    console.print()

    full_log = {
        "test": "qa_world",
        "model": args.model,
        "model_name": model_config["name"],
        "timestamp": datetime.now().isoformat(),
        "all_passed": all_passed,
        "scenarios": scenarios_log,
    }

    # Write JSON log
    LOGS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"qa_world_{args.model}_{timestamp}.json"
    log_path.write_text(json.dumps(full_log, indent=2))

    if all_passed:
        console.print("[bold green]All 4 scenarios passed.[/bold green]")
    else:
        console.print("[bold red]Some scenarios failed.[/bold red]")

    console.print(f"[dim]Log: {log_path}[/dim]")

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
