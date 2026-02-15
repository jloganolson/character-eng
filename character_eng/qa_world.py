"""Integration test for the world state director.

Runs 4 sequential scenarios against the real LLM, validating schema,
index ranges, and non-empty events. Updates are applied cumulatively.

Usage: uv run -m character_eng.qa_world
"""

import sys

from rich.console import Console
from rich.panel import Panel

from character_eng.world import WorldState, WorldUpdate, director_call

console = Console()

SCENARIOS = [
    "The orb rolls off the table and onto the floor",
    "A cat wanders into the room and sits near the orb",
    "The orb starts glowing brightly",
    "The person picks up the orb and places it back on the desk",
]


def validate_update(update: WorldUpdate, world: WorldState, scenario_idx: int) -> list[str]:
    """Return a list of validation errors (empty if all good)."""
    errors = []

    if not isinstance(update.remove_facts, list):
        errors.append(f"remove_facts is not a list: {type(update.remove_facts)}")
    if not isinstance(update.add_facts, list):
        errors.append(f"add_facts is not a list: {type(update.add_facts)}")
    if not isinstance(update.events, list):
        errors.append(f"events is not a list: {type(update.events)}")

    # Check index ranges
    for idx in update.remove_facts:
        if not isinstance(idx, int):
            errors.append(f"remove_facts contains non-int: {idx}")
        elif idx < 0 or idx >= len(world.dynamic):
            errors.append(f"remove_facts index {idx} out of range (0-{len(world.dynamic) - 1})")

    # Events should not be empty
    if not update.events:
        errors.append("events list is empty — expected at least one event")

    # All strings should be non-empty
    for fact in update.add_facts:
        if not isinstance(fact, str) or not fact.strip():
            errors.append(f"add_facts contains empty/non-string: {fact!r}")
    for event in update.events:
        if not isinstance(event, str) or not event.strip():
            errors.append(f"events contains empty/non-string: {event!r}")

    return errors


def main():
    console.print(Panel("[bold]World State Director — QA Integration Test[/bold]", border_style="green"))

    world = WorldState(
        static=[
            "Greg is a robot head bolted to a desk",
            "Greg has no body, arms, or hands",
        ],
        dynamic=[
            "A mysterious orb sits on the desk in front of Greg",
            "The orb swirls with faint light",
            "A person is standing across the desk from Greg",
            "The room is quiet",
        ],
    )

    console.print(world.show())
    console.print()

    all_passed = True

    for i, scenario in enumerate(SCENARIOS):
        console.print(f"[bold cyan]Scenario {i + 1}:[/bold cyan] {scenario}")

        try:
            update = director_call(world, scenario)
        except Exception as e:
            console.print(f"  [red]FAIL — director_call raised: {e}[/red]")
            all_passed = False
            continue

        errors = validate_update(update, world, i)

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
        console.print()

    console.print(world.show())
    console.print()

    if all_passed:
        console.print("[bold green]All 4 scenarios passed.[/bold green]")
    else:
        console.print("[bold red]Some scenarios failed.[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
