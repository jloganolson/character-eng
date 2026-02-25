from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from rich.panel import Panel

from character_eng.world import _load_prompt_file, _make_client

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
CHARACTERS_DIR = PROMPTS_DIR / "characters"


# --- Data classes ---


@dataclass
class StageExit:
    condition: str
    goto: str
    label: str = ""


@dataclass
class Stage:
    name: str
    goal: str
    exits: list[StageExit] = field(default_factory=list)


@dataclass
class ScenarioScript:
    name: str
    stages: dict[str, Stage] = field(default_factory=dict)
    start: str = ""
    current_stage: str = ""

    @property
    def active_stage(self) -> Stage | None:
        return self.stages.get(self.current_stage)

    def advance_to(self, name: str) -> bool:
        if name in self.stages:
            self.current_stage = name
            return True
        return False

    def render(self) -> str:
        lines: list[str] = []
        for sname, stage in self.stages.items():
            marker = "→" if sname == self.current_stage else " "
            lines.append(f"{marker} {sname}: {stage.goal}")
            for i, exit in enumerate(stage.exits):
                lines.append(f"    exit {i}: [{exit.condition}] → {exit.goto}")
        return "\n".join(lines)

    def show(self) -> Panel:
        body = self.render() if self.stages else "[dim]No scenario loaded.[/dim]"
        return Panel(body, title="Scenario", border_style="cyan")


@dataclass
class DirectorResult:
    thought: str
    status: str  # "hold" or "advance"
    exit_index: int = -1  # which exit matched (-1 for hold)


# --- File loading ---


def load_scenario_script(character: str) -> ScenarioScript | None:
    """Load a character's scenario_script.toml. Returns None if not found."""
    path = CHARACTERS_DIR / character / "scenario_script.toml"
    if not path.exists():
        return None

    data = tomllib.loads(path.read_text())

    scenario_data = data.get("scenario", {})
    name = scenario_data.get("name", character)
    start = scenario_data.get("start", "")

    stages: dict[str, Stage] = {}
    for stage_data in data.get("stage", []):
        sname = stage_data.get("name", "")
        goal = stage_data.get("goal", "")
        exits: list[StageExit] = []
        for exit_data in stage_data.get("exit", []):
            exits.append(StageExit(
                condition=exit_data.get("condition", ""),
                goto=exit_data.get("goto", ""),
                label=exit_data.get("label", ""),
            ))
        stages[sname] = Stage(name=sname, goal=goal, exits=exits)

    return ScenarioScript(
        name=name,
        stages=stages,
        start=start,
        current_stage=start,
    )


# --- Director LLM call ---


def director_call(
    scenario: ScenarioScript,
    world,
    people,
    history: list[dict],
    model_config: dict,
    recent_n: int = 10,
) -> DirectorResult:
    """Ask the director LLM to evaluate stage transitions."""
    stage = scenario.active_stage
    if stage is None:
        return DirectorResult(thought="No active stage.", status="hold")

    context_parts: list[str] = []

    context_parts.append(f"=== CURRENT STAGE ===")
    context_parts.append(f"Stage: {stage.name}")
    context_parts.append(f"Goal: {stage.goal}")

    if stage.exits:
        context_parts.append("\nExit conditions (check in order, first match wins):")
        for i, exit in enumerate(stage.exits):
            context_parts.append(f"  {i}. [{exit.condition}] → {exit.goto}")

    if world is not None:
        context_parts.append(f"\n=== WORLD STATE ===")
        context_parts.append(world.render())

    if people is not None:
        people_text = people.render()
        if people_text:
            context_parts.append(f"\n=== PEOPLE PRESENT ===")
            context_parts.append(people_text)

    recent = history[-recent_n:] if len(history) > recent_n else history
    if recent:
        context_parts.append(f"\n=== RECENT CONVERSATION ===")
        for msg in recent:
            role = msg["role"].upper()
            context_parts.append(f"{role}: {msg['content']}")

    user_message = "\n".join(context_parts)

    client = _make_client(model_config)
    response = client.chat.completions.create(
        model=model_config["model"],
        messages=[
            {"role": "system", "content": _load_prompt_file("director_system.txt")},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,
        response_format={"type": "json_object"},
    )

    data = json.loads(response.choices[0].message.content)

    return DirectorResult(
        thought=data.get("thought", ""),
        status=data.get("status", "hold"),
        exit_index=data.get("exit_index", -1),
    )
