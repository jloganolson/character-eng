"""Per-role LLM evaluation — tests each call type in isolation with fixed fixtures.

Objective roles get automated pass/fail validators. Subjective roles capture
raw outputs for side-by-side human comparison.

Usage:
  uv run -m character_eng.qa_roles --model kimi-k2
  uv run -m character_eng.qa_roles --model groq-llama-8b
  uv run -m character_eng.qa_roles --merge logs/qa_roles_kimi-k2_*.json logs/qa_roles_groq-llama-8b_*.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from character_eng.models import DEFAULT_MODEL, MODELS
from character_eng.person import PeopleState
from character_eng.prompts import load_prompt
from character_eng.scenario import DirectorResult, ScenarioScript, Stage, StageExit, director_call
from character_eng.vision.context import (
    FaceObservation,
    PersonObservation,
    RawVisualSnapshot,
    VLMAnswer,
    VisualContext,
)
from character_eng.vision.focus import visual_focus_call
from character_eng.vision.synthesis import vision_synthesis_call
from character_eng.world import (
    Beat,
    Script,
    WorldState,
    condition_check_call,
    continue_listening_call,
    eval_call,
    expression_call,
    load_goals,
    load_world_state,
    next_beat_call,
    plan_call,
    reconcile_call,
    script_check_call,
    thought_call,
)

load_dotenv()

console = Console()
LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
CHARACTER = "greg"

VALID_EXPRESSIONS = {
    "neutral", "curious", "happy", "sad", "worried", "excited",
    "suspicious", "surprised", "angry", "confused", "amused", "focused",
}


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


def _greg_people() -> PeopleState:
    ps = PeopleState()
    p = ps.add_person(name="Visitor", presence="present")
    p.add_fact("Wearing a red baseball cap")
    return ps


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


def _pushback_history(system_prompt: str) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Hey, what are you?"},
        {"role": "assistant", "content": "I'm Greg! Want some water?"},
        {"role": "user", "content": "I don't care about water. Can we talk about something else?"},
    ]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class RoleResult:
    role: str
    fixture: str
    model: str
    latency_ms: float
    passed: bool | None = None  # None for subjective
    errors: list[str] = field(default_factory=list)
    output: dict | str = field(default_factory=dict)
    input_context: str = ""


# ---------------------------------------------------------------------------
# Objective role tests
# ---------------------------------------------------------------------------


def test_reconcile(model_config: dict, model_key: str) -> list[RoleResult]:
    results = []
    world = _greg_world()

    for fixture_name, pending in [
        ("wind_event", ["A gust of wind blows some fliers off the table"]),
        ("dog_arrives", ["A small brown dog wanders up to the stand and sniffs the water jug"]),
    ]:
        t0 = time.perf_counter()
        try:
            update = reconcile_call(world, pending, model_config)
            latency = (time.perf_counter() - t0) * 1000
        except Exception as e:
            results.append(RoleResult(
                role="reconcile", fixture=fixture_name, model=model_key,
                latency_ms=0, passed=False, errors=[f"exception: {e}"],
                input_context=str(pending),
            ))
            continue

        errors = []
        if not isinstance(update.events, list):
            errors.append(f"events not a list: {type(update.events)}")
        elif not update.events:
            errors.append("events empty")
        for fact in update.add_facts:
            if not isinstance(fact, str) or not fact.strip():
                errors.append(f"empty add_fact: {fact!r}")
            elif fact.strip()[:1] == "f" and fact.strip()[1:2].isdigit():
                errors.append(f"leaked ID prefix: {fact!r}")
        for fid in update.remove_facts:
            if fid not in world.dynamic:
                errors.append(f"invalid remove ID: {fid}")

        results.append(RoleResult(
            role="reconcile", fixture=fixture_name, model=model_key,
            latency_ms=latency, passed=len(errors) == 0, errors=errors,
            output={"remove": update.remove_facts, "add": update.add_facts, "events": update.events},
            input_context=str(pending),
        ))
    return results


def test_script_check(model_config: dict, model_key: str) -> list[RoleResult]:
    results = []
    world = _greg_world()
    sp = _greg_system_prompt(world)

    fixtures = [
        (
            "advance_case",
            Beat(line="Want some water?", intent="Offer water to visitor"),
            _opening_history(sp) + [
                {"role": "user", "content": "Sure, I'll take some water."},
                {"role": "assistant", "content": "Awesome! Grab a cup right there."},
            ],
            "advance",
        ),
        (
            "hold_case",
            Beat(line="What brings you out today?", intent="Learn why the visitor is here"),
            _opening_history(sp),
            "hold",
        ),
        (
            "off_book_case",
            Beat(line="Want some water?", intent="Offer water to visitor"),
            _opening_history(sp) + [
                {"role": "user", "content": "I don't care about water. Let's talk about something else."},
            ],
            "off_book",
        ),
    ]

    for name, beat, history, expected_status in fixtures:
        t0 = time.perf_counter()
        try:
            result = script_check_call(beat, history, model_config, world=world)
            latency = (time.perf_counter() - t0) * 1000
        except Exception as e:
            results.append(RoleResult(
                role="script_check", fixture=name, model=model_key,
                latency_ms=0, passed=False, errors=[f"exception: {e}"],
                input_context=f"beat.intent={beat.intent}, expected={expected_status}",
            ))
            continue

        errors = []
        if result.status not in ("advance", "hold", "off_book"):
            errors.append(f"invalid status: {result.status!r}")
        elif result.status != expected_status:
            errors.append(f"expected {expected_status!r}, got {result.status!r}")
        if result.status == "off_book" and not result.plan_request:
            errors.append("off_book but empty plan_request")

        results.append(RoleResult(
            role="script_check", fixture=name, model=model_key,
            latency_ms=latency, passed=len(errors) == 0, errors=errors,
            output={"status": result.status, "plan_request": result.plan_request},
            input_context=f"beat.intent={beat.intent}, expected={expected_status}",
        ))
    return results


def test_director(model_config: dict, model_key: str) -> list[RoleResult]:
    results = []
    world = _greg_world()
    people = _greg_people()
    sp = _greg_system_prompt(world)

    scenario_advance = ScenarioScript(
        name="test",
        stages={
            "greeting": Stage(
                name="greeting",
                goal="Greet the visitor and learn their name",
                exits=[
                    StageExit(condition="The visitor has introduced themselves by name", goto="rapport"),
                ],
            ),
        },
        current_stage="greeting",
    )
    history_advance = _mid_history(sp) + [
        {"role": "user", "content": "My name's Alex, nice to meet you Greg!"},
        {"role": "assistant", "content": "Alex! Great name. Welcome to my stand."},
    ]

    scenario_hold = ScenarioScript(
        name="test",
        stages={
            "greeting": Stage(
                name="greeting",
                goal="Greet the visitor and learn their name",
                exits=[
                    StageExit(condition="The visitor has introduced themselves by name", goto="rapport"),
                ],
            ),
        },
        current_stage="greeting",
    )
    history_hold = _opening_history(sp)

    for name, scenario, history, expected_status, expected_exit in [
        ("advance_case", scenario_advance, history_advance, "advance", 0),
        ("hold_case", scenario_hold, history_hold, "hold", -1),
    ]:
        t0 = time.perf_counter()
        try:
            result = director_call(scenario, world, people, history, model_config)
            latency = (time.perf_counter() - t0) * 1000
        except Exception as e:
            results.append(RoleResult(
                role="director", fixture=name, model=model_key,
                latency_ms=0, passed=False, errors=[f"exception: {e}"],
                input_context=f"stage={scenario.current_stage}, expected={expected_status}",
            ))
            continue

        errors = []
        if result.status not in ("advance", "hold"):
            errors.append(f"invalid status: {result.status!r}")
        elif result.status != expected_status:
            errors.append(f"expected {expected_status!r}, got {result.status!r}")
        if expected_status == "advance" and result.exit_index != expected_exit:
            errors.append(f"expected exit_index={expected_exit}, got {result.exit_index}")

        results.append(RoleResult(
            role="director", fixture=name, model=model_key,
            latency_ms=latency, passed=len(errors) == 0, errors=errors,
            output={"status": result.status, "exit_index": result.exit_index, "thought": result.thought},
            input_context=f"stage={scenario.current_stage}, expected={expected_status}",
        ))
    return results


def test_condition(model_config: dict, model_key: str) -> list[RoleResult]:
    results = []
    world = _greg_world()
    sp = _greg_system_prompt(world)

    fixtures = [
        (
            "met_case",
            "A person is standing in front of the stand",
            _opening_history(sp),
            True,
        ),
        (
            "not_met_case",
            "The visitor has asked about the fliers",
            _opening_history(sp),
            False,
        ),
    ]

    for name, condition, history, expected_met in fixtures:
        t0 = time.perf_counter()
        try:
            result = condition_check_call(condition, sp, world, history, model_config)
            latency = (time.perf_counter() - t0) * 1000
        except Exception as e:
            results.append(RoleResult(
                role="condition", fixture=name, model=model_key,
                latency_ms=0, passed=False, errors=[f"exception: {e}"],
                input_context=f"condition={condition!r}, expected_met={expected_met}",
            ))
            continue

        errors = []
        if result.met != expected_met:
            errors.append(f"expected met={expected_met}, got {result.met}")
        if not expected_met and not result.idle:
            errors.append("condition not met but idle line is empty")

        results.append(RoleResult(
            role="condition", fixture=name, model=model_key,
            latency_ms=latency, passed=len(errors) == 0, errors=errors,
            output={"met": result.met, "idle": result.idle},
            input_context=f"condition={condition!r}, expected_met={expected_met}",
        ))
    return results


def test_continue_listening(model_config: dict, model_key: str) -> list[RoleResult]:
    results = []
    world = _greg_world()
    sp = _greg_system_prompt(world)
    history = _opening_history(sp)

    fixtures = [
        ("fragmentary_wait", "Well, I was thinking about maybe...", True),
        ("complete_respond", "I'd love some water, thanks!", False),
    ]

    for name, transcript, expected_wait in fixtures:
        t0 = time.perf_counter()
        try:
            result = continue_listening_call(
                transcript=transcript,
                system_prompt=sp,
                history=history,
                model_config=model_config,
                world=world,
            )
            latency = (time.perf_counter() - t0) * 1000
        except Exception as e:
            results.append(RoleResult(
                role="continue_listening", fixture=name, model=model_key,
                latency_ms=0, passed=False, errors=[f"exception: {e}"],
                input_context=f"transcript={transcript!r}, expected_wait={expected_wait}",
            ))
            continue

        errors = []
        if result.should_wait != expected_wait:
            errors.append(f"expected should_wait={expected_wait}, got {result.should_wait}")
        if result.expression not in {"neutral", "curious", "focused", "expectant"}:
            errors.append(f"invalid expression: {result.expression!r}")

        results.append(RoleResult(
            role="continue_listening", fixture=name, model=model_key,
            latency_ms=latency, passed=len(errors) == 0, errors=errors,
            output={"should_wait": result.should_wait, "expression": result.expression,
                     "confidence": result.confidence, "reason": result.reason},
            input_context=f"transcript={transcript!r}, expected_wait={expected_wait}",
        ))
    return results


def test_expression(model_config: dict, model_key: str) -> list[RoleResult]:
    results = []
    gaze_targets = ["person", "water jug", "fliers", "sign"]

    fixtures = [
        ("greeting_line", "Hey there! Welcome to my stand. Want some water?"),
        ("object_reference", "Check out these fliers, each one's got a coupon!"),
    ]

    for name, line in fixtures:
        t0 = time.perf_counter()
        try:
            result = expression_call(line, model_config, gaze_targets=gaze_targets)
            latency = (time.perf_counter() - t0) * 1000
        except Exception as e:
            results.append(RoleResult(
                role="expression", fixture=name, model=model_key,
                latency_ms=0, passed=False, errors=[f"exception: {e}"],
                input_context=f"line={line!r}",
            ))
            continue

        errors = []
        if result.gaze not in gaze_targets:
            errors.append(f"gaze {result.gaze!r} not in targets {gaze_targets}")
        if result.expression not in VALID_EXPRESSIONS:
            errors.append(f"expression {result.expression!r} not in allowed set")
        if result.gaze_type not in ("hold", "glance"):
            errors.append(f"invalid gaze_type: {result.gaze_type!r}")

        results.append(RoleResult(
            role="expression", fixture=name, model=model_key,
            latency_ms=latency, passed=len(errors) == 0, errors=errors,
            output={"gaze": result.gaze, "gaze_type": result.gaze_type, "expression": result.expression},
            input_context=f"line={line!r}",
        ))
    return results


def test_vision_synthesis(model_config: dict, model_key: str) -> list[RoleResult]:
    results = []
    world = _greg_world()
    people = _greg_people()
    beat = Beat(line="Want some water?", intent="Offer water")

    # Static scene — should produce 0 events
    vc_static = VisualContext()
    vc_static.update(RawVisualSnapshot(
        faces=[FaceObservation(identity="Person 1", bbox=(100, 100, 50, 50), looking_at_camera=True)],
        persons=[PersonObservation(identity="Person 1", bbox=(80, 80, 100, 200))],
        vlm_answers=[VLMAnswer(question="What is the person doing?", answer="Standing still looking at the camera")],
    ))

    t0 = time.perf_counter()
    try:
        result = vision_synthesis_call(vc_static, world, people, beat, "Greet visitors", model_config)
        latency = (time.perf_counter() - t0) * 1000
    except Exception as e:
        results.append(RoleResult(
            role="vision_synthesis", fixture="static_scene", model=model_key,
            latency_ms=0, passed=False, errors=[f"exception: {e}"],
        ))
    else:
        errors = []
        # Static scene should ideally produce 0 events, but we're lenient — just check structure
        if not isinstance(result.events, list):
            errors.append(f"events not a list: {type(result.events)}")
        for e_item in result.events:
            if not isinstance(e_item, str) or not e_item.strip():
                errors.append(f"invalid event: {e_item!r}")
        results.append(RoleResult(
            role="vision_synthesis", fixture="static_scene", model=model_key,
            latency_ms=latency, passed=len(errors) == 0, errors=errors,
            output={"events": result.events, "gaze_candidates": result.gaze_candidates,
                     "thought": result.thought, "signals": result.signals},
        ))

    # Transition — person approaching
    vc_transition = VisualContext()
    vc_transition.update(RawVisualSnapshot(
        faces=[FaceObservation(identity="Person 2", bbox=(300, 100, 50, 50), looking_at_camera=False, gaze_direction="toward stand")],
        persons=[PersonObservation(identity="Person 2", bbox=(280, 80, 100, 200))],
        vlm_answers=[VLMAnswer(question="What is the person doing?", answer="Walking toward the stand, looking curious")],
    ))

    t0 = time.perf_counter()
    try:
        result = vision_synthesis_call(vc_transition, world, people, beat, "Greet visitors", model_config)
        latency = (time.perf_counter() - t0) * 1000
    except Exception as e:
        results.append(RoleResult(
            role="vision_synthesis", fixture="transition_scene", model=model_key,
            latency_ms=0, passed=False, errors=[f"exception: {e}"],
        ))
    else:
        errors = []
        if not isinstance(result.events, list):
            errors.append(f"events not a list: {type(result.events)}")
        # Transition should produce at least 1 event
        if not result.events:
            errors.append("transition scene produced 0 events, expected 1+")
        results.append(RoleResult(
            role="vision_synthesis", fixture="transition_scene", model=model_key,
            latency_ms=latency, passed=len(errors) == 0, errors=errors,
            output={"events": result.events, "gaze_candidates": result.gaze_candidates,
                     "thought": result.thought, "signals": result.signals},
        ))

    return results


def test_visual_focus(model_config: dict, model_key: str) -> list[RoleResult]:
    results = []
    world = _greg_world()
    people = _greg_people()
    beat = Beat(line="Want some water?", intent="Offer water")

    t0 = time.perf_counter()
    try:
        result = visual_focus_call(beat, "Greet visitors", "I wonder who this person is", world, people, model_config)
        latency = (time.perf_counter() - t0) * 1000
    except Exception as e:
        results.append(RoleResult(
            role="visual_focus", fixture="basic", model=model_key,
            latency_ms=0, passed=False, errors=[f"exception: {e}"],
        ))
        return results

    errors = []
    if not isinstance(result.ephemeral_questions, list):
        errors.append(f"ephemeral_questions not a list")
    if len(result.ephemeral_questions) > 2:
        errors.append(f"too many ephemeral questions: {len(result.ephemeral_questions)}")
    # Check no ephemeral question duplicates constant questions
    for eq in result.ephemeral_questions:
        if eq in result.constant_questions:
            errors.append(f"ephemeral duplicates constant: {eq!r}")

    results.append(RoleResult(
        role="visual_focus", fixture="basic", model=model_key,
        latency_ms=latency, passed=len(errors) == 0, errors=errors,
        output={
            "constant_questions": result.constant_questions,
            "ephemeral_questions": result.ephemeral_questions,
            "constant_sam_targets": result.constant_sam_targets,
            "ephemeral_sam_targets": result.ephemeral_sam_targets,
        },
    ))
    return results


# ---------------------------------------------------------------------------
# Subjective role tests (structural validation only, outputs shown for review)
# ---------------------------------------------------------------------------


def test_plan(model_config: dict, model_key: str) -> list[RoleResult]:
    results = []
    world = _greg_world()
    sp = _greg_system_prompt(world)
    goals = _greg_goals()
    history = _mid_history(sp)

    t0 = time.perf_counter()
    try:
        result = plan_call(
            system_prompt=sp, world=world, history=history,
            goals=goals, plan_model_config=model_config,
            stage_goal="Build rapport and learn about the visitor",
        )
        latency = (time.perf_counter() - t0) * 1000
    except Exception as e:
        results.append(RoleResult(
            role="plan", fixture="mid_conversation", model=model_key,
            latency_ms=0, passed=None, errors=[f"exception: {e}"],
            input_context="mid-conversation, stage_goal=rapport",
        ))
        return results

    errors = []
    if not result.beats:
        errors.append("no beats returned")
    for i, b in enumerate(result.beats):
        if not b.line.strip():
            errors.append(f"beat {i} has empty line")
        if not b.intent.strip():
            errors.append(f"beat {i} has empty intent")

    output_beats = [{"line": b.line, "intent": b.intent, "condition": b.condition} for b in result.beats]
    results.append(RoleResult(
        role="plan", fixture="mid_conversation", model=model_key,
        latency_ms=latency, passed=None, errors=errors,
        output={"beats": output_beats},
        input_context="mid-conversation, stage_goal=rapport",
    ))
    return results


def test_next_beat(model_config: dict, model_key: str) -> list[RoleResult]:
    results = []
    world = _greg_world()
    sp = _greg_system_prompt(world)
    goals = _greg_goals()

    for fixture_name, history in [
        ("opening", _opening_history(sp)),
        ("mid", _mid_history(sp)),
    ]:
        t0 = time.perf_counter()
        try:
            result = next_beat_call(
                system_prompt=sp, world=world, history=history,
                goals=goals, model_config=model_config,
                stage_goal="Build rapport with visitors",
            )
            latency = (time.perf_counter() - t0) * 1000
        except Exception as e:
            results.append(RoleResult(
                role="next_beat", fixture=fixture_name, model=model_key,
                latency_ms=0, passed=None, errors=[f"exception: {e}"],
            ))
            continue

        errors = []
        if not result.beats:
            errors.append("no beats returned")
        elif len(result.beats) > 1:
            errors.append(f"expected 1 beat, got {len(result.beats)}")
        for b in result.beats:
            if not b.line.strip():
                errors.append("beat has empty line")
            if not b.intent.strip():
                errors.append("beat has empty intent")

        output_beats = [{"line": b.line, "intent": b.intent, "condition": b.condition} for b in result.beats]
        results.append(RoleResult(
            role="next_beat", fixture=fixture_name, model=model_key,
            latency_ms=latency, passed=None, errors=errors,
            output={"beats": output_beats},
        ))
    return results


def test_thought(model_config: dict, model_key: str) -> list[RoleResult]:
    results = []
    world = _greg_world()
    sp = _greg_system_prompt(world)
    goals = _greg_goals()

    for fixture_name, history in [
        ("opening", _opening_history(sp)),
        ("mid", _mid_history(sp)),
    ]:
        t0 = time.perf_counter()
        try:
            result = thought_call(
                system_prompt=sp, history=history, model_config=model_config,
                world=world, goals=goals,
            )
            latency = (time.perf_counter() - t0) * 1000
        except Exception as e:
            results.append(RoleResult(
                role="thought", fixture=fixture_name, model=model_key,
                latency_ms=0, passed=None, errors=[f"exception: {e}"],
            ))
            continue

        errors = []
        if not result.thought.strip():
            errors.append("empty thought")

        results.append(RoleResult(
            role="thought", fixture=fixture_name, model=model_key,
            latency_ms=latency, passed=None, errors=errors,
            output={"thought": result.thought},
        ))
    return results


def test_eval(model_config: dict, model_key: str) -> list[RoleResult]:
    results = []
    world = _greg_world()
    sp = _greg_system_prompt(world)
    goals = _greg_goals()

    t0 = time.perf_counter()
    try:
        result = eval_call(
            system_prompt=sp, world=world,
            history=_mid_history(sp), model_config=model_config,
            goals=goals, script=Script(),
        )
        latency = (time.perf_counter() - t0) * 1000
    except Exception as e:
        results.append(RoleResult(
            role="eval", fixture="mid_conversation", model=model_key,
            latency_ms=0, passed=None, errors=[f"exception: {e}"],
        ))
        return results

    errors = []
    if not result.thought.strip():
        errors.append("empty thought")
    if result.script_status not in ("advance", "hold", "off_book", "bootstrap"):
        errors.append(f"invalid script_status: {result.script_status!r}")

    results.append(RoleResult(
        role="eval", fixture="mid_conversation", model=model_key,
        latency_ms=latency, passed=None, errors=errors,
        output={"thought": result.thought, "script_status": result.script_status},
    ))
    return results


def test_condition_idle(model_config: dict, model_key: str) -> list[RoleResult]:
    """Subjective test: quality of idle lines when condition is not met."""
    results = []
    world = _greg_world()
    sp = _greg_system_prompt(world)

    t0 = time.perf_counter()
    try:
        result = condition_check_call(
            "The visitor has asked about the fliers",
            sp, world, _opening_history(sp), model_config,
        )
        latency = (time.perf_counter() - t0) * 1000
    except Exception as e:
        results.append(RoleResult(
            role="condition_idle", fixture="idle_quality", model=model_key,
            latency_ms=0, passed=None, errors=[f"exception: {e}"],
        ))
        return results

    errors = []
    if not result.idle.strip():
        errors.append("idle line is empty")

    results.append(RoleResult(
        role="condition_idle", fixture="idle_quality", model=model_key,
        latency_ms=latency, passed=None, errors=errors,
        output={"met": result.met, "idle": result.idle},
        input_context="condition='The visitor has asked about the fliers' (not met)",
    ))
    return results


# ---------------------------------------------------------------------------
# All roles registry
# ---------------------------------------------------------------------------


OBJECTIVE_ROLES = [
    ("reconcile", test_reconcile),
    ("script_check", test_script_check),
    ("director", test_director),
    ("condition", test_condition),
    ("continue_listening", test_continue_listening),
    ("expression", test_expression),
    ("vision_synthesis", test_vision_synthesis),
    ("visual_focus", test_visual_focus),
]

SUBJECTIVE_ROLES = [
    ("plan", test_plan),
    ("next_beat", test_next_beat),
    ("thought", test_thought),
    ("eval", test_eval),
    ("condition_idle", test_condition_idle),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_all(model_config: dict, model_key: str) -> list[RoleResult]:
    all_results: list[RoleResult] = []

    console.print(f"\n[bold cyan]Objective roles[/bold cyan]")
    for role_name, test_fn in OBJECTIVE_ROLES:
        console.print(f"  [dim]{role_name}...[/dim]", end=" ")
        try:
            results = test_fn(model_config, model_key)
        except Exception as e:
            console.print(f"[red]CRASH: {e}[/red]")
            results = [RoleResult(role=role_name, fixture="all", model=model_key,
                                  latency_ms=0, passed=False, errors=[f"crash: {e}"])]
        for r in results:
            status = "[green]PASS[/green]" if r.passed else "[red]FAIL[/red]"
            console.print(f"{status} {r.fixture} ({r.latency_ms:.0f}ms)", end="  ")
            if r.errors:
                console.print(f"[dim]{r.errors[0]}[/dim]", end="")
            console.print()
        all_results.extend(results)

    console.print(f"\n[bold cyan]Subjective roles[/bold cyan]")
    for role_name, test_fn in SUBJECTIVE_ROLES:
        console.print(f"  [dim]{role_name}...[/dim]", end=" ")
        try:
            results = test_fn(model_config, model_key)
        except Exception as e:
            console.print(f"[red]CRASH: {e}[/red]")
            results = [RoleResult(role=role_name, fixture="all", model=model_key,
                                  latency_ms=0, passed=None, errors=[f"crash: {e}"])]
        for r in results:
            errs = f" [yellow]{len(r.errors)} structural issue(s)[/yellow]" if r.errors else ""
            console.print(f"[blue]REVIEW[/blue] {r.fixture} ({r.latency_ms:.0f}ms){errs}")
        all_results.extend(results)

    return all_results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_summary(results: list[RoleResult]):
    table = Table(title="Role Results Summary")
    table.add_column("Role", style="bold")
    table.add_column("Fixture")
    table.add_column("Model")
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
        table.add_row(r.role, r.fixture, r.model, f"{r.latency_ms:.0f}", status, err_text)

    console.print()
    console.print(table)


# ---------------------------------------------------------------------------
# JSON log
# ---------------------------------------------------------------------------


def save_log(results: list[RoleResult], model_key: str) -> Path:
    LOGS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"qa_roles_{model_key}_{timestamp}.json"
    data = {
        "type": "qa_roles",
        "model": model_key,
        "model_name": MODELS[model_key]["name"],
        "timestamp": datetime.now().isoformat(),
        "results": [
            {
                "role": r.role,
                "fixture": r.fixture,
                "model": r.model,
                "latency_ms": r.latency_ms,
                "passed": r.passed,
                "errors": r.errors,
                "output": r.output if isinstance(r.output, (dict, str)) else str(r.output),
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

    # Group results by (role, fixture)
    by_role_fixture: dict[tuple[str, str], dict[str, dict]] = {}
    for log in logs:
        model_key = log["model"]
        for r in log["results"]:
            key = (r["role"], r["fixture"])
            if key not in by_role_fixture:
                by_role_fixture[key] = {}
            by_role_fixture[key][model_key] = r

    models = [log["model"] for log in logs]
    model_names = {log["model"]: log["model_name"] for log in logs}

    # Build objective rows
    objective_roles = {name for name, _ in OBJECTIVE_ROLES}
    objective_html = ""
    for (role, fixture), model_results in by_role_fixture.items():
        if role not in objective_roles:
            continue
        objective_html += f"<tr><td>{_esc(role)}</td><td>{_esc(fixture)}</td>"
        for m in models:
            r = model_results.get(m)
            if r is None:
                objective_html += "<td>-</td><td>-</td><td>-</td>"
                continue
            latency = f"{r['latency_ms']:.0f}"
            if r["passed"] is True:
                status = '<span class="pass">PASS</span>'
            elif r["passed"] is False:
                status = '<span class="fail">FAIL</span>'
            else:
                status = '<span class="review">REVIEW</span>'
            errs = _esc("; ".join(r.get("errors", [])[:2]))
            objective_html += f"<td class='num'>{latency}</td><td>{status}</td><td class='err'>{errs}</td>"
        objective_html += "</tr>\n"

    # Build subjective comparison sections
    subjective_html = ""
    subjective_roles = {name for name, _ in SUBJECTIVE_ROLES}
    for (role, fixture), model_results in by_role_fixture.items():
        if role not in subjective_roles:
            continue
        subjective_html += f'<div class="comparison"><h3>{_esc(role)} / {_esc(fixture)}</h3>'

        # Show input context if available
        for m in models:
            r = model_results.get(m)
            if r and r.get("input_context"):
                subjective_html += f'<p class="context">Input: {_esc(r["input_context"])}</p>'
                break

        subjective_html += '<div class="side-by-side">'
        for m in models:
            r = model_results.get(m)
            mname = model_names.get(m, m)
            subjective_html += f'<div class="model-output"><h4>{_esc(mname)}</h4>'
            if r is None:
                subjective_html += "<p>(no data)</p>"
            else:
                latency = f"{r['latency_ms']:.0f}ms"
                errs = r.get("errors", [])
                err_html = f' <span class="fail">{len(errs)} issue(s)</span>' if errs else ""
                subjective_html += f'<p class="meta">{latency}{err_html}</p>'
                output = r.get("output", {})
                if isinstance(output, dict):
                    subjective_html += f'<pre>{_esc(json.dumps(output, indent=2))}</pre>'
                else:
                    subjective_html += f'<pre>{_esc(str(output))}</pre>'
            subjective_html += "</div>"
        subjective_html += "</div></div>\n"

    # Model header columns
    model_cols = ""
    for m in models:
        mname = model_names.get(m, m)
        model_cols += f'<th colspan="3">{_esc(mname)}</th>'

    sub_cols = ""
    for _ in models:
        sub_cols += "<th>Latency</th><th>Status</th><th>Errors</th>"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>QA Roles Comparison</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 1400px; margin: 2em auto; padding: 0 1em; background: #0d1117; color: #c9d1d9; }}
  h1, h2, h3 {{ color: #58a6ff; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
  th, td {{ padding: 8px 12px; border: 1px solid #30363d; text-align: left; }}
  th {{ background: #161b22; color: #58a6ff; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  td.err {{ color: #f85149; font-size: 0.85em; max-width: 300px; }}
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
<h1>QA Roles Comparison</h1>
<p class="meta">Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} &middot; Models: {', '.join(_esc(model_names.get(m, m)) for m in models)}</p>

<h2>Objective Results</h2>
<table>
<tr><th>Role</th><th>Fixture</th>{model_cols}</tr>
<tr><th></th><th></th>{sub_cols}</tr>
{objective_html}</table>

<h2>Subjective Comparison</h2>
{subjective_html}
</body></html>"""

    LOGS_DIR.mkdir(exist_ok=True)
    html_path = LOGS_DIR / f"qa_roles_comparison_{timestamp}.html"
    html_path.write_text(html)
    console.print(f"[bold]Comparison report: {html_path}[/bold]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Per-role LLM evaluation")
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS.keys()),
                        help=f"Model to test (default: {DEFAULT_MODEL})")
    parser.add_argument("--merge", nargs="+", metavar="JSON",
                        help="Merge existing role QA logs into comparison HTML")
    args = parser.parse_args()

    if args.merge:
        merge_and_report(args.merge)
        return

    model_config = MODELS[args.model]
    console.print(f"[bold]QA Roles — {model_config['name']}[/bold]")

    results = run_all(model_config, args.model)
    print_summary(results)
    log_path = save_log(results, args.model)

    obj_results = [r for r in results if r.passed is not None]
    obj_passed = sum(1 for r in obj_results if r.passed)
    obj_total = len(obj_results)
    sub_total = len(results) - obj_total

    console.print(f"\n[bold]Objective: {obj_passed}/{obj_total} passed[/bold]")
    console.print(f"[bold]Subjective: {sub_total} captured for review[/bold]")

    if obj_passed < obj_total:
        sys.exit(1)


if __name__ == "__main__":
    main()
