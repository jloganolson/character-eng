from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Iterable

from character_eng.config import load_config
from character_eng.models import MODELS
from character_eng.robot_sim import RobotSimController
from character_eng.world import RobotActionResult, RobotActionTrigger, robot_action_call


ActionDecider = Callable[[RobotActionTrigger, dict], RobotActionResult]


@dataclass(frozen=True)
class RobotSimEvalCase:
    name: str
    trigger: RobotActionTrigger
    expected_action: str
    interaction: str = "none"
    notes: str = ""


@dataclass
class RobotSimEvalTrace:
    step: str
    snapshot: dict


@dataclass
class RobotSimEvalResult:
    case: str
    expected_action: str
    actual_action: str
    passed: bool
    reason: str
    notes: str = ""
    traces: list[RobotSimEvalTrace] = field(default_factory=list)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["traces"] = [
            {"step": trace.step, "snapshot": trace.snapshot}
            for trace in self.traces
        ]
        return payload


def default_robot_sim_eval_cases() -> list[RobotSimEvalCase]:
    return [
        RobotSimEvalCase(
            name="visual_arrival",
            trigger=RobotActionTrigger(
                source="vision",
                kind="perception_batch",
                summary="A person just stepped into view.",
                details=["person_visible"],
                people_text="- Visitor is present",
            ),
            expected_action="wave",
            notes="First visual notice should feel like a greeting beat.",
        ),
        RobotSimEvalCase(
            name="greeting_dialogue",
            trigger=RobotActionTrigger(
                source="dialogue",
                kind="assistant_reply",
                summary="Assistant produced a spoken reply.",
                latest_user_message="hi greg",
                assistant_reply="Hey. Good to see somebody.",
                history=[
                    {"role": "user", "content": "hi greg"},
                    {"role": "assistant", "content": "Hey. Good to see somebody."},
                ],
                people_text="- Visitor is present",
            ),
            expected_action="wave",
            notes="Plain greeting should still map to wave.",
        ),
        RobotSimEvalCase(
            name="introduction_by_name",
            trigger=RobotActionTrigger(
                source="dialogue",
                kind="assistant_reply",
                summary="Assistant produced a spoken reply.",
                latest_user_message="Hi, I am Maya.",
                assistant_reply="Nice to meet you, Maya.",
                history=[
                    {"role": "user", "content": "Hi, I am Maya."},
                    {"role": "assistant", "content": "Nice to meet you, Maya."},
                ],
                people_text="- Maya is present",
            ),
            expected_action="offer_fistbump",
            interaction="contact",
            notes="Greeting plus name-sharing should prioritize the introduction beat.",
        ),
        RobotSimEvalCase(
            name="introduction_with_task_pivot",
            trigger=RobotActionTrigger(
                source="dialogue",
                kind="assistant_reply",
                summary="Assistant produced a spoken reply.",
                latest_user_message="Hey, I'm Logan. Do you know what time it is?",
                assistant_reply="Nice to meet you, Logan. I don't know the time yet.",
                history=[
                    {"role": "user", "content": "Hey, I'm Logan. Do you know what time it is?"},
                    {"role": "assistant", "content": "Nice to meet you, Logan. I don't know the time yet."},
                ],
                people_text="- Logan is present",
            ),
            expected_action="offer_fistbump",
            interaction="contact",
            notes="Task content should not erase an introduction beat.",
        ),
        RobotSimEvalCase(
            name="plain_factual_qa",
            trigger=RobotActionTrigger(
                source="dialogue",
                kind="assistant_reply",
                summary="Assistant produced a spoken reply.",
                latest_user_message="What time is it?",
                assistant_reply="I don't know yet.",
                history=[
                    {"role": "user", "content": "What time is it?"},
                    {"role": "assistant", "content": "I don't know yet."},
                ],
                people_text="- Visitor is present",
            ),
            expected_action="none",
            notes="Ordinary Q and A should stay still.",
        ),
        RobotSimEvalCase(
            name="introduction_timeout",
            trigger=RobotActionTrigger(
                source="dialogue",
                kind="assistant_reply",
                summary="Assistant produced a spoken reply.",
                latest_user_message="I'm Robin.",
                assistant_reply="Nice to meet you, Robin.",
                history=[
                    {"role": "user", "content": "I'm Robin."},
                    {"role": "assistant", "content": "Nice to meet you, Robin."},
                ],
                people_text="- Robin is present",
            ),
            expected_action="offer_fistbump",
            interaction="timeout",
            notes="Timeout path should also be easy to rehearse.",
        ),
    ]


def select_robot_sim_eval_cases(
    cases: Iterable[RobotSimEvalCase],
    selected_names: Iterable[str],
) -> list[RobotSimEvalCase]:
    requested = [str(name or "").strip() for name in selected_names if str(name or "").strip()]
    if not requested:
        return list(cases)
    available = {case.name: case for case in cases}
    missing = sorted(name for name in requested if name not in available)
    if missing:
        raise KeyError(f"unknown sim-eval case(s): {', '.join(missing)}")
    return [available[name] for name in requested]


def load_default_model_config(model_key: str = "") -> dict:
    config = load_config()
    resolved_key = model_key.strip() or config.models.micro_model
    if resolved_key not in MODELS:
        raise KeyError(f"unknown model key: {resolved_key}")
    model_config = dict(MODELS[resolved_key])
    api_key_env = str(model_config.get("api_key_env") or "").strip()
    if api_key_env and not os.environ.get(api_key_env):
        raise RuntimeError(
            f"missing API key for model '{resolved_key}': set {api_key_env}"
        )
    return model_config


def default_action_decider(model_config: dict) -> ActionDecider:
    def _decide(trigger: RobotActionTrigger, robot_state: dict) -> RobotActionResult:
        return robot_action_call(
            model_config=model_config,
            trigger=trigger,
            robot_state=robot_state,
        )

    return _decide


def run_robot_sim_eval_case(
    case: RobotSimEvalCase,
    *,
    decide_action: ActionDecider,
    controller: RobotSimController | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> RobotSimEvalResult:
    robot = controller or RobotSimController()
    traces: list[RobotSimEvalTrace] = []
    decision = decide_action(case.trigger, robot.snapshot())
    traces.append(RobotSimEvalTrace(step="decision", snapshot={"action": decision.action, "reason": decision.reason}))

    if decision.action == "wave":
        traces.append(
            RobotSimEvalTrace(
                step="after_apply",
                snapshot=robot.trigger_wave(source=f"sim_eval:{case.name}"),
            )
        )
    elif decision.action == "offer_fistbump":
        traces.append(
            RobotSimEvalTrace(
                step="after_apply",
                snapshot=robot.offer_fistbump(source=f"sim_eval:{case.name}"),
            )
        )
        session_id = traces[-1].snapshot["fistbump"]["session_id"]
        if case.interaction == "contact":
            traces.append(
                RobotSimEvalTrace(
                    step="after_contact",
                    snapshot=robot.register_fistbump_contact(
                        session_id=session_id,
                        source=f"sim_eval:{case.name}:contact",
                    ),
                )
            )
            sleep_fn(getattr(robot, "_fistbump_result_hold_s", 0.0) + 0.01)
            traces.append(
                RobotSimEvalTrace(step="after_retract", snapshot=robot.snapshot())
            )
        elif case.interaction == "timeout":
            sleep_fn(getattr(robot, "_fistbump_timeout_s", 0.0) + 0.01)
            traces.append(
                RobotSimEvalTrace(step="after_timeout", snapshot=robot.snapshot())
            )
            sleep_fn(getattr(robot, "_fistbump_result_hold_s", 0.0) + 0.01)
            traces.append(
                RobotSimEvalTrace(step="after_retract", snapshot=robot.snapshot())
            )
    else:
        traces.append(RobotSimEvalTrace(step="after_apply", snapshot=robot.snapshot()))

    return RobotSimEvalResult(
        case=case.name,
        expected_action=case.expected_action,
        actual_action=decision.action,
        passed=decision.action == case.expected_action,
        reason=decision.reason,
        notes=case.notes,
        traces=traces,
    )


def run_robot_sim_eval_cases(
    cases: list[RobotSimEvalCase],
    *,
    decide_action: ActionDecider,
    controller_factory: Callable[[], RobotSimController] = RobotSimController,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> list[RobotSimEvalResult]:
    results: list[RobotSimEvalResult] = []
    for case in cases:
        results.append(
            run_robot_sim_eval_case(
                case,
                decide_action=decide_action,
                controller=controller_factory(),
                sleep_fn=sleep_fn,
            )
        )
    return results


def render_robot_sim_eval_text(results: list[RobotSimEvalResult]) -> str:
    lines: list[str] = []
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        lines.append(
            f"{status} {result.case}: expected {result.expected_action}, got {result.actual_action}"
        )
        if result.reason:
            lines.append(f"  reason: {result.reason}")
        if result.notes:
            lines.append(f"  note: {result.notes}")
        for trace in result.traces:
            if trace.step == "decision":
                continue
            lines.append(f"  {trace.step}: {_summarize_trace_snapshot(trace.snapshot)}")
    return "\n".join(lines)


def render_robot_sim_eval_json(results: list[RobotSimEvalResult]) -> str:
    return json.dumps([result.to_dict() for result in results], indent=2)


def robot_sim_eval_passed(results: Iterable[RobotSimEvalResult]) -> bool:
    return all(result.passed for result in results)


def _summarize_trace_snapshot(snapshot: dict) -> str:
    if not isinstance(snapshot, dict):
        return "state updated"
    fistbump = snapshot.get("fistbump")
    if isinstance(fistbump, dict):
        fist_state = str(fistbump.get("state") or "").strip()
        if fist_state and fist_state != "idle":
            return f"fistbump={fist_state}"
    wave = snapshot.get("wave")
    if isinstance(wave, dict) and bool(wave.get("active")):
        return "wave=active"
    headline = str(snapshot.get("headline") or "").strip()
    return headline or "state updated"
