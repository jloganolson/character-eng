import time

from character_eng.robot_sim import RobotSimController
from character_eng.sim_eval import (
    RobotSimEvalCase,
    default_robot_sim_eval_cases,
    render_robot_sim_eval_text,
    robot_sim_eval_passed,
    run_robot_sim_eval_case,
    run_robot_sim_eval_cases,
    select_robot_sim_eval_cases,
)
from character_eng.world import RobotActionResult, RobotActionTrigger


def _fake_decider(action: str, reason: str = "test action"):
    def _decide(_trigger, _robot_state):
        return RobotActionResult(action=action, reason=reason)

    return _decide


def test_default_robot_sim_eval_cases_cover_core_social_beats():
    cases = {case.name: case for case in default_robot_sim_eval_cases()}

    assert cases["visual_arrival"].expected_action == "wave"
    assert cases["greeting_dialogue"].expected_action == "wave"
    assert cases["introduction_by_name"].expected_action == "offer_fistbump"
    assert cases["introduction_with_task_pivot"].expected_action == "offer_fistbump"
    assert cases["plain_factual_qa"].expected_action == "none"
    assert cases["introduction_timeout"].interaction == "timeout"


def test_select_robot_sim_eval_cases_preserves_requested_order():
    cases = default_robot_sim_eval_cases()

    selected = select_robot_sim_eval_cases(
        cases,
        ["plain_factual_qa", "visual_arrival"],
    )

    assert [case.name for case in selected] == ["plain_factual_qa", "visual_arrival"]


def test_run_robot_sim_eval_case_contact_flow_retracts_to_idle():
    case = RobotSimEvalCase(
        name="contact",
        trigger=RobotActionTrigger(
            source="dialogue",
            kind="assistant_reply",
            summary="Assistant produced a spoken reply.",
            latest_user_message="I'm Logan.",
            assistant_reply="Nice to meet you, Logan.",
        ),
        expected_action="offer_fistbump",
        interaction="contact",
    )

    result = run_robot_sim_eval_case(
        case,
        decide_action=_fake_decider("offer_fistbump", "introduction beat"),
        controller=RobotSimController(fistbump_result_hold_s=0.01),
        sleep_fn=time.sleep,
    )

    assert result.passed is True
    assert result.actual_action == "offer_fistbump"
    assert result.traces[-1].step == "after_retract"
    assert result.traces[-1].snapshot["fistbump"]["state"] == "idle"


def test_run_robot_sim_eval_case_timeout_flow_records_timeout_then_idle():
    case = RobotSimEvalCase(
        name="timeout",
        trigger=RobotActionTrigger(
            source="dialogue",
            kind="assistant_reply",
            summary="Assistant produced a spoken reply.",
            latest_user_message="I'm Robin.",
            assistant_reply="Nice to meet you, Robin.",
        ),
        expected_action="offer_fistbump",
        interaction="timeout",
    )

    result = run_robot_sim_eval_case(
        case,
        decide_action=_fake_decider("offer_fistbump", "introduction beat"),
        controller=RobotSimController(fistbump_timeout_s=0.01, fistbump_result_hold_s=0.01),
        sleep_fn=time.sleep,
    )

    assert result.passed is True
    assert result.traces[-2].step == "after_timeout"
    assert result.traces[-2].snapshot["fistbump"]["state"] == "timed_out"
    assert result.traces[-1].snapshot["fistbump"]["state"] == "idle"
    rendered = render_robot_sim_eval_text([result])
    assert "after_timeout: fistbump=timed_out" in rendered


def test_run_robot_sim_eval_cases_reports_failures_in_text_output():
    case = RobotSimEvalCase(
        name="plain_qa",
        trigger=RobotActionTrigger(
            source="dialogue",
            kind="assistant_reply",
            summary="Assistant produced a spoken reply.",
            latest_user_message="What time is it?",
            assistant_reply="I don't know yet.",
        ),
        expected_action="none",
    )

    results = run_robot_sim_eval_cases(
        [case],
        decide_action=_fake_decider("wave", "wrongly over-animated"),
    )

    assert robot_sim_eval_passed(results) is False
    rendered = render_robot_sim_eval_text(results)
    assert "FAIL plain_qa: expected none, got wave" in rendered
    assert "wrongly over-animated" in rendered
