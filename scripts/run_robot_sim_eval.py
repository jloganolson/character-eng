#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from character_eng.robot_sim import RobotSimController
from character_eng.sim_eval import (
    default_action_decider,
    default_robot_sim_eval_cases,
    load_default_model_config,
    render_robot_sim_eval_json,
    render_robot_sim_eval_text,
    robot_sim_eval_passed,
    run_robot_sim_eval_cases,
    select_robot_sim_eval_cases,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the robot sim evaluation harness against the live robot-action controller."
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Run only the named canned case. Repeat to run multiple cases.",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="List the canned robot sim eval cases and exit.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Render machine-readable JSON instead of text.",
    )
    parser.add_argument(
        "--model-key",
        default="",
        help="Override the configured model key used for robot_action_call.",
    )
    parser.add_argument(
        "--wave-duration-s",
        type=float,
        default=0.2,
        help="Wave duration for sim playback. Defaults to a fast eval value.",
    )
    parser.add_argument(
        "--fistbump-timeout-s",
        type=float,
        default=0.35,
        help="Fist-bump timeout for sim playback. Defaults to a fast eval value.",
    )
    parser.add_argument(
        "--fistbump-result-hold-s",
        type=float,
        default=0.1,
        help="How long contacted/timed-out fist-bump states are held before retracting.",
    )
    return parser


def _print_case_catalog() -> int:
    for case in default_robot_sim_eval_cases():
        print(f"{case.name}: {case.expected_action}")
        if case.notes:
            print(f"  {case.notes}")
    return 0


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.list_cases:
        return _print_case_catalog()

    cases = select_robot_sim_eval_cases(default_robot_sim_eval_cases(), args.case)
    model_config = load_default_model_config(args.model_key)
    decide_action = default_action_decider(model_config)

    def _controller_factory() -> RobotSimController:
        return RobotSimController(
            wave_duration_s=args.wave_duration_s,
            fistbump_timeout_s=args.fistbump_timeout_s,
            fistbump_result_hold_s=args.fistbump_result_hold_s,
        )

    results = run_robot_sim_eval_cases(
        cases,
        decide_action=decide_action,
        controller_factory=_controller_factory,
    )
    if args.json:
        print(render_robot_sim_eval_json(results))
    else:
        print(render_robot_sim_eval_text(results))
    return 0 if robot_sim_eval_passed(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
