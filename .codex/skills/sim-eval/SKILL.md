---
name: sim-eval
description: Use when the user wants a controlled simulation or offline evaluation pass without live mic/camera input or archive data. Triggers on requests like run the sim harness, try this without going live, evaluate the robot behavior offline, rehearse prompt changes, or do a subjective sim review.
---

# Sim Eval

Use this skill for controlled runtime-style evaluation between unit tests and a live session.

## Inputs

Usually enough:

- the behavior or feature to evaluate
- optional case names or a short scenario summary

## Workflow

1. Start with the canned harness:
   - `./scripts/robot_sim_eval.sh`
2. List or focus cases as needed:
   - `./scripts/robot_sim_eval.sh --list-cases`
   - `./scripts/robot_sim_eval.sh --case introduction_by_name`
3. Use the fast default timings for prompt iteration unless you are specifically evaluating UX timing.
4. If behavior is wrong, add or adjust a deterministic case in `character_eng/sim_eval.py` and `tests/test_sim_eval.py` before escalating to a live run.
5. Use a live dashboard session only after the sim harness looks clean.

## Repo-specific paths

- Playbook: `docs/robot_sim_eval_playbook.md`
- Harness: `character_eng/sim_eval.py`
- CLI: `scripts/run_robot_sim_eval.py`
- Shell wrapper: `scripts/robot_sim_eval.sh`
- Prompt: `prompts/robot_action_system.txt`
- State machine: `character_eng/robot_sim.py`
- Runtime integration: `character_eng/world.py` and `character_eng/__main__.py`
- Regression tests: `tests/test_sim_eval.py`

## Expected output

Each sim-eval pass should produce:

1. The observed robot-action decision for each case
2. Relevant sim state traces like contact, timeout, and retract
3. A pass/fail result against the canonical pack
4. Any prompt or runtime fix plus verification
