# Robot Sim Eval Playbook

Use this when you want a controlled rehearsal path between unit tests and a messy live runtime session.
If you already have an exported archive, prefer the archive-debug workflow instead. This playbook is for trying behavior without live/archive data.

## When to use it

- prompt iteration for robot actions without mic or camera noise
- checking whether a new social behavior is too eager, too timid, or clearly wrong
- rehearsing interaction flows like wave, fist-bump contact, or timeout
- adding a small regression before or after changing prompts or runtime wiring

## Default command

```bash
./scripts/robot_sim_eval.sh
```

That runs the canned robot-action pack against the real `robot_action_call()` controller and the real `RobotSimController`, but with fast sim timings so the pass is quick.

## Useful variants

List the available canned cases:

```bash
./scripts/robot_sim_eval.sh --list-cases
```

Run only focused introduction cases while iterating on prompt wording:

```bash
./scripts/robot_sim_eval.sh \
  --case introduction_by_name \
  --case introduction_with_task_pivot
```

Capture structured output for comparison or logs:

```bash
./scripts/robot_sim_eval.sh --json
```

Use the real configured robot-action model by default, or override it:

```bash
./scripts/robot_sim_eval.sh --model-key gpt5_mini
```

## Canonical pack

Keep the canned pack small and stable. Right now it covers:

- `visual_arrival` -> `wave`
- `greeting_dialogue` -> `wave`
- `introduction_by_name` -> `offer_fistbump`
- `introduction_with_task_pivot` -> `offer_fistbump`
- `plain_factual_qa` -> `none`
- `introduction_timeout` -> `offer_fistbump`, then timeout/retract in the sim

If a prompt change breaks one of these, fix that before doing broader live evaluation.

## Workflow

1. Start with a focused case or two while changing the prompt or runtime.
2. Run the full canned pack before considering the change stable.
3. If behavior is wrong, add or adjust a deterministic test in `tests/test_sim_eval.py`.
4. If behavior is still ambiguous, move to a live dashboard run only after the harness looks clean.

## Repo paths

- Harness module: `character_eng/sim_eval.py`
- CLI entry: `scripts/run_robot_sim_eval.py`
- Shell wrapper: `scripts/robot_sim_eval.sh`
- Prompt: `prompts/robot_action_system.txt`
- Runtime action path: `character_eng/world.py` and `character_eng/__main__.py`
- Regression tests: `tests/test_sim_eval.py`
