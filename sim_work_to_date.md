# Sim Work To Date

This file is the handoff point for the current robot-sim work on branch `robot-sim`.

## Current status

- Branch: `robot-sim`
- Base checkpoint commit for the sim/runtime/eval work: `0d734bd` (`Add robot sim runtime and eval harness`)
- Current dashboard visual is still the placeholder wireframe/test-panel version.
- I started a more graphical dashboard pass and then explicitly backed it out before committing.

## What is already built

### Runtime / backend

- A stateful robot simulation controller exists in `character_eng/robot_sim.py`.
- It tracks:
  - face state: gaze, gaze type, expression
  - wave: one-shot timer-based action
  - fist bump: offer, contact, timeout, cancel, retract
- The dashboard server exposes robot-sim state/action endpoints.
- The runtime pushes expression/gaze into the robot sim automatically.
- Robot actions are chosen by an LLM-driven controller, not string matching.

### Robot action architecture

- Robot actions now use a normalized trigger path rather than separate hardcoded dialogue/vision logic.
- Main types/files:
  - `character_eng/world.py`
  - `character_eng/__main__.py`
  - `prompts/robot_action_system.txt`
- Dialogue and perception both feed the same action decision path.
- Current supported LLM-chosen actions:
  - `none`
  - `wave`
  - `offer_fistbump`

### Dashboard / testing panel

- The dashboard includes a Robot Sim panel in `character_eng/dashboard/index.html`.
- It supports:
  - `minimized`
  - `column`
  - `fullscreen`
- It currently renders:
  - face state text
  - gaze/expression labels
  - wave state
  - fist-bump state
  - manual controls for wave / offer / contact / timeout / cancel / refresh
- This panel is intentionally usable for interactive testing, but the visuals are still placeholder/debug oriented.

### Offline eval path

- A reusable sim harness exists in `character_eng/sim_eval.py`.
- CLI:
  - `scripts/run_robot_sim_eval.py`
  - `scripts/robot_sim_eval.sh`
- Playbook:
  - `docs/robot_sim_eval_playbook.md`
- Codex skill:
  - `.codex/skills/sim-eval/SKILL.md`

## Canonical eval pack

The sim harness currently covers these canned cases:

- `visual_arrival` -> `wave`
- `greeting_dialogue` -> `wave`
- `introduction_by_name` -> `offer_fistbump`
- `introduction_with_task_pivot` -> `offer_fistbump`
- `plain_factual_qa` -> `none`
- `introduction_timeout` -> `offer_fistbump` then timeout/retract

Run it with:

```bash
./scripts/robot_sim_eval.sh
```

## Verification status

At the checkpoint commit, verification was:

- `uv run pytest -q` -> passing
- `./scripts/robot_sim_eval.sh` -> passing on the canonical pack

This means the runtime wiring and offline evaluation path are in good shape.
What has not happened yet is a strong live human-in-the-loop evaluation pass.

## Intentional rollback from the aborted visual pass

The next graphical sim pass should start fresh from the current committed dashboard state.

Important: do not assume there is partially finished graphical code waiting in the dashboard.
That work was reverted before this handoff commit.

## Recommended next step

The next session should focus on replacing the placeholder wireframe visuals in `character_eng/dashboard/index.html` while keeping the current runtime contract unchanged.

That means:

- keep the same robot-sim endpoints
- keep the same state payload shape
- keep the same manual controls
- improve only the presentation/rendering layer first

## Files most relevant for the next pass

- `character_eng/dashboard/index.html`
- `character_eng/robot_sim.py`
- `character_eng/world.py`
- `character_eng/__main__.py`
- `prompts/robot_action_system.txt`
- `character_eng/sim_eval.py`
- `tests/test_sim_eval.py`
- `tests/test_robot_sim.py`
- `tests/test_runtime_robot_actions.py`
- `tests/test_dashboard.py`
- `QA_TODO.md`

## Constraints for the next graphical pass

- Preserve the existing panel modes: `minimized`, `column`, `fullscreen`
- Preserve the current action/state contract
- Do not regress sidebar docking or fullscreen behavior
- Keep the panel useful for testing even before it becomes polished
- Prefer a clear visual mapping for:
  - expression
  - gaze direction
  - wave
  - fist-bump offer/contact/timeout

## If you want to re-brief Codex next time

Point the next session at this file and then specify:

1. the visual direction you want
2. whether to keep the current HTML/CSS-only approach or introduce canvas/svg
3. which expressions matter most to make legible first
4. whether wave/fist-bump should read more toy-like, robot-like, or character-like
