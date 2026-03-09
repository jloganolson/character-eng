# Current State

## Local Product

The local-first path is now the mainline:

- warm boot with voice + vision
- heavy/hot split scripts
- paused-ready startup
- live dashboard stream view
- prompt inspector in the live dashboard
- restart/pause/resume working against the real runtime
- Playwright coverage for both the trace viewer and live dashboard

Recent commits in this phase:

- `d5b4e3d` Simplify dashboard runtime controls
- `2adadc3` Improve dashboard timeline navigation
- `48eb130` Add prompt details to dashboard timeline
- `3a51cb4` Use shift-wheel for timeline panning
- `fc9123c` Fix live dashboard restart and input flow

## Runtime Now

Current live app URL:

- dashboard: `http://127.0.0.1:56029/`
- vision UI: `http://127.0.0.1:7860/`

Heavy services remain reusable across app restarts:

- `./scripts/run_heavy.sh`
- `./scripts/run_hot.sh`
- `./scripts/stop_local.sh`

## Validation

Most recent focused validation:

```bash
uv run pytest tests/test_dashboard.py tests/test_runtime_controls.py tests/test_dashboard_playwright.py
```

Result:

- `19 passed`

Also validated live:

- heavy boot
- hot app restart
- dashboard `/resume`
- dashboard `/restart`
- post-resume live turn path

## Remote / Deploy Archive

The remote/browser-bridge and deploy residue has been intentionally moved out of the repo worktree so this branch can stay focused on the core experience.

Filesystem archive:

- `/home/logan/Projects/character-eng-archive/remote-bridge-2026-03-09/`

Git stash reference:

- `stash@{0}` `On full-automation-tinker: archive remote bridge and deploy residue`

That archived bucket includes the old:

- `character_eng/bridge.py`
- `character_eng/browser_voice.py`
- `tests/test_bridge.py`
- `Dockerfile`
- `.github/`
- `deploy/`
- `scripts/validate.sh`
- `scripts/vision_smoke.sh`
- `vision_plan.md`
- vision log files

## Remaining Dirty State

Still dirty in the repo:

- `CLAUDE.md` (unrelated local edit)
- `.claude/` (local workspace residue; do not commit)

## Burndown

What is actually left now:

1. Keep iterating on the core local experience.
   Focus:
   - spoken line quality
   - scenario/stage behavior
   - vision usefulness and grounding
   - dashboard readability

2. Clean docs around the new local workflow.
   Files:
   - `README.md`
   - this file
   - `AGENT_START.md`
   - maybe `CLAUDE.md` once the product direction is stable

3. Later, decide whether to revive the archived remote/deploy bucket.
   That is now a deliberate future decision, not active branch clutter.
