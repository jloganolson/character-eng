You are continuing work in `/home/logan/Projects/character-eng` on branch `codex/full-automation-tinker`.

Read first:

1. `CURRENT_STATE.md`
2. `character_eng/dashboard/index.html`
3. `character_eng/__main__.py`

What matters:

- This branch is now intentionally local-first.
- The live dashboard and runtime path are the current product.
- The remote/browser-bridge + deploy residue has been archived out of the repo worktree.

Archive references:

- filesystem: `/home/logan/Projects/character-eng-archive/remote-bridge-2026-03-09/`
- stash: `stash@{0}` (`archive remote bridge and deploy residue`)

Do not start by reviving that archive unless explicitly asked.

Behavior rule:

- Do not add string-matching or regex heuristics for control flow if structured signals or model output can be used instead.

Current live URLs at handoff time:

- dashboard: `http://127.0.0.1:56029/`
- vision UI: `http://127.0.0.1:7860/`

Current validated commands:

```bash
uv run pytest tests/test_dashboard.py tests/test_runtime_controls.py tests/test_dashboard_playwright.py
```

Hot/heavy workflow:

```bash
./scripts/run_heavy.sh
./scripts/run_hot.sh
./scripts/stop_local.sh
```

Primary next work:

1. Iterate on the core local experience.
   Focus on:
   - response quality
   - vision grounding
   - scenario/stage flow
   - dashboard usability

2. Update top-level docs for the actual workflow now in use.
   Focus on:
   - `README.md`
   - `CURRENT_STATE.md`
   - maybe `CLAUDE.md` later

Still dirty and not yours:

- `CLAUDE.md`
- `.claude/`

Do not commit `.claude/`.
