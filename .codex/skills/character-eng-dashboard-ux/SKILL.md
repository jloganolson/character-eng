---
name: character-eng-dashboard-ux
description: Iterate on the Character Engine dashboard UX using the offline archive-first workflow. Use when working in `character-eng-uv-v3` on `character_eng/dashboard/index.html`, archive-only dashboard behavior, timeline/layout polish, status/sidebar UX, or Playwright verification around the saved dashboard fixture.
---

# Character Engine Dashboard UX

## Start Here

- Open `/home/logan/Projects/character-eng-uv-v3/docs/dashboard_ux_workflow.md` first.
- Prefer the offline loop before touching the full live stack.
- From the repo root, run `./scripts/run_hot_offline.sh`.
- Expect the browser to land on `http://127.0.0.1:7862/?archive=c206a6db`.
- Confirm the boot overlay is hidden, the archive label shows `sample`, and the timeline is populated before editing.

## Working Rules

- Treat the current process state and the recorded archive state as separate truths.
- Keep the current status panels truthful to the current offline `archive-only` process.
- Allow recorded live-session events to remain in the timeline and history views.
- Do not let archived `runtime_controls` overwrite current runtime/sidebar state.
- Keep live vision/model polling suppressed while browsing archives.

## Main Files

- `/home/logan/Projects/character-eng-uv-v3/character_eng/dashboard/index.html`
- `/home/logan/Projects/character-eng-uv-v3/scripts/run_hot_offline.sh`
- `/home/logan/Projects/character-eng-uv-v3/character_eng/dashboard/server.py`
- `/home/logan/Projects/character-eng-uv-v3/character_eng/__main__.py`
- `/home/logan/Projects/character-eng-uv-v3/tests/test_dashboard_playwright.py`

## Verification

- Run `uv run pytest tests/test_dashboard_playwright.py -q -k archive_query_autoload_hides_boot_overlay` after changing archive boot, sidebar/runtime state, or offline archive UX.
- Run `uv run pytest tests/test_dashboard.py -q` for the dashboard server/state slice when touching bootstrapping or archive routes.

## Handoff

- Start with the offline archive fixture unless the task explicitly requires live voice, live vision, or live model behavior.
- Summarize any UX issues as either `current offline state confusion` or `recorded archive presentation` so future agents do not mix them together.
