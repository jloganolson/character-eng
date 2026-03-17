# Dashboard UX Workflow

Use this when iterating on the HTML dashboard without booting the full live stack.

## Default Offline Loop

Start the offline dashboard:

```bash
./scripts/run_hot_offline.sh
```

That starts `character_eng` in `--archive-only` mode and auto-loads the default sample archive.

Current default archive:

- Session id: `c206a6db`
- Title: `sample`
- Archive root: `history/sessions/c206a6db`

The browser should land on:

```text
http://127.0.0.1:7862/?archive=c206a6db
```

Expected state:

- boot overlay hidden
- archive header shows `Browsing archive: sample`
- archive action button shows `Return To Live`
- timeline is populated from the saved session

## Switching Fixtures

Override the default archive for one run:

```bash
DASHBOARD_DEFAULT_ARCHIVE=<session_id> ./scripts/run_hot_offline.sh
```

Example:

```bash
DASHBOARD_DEFAULT_ARCHIVE=c206a6db ./scripts/run_hot_offline.sh
```

## What Archive-Only Means

`archive-only` is the current process state. It means:

- no live conversation runtime
- no live voice startup
- no live vision/model startup
- dashboard and history APIs stay available

An archive may still contain recorded events from the original live session, including recorded `runtime_controls`. The dashboard should treat those as timeline/history data, not as the current process state.

## UX Rule For Agents

When browsing an archive:

- current status panels should describe the current offline process
- timeline cards can still show recorded live-session events
- archived `runtime_controls` must not overwrite the current archive-only status panel state
- live vision model polling should stay suppressed

This avoids the common confusion between:

- "what this saved session originally did"
- "what the dashboard process is doing right now"

## Files To Edit

Primary dashboard surface:

- `character_eng/dashboard/index.html`

Offline bootstrap path:

- `scripts/run_hot_offline.sh`
- `character_eng/dashboard/server.py`
- `character_eng/__main__.py`

## Verification

Fast browser verification:

```bash
uv run pytest tests/test_dashboard_playwright.py -q -k archive_query_autoload_hides_boot_overlay
```

Useful dashboard slice:

```bash
uv run pytest tests/test_dashboard.py -q
```

## Agent Handoff Checklist

Before changing layout or controls:

1. Start offline mode with `./scripts/run_hot_offline.sh`.
2. Confirm the page auto-loads `c206a6db`.
3. Confirm the boot overlay is hidden after archive hydration.
4. Confirm the runtime sidebar still says `archive only`, not recorded live voice/vision state.
5. Run the Playwright archive autoload test after UI changes.

## Known Good Baseline

As of March 17, 2026:

- offline archive autoload is wired through `DASHBOARD_DEFAULT_ARCHIVE`
- the boot overlay clears after archive hydration
- archived runtime events stay in the timeline but do not drive current runtime status panels
