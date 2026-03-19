# Dashboard UX Workflow

Use this when iterating on the dashboard UI without booting the full live stack.

## Offline Loop

Start the archive-only dashboard:

```bash
./scripts/run_hot_offline.sh
```

`./scripts/run_hot_offline.sh` now auto-installs the repo-owned canonical archive fixture into local `history/sessions/` if it is missing.

In archive-only mode, the dashboard now prefers the canonical real-data archive `greg-offline-canonical-v1` when it exists, then falls back to the latest saved archive with real data instead of relying on the stale old `uv-v3` sample id.

There is also a tiny offline picker in the header:

- `offline` frame dropdown for the main UX states
- archive dropdown for switching the real archive backing the offline view

You can still use direct URLs when you want them:

```text
http://127.0.0.1:7862/?fixture=loading
http://127.0.0.1:7862/?fixture=ready
http://127.0.0.1:7862/?fixture=startup-paused
http://127.0.0.1:7862/?fixture=paused
http://127.0.0.1:7862/?fixture=live
http://127.0.0.1:7862/?fixture=archive-only
```

These fixtures are intentionally dashboard-local:

- they live only in `character_eng/dashboard/index.html`
- they freeze the page into a representative UX frame
- they do not alter runtime behavior elsewhere in the app

## Data-Backed Archive Frame

Archive browsing is the one frame that still wants real session data, and it is now the default offline landing view.

The canonical fixture was captured from session `962fef15`, and it is installed locally as `greg-offline-canonical-v1`.

Use:

```text
http://127.0.0.1:7862/?archive=<session_id>
```

You can combine it with archive-only mode if you want the old offline archive workflow:

```text
http://127.0.0.1:7862/?fixture=archive-only&archive=<session_id>
```

## Verification

```bash
uv run pytest tests/test_dashboard.py -q
uv run pytest tests/test_dashboard_playwright.py -q -k "offline_fixture or archive_query_autoload"
```
