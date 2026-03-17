# Vision Snippets

Typed snippet QA lives here. The storage substrate is still the shared history/snippet system, but each snippet mode gets its own evaluator and expectations.

## Vision state fixtures

- Fixtures for the `vision_state_turn` snippet type live under `fixtures/moments/...`.
- Each fixture snippet directory should include:
  - `manifest.json`
  - `checkpoint.json`
- `manifest.json` must set `capture_mode` to `vision_state_turn` and include `bundle.vision_state.task_answers` plus `bundle.vision_state.expected_update`.

## Local commands

Unit coverage for the typed evaluator:

```bash
uv run pytest tests/test_qa_vision_snippets.py
```

Browser coverage for dashboard capture defaults:

```bash
uv run pytest tests/test_dashboard_playwright.py -k vision_state_event_defaults_snippet_mode
```

Run the evaluator against the local fixture corpus and emit an HTML report:

```bash
uv run -m character_eng.qa_vision_snippets \
  --root tests/vision_snippets/fixtures \
  --tag vision-state \
  --report logs/qa_vision_snippets_fixture_report.html
```

By default this uses the runtime `MICRO_MODEL`, so the snippet report follows the same structured-update model path as production. Override `--model` only when you explicitly want an A/B comparison.

That report is for hands-on semantic checks. The unit tests mock the interpreter so the regression suite stays deterministic.
