# Vision System Change Log

Date: 2026-03-16

## Goal

Make the vision stack easier to reason about and easier to inspect by:

- adding deterministic raw-vision triggers that scenario logic can depend on
- making core VLM work first-class instead of opaque question strings
- routing system-grade VLM answers into direct world/person updates
- making the dashboard expose the raw inputs and the follow-on interpretation clearly

## Implementation Notes

### Slice 1: Deterministic triggers and structured VLM tasks

- Added stage/global `vision_trigger` support with frame debouncing in the scenario schema.
- Evaluated those triggers directly in `VisionManager` against raw SAM3 / presence / VLM outputs.
- Added named VLM task specs with `task_id`, `label`, `target`, `cadence_s`, and `interpret_as`.
- Introduced system tasks:
  - `world_state`
  - `person_description_static`
  - `person_description_dynamic`
- Added nearest-person targeting in the vision service by cropping the frame to the primary tracked person when appropriate.
- Removed gaze from the default VLM prompt path and from scenario reliance.

### Slice 2: Direct interpretation path

- Added a dedicated vision-state interpreter prompt and call path.
- Changed system-grade VLM task answers to flow into a structured `WorldUpdate` / `person_updates` result instead of generic pending text only.
- Applied those updates directly in `process_perception` via a new `vision_state_update` event kind.
- Emitted raw dashboard events for:
  - deterministic trigger firings
  - direct vision-state updates

### Slice 3: Typed vision-state snippets and evaluator

- Added `vision_state_turn` as a first-class snippet capture mode in the dashboard.
- Captured system VLM task answers plus expected world/person deltas into `bundle.vision_state`.
- Added a dedicated evaluator module: `character_eng.qa_vision_snippets`.
- Added a typed fixture corpus under `tests/vision_snippets/fixtures`.
- Added a local HTML report flow for hands-on semantic checks.
- Tightened evaluator diffing so extra world facts, extra events, and unexpected person updates are flagged as review failures instead of being silently tolerated.

## Files Touched

- `character_eng/scenario.py`
- `character_eng/vision/vlm.py`
- `character_eng/vision/focus.py`
- `character_eng/vision/client.py`
- `character_eng/vision/context.py`
- `character_eng/vision/manager.py`
- `character_eng/vision/interpret.py`
- `character_eng/perception.py`
- `services/vision/app.py`
- `services/vision/face_tracker.py`
- `character_eng/dashboard/index.html`
- `character_eng/dashboard/stream_schema.json`
- `character_eng/qa_vision_snippets.py`
- `prompts/vision_constant_questions.txt`
- `prompts/vision_synthesis.txt`
- `prompts/vision_state_update.txt`
- `prompts/characters/greg/scenario_script.toml`
- `tests/test_qa_vision_snippets.py`
- `tests/vision_snippets/README.md`
- `tests/vision_snippets/fixtures/moments/visitor_backpack_state_update/manifest.json`
- `tests/vision_snippets/fixtures/moments/visitor_backpack_state_update/checkpoint.json`

## Test / Validation Log

### Automated

- `uv run pytest tests/test_scenario.py tests/test_vision.py tests/test_prompt_hot_reload.py`
  - passed
- `uv run pytest tests/test_vision_interpret.py tests/test_perception.py`
  - passed
- `uv run pytest tests/test_dashboard.py`
  - passed
- `uv run pytest tests/test_dashboard_playwright.py -k vision_panel_stays_visible_without_source`
  - passed
- `uv run pytest tests/test_qa_vision_snippets.py tests/test_vision_interpret.py tests/test_dashboard.py`
  - passed
- `uv run pytest tests/test_dashboard_playwright.py -k 'vision_state_event_defaults_snippet_mode or vision_panel_stays_visible_without_source'`
  - passed
- `uv run pytest tests/test_dashboard_playwright.py tests/test_qa_full_stack_playwright.py`
  - passed
- `uv run pytest tests/test_qa_full_stack.py`
  - passed
- `uv run pytest tests/test_scenario.py tests/test_e2e.py`
  - passed
- `uv run pytest tests/test_runtime_world_sync.py tests/test_runtime_controls.py`
  - passed
- `uv run pytest tests/test_e2e.py`
  - passed
- `uv run -m character_eng.qa_vision_snippets --root tests/vision_snippets/fixtures --tag vision-state --model gemini-2.5-flash --report logs/qa_vision_snippets_fixture_report.html`
  - passed
  - report: `logs/qa_vision_snippets_fixture_report.html`

### Runtime issues found during validation

- The mock-vision e2e path initially failed because `_ensure_vision_manager()` was using `scenario` without accepting it as an argument.
- Fixed by threading `scenario` through the helper and its call sites.
- The mock-vision dialogue sim later failed because `director_call()` was passing through a string `exit_index`, and `run_post_response()` compares that field numerically.
- Fixed by coercing `exit_index` to `int` with a safe fallback to `-1`, and added a regression test for the string case.

## Quality Notes

- No browser/runtime regressions were found in the dashboard Playwright suite after the new raw event types were added.
- The local mock-vision dialogue e2e completed successfully after the `scenario` wiring fix, so the end-to-end loop still boots, plans, consumes vision, and exits cleanly.
- The typed vision-state snippet evaluator now gives us a repeatable way to spot semantic drift in those system VLM updates without forcing that use case into the barge-in evaluator shape.
- The current fixture corpus is intentionally small. The next quality step is to add real captured `vision_state_turn` snippets from hands-on sessions so the report reflects actual scene ambiguity instead of only synthetic fixture cases.
