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

## Post-Commit Validation

Date: 2026-03-17

### Broader regression sweep

- `uv run pytest tests/test_scenario.py tests/test_vision.py tests/test_vision_interpret.py tests/test_perception.py tests/test_dashboard.py tests/test_qa_vision_snippets.py tests/test_runtime_world_sync.py tests/test_runtime_controls.py tests/test_qa_full_stack.py`
  - passed
  - 98 tests passed
- `uv run pytest tests/test_dashboard_playwright.py tests/test_qa_full_stack_playwright.py`
  - passed
  - 15 tests passed
- `uv run pytest tests/test_e2e.py`
  - passed
  - includes the live mock-vision dialogue sim

### Snippet evaluator result from committed state

- `uv run -m character_eng.qa_vision_snippets --root tests/vision_snippets/fixtures --tag vision-state --report logs/qa_vision_snippets_fixture_report.html`
  - review, not pass
  - report: `logs/qa_vision_snippets_fixture_report.html`

Observed mismatch on the synthetic fixture:

- the micro-model removed the old world fact `f1`
- the micro-model removed the old person fact `p1f1`
- it paraphrased the new person fact as `The highlighted visitor is wearing a black backpack.` instead of the fixture’s exact expectation

Interpretation:

- runtime stability is good; the full app/browser/e2e sweep stayed green
- semantic behavior under the real micro-model is more update-oriented than the synthetic fixture expected
- this is exactly the kind of thing the typed snippet evaluator is supposed to reveal

### Vision raw lane UX follow-up

- The dashboard now makes the raw lane explicit as `vision raw` instead of the internal `vision_raw` label.
- When synthesized vision passes exist but no raw loop events are present, the timeline now renders a visible placeholder lane instead of silently omitting raw vision data.
- That makes the absence of SAM3 / VLM / trigger / state-update events diagnosable during hands-on testing instead of looking like the UI forgot the lane entirely.

Validation:

- `uv run pytest tests/test_dashboard.py`
  - passed
- `uv run pytest tests/test_dashboard_playwright.py -k 'vision_raw_lane_is_explicit_and_shows_placeholder_or_events or vision_panel_stays_visible_without_source or vision_state_event_defaults_snippet_mode'`
  - passed

### Vision raw cadence follow-up

- Hands-on validation exposed a real runtime bug in `apply_listening_expression()`: `ExpressionResult` was referenced without being imported in `character_eng/__main__.py`.
- Fixed by importing `ExpressionResult` from `character_eng.world`.
- The first raw-cadence attempt was incomplete: the manager was waking up faster, but raw polling and `vision_synthesis_call()` were still on the same thread, so a 2-3s synthesis call still stalled raw events.
- Fixed by splitting `VisionManager` into:
  - a fast raw poll loop
  - a separate synthesis loop that consumes the latest snapshot on the slower cadence
- Added `vision.raw_poll_interval` to config with a default of `0.2s`, while keeping `vision.synthesis_min_interval` at `0.75s`.
- Added a regression test proving raw dashboard events can outpace synthesis:
  - `tests/test_vision.py::test_vision_manager_raw_poll_is_faster_than_synthesis`

Validation:

- `uv run pytest tests/test_vision.py -k 'config_vision or emits_raw_dashboard_stage_events or raw_poll_is_faster_than_synthesis or carries_structured_synthesis_signals or emits_direct_state_updates_from_system_tasks or dashboard_snapshot_includes_objects_and_focus'`
  - passed
- `uv run pytest tests/test_e2e.py -k vision`
  - passed
- Hands-on live run against the running vision service:
  - initial bad behavior before the two-loop fix: raw events arrived roughly every 3s
  - after the two-loop fix: `vision_snapshot_read` / `sam3_detection` arrived about every `0.2s`
  - measured from `/state` on the live dashboard:
    - sample 1: `26` `vision_snapshot_read`, `26` `sam3_detection`
    - sample 2: `37` `vision_snapshot_read`, `37` `sam3_detection`
    - sample 3: `46` `vision_snapshot_read`, `46` `sam3_detection`
- Playwright check against the live dashboard:
  - confirmed `[data-stream-lane='vision_raw']` is visible
  - confirmed raw cards render in that lane, including `vision_snapshot_read`, `face_tracking`, and `sam3_detection`
  - sample counts from the rendered DOM:
    - `sam3_cards = 137`
    - `snapshot_cards = 137`
  - screenshot: `logs/playwright_live_vision_raw_check.png`
