# Dashboard Timeline Perf Plan

## Goal

Make the dashboard stay responsive at 10-minute archive/live sessions with dense vision activity, while preserving smooth scrolling, scrubbing, selection, and inspector interaction.

This is primarily a rendering-architecture problem, not a WebAssembly problem.

## Current Problems

- `streamEvents` grows without a live cap.
- `renderStream()` rebuilds the whole timeline DOM with `innerHTML`.
- Card/dot listeners are rebound after every full render.
- Several polling loops keep running even when the tab is hidden or when archive mode does not need them.
- Dense lanes like `vision_raw` are especially expensive as DOM.

## Target Architecture

- Keep event/state data in JS.
- Use canvas/WebGL-style rendering for dense graphical timeline layers.
- Keep high-value rich UI in DOM:
  - inspector
  - selected card overlays
  - controls
  - modals
- Use virtualization for card-heavy lanes that stay in DOM.

## Phase 1: Stop Wasted Work

- Pause polling when `document.hidden`.
- Disable live-only polling while browsing archives.
- Throttle timeline rendering through a single `requestAnimationFrame` scheduler.
- Avoid repeated full `querySelectorAll()` sweeps where possible.

Acceptance:

- Hidden dashboard tabs should produce minimal CPU activity.
- Archive mode should not continue unnecessary model/runtime polling.

## Phase 2: Bound Live Growth

- Add a rolling cap for in-memory live events.
- Preserve full fidelity in saved archives/history, but do not require the live page to keep every event forever.
- Consider per-lane or per-type caps, with a larger allowance for conversational cards and a tighter allowance for dense raw vision events.

Acceptance:

- A 10-minute live run should not make the live dashboard heap or DOM grow without bound.

## Phase 3: Event Delegation

- Replace per-card/per-dot listeners with delegated handlers on the lane container.
- Keep per-event lookup in scene/event maps rather than attaching closures to every node.

Acceptance:

- Re-render cost should not scale linearly with listener rebinding.

## Phase 4: Canvas Timeline Surfaces

Move these first:

- timeline ruler
- playhead / scrub cursor
- turn rails
- `vision_raw` dense dot lane

Why first:

- These are graphical, dense, and do not need rich DOM semantics.
- This should remove the worst DOM pressure quickly.

Acceptance:

- `vision_raw` should remain smooth with very dense archives.
- Scrubbing and hover should stay responsive.

## Phase 5: Virtualized Card Lanes

Keep cards in DOM, but only render the visible time window plus overscan.

Priority lanes:

- `chat`
- `planner`
- `world`
- `vision`

Approach:

- Maintain full lane event arrays in JS.
- Compute visible time bounds from scroll position and scale.
- Render only intersecting cards plus a modest left/right overscan window.

Acceptance:

- DOM node count should remain roughly stable as archive duration grows.
- Scrolling should not stall when many cards exist outside the viewport.

## Phase 6: Profiling Harness

Create a repeatable perf check using a long archive:

- 10-minute archive target
- dense `vision_raw`
- representative chat/planner/world activity

Track:

- node count
- frame rate while scrolling
- frame rate while scrubbing
- main-thread time during live event ingest
- memory trend over a 10-minute open tab

## What Not To Do First

- Do not jump to full-canvas cards immediately.
- Do not introduce WebAssembly first.
- Do not optimize scene computation before fixing DOM churn and polling waste.

## Likely Final Hybrid

- Canvas/WebGL:
  - ruler
  - playhead
  - turn rails
  - dense dot lanes
- Virtualized DOM:
  - rich text cards
  - selected/hovered event overlays
- DOM:
  - inspector
  - controls
  - modals

## Suggested Implementation Order

1. Hidden-tab and archive-mode polling cleanup.
2. `requestAnimationFrame` render scheduling.
3. Event delegation.
4. Live event cap.
5. Canvas ruler/playhead/rails.
6. Canvas `vision_raw`.
7. Virtualized DOM cards.
8. Long-archive profiling pass.

## Open Questions

- Whether `vision_raw` should become pure canvas or canvas plus a DOM overlay for selected items.
- Whether turn rails should remain separately selectable once canvas-backed.
- How aggressively to cap live event history before relying on archive save/load for deep inspection.
