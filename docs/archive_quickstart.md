# Archive Quickstart

Use this when a live run goes bad and you want to iterate from the exported archive instead of trying to reproduce the issue live.

For the deeper workflow, see [archive_debug_playbook.md](./archive_debug_playbook.md).

## Minimal input

Usually this is enough:

- archive id or archive path
- one-line issue summary

Example:

```bash
./scripts/archive_debug.sh 126f4a56_2026-03-26-08-29-greg-export_1774539032 \
  "name acknowledged in chat but not sticking in state"
```

## The 5-step loop

### 1. Run triage

```bash
./scripts/archive_debug.sh <archive-id-or-path> "<issue summary>"
```

This gives you:

- dialogue timeline
- name/state timeline
- state fidelity flags
- recommended next pass

### 2. Find the first wrong thing

Do not start with prompt edits.

First answer:

1. What is the earliest incorrect event?
2. Which subsystem introduced it?
3. Did chat misunderstand the user, or did state fail after chat got it right?

Use this simple sort:

- transcript wrong: STT/audio problem
- transcript right, reply wrong: chat/prompt problem
- reply right, state wrong: state-mutation problem
- state right, UI wrong: rendering/dashboard problem

### 3. Turn that moment into a replayable test

Examples:

- bad `vision_state_update`: add or tighten a fixture in `tests/vision_snippets/fixtures/...`
- bad transcript-to-state behavior: add a unit or integration test around transcript promotion

If you cannot replay the bug offline, iteration will stay slow.

### 4. Add the smallest guard that fixes the real failure

Prefer deterministic fixes before prompt tuning when the model is clearly overreaching.

Examples:

- only allow `set_name` when the source text explicitly contains the name
- strip speculative state facts like `could be`, `possibly`, `suggesting`
- prefer transcript self-identification over inferred visual identity

### 5. Re-run offline, then go back live

Before returning to live testing:

- run the targeted tests
- rerun the archive command
- confirm the earliest bad event no longer happens

## What “good” looks like

A successful archive-debug pass should produce:

1. a short diagnosis
2. a replayable fixture or test
3. a minimal verified fix

If you only have a theory, you are not done yet.

## When to pull in media

Only inspect audio/video if `events.jsonl` is not enough to answer:

- did STT hear the user correctly?
- was the visual cue actually present?
- did the VLM have plausible evidence?

If the transcript, prompt trace, and state events already show the bug, stay in JSON first.

## Concrete lesson from the March 26 archive

The naming archive turned out to be two bugs, not one:

- vision/state hallucinated `p1 -> Greg`
- later the user said `my name is logan`
- chat understood `Logan`
- tracked state still failed to settle correctly

That is why archive-first iteration matters: it tells you whether you have a prompt problem, a state problem, or multiple chained problems.
