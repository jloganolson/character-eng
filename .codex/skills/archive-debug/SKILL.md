---
name: archive-debug
description: Use when the user wants to debug from an exported archive, history session, snippet, or offline replay instead of live data. Triggers on requests like debug from this archive, replay this session offline, use the export, iterate from history, or archive-based debugging. Default inputs are an archive id or path plus a one-sentence issue summary.
---

# Archive Debug

Use this skill for archive-first debugging. The default assumption is that an archive ref or path plus a short issue summary is sufficient to begin.

## Inputs

Usually ask for only:

- archive id or archive path
- short issue summary

Only ask for more if the archive is missing `manifest.json`, `events.jsonl`, or `checkpoint.json`, or if the issue depends on external context not captured in history.

## Workflow

1. Resolve the archive from the provided id or path.
2. Run the repo triage command if present:
   - `./scripts/archive_debug.sh <archive-ref-or-path> "<issue summary>"`
3. Use `events.jsonl` first:
   - inspect transcript, assistant reply, `people_state`, `world_state`, `vision_state_update`, `reconcile`, and prompt traces
4. Classify the bug before changing anything:
   - transcript/STT
   - prompt/chat behavior
   - state mutation
   - UI/rendering
5. Turn the earliest incorrect event into a replayable fixture or test.
6. Prefer deterministic guards before prompt tuning when the model is clearly overreaching.
7. Pull in media only when JSON leaves ambiguity about STT or visual grounding.

## Repo-specific paths

If working inside `character-eng`, use these:

- Quickstart: `docs/archive_quickstart.md`
- Playbook: `docs/archive_debug_playbook.md`
- Postmortem: `docs/archive_debug_postmortem.md`
- Quick triage: `scripts/archive_debug.sh`
- Naming-specific timeline: `scripts/debug_archive_naming.py`
- Vision fixture QA: `tests/vision_snippets/fixtures/` and `character_eng/qa_vision_snippets.py`
- Repo reference: read `references/character-eng.md` when you need the exact archive-debug loop for this codebase

## Expected output

Each archive-debug pass should produce:

1. The earliest bad event
2. The subsystem that introduced it
3. A replayable regression artifact
4. The minimal fix and verification

If the user gives only an archive id and issue summary, that is normally enough to start.
