# Archive Debug Playbook

If you just want the short operator checklist, start with [archive_quickstart.md](./archive_quickstart.md).
If you are invoking Codex or another repo-aware agent, the matching skill name is `archive-debug`.

Use this when a live session goes wrong, you export an archive, and you want fast offline iteration instead of guessing against live state.

## Minimal handoff

Usually this is enough:

- archive id or archive path
- one-sentence issue summary

Example:

```bash
./scripts/archive_debug.sh 126f4a56_2026-03-26-08-29-greg-export_1774539032 \
  "name is acknowledged in chat but not sticking in state"
```

That should be the default handoff to another engineer or to Codex. Do not start by re-explaining the whole session unless the archive is incomplete.

## What the archive should answer first

Before touching prompts or media, answer these questions from `events.jsonl`:

1. What is the earliest incorrect event?
2. Is the error in transcript, prompt construction, state mutation, or UI rendering?
3. Did the model understand the user correctly but fail to write state, or did it misunderstand the input?
4. Is there one bug or multiple bugs chained together?

For the March 26, 2026 naming archive, the answer was:

- vision first hallucinated `p1 -> Greg`
- later the transcript clearly contained `my name is logan`
- chat used `Logan`
- tracked `people_state` did not absorb `Logan`

That means prompt-only iteration would have been too blunt. The failure was split across two subsystems.

The same archive can also surface quality issues beyond the headline bug, for example:

- speculative state facts such as `appears to`, `possibly`, or `might`
- summary labels that jump from `Person 1` / `Person 2` to a human name without evidence

That is useful because it lets you mine one exported archive for multiple offline regressions instead of treating it as a one-bug artifact.

## Default workflow

### 1. Run the archive triage command

```bash
./scripts/archive_debug.sh <archive-id-or-path> "<issue summary>"
```

This gives you:

- archive metadata
- event-type counts
- dialogue timeline
- name/state timeline
- state fidelity flags for suspicious world/person updates
- recommended next pass

### 2. Identify the failing source of truth

Use the timeline to sort the bug into one of these buckets:

- `user_transcript_final` is wrong: this is STT or audio capture
- transcript is right but assistant reply is wrong: this is prompt/chat behavior
- assistant reply is right but `people_state` or `world_state` is wrong: this is state mutation
- state is right but UI is wrong: this is dashboard/rendering

Do not tune prompts until you know which bucket you are in.

### 3. Turn the bad moment into a fixture

For structured visual bugs:

- extract the bad `vision_state_update` inputs
- create or update a `tests/vision_snippets/fixtures/...` fixture
- define the expected update that should have happened

For transcript/state bugs:

- extract the transcript text
- add a unit or small integration case that applies the transcript to `PeopleState`

The point is to make the failure replayable without live hardware.

### 4. Add deterministic guards before prompt tuning

If the model is clearly overreaching, prefer code-level sanitizers first.

Examples:

- strip `set_name` unless the source text contains explicit naming evidence
- prefer transcript-based name capture over inferred visual identity
- refuse unsupported presence labels or malformed IDs

Only after that should you edit prompts.

### 5. Use media only when JSON is not enough

Bring in audio or video when you need to answer one of these:

- did STT hear the user correctly?
- was the visual cue actually present?
- did the VLM have plausible evidence?

If `events.jsonl` already shows the transcript and prompt trace clearly, media is optional.

## Fast decision table

- Transcript wrong: inspect audio and STT first
- Transcript right, reply wrong: inspect prompt blocks and chat context
- Reply right, state wrong: inspect `vision_state_update`, `reconcile`, or transcript promotion code
- State right, dashboard wrong: inspect archive rendering and event mapping

## Recommended outputs from an archive-debug pass

Every pass should produce all three:

1. A short written diagnosis
2. A replayable fixture or test
3. A minimal fix with verification

If you only have a diagnosis, you do not yet have a stable workflow.

## Commands

Quick triage:

```bash
./scripts/archive_debug.sh <archive-id-or-path> "<issue summary>"
```

Name timeline only:

```bash
python3 scripts/debug_archive_naming.py <archive-path>
```

Run snippet QA fixtures:

```bash
uv run pytest tests/test_qa_vision_snippets.py
```

Render the fixture corpus against the current interpreter:

```bash
uv run -m character_eng.qa_vision_snippets \
  --root tests/vision_snippets/fixtures \
  --tag vision-state \
  --report logs/qa_vision_snippets_fixture_report.html
```
