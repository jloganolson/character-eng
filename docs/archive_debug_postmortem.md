# Archive Debug Postmortem

This document explains what went wrong in the March 26, 2026 archive, what we fixed, and how the new archive-debug workflow helped.

Use this as a concrete example of why exported archives are worth treating as a first-class debugging surface instead of just a record of a bad run.

## The original problem

The user reported a naming failure:

- Greg acknowledged the user's name in chat
- but the name was not reliably sticking in tracked state

At first glance, that sounds like one bug. The archive showed it was actually multiple bugs chained together.

## What the archive revealed

Running:

```bash
./scripts/archive_debug.sh 126f4a56_2026-03-26-08-29-greg-export_1774539032 \
  "name acknowledged in chat but not sticking in state"
```

surfaced this sequence:

1. `people_state` first flipped `p1 -> Greg`
2. `vision_state_update` then reinforced `set_name p1 -> Greg`
3. later the user clearly said `my name is logan`
4. chat replied with `Logan`
5. tracked state did not settle cleanly from that evidence

The important lesson: this was not just a prompt problem. It was a state mutation problem introduced upstream of chat.

## The second problem hiding in the same archive

Once the archive-debug command grew a state-fidelity pass, the same export also showed a second class of issues:

- person facts like `appears to`, `could be`, `possibly`
- world facts like `suggesting an entry or exit`
- age estimates and speculative visual inferences getting stored as state

That mattered because even if the naming bug were fixed, the system would still be carrying low-confidence facts as if they were stable truth.

So one archive gave us two regressions:

1. unsupported name assignment
2. speculative state drift

## What we fixed

### 1. Name writes now require explicit evidence

We added name-evidence guards so vision and reconcile cannot assign a name unless the source evidence explicitly contains that name.

Examples:

- good: transcript says `my name is Logan`
- bad: vision summary says `Updated person description for Greg` without any actual name evidence

Result:

- `p1 -> Greg` hallucinations from visual inference get stripped instead of persisted

Relevant files:

- [name_memory.py](/home/logan/Projects/character-eng/character_eng/name_memory.py)
- [interpret.py](/home/logan/Projects/character-eng/character_eng/vision/interpret.py)
- [world.py](/home/logan/Projects/character-eng/character_eng/world.py)

### 2. Transcript self-identification now promotes into state

When the user says something like `my name is Logan`, that evidence now gets promoted into tracked `people_state` before response generation.

Result:

- if chat understood the self-identification, state is much less likely to lag behind it

Relevant file:

- [__main__.py](/home/logan/Projects/character-eng/character_eng/__main__.py)

### 3. Speculative facts get trimmed or dropped before entering state

We added shared state-fidelity sanitization so clearly hedged claims do not persist as stable facts.

Examples:

- `A window allows natural light, and there's a door on the right, suggesting an entry or exit to this space.`
  became
  `A window allows natural light, and there's a door on the right`

- `An accessory could be a string around their neck, possibly for holding keys or earbuds.`
  became
  dropped entirely

- `The nearest visible person appears to have short brown hair...`
  became
  `The nearest visible person has short brown hair...`

Result:

- state is more grounded
- archive analysis and runtime sanitization now point in the same direction

Relevant files:

- [state_fidelity.py](/home/logan/Projects/character-eng/character_eng/state_fidelity.py)
- [archive_analysis.py](/home/logan/Projects/character-eng/character_eng/archive_analysis.py)
- [interpret.py](/home/logan/Projects/character-eng/character_eng/vision/interpret.py)
- [world.py](/home/logan/Projects/character-eng/character_eng/world.py)

## How the tool helped

The shell tool helped by being fast and deterministic.

What it does well:

- prints the dialogue timeline
- prints name/state transitions
- flags suspicious world/person state mutations
- gives a compact first pass over the archive

Why that mattered here:

- it quickly showed the earliest wrong event was a state mutation, not a bad chat response
- it exposed that the same archive had broader state-quality issues beyond naming

Relevant file:

- [archive_debug.sh](/home/logan/Projects/character-eng/scripts/archive_debug.sh)

## How the skill helped

The `archive-debug` skill helped by adding workflow judgment around the tool.

What it contributed:

- what minimal handoff is enough
- when to inspect `events.jsonl` first
- how to classify the failure as transcript vs prompt vs state vs UI
- when to make a deterministic guard instead of prompt-tuning
- how to turn the earliest bad event into a replayable test

Why that mattered here:

- it kept the work from collapsing into "just tweak the prompt"
- it led directly to regression fixtures and targeted tests

Relevant file:

- [archive-debug skill](/home/logan/.codex/skills/archive-debug/SKILL.md)

## How the tool and skill fit together

They are related, but they do different jobs.

- `archive_debug.sh` is the probe
- `archive-debug` is the workflow

The tool answers:

- what happened?

The skill answers:

- what should we do next?

That split turned out to be useful in practice:

- humans can run the script directly
- an agent can use the script output as one input into a larger debugging loop

## What this changed about the workflow

Before:

- exported archives were evidence
- debugging still depended too much on live reproduction and guessing

After:

- archive id/path plus one sentence is usually enough to start
- the earliest bad event can be found offline
- that event can be converted into a test or fixture
- fixes can be verified without hardware or live timing

## The main takeaway

The biggest win was not just "naming is better."

The biggest win was proving that one exported archive can drive:

1. diagnosis
2. tests
3. deterministic fixes
4. broader best practices

That is the real value of archive-debug as a workflow.
