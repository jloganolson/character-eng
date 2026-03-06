# Creative Collaborator Guide

How to build characters, design scenarios, and test interactions with character-eng.

## How It Works (30-Second Version)

You write a **character** (personality, scenario, world facts) and a **scenario script** (stage graph with exit conditions). The engine runs the character live:

1. Character generates dialogue guided by **beats** (scripted intents with example lines)
2. An **eval** tracks whether each beat's intent was accomplished
3. A **director** watches for stage transitions and advances the narrative
4. A **reconciler** converts natural-language events into structured world state
5. An **expression system** picks gaze target + facial emotion after each line

All LLM calls use an 8B model via Groq (~100ms). You never wait.

---

## Architecture at a Glance

```
┌─────────────────────────────────────────────────────────┐
│  User Input (text or voice)                              │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│  Chat (streaming 8B)                                     │
│  System prompt = character.txt + scenario.txt + world    │
│  + beat guidance (intent + example line)                  │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│  Parallel Post-Response (4 concurrent 8B calls, ~350ms)  │
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐ │
│  │ script   │  │ thought  │  │ director │  │ express │ │
│  │ check    │  │ (inner   │  │ (stage   │  │ (gaze + │ │
│  │ (advance │  │ monolog) │  │ transit) │  │ emotion)│ │
│  │ /hold/   │  │          │  │          │  │         │ │
│  │ off_book)│  │          │  │          │  │         │ │
│  └──────────┘  └──────────┘  └──────────┘  └─────────┘ │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│  Replan if needed (single_beat_call, 8B, ~350ms)         │
│  Reconcile world state (background thread, 8B)           │
└─────────────────────────────────────────────────────────┘
```

**Vision** (optional) adds a camera → face/person/object tracking → VLM questions → synthesis loop that generates perception events every 750ms.

---

## Project Layout — What You Touch

```
prompts/
├── characters/
│   └── greg/                    ← one directory per character
│       ├── prompt.txt           ← master template (macros)
│       ├── character.txt        ← personality & goals
│       ├── scenario.txt         ← current situation
│       ├── world_static.txt     ← permanent scene facts
│       ├── world_dynamic.txt    ← mutable starting facts
│       ├── scenario_script.toml ← stage graph
│       └── sims/
│           └── walkup.sim.txt   ← scripted test scenario
├── global_rules.txt             ← behavioral rules for ALL characters
├── beat_guide.txt               ← how beats are delivered
├── plan_system.txt              ← beat generation prompt
├── eval_system.txt              ← script tracking prompt
├── reconcile_system.txt         ← world state update prompt
├── director_system.txt          ← stage transition prompt
├── expression_system.txt        ← gaze + emotion prompt
├── think_system.txt             ← inner monologue prompt
├── condition_system.txt         ← beat precondition prompt
├── vision_synthesis.txt         ← visual event distillation
└── visual_focus.txt             ← VLM question planning
```

**Key principle**: All prompts are read from disk on every call. Edit any `.txt` file and it takes effect on the next LLM call — no restart needed.

---

## Creating a Character

### Quick start — scaffolding script

```bash
./scripts/new_character.sh yourchar
```

This creates the full directory with template files. Edit them to customize.

### Manual setup

#### 1. Create the directory

```bash
mkdir -p prompts/characters/yourchar/sims
```

### 2. `character.txt` — Who They Are

Plain text. Define personality, background, quirks, goals.

```
You are Mara, a fortune teller at a carnival booth. You sit behind a small
round table with a crystal ball, a deck of tarot cards, and a velvet curtain
behind you. You speak with dramatic flair but you're actually quite practical
underneath the mysticism.

You believe your readings have real value — not because of magic, but because
you're genuinely perceptive about people. You read body language, ask the right
questions, and reflect things back in a way that helps people think.

Long-term goal: Help people gain genuine insight about their lives
Short-term goal: Get this customer to sit down and try a reading
```

**Guidelines**:
- 2-3 sentences of core identity
- Concrete examples of how they'd respond to common situations
- Physical constraints (what they can/can't do)
- Long-term goal (permanent, drives overall behavior)
- Short-term goal (changes based on events)

### 3. `scenario.txt` — The Current Situation

```
You're set up at a small carnival on a Saturday afternoon. Business has been
slow today — only two readings so far. A new person is walking past your
booth. You need to catch their attention before they walk by.

Your booth has a hand-painted sign that reads "MYSTIC MARA — Past, Present,
Future — $5" with stars painted around it. The crystal ball is actually a
decorative bowling ball, but it looks the part.
```

### 4. `world_static.txt` — Permanent Facts

One fact per line. Present tense.

```
Mara sits behind a round table in a carnival booth
A crystal ball sits on the table (actually a decorative bowling ball)
A deck of tarot cards is on the table
A velvet curtain hangs behind Mara
A hand-painted sign reads "MYSTIC MARA — Past, Present, Future — $5"
The booth is at a small weekend carnival
Carnival music plays faintly in the background
```

### 5. `world_dynamic.txt` — Starting State (Mutable)

```
It is a warm Saturday afternoon
The carnival is moderately busy
Mara has done two readings today
No one is at the booth — Mara is alone
Mara's tip jar has $3 in it
```

These facts get IDs (`f1`, `f2`, ...) and the reconciler can add/remove them as events happen. When someone sits down, the reconciler removes "No one is at the booth" and adds "A customer is seated at the table."

### 6. `scenario_script.toml` — Stage Graph

Defines the narrative arc as stages with exit conditions.

```toml
[scenario]
name = "Mara's Fortune Booth"
start = "idle"

[[stage]]
name = "idle"
goal = "Mara is alone, trying to attract customers. She calls out to passersby and fidgets with her cards."

[[stage.exit]]
condition = "Someone shows interest or approaches the booth"
goto = "hook"
label = "interested"

[[stage.exit]]
condition = "Someone walks past without stopping"
goto = "idle"
label = "ignored"

[[stage]]
name = "hook"
goal = "Someone is nearby. Mara makes her pitch — dramatic but not pushy. She wants them to sit down."

[[stage.exit]]
condition = "The person sits down or agrees to a reading"
goto = "reading"
label = "sits down"

[[stage.exit]]
condition = "The person declines or walks away"
goto = "idle"
label = "declines"

[[stage]]
name = "reading"
goal = "Mara performs a reading. She asks perceptive questions, interprets cards, and tries to give genuine insight."

[[stage.exit]]
condition = "The reading feels complete and the person seems satisfied"
goto = "wrap_up"
label = "done"

[[stage]]
name = "wrap_up"
goal = "Mara wraps up warmly, mentions the tip jar, and thanks the customer."

[[stage.exit]]
condition = "The person leaves"
goto = "idle"
label = "leaves"
```

**How stages work**:
- Each stage has a `goal` — this guides beat generation
- Exit conditions are checked by the director after every response
- First matching condition wins (checked in order)
- `label` shows in the HUD as manual triggers (`1: interested  2: ignored`)
- Stage transitions trigger replanning — new beats generated for new goal

### 7. `prompt.txt` — Master Template

Use macros to assemble the full system prompt.

```
{{global_rules}}

## Character
{{character}}

## Current Scenario
{{scenario}}

## World State
{{world}}

## People Present
{{people}}

## What You See
{{vision}}

Play this character. Respond to what the user says and does, staying true to
the personality and situation above.
```

Available macros: `{{global_rules}}`, `{{character}}`, `{{scenario}}`, `{{world}}`, `{{people}}`, `{{vision}}`.

### 8. Try It

```bash
uv run -m character_eng
# Select your character from the menu
```

---

## Writing Sim Scripts (Test Without a Person)

Sim scripts replay scripted interactions. Great for testing characters without needing a real conversation partner.

Create `prompts/characters/yourchar/sims/test.sim.txt`:

```
# Format: time_offset | description
# Quoted descriptions become user dialogue
# Unquoted descriptions become narrator events ([brackets])

0.0 | A person walks up to the booth and looks at the sign
3.0 | "What is this?"
6.0 | "How much for a reading?"
9.0 | The person sits down at the table
11.0 | "Okay, let's do it"
15.0 | The person looks nervous
18.0 | "I've been thinking about changing jobs"
22.0 | The person nods slowly
25.0 | "That's... actually really helpful"
28.0 | The person puts $5 on the table
30.0 | "Thanks, Mara"
33.0 | The person stands up and waves goodbye
36.0 | The person walks away into the carnival crowd
```

Run it:
```bash
# In-app:
/sim test

# From the CLI (non-interactive, runs and exits):
uv run -m character_eng --character yourchar --sim test
```

The engine replays events at their time offsets. Quoted lines become user dialogue; unquoted lines become `[narrator events]` that the character reacts to.

---

## Vision Mock Testing (No Camera Needed)

Test the vision system using pre-recorded visual snapshots:

```bash
# Use a built-in replay
uv run -m character_eng --vision-mock walkup.json

# Use a custom replay file
uv run -m character_eng --vision-mock path/to/your/replay.json
```

Replay files live in `services/vision/replays/` and define timestamped snapshots:

```json
{
  "loop": false,
  "snapshots": [
    {
      "at": 0.0,
      "faces": [],
      "persons": [],
      "objects": [{"label": "sign", "bbox": [100, 200, 50, 30], "confidence": 0.9}],
      "vlm_answers": [{"question": "How many people?", "answer": "None", "elapsed": 0.4, "slot_type": "constant"}]
    },
    {
      "at": 5.0,
      "faces": [{"identity": "Person 1", "bbox": [300, 100, 80, 100], "age": 30, "gender": "M", "confidence": 0.9, "gaze_direction": "at camera", "looking_at_camera": true}],
      "persons": [{"identity": "Person 1", "bbox": [280, 80, 120, 300], "confidence": 0.9}],
      "objects": [{"label": "sign", "bbox": [100, 200, 50, 30], "confidence": 0.9}],
      "vlm_answers": [{"question": "How many people?", "answer": "One person at the booth", "elapsed": 0.4, "slot_type": "constant"}]
    }
  ]
}
```

Each snapshot is served until the next timestamp is reached. The app can't tell the difference from a real camera.

---

## In-App Commands

| Command | What it does |
|---------|-------------|
| `/beat` | Deliver the next scripted beat |
| `/world <text>` | Inject a world event (character reacts immediately) |
| `/see <text>` | Inject a perception event |
| `/sim <name>` | Run a sim script |
| `/info` | Show world state, script, goals |
| `/trace` | Show system prompt, message history |
| `/vision` | Show current visual context and gaze targets |
| `/reload` | Hot-reload all prompts from disk |
| `/back` | Return to character select |
| `1`-`4` | Trigger stage transitions manually |
| `/quit` | Exit |

**Voice mode hotkeys**: `i`=info, `b`=beat, `t`=trace, `1-4`=triggers, `q`=quit, `Escape`=toggle voice

---

## Prompt Tuning Guide

All system prompts in `prompts/*.txt` are reloaded on every call. Edit and test in real-time — no restart needed. Use `/reload` to pick up changes to character files.

### What Each Prompt Controls

| Prompt | Controls | Tune when... |
|--------|----------|--------------|
| `global_rules.txt` | Response length, format rules, pause behavior | Character talks too much, adds actions/emotes |
| `plan_system.txt` | Beat generation (how many, what kind) | Beats are too generic, too many, wrong tone |
| `eval_system.txt` | Script tracking (advance/hold/off_book) | Beats advance too fast or get stuck |
| `director_system.txt` | Stage transitions | Stages change too early or never change |
| `expression_system.txt` | Gaze + facial expression | Character stares at wrong thing, wrong emotions |
| `think_system.txt` | Inner monologue | Thoughts are repetitive or generic |
| `reconcile_system.txt` | World state updates | Facts aren't tracked properly |
| `condition_system.txt` | Beat precondition checks | Beats fire when they shouldn't |
| `beat_guide.txt` | How beats are delivered | Character ignores user input during beats |
| `vision_synthesis.txt` | Visual event distillation | Too many/few vision events, wrong focus |
| `visual_focus.txt` | VLM question generation | Wrong questions asked of the camera |

### Common Adjustments

**Character talks too long**: Edit `global_rules.txt` — tighten the "1-2 short sentences" rule.

**Character ignores user input**: Edit `beat_guide.txt` — strengthen "Acknowledge what the user said."

**Stages transition too easily**: Edit `director_system.txt` — add "Be conservative, require clear evidence."

**Inner thoughts are repetitive**: Edit `think_system.txt` — the non-repetition rules are critical. Strengthen them.

**Vision generates too many events**: Edit `vision_synthesis.txt` — emphasize "Output 0 events if nothing new."

---

## How the Conversation Flows

### Boot
1. Load character files → assemble system prompt
2. Load scenario → set initial stage
3. Generate first beat (synchronous, 8B, ~350ms)
4. Arm auto-beat timer (voice mode: fires after 4s silence)

### Each Turn
1. **User speaks** → stream character response (8B, guided by current beat)
2. **Post-response** (4 parallel 8B calls, ~350ms total):
   - Script check: did the beat's intent get accomplished?
   - Thought: inner monologue
   - Director: any stage exit condition met?
   - Expression: gaze target + facial emotion
3. **Replan** if script exhausted or conversation went off-book
4. **Reconcile** world state in background

### Stage Transitions
- Director detects exit condition → advance to new stage
- Old beats discarded → new beats generated for new stage goal
- Character's behavior shifts to match new stage

### Off-Book Detection
- If conversation diverges from planned beats, eval returns "off_book"
- New beat generated matching the actual conversation direction
- Character naturally adapts without losing coherence

---

## Testing Checklist

```bash
# Unit tests (fast, mocked)
uv run pytest

# Smoke test (real LLM calls, automated)
uv run -m character_eng --smoke

# QA chat (parses test_plan.md, validates behavior)
uv run -m character_eng.qa_chat

# World reconciler test (if you changed world logic)
uv run -m character_eng.qa_world

# Interactive test
uv run -m character_eng

# Auto-select character (skip menu)
uv run -m character_eng --character mara

# Run sim non-interactively (runs and exits)
uv run -m character_eng --character mara --sim curious

# Vision mock test
uv run -m character_eng --vision-mock walkup.json

# Combined: sim + vision mock (full automated demo)
uv run -m character_eng --character greg --sim walkup --vision-mock walkup.json

# Persona stress test (7 parallel personas, HTML report)
uv run -m character_eng.qa_personas --character greg --turns 10 --open
```

---

## Setup for a New Collaborator

```bash
# Clone and install
git clone <repo> && cd character-eng
uv sync

# Set up API key (Groq, free tier works)
echo "GROQ_API_KEY=your-key" > .env

# Run
uv run -m character_eng

# Optional: voice mode
uv sync --extra voice
uv run -m character_eng --voice

# Optional: vision (requires NVIDIA GPU)
cd services/vision && uv sync && cd ../..
uv run -m character_eng --vision
```

---

## Quick Reference: File Formats

### Character files — plain text, no special syntax
- `character.txt` — personality, goals, constraints
- `scenario.txt` — current situation, context
- `world_static.txt` — permanent facts (one per line)
- `world_dynamic.txt` — mutable facts (one per line)

### `prompt.txt` — template with `{{macros}}`
- `{{global_rules}}`, `{{character}}`, `{{scenario}}`, `{{world}}`, `{{people}}`, `{{vision}}`

### `scenario_script.toml` — TOML stage graph
- `[scenario]` with `name` and `start`
- `[[stage]]` with `name`, `goal`, and `[[stage.exit]]` (condition, goto, label)

### `.sim.txt` — timed events
- `time_offset | description` per line
- Quoted = dialogue, unquoted = narrator event
- Comments with `#`

### Vision replay `.json` — timestamped snapshots
- `snapshots[].at` = seconds from start
- Each snapshot has `faces`, `persons`, `objects`, `vlm_answers`

---

## Authoring Vision Replays

Vision replay files simulate what a camera would see. They live in `services/vision/replays/`.

### Snapshot anatomy

Each snapshot represents the visual state at a point in time:

```json
{
  "at": 5.0,
  "faces": [
    {
      "identity": "Person 1",
      "bbox": [x, y, width, height],
      "age": 28,
      "gender": "F",
      "confidence": 0.9,
      "gaze_direction": "at camera",
      "looking_at_camera": true
    }
  ],
  "persons": [
    {
      "identity": "Person 1",
      "bbox": [x, y, width, height],
      "confidence": 0.9
    }
  ],
  "objects": [
    {"label": "cup", "bbox": [x, y, w, h], "confidence": 0.85}
  ],
  "vlm_answers": [
    {
      "question": "What is the person doing?",
      "answer": "The person is drinking from a cup.",
      "elapsed": 0.4,
      "slot_type": "constant"
    }
  ]
}
```

### Tips for realistic replays

- **People approach gradually**: Start with empty `faces`/`persons`, then add them with low confidence at the edge of frame, increasing confidence and shifting bbox toward center as they approach.
- **Gaze matters**: Use `"gaze_direction": "away"` when approaching, `"at camera"` when engaged.
- **Objects appear/disappear**: Add objects when they become relevant (dollar bill when someone pays, cup when they're drinking).
- **VLM answers add context**: The synthesis LLM reads these to understand what's happening. Make them descriptive.
- **End clean**: Remove people and objects when the interaction ends, returning to the static scene.
- **Use `"loop": true`** for background ambience (people walking by repeatedly).

### Pair with sim scripts

For the richest testing, combine a vision replay with a sim script:

```bash
uv run -m character_eng --character greg --sim walkup --vision-mock walkup.json
```

The sim provides dialogue and narrator events. The vision replay provides visual perception. Together they create a full automated scenario.
