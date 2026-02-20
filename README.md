# Character Engine

Interactive NPC chat CLI with selectable LLM backend (Cerebras, Groq, Google Gemini, or local vLLM models). Optional **voice mode** (Deepgram STT + ElevenLabs TTS) lets you speak to characters and hear them respond. Characters have personalities, world state that evolves during conversation, and a two-speed script system — a fast eval (every turn, non-blocking) tracks progress through premeditated dialogue beats, while a slow planner (Gemini Flash 3) generates multi-beat conversation scripts. Eval and plan run in background threads so the user can keep chatting without waiting. During conversation, beats use LLM-guided delivery: the beat's intent guides the LLM to respond naturally to the user while serving the beat's purpose (no verbatim pasting). The `/beat` command (autonomous time-passing) still uses verbatim delivery for TTS pre-rendering. Each beat includes gaze and expression data for animation.

World state uses a two-speed update system: the fast path immediately stores changes as pending and lets the character react (no LLM call needed), while the slow path reconciles pending changes into structured state mutations in the background using stable fact IDs (`f1`, `f2`, ...) — eliminating index-shift bugs that plagued small models.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

For voice mode (optional):

```bash
uv sync --extra voice
```

Add API keys to `.env` (at least one LLM key required):

```
CEREBRAS_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here

# Voice mode (optional)
DEEPGRAM_API_KEY=your_key_here
ELEVENLABS_API_KEY=your_key_here
```

If multiple keys are set, you'll choose a model at startup. If only one is set, it's auto-selected. The planner and reconciler require `GEMINI_API_KEY` — if not set, the script system runs in eval-only mode and the reconciler falls back to the chat model. Background eval uses `GROQ_API_KEY` for faster thinking via Groq Llama 70B — if not set, falls back to the chat model.

## Run

```bash
uv run -m character_eng          # text mode
uv run -m character_eng --voice  # voice mode (speak to chat, hear responses)
```

Pick a character from the menu, then chat. The character evaluates each turn in the background, tracks a script of dialogue beats, and adapts naturally. In-session commands:

| Command | What it does |
|---------|-------------|
| `/world` | Show current world state (facts with stable IDs, pending changes, events) |
| `/world <text>` | Describe a change — character reacts immediately, reconciler updates state in background |
| `/beat` | Deliver next scripted beat (time passes) — checks conditions, shows idle line if not met |
| `/plan` | Show current script (beat list with current beat highlighted) |
| `/goals` | Show character goals (long-term) |
| `/voice` | Toggle voice mode on/off mid-session |
| `/devices` | List audio input/output devices |
| `/reload` | Reload all prompt files from disk (for live editing) |
| `/trace` | Show system prompt, model, and token usage |
| `/back` | Return to character select |
| `/quit` | Exit |

Unknown `/` commands show an error with a `/help` hint.

## How the script system works

Each conversation turn follows one of two paths:

**Beat exists (conversation turn)**: The beat's intent and example line are injected as guidance, then the LLM generates a response that reacts to the user's input while serving the beat's purpose. This means the character acknowledges what the user said instead of barreling through a script. After delivery, a background eval (using Groq Llama 70B for speed) decides: `advance` (move to next beat), `hold` (stay on current beat), or `off_book` (conversation diverged, trigger replanning). The eval runs asynchronously — results are applied at the start of the next turn, and stale results are discarded if the user typed again before the eval finished. The eval also detects user pushback — if the user explicitly rejects or redirects away from the current topic, the eval goes `off_book` to follow the user's interest rather than forcing the script. Each beat also carries `gaze` and `expression` data for animation.

**Beat exists (`/beat` command)**: The pre-rendered beat line is pasted verbatim — no LLM call needed. This is for autonomous time-passing and enables TTS pre-rendering. Beats can have conditions (natural language preconditions) — if a condition isn't met, the character shows an in-character idle line instead.

**No beat** (script empty or planner unavailable): Falls back to LLM streaming for the response.

## How world updates work

World changes (`/world <text>`) use a two-speed system:

**Fast path** (immediate, no LLM): The change is stored as pending, a narrator message is injected into the conversation, and the character reacts via LLM streaming. This happens instantly.

**Slow path** (background thread): A reconciler LLM (using `PLAN_MODEL` if available, otherwise the chat model) processes all pending changes against ID-tagged dynamic facts, producing structured mutations (add/remove facts by stable ID, events). Results are applied at the start of the next turn with a "Reconciled" diff panel. Multiple `/world` commands before reconciliation accumulate — the next reconcile batch processes all.

## Voice mode

Voice mode adds speech I/O as a layer around the existing text pipeline. Everything works the same — voice just replaces keyboard input with microphone and adds TTS output.

**Start with voice**: `uv run -m character_eng --voice`
**Toggle mid-session**: `/voice` command or `Escape` hotkey

### How it works

1. **Mic** (sounddevice 16kHz) captures audio and streams it to Deepgram
2. **Deepgram** (nova-3 model) detects speech turns and returns transcripts
3. Transcript replaces keyboard input — goes through the same `ChatSession.send()` pipeline
4. **LLM response chunks** stream simultaneously to the terminal and to ElevenLabs TTS
5. **ElevenLabs** (eleven_flash_v2_5) streams PCM audio back to the speaker in real-time
6. **Auto-beat**: 4 seconds after the character finishes speaking, `/beat` fires automatically (advancing the script)

### Barge-in

If you start speaking while the character is responding, barge-in kicks in:
- Deepgram detects speech start → cancels LLM stream + TTS + speaker playback
- Your full utterance is captured and processed as the next input
- Partial LLM responses are still recorded in conversation history

### Voice hotkeys

In voice mode, single keystrokes trigger commands (no Enter needed):

| Key | Command |
|-----|---------|
| `w` | `/world` |
| `b` | `/beat` |
| `p` | `/plan` |
| `g` | `/goals` |
| `t` | `/trace` |
| `Escape` | Toggle voice off |

### Audio devices

When voice mode starts, the active mic and speaker are printed. Use `/devices` to list all available audio devices with their indices. The app uses your system defaults — change them in your OS audio settings.

### Requirements

Voice mode requires `uv sync --extra voice` and two API keys in `.env`:
- `DEEPGRAM_API_KEY` — for speech-to-text
- `ELEVENLABS_API_KEY` — for text-to-speech

Run `uv run -m character_eng.qa_voice` to verify connectivity (no mic needed).

If voice deps or keys are missing, the app falls back to text mode gracefully.

## Editing prompts

Characters live in `prompts/characters/<name>/` with these files:

| File | Purpose |
|------|---------|
| `prompt.txt` | Main template — uses `{{global_rules}}`, `{{character}}`, `{{scenario}}`, `{{world}}` macros |
| `character.txt` | Personality, voice, and goals (optional `Long-term goal:` line) |
| `scenario.txt` | Situation and context |
| `world_static.txt` | Permanent facts (one per line, optional) |
| `world_dynamic.txt` | Initial mutable state (one per line, loaded with stable IDs `f1`, `f2`, ...) |

`prompts/global_rules.txt` contains rules shared across all characters.

System-level prompts are also editable files in `prompts/`:

| File | Purpose |
|------|---------|
| `reconcile_system.txt` | System prompt for the world-state reconciler LLM (processes pending changes using stable fact IDs) |
| `eval_system.txt` | System prompt for the character eval LLM (script tracking + inner monologue) |
| `condition_system.txt` | System prompt for the condition checker LLM (gates beat delivery) |
| `plan_system.txt` | System prompt for the conversation planner LLM (beat generation with conditions) |
| `beat_guide.txt` | Template for beat guidance injected before LLM-guided delivery (`{intent}` and `{line}` placeholders) |

**Live editing works** — edit any character prompt file in your editor, then type `/reload` in the chat to pick up changes without restarting. The conversation resets but you keep your place in the character menu. Reconciler, eval, and plan system prompts are re-read on every call, so edits take effect immediately without `/reload`.

### Adding a new character

Create a directory under `prompts/characters/` with at least a `prompt.txt`. No code changes needed. Example:

```
prompts/characters/my_npc/
  prompt.txt
  character.txt
  scenario.txt
```

## Local models

Local models run via vLLM on your GPU. You can start vLLM two ways:

**From the app** (recommended) — when local models are configured but vLLM isn't running, the model menu shows a hint and an `s` option to start a local model. Pick a model, and vLLM launches with live output streaming. Once healthy, the model is auto-selected. The vLLM process is cleaned up automatically when you quit the app.

**Standalone** — use `serve.py` directly:

```bash
# Start vLLM server (interactive model picker)
uv run python -m character_eng.serve

# Start a specific model directly
uv run python -m character_eng.serve lfm-2.6b

# Kill the server
lsof -ti:8000 | xargs kill -9
```

Available local models: LFM-2 2.6B (`lfm-2.6b`), LFM-2.5 1.2B (`lfm-1.2b`). The server auto-kills any existing process on port 8000 before starting.

## Testing

```bash
# Unit tests (mocked, fast)
uv run pytest

# Integration tests (hit real LLM, default model: cerebras-llama)
uv run -m character_eng.qa_world                       # world state reconciler
uv run -m character_eng.qa_chat                        # end-to-end chat, world, eval
uv run -m character_eng.qa_world --model groq-llama    # test with Groq Llama
uv run -m character_eng.qa_chat --model groq-llama     # test with Groq Llama

# Voice connectivity test (Deepgram + ElevenLabs, no mic needed)
uv run -m character_eng.qa_voice

# Persona QA (6 parallel personas, HTML report)
uv run -m character_eng.qa_personas                        # default model + character
uv run -m character_eng.qa_personas --model groq-llama     # test with Groq Llama
uv run -m character_eng.qa_personas --turns 5              # quick run with fewer turns
uv run -m character_eng.qa_personas --turns 5 --open       # run and open report in browser

# Open HTML reports in browser (starts local server for annotation save)
uv run -m character_eng.open_report                        # latest report in logs/
uv run -m character_eng.open_report logs/some_report.html  # specific file
uv run -m character_eng.open_report --test                 # generate + serve a spoofed test report
```

`test_plan.md` defines the QA chat scenarios in a human-editable format — add new test sections without touching code. Supports `send:`, `world:`, `script:` (load beats for eval tracking), `beat` (run eval), and `expect:` commands including `eval_status:<status>` to assert eval outcomes.

### Persona QA

Automated "subjective" testing — 6 LLM-driven personas chat with the character in parallel, exercising the full async loop (background eval/plan/reconcile). Produces an HTML report for human review.

| Persona | Type | Turns | Behavior |
|---------|------|-------|----------|
| AFK Andy | LLM | 12 | Inaction griefer — "k", "idk", "sure", sometimes does nothing |
| Chaos Carl | LLM | 12 | Overaction griefer — ~40% `/world` (fire, explosions), rest bizarre messages |
| Casual Player | LLM | 15 | Normal user — asks questions, engages naturally |
| Bored Player | LLM | 12 | Not having fun — "this is boring", redirects, dismisses |
| Beat Runner | Scripted | 10 | Every turn: `/beat` — tests autonomous time-passing |
| Interrupter | Hybrid | 12 | Alternates message→beat→message→beat — tests script interruption |

Each persona runs in its own `ConversationDriver` with fully isolated threading state (locks, result slots, context version). The HTML report shows per-persona conversation cards with action type badges, character responses, eval details in collapsible sections, and stale discard markers.

## Benchmarking

Measure real-world latency across all available models for the three LLM call patterns: streaming chat, structured JSON reconcile calls, and structured JSON eval calls.

```bash
# Benchmark all available models (3 runs per scenario)
uv run -m character_eng.benchmark

# Benchmark a specific model with more runs
uv run -m character_eng.benchmark --model cerebras-llama --runs 5
```

Prints per-run details (TTFT, total time, response text) and a summary table with averages and standard deviations. Full results are saved to `logs/benchmark_*.json`.

## Logs

Every chat session and QA run saves logs to `logs/` (gitignored). Chat sessions save both a JSON log and an HTML report; persona QA saves an HTML report. The log path is printed when the session ends.

HTML reports (chat sessions and persona QA) include an **annotation UI** — click the pencil icon on any turn to leave a note, then click "Export annotations" in the sticky bottom toolbar to download a `.annotations.json` file. The JSON contains a notes summary plus full conversations for annotated personas/sessions only. Useful for reviewing conversation quality, marking issues, and feeding annotations back into Claude Code.
