# Character Engine

Interactive NPC chat CLI with selectable LLM backend (Cerebras, Groq, Google Gemini, or local vLLM models). Optional **voice mode** (Deepgram STT + ElevenLabs TTS) lets you speak to characters and hear them respond. Characters have personalities, world state that evolves during conversation, and a script system driven entirely by 8B microservices — post-response calls (eval + director) fire in parallel for ~350ms total instead of ~1050ms sequential. A single-beat 8B planner generates one beat at a time synchronously, eliminating background plan threads entirely. During conversation, beats use LLM-guided delivery: the beat's intent guides the LLM to respond naturally to the user while serving the beat's purpose (no verbatim pasting). The `/beat` command (autonomous time-passing) still uses verbatim delivery for TTS pre-rendering. After each character response, lightweight 8B microservices derive gaze/expression, evaluate script progress, and check scenario exit conditions — all running concurrently for immediate effect.

**Scenario director**: Characters can have a branching scenario script (TOML) defining stages with goals and exit conditions. A director microservice (8B, synchronous) evaluates exit conditions after each response with immediate effect — no background delay.

**Person tracking**: Individual people are tracked alongside world state with scoped fact IDs (`p1f1`, `p1f2`, ...). The reconciler LLM handles person-fact assignments — no heuristic matching needed.

**Perception layer**: The `/see` command lets you describe what the character perceives, and `/sim` replays scripted perception events from `.sim.txt` files with timing.

World state uses a two-speed update system: the fast path immediately stores changes as pending and lets the character react (no LLM call needed), while the slow path reconciles pending changes into structured state mutations in the background using stable fact IDs (`f1`, `f2`, ...) — eliminating index-shift bugs that plagued small models.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

For local Qwen3-TTS on GPU (optional, ~2GB VRAM):

```bash
uv sync --extra local-tts
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

If multiple keys are set, you'll choose a model at startup. If only one is set, it's auto-selected. All LLM calls (chat, eval, plan, reconcile) use the 8B chat model — no separate big model needed.

### Configuration (optional)

Copy `config.example.toml` to `config.toml` for persistent settings (gitignored):

```bash
cp config.example.toml config.toml
```

```toml
[voice]
input_device = "reSpeaker"   # by name substring (case-insensitive), or integer index
output_device = "reSpeaker"  # route TTS through device for hardware AEC reference
enabled = true               # start in voice mode by default
aec = true                   # WebRTC AEC3 echo cancellation (keeps mic live during playback)

# Use Qwen3-TTS on GPU (requires uv sync --extra local-tts)
# tts_backend = "qwen"
# ref_audio = "/path/to/reference.wav"
# ref_text = ""               # transcript of ref_audio for ICL mode (higher quality cloning)
# tts_model = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
# tts_device = "cuda:0"

# Use Pocket-TTS server (auto-starts if not running, auto-stops on exit)
# tts_backend = "pocket"
# tts_server_url = "http://localhost:8003"
# ref_audio = "/path/to/reference.wav"
# pocket_voice = "voices/greg.safetensors"  # default
```

The app works without `config.toml` — all settings have defaults. The `--voice` flag still works as an override.

## Run

```bash
uv run -m character_eng          # text mode
uv run -m character_eng --voice  # voice mode (speak to chat, hear responses)
uv run -m character_eng --smoke  # smoke test (auto greg, scripted inputs, exit)
```

Pick a character from the menu, then chat. After each response, synchronous microservices evaluate script progress and check scenario exits with immediate effect. In-session commands:

| Command | What it does |
|---------|-------------|
| `/info` | Show all state (world, plan, goals, stage, people) in one dump |
| `/world <text>` | Describe a change — character reacts immediately, reconciler updates state in background |
| `/beat` | Deliver next scripted beat (time passes) — checks conditions, shows idle line if not met |
| `/see <text>` | Describe something the character perceives — character reacts, reconciler processes |
| `/sim <name>` | Replay a sim script with timing (quoted lines = dialogue, unquoted = perception) |
| `1`-`4` | Trigger a stage exit by number (shown in the HUD line before each prompt) |
| `/voice` | Toggle voice mode on/off mid-session |
| `/devices` | List audio input/output devices |
| `/reload` | Reload all prompt files from disk (for live editing) |
| `/trace` | Show system prompt, model, and token usage |
| `/back` | Return to character select |
| `/quit` | Exit |

A **stage HUD** line is printed before every input prompt showing the current scenario stage and numbered exits (e.g. `[watching] 1: notices  2: ignores`). Type the number to trigger that exit.

Unknown `/` commands show an error with a `/help` hint.

## How the script system works

Each conversation turn follows one of two paths:

**Beat exists (conversation turn)**: The beat's intent and example line are injected as guidance, then the LLM generates a response that reacts to the user's input while serving the beat's purpose. This means the character acknowledges what the user said instead of barreling through a script. After delivery, a synchronous eval microservice (8B) decides: `advance` (move to next beat), `hold` (stay on current beat), or `off_book` (conversation diverged, trigger replanning). The eval fires immediately after each response — no background delay, no stale discard needed. The eval is split into two focused calls: `script_check_call` (classification only — did the beat's intent get accomplished?) and `thought_call` (brief inner monologue). This split lets the small model focus on classification without creative generation drowning out the signal. The eval also detects user pushback — if the user explicitly rejects or redirects away from the current topic, the eval goes `off_book` to follow the user's interest rather than forcing the script.

**Beat exists (`/beat` command)**: The pre-rendered beat line is pasted verbatim — no LLM call needed. This is for autonomous time-passing and enables TTS pre-rendering. Beats can have conditions (natural language preconditions) — if a condition isn't met, the character shows an in-character idle line instead.

**No beat** (script empty or planner unavailable): Falls back to LLM streaming for the response.

## How world updates work

World changes (`/world <text>`) use a two-speed system:

**Fast path** (immediate, no LLM): The change is stored as pending, a narrator message is injected into the conversation, and the character reacts via LLM streaming. This happens instantly.

**Slow path** (background thread): A reconciler LLM (8B) processes all pending changes against ID-tagged dynamic facts, producing structured mutations (add/remove facts by stable ID, events). Results are applied at the start of the next turn with a "Reconciled" diff panel. Multiple `/world` commands before reconciliation accumulate — the next reconcile batch processes all.

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
- Deepgram detects speech start → cancels LLM stream + TTS + speaker playback (only while TTS is actively generating, guarded by `_is_speaking` flag to ignore ambient noise)
- Your full utterance is captured and processed as the next input
- Partial LLM responses are still recorded in conversation history
- The character is told it was interrupted — it acknowledges briefly and yields the floor
- Auto-beat is suppressed after the barge-in response, giving you silence until you speak again
- Conversation history is truncated to approximate what was actually spoken (using speaker playback ratio), so the LLM knows where it was cut off

### Voice hotkeys

In voice mode, single keystrokes trigger commands (no Enter needed):

| Key | Command |
|-----|---------|
| `i` | `/info` |
| `b` | `/beat` |
| `t` | `/trace` |
| `1`-`4` | Stage triggers |
| `q` | Quit |
| `Escape` | Toggle voice off |

### Audio devices

When voice mode starts, the active mic and speaker are printed. Use `/devices` in-app or `uv run -m character_eng.devices` from the terminal to list all available audio devices with their indices. To use a specific device, set `input_device` and/or `output_device` in `config.toml` — either as an integer index or a name substring (case-insensitive, e.g. `"reSpeaker"`). Name-based matching survives USB re-enumeration across reboots/suspends. Without config, the app uses your system defaults.

### Echo cancellation

Voice mode uses **WebRTC AEC3** (via LiveKit) by default — the mic stays live during robot speech, enabling natural barge-in. The AEC receives a reference signal of what the speaker is playing and cancels it from the mic input before sending to Deepgram.

Set `aec = false` in `config.toml` to fall back to mic-muting during playback (legacy behavior, no barge-in).

For hardware AEC devices (e.g. reSpeaker XVF3800), route TTS through the device and disable software AEC:

```toml
[voice]
input_device = "reSpeaker"
output_device = "reSpeaker"
aec = false
```

### TTS backends

Voice mode supports three TTS backends, configured via `tts_backend` in `config.toml`:

**ElevenLabs (default)**: Cloud TTS, low latency, streaming text input. Requires `ELEVENLABS_API_KEY` in `.env`.

**Qwen3-TTS** (`tts_backend = "qwen"`): GPU TTS via vendored Qwen3-TTS package. Requires `uv sync --extra local-tts`, a reference audio WAV for voice cloning, and ~2GB VRAM for the 0.6B model. No `ELEVENLABS_API_KEY` needed. Set `ref_audio` in `config.toml`. Two cloning modes: with `ref_text` (transcript of ref_audio) uses ICL mode for higher quality; without uses x-vector-only mode. Model loads once on first speech and persists across utterances. Uses Flash Attention 2 for optimized inference (`flash-attn` included in `local-tts` extras; `torch` pinned to 2.8.x for wheel compatibility). `"local"` is accepted as a legacy alias for `"qwen"`.

**Pocket-TTS** (`tts_backend = "pocket"`): Streaming HTTP client for Pocket-TTS server (100M causal transformer). True autoregressive streaming with flat ~80ms TTFA. Supports voice cloning via `ref_audio` (WAV upload) or `pocket_voice` (path to `.safetensors` voice file passed to `pocket-tts serve --voice`). The server **auto-starts** when not already running and **auto-stops** on exit. For faster startup, run the server persistently in the background (`pocket-tts serve --port 8003 &`). Set `tts_server_url` in `config.toml` (default: `http://localhost:8003`). Only needs `requests` (included in main dependencies).

All backends use the same 4-method interface (`send_text`, `flush`, `wait_for_done`, `close`) — no changes to the main conversation loop.

### Requirements

Voice mode requires at least:
- `DEEPGRAM_API_KEY` — for speech-to-text (always required)
- `ELEVENLABS_API_KEY` — for ElevenLabs TTS (only when `tts_backend = "elevenlabs"`)

Run `uv run -m character_eng.qa_voice` to verify connectivity (no mic needed).

If voice deps or keys are missing, the app falls back to text mode gracefully.

## Editing prompts

Characters live in `prompts/characters/<name>/` with these files:

| File | Purpose |
|------|---------|
| `prompt.txt` | Main template — uses `{{global_rules}}`, `{{character}}`, `{{scenario}}`, `{{world}}`, `{{people}}` macros |
| `character.txt` | Personality, voice, and goals (optional `Long-term goal:` line) |
| `scenario.txt` | Situation and context |
| `world_static.txt` | Permanent facts (one per line, optional) |
| `world_dynamic.txt` | Initial mutable state (one per line, loaded with stable IDs `f1`, `f2`, ...) |
| `scenario_script.toml` | Branching scenario stages with exit conditions and optional HUD labels (parsed by director + number triggers) |
| `sims/*.sim.txt` | Sim scripts for `/sim` command — `time_offset \| description` per line (optional) |

`prompts/global_rules.txt` contains rules shared across all characters.

System-level prompts are also editable files in `prompts/`:

| File | Purpose |
|------|---------|
| `reconcile_system.txt` | System prompt for the world-state reconciler LLM (processes pending changes using stable fact IDs, including person-scoped IDs) |
| `eval_system.txt` | System prompt for the thought microservice (inner monologue generation) |
| `condition_system.txt` | System prompt for the condition checker LLM (gates beat delivery) |
| `plan_system.txt` | System prompt for the conversation planner LLM (beat generation with conditions, stage-goal and people aware) |
| `director_system.txt` | System prompt for the director microservice (evaluates stage exit conditions, 8B sync) |
| `beat_guide.txt` | Template for beat guidance injected before LLM-guided delivery (`{intent}` and `{line}` placeholders) |

**Live editing works** — edit any character prompt file in your editor, then type `/reload` in the chat to pick up changes without restarting. The conversation resets but you keep your place in the character menu. Reconciler, eval, plan, and director system prompts are re-read on every call, so edits take effect immediately without `/reload`.

### Adding a new character

Create a directory under `prompts/characters/` with at least a `prompt.txt`. No code changes needed. Example:

```
prompts/characters/my_npc/
  prompt.txt
  character.txt
  scenario.txt
```

## Local models

Local models run via vLLM on your GPU. Start vLLM with `serve.py`:

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
# Unit tests + e2e smoke (includes real LLM smoke test when API keys are set)
uv run pytest

# E2e smoke test standalone (sends real messages, exercises full startup path)
uv run -m character_eng --smoke

# Integration tests (hit real LLM, default model: cerebras-llama)
uv run -m character_eng.qa_world                       # world state reconciler
uv run -m character_eng.qa_chat                        # end-to-end chat, world, eval
uv run -m character_eng.qa_world --model groq-llama    # test with Groq Llama
uv run -m character_eng.qa_chat --model groq-llama     # test with Groq Llama

# Voice connectivity test (Deepgram + ElevenLabs, no mic needed)
uv run -m character_eng.qa_voice

# Persona QA (7 parallel personas, HTML report)
uv run -m character_eng.qa_personas                        # default model + character
uv run -m character_eng.qa_personas --model groq-llama     # test with Groq Llama
uv run -m character_eng.qa_personas --turns 5              # quick run with fewer turns
uv run -m character_eng.qa_personas --turns 5 --open       # run and open report in browser

# Open HTML reports in browser (starts local server for annotation save)
uv run -m character_eng.open_report                        # latest report in logs/
uv run -m character_eng.open_report logs/some_report.html  # specific file
uv run -m character_eng.open_report --test                 # generate + serve a spoofed test report
```

`test_plan.md` defines the QA chat scenarios in a human-editable format — add new test sections without touching code. Supports `send:`, `world:`, `script:` (load beats for eval tracking), `beat` (run eval), and `expect:` commands including `eval_status:<status>` to assert eval outcomes. Eval now runs synchronously via split microservices (script_check + thought) so results are immediate.

### Persona QA

Automated "subjective" testing — 7 LLM-driven personas chat with the character in parallel, exercising the full loop (parallel eval/director microservices + background reconcile). Produces an HTML report for human review.

| Persona | Type | Turns | Behavior |
|---------|------|-------|----------|
| AFK Andy | LLM | 12 | Inaction griefer — "k", "idk", "sure", sometimes does nothing |
| Chaos Carl | LLM | 12 | Overaction griefer — ~40% `/world` (fire, explosions), rest bizarre messages |
| Casual Player | LLM | 15 | Normal user — asks questions, engages naturally |
| Bored Player | LLM | 12 | Not having fun — "this is boring", redirects, dismisses |
| Beat Runner | Scripted | 10 | Every turn: `/beat` — tests autonomous time-passing |
| Interrupter | Hybrid | 12 | Alternates message→beat→message→beat — tests script interruption |
| Scene Observer | LLM | 15 | Perception tester — mix of `/see` and messages, simulates approach/interact/leave |

Each persona runs in its own `ConversationDriver` with fully isolated state (locks, result slots, context version, people state, scenario script). The HTML report shows per-persona conversation cards with action type badges, character responses, eval details in collapsible sections.

## Benchmarking

Measure real-world latency across all available models for LLM call patterns: streaming chat and structured JSON reconcile calls.

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
