# Character Engine

Quick operator guide: [QUICKSTART.md](/home/logan/Projects/character-eng/QUICKSTART.md)

Interactive NPC chat CLI with two-tier LLM routing (Cerebras, Groq, Google Gemini, or local vLLM models). **Chat** uses Kimi K2 for streaming dialogue; **microservices** (eval, director, planning, reconciliation, expression) use 8B for fast structured JSON. Optional **voice mode** (Deepgram STT + ElevenLabs TTS) lets you speak to characters and hear them respond. Characters have personalities, world state that evolves during conversation, and a script system driven by parallel microservices — post-response calls (eval + director) fire concurrently for ~350ms total instead of ~1050ms sequential. A single-beat planner generates one beat at a time synchronously, eliminating background plan threads entirely. During conversation, beats use LLM-guided delivery: the beat's intent guides the LLM to respond naturally to the user while serving the beat's purpose (no verbatim pasting). The `/beat` command (autonomous time-passing) still uses verbatim delivery for TTS pre-rendering. After each character response, lightweight microservices derive gaze/expression, evaluate script progress, and check scenario exit conditions — all running concurrently for immediate effect.

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

If multiple keys are set, you'll choose a model at startup. If only one is set, it's auto-selected. Defaults live in `config.toml` under `[models]`, with `CHARACTER_ENG_CHAT_MODEL`, `CHARACTER_ENG_MICRO_MODEL`, and `CHARACTER_ENG_BIG_MODEL` still available as env overrides.

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

[models]
chat_model = "groq-llama-8b"
micro_model = "groq-llama-8b"
big_model = "groq-llama"
```

The app works without `config.toml` — all settings have defaults. The `--voice` flag still works as an override.

## Run

```bash
./scripts/run_live.sh                   # app/dashboard only: voice + vision + paused-ready startup
./scripts/stop_live.sh                  # stop only the app/dashboard layer
./scripts/run_local.sh                  # one-command local stack: voice + vision + paused-ready startup
./scripts/run_heavy.sh                  # keep vLLM/vision/Pocket-TTS hot in the background
./scripts/vision_smoke.sh               # one-shot heavy stack readiness check with model-status output
./scripts/run_hot_offline.sh            # archive-only dashboard loop; auto-installs canonical archive fixture; use ?fixture=... or ?archive=...
./scripts/run_hot.sh                    # restart only the fast app/UI layer
./scripts/stop_local.sh                 # stop the app plus heavy local services
uv run -m character_eng                  # text mode (dashboard auto-opens at :7862)
uv run -m character_eng --no-dashboard   # text mode without dashboard
uv run -m character_eng --voice          # voice mode (speak to chat, hear responses)
uv run -m character_eng --vision         # vision mode (live camera + visual intelligence)
uv run -m character_eng --voice --vision # voice + vision
uv run -m character_eng --vision-mock walkup.json  # vision with mock replay (no camera)
uv run -m character_eng --browser        # browser mic/camera via WebSocket (remote mode)
uv run -m character_eng --browser --vision  # full remote: browser audio + camera → vision
uv run -m character_eng --smoke          # smoke test (auto greg, scripted inputs, exit)
```

`./scripts/run_local.sh` is the recommended hands-on entrypoint. By default it does a clean local restart first: it stops older character-eng app processes plus anything still bound to the usual local ports for dashboard/bridge (`:7862`), vision (`:7860`), vLLM (`:8000`), and Pocket-TTS (`:8003`), then launches the app with voice, vision, and startup-paused enabled.

If you only want to manage the app/dashboard layer and leave heavy services alone, use:

```bash
./scripts/run_live.sh
./scripts/stop_live.sh
```

`run_live.sh` is a thin wrapper around `uv run -m character_eng` with the usual `greg + voice + vision + start-paused` defaults. `stop_live.sh` only stops the app/dashboard layer; it does not tear down vLLM, Pocket-TTS, or the vision service.

Useful variants:

```bash
CLEAN_START=0 ./scripts/run_local.sh     # reuse existing local services
KEEP_POCKET=1 ./scripts/run_local.sh     # keep an already-running Pocket-TTS server
./scripts/run_local.sh --no-dashboard    # same clean start, but CLI only
```

For the simplest persistent-dev loop:

```bash
./scripts/run_heavy.sh
./scripts/run_hot.sh
```

`run_heavy.sh` starts or reuses the slow stack (`vLLM`, vision service, `Pocket-TTS`) and now waits until those services are actually reachable before it returns. `run_hot.sh` is intentionally strict: it only restarts the fast character app/dashboard layer and will fail fast unless the heavy stack is already up. When you are done:

```bash
./scripts/stop_local.sh
```

For a local heavy-stack simulation of the lower-latency WebRTC transport, use:

```bash
./scripts/run_hot_webrtc.sh
```

That mode keeps the full-local heavy stack on the same machine, starts a disposable LiveKit dev server, and runs the app in `remote_hot_webrtc` mode.
The Vision panel shows a LiveKit preview of the outgoing camera stream, while the app injects those frames into the vision service.

For the GCP-backed WebRTC hybrid path, use:

```bash
./deploy/gcp/doctor.sh
./scripts/run_hot_remote_webrtc.sh
./scripts/stop_hot_remote_webrtc.sh
```

That keeps the app local while using remote vision + Pocket-TTS over an SSH tunnel and a remote LiveKit server on the GCP VM.
This is the default hosted-heavy workflow.

To allowlist another engineer for that path by Gmail address:

```bash
./deploy/gcp/onboard_user.sh teammate@gmail.com
./deploy/gcp/offboard_user.sh teammate@gmail.com
```

For faster heavy-runtime debugging, use:

```bash
./scripts/vision_smoke.sh
TEARDOWN=1 ./scripts/vision_smoke.sh
```

`vision_smoke.sh` runs the heavy stack in one-shot mode, prints the raw `/model_status` payload on success, and exits. By default it leaves the heavy services warm for follow-up debugging; set `TEARDOWN=1` to stop only the services started by that smoke run.

If `sam3` is part of the check, make sure the environment has Hugging Face access for the gated `facebook/sam3` repo, for example by exporting `HF_TOKEN` (or another `huggingface_hub`-compatible auth env var) before running the smoke command.

Pick a character from the menu, then chat. After each response, synchronous microservices evaluate script progress and check scenario exits with immediate effect. In-session commands:

| Command | What it does |
|---------|-------------|
| `/info` | Show all state (world, plan, goals, stage, people) in one dump |
| `/world <text>` | Describe a change — character reacts immediately, reconciler updates state in background |
| `/beat` | Deliver next scripted beat (time passes) — checks conditions, shows idle line if not met |
| `/see <text>` | Describe something the character perceives — character reacts, reconciler processes |
| `/sim <name>` | Replay a sim script with timing (quoted lines = dialogue, unquoted = perception) |
| `1`-`4` | Trigger a stage exit by number (shown in the HUD line before each prompt) |
| `/vision` | Show current visual context and gaze targets (requires `--vision`) |
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

**Pocket-TTS** (`tts_backend = "pocket"`): Streaming HTTP client for Pocket-TTS server (100M causal transformer). True autoregressive streaming with flat ~80ms TTFA. Supports voice cloning via `ref_audio` (WAV upload) or `pocket_voice` (path to `.safetensors` voice file passed to `pocket-tts serve --voice`). The server **auto-starts** when not already running and **auto-stops** on exit. If `pocket-tts` is not already installed, character-eng now bootstraps it into a dedicated runtime tool directory instead of baking it into the core app environment. For faster startup, run the server persistently in the background (`pocket-tts serve --port 8003 &`) or preinstall it with `uv sync --extra pocket-server`. Set `tts_server_url` in `config.toml` (default: `http://localhost:8003`).

All backends use the same 4-method interface (`send_text`, `flush`, `wait_for_done`, `close`) — no changes to the main conversation loop.

### Requirements

Voice mode requires at least:
- `DEEPGRAM_API_KEY` — for speech-to-text (always required)
- `ELEVENLABS_API_KEY` — for ElevenLabs TTS (only when `tts_backend = "elevenlabs"`)

Run `uv run -m character_eng.qa_voice` to verify connectivity (no mic needed).

If voice deps or keys are missing, the app falls back to text mode gracefully.

## Vision mode

Vision mode adds live visual intelligence from a webcam. The character can see people approaching, track their identities, detect objects, and ask VLM questions about what's in frame.

**Architecture**: Vision runs as a standalone service (`services/vision/`) with its own Python venv and heavy GPU dependencies (torch, SAM3, InsightFace, torchreid). This avoids dependency conflicts entirely. character-eng gets a thin HTTP client — zero new deps added to the main app.

### Setup

```bash
# Install vision service deps (first time only)
cd services/vision && uv sync && cd ../..

# Start vLLM for the VLM model (if using VLM questions)
# Uses ~6GB VRAM for LFM2-VL-3B
cd services/vision && ./start.sh && cd ../..

# Or just run character-eng with --vision (auto-starts the service)
uv run -m character_eng --vision
```

### Configuration

Add a `[vision]` section to `config.toml`:

```toml
[vision]
enabled = true                       # same as --vision flag
service_url = "http://localhost:7860" # vision service URL
service_port = 7860
auto_launch = true                   # auto-start if not running
synthesis_min_interval = 0.75        # seconds between synthesis cycles
```

### Mock vision (no camera)

Test the vision system without a camera or GPU using pre-recorded visual replays:

```bash
# Use a built-in replay (looks in services/vision/replays/)
uv run -m character_eng --vision-mock walkup.json

# Use a custom replay file
uv run -m character_eng --vision-mock /path/to/replay.json
```

Replay files are JSON with timestamped snapshots. See `services/vision/replays/` for examples. The mock server serves the same endpoints as the real vision service — the app can't tell the difference.

### How it works

1. **Vision service** (separate process, ~10.5GB VRAM) captures camera, runs face tracking (InsightFace), person tracking (SAM3 + ReID), and VLM questions
2. **VisionManager** (background thread in character-eng) polls `/snapshot` every 750ms, runs `vision_synthesis_call` (8B via Groq) to distill raw visual data into perception events
3. **Perception events** flow through the existing `process_perception()` pipeline — same as `/see` commands but automatic
4. **Gaze targets** update dynamically from detected people/objects instead of static defaults
5. **Visual focus** (`visual_focus_call`, 8B) generates context-driven VLM questions based on the character's current beat/stage intent

### GPU budget (~10.5GB on RTX 4090)

| Component | VRAM |
|-----------|------|
| SAM3 | ~3.4GB |
| InsightFace buffalo_s | ~0.8GB |
| ReID ResNet50 | ~0.25GB |
| LFM2-VL-3B (vLLM) | ~6GB |
| **Total** | **~10.5GB** |

character-eng process: zero GPU usage (text LLM via Groq).

## Dashboard

A real-time HTML dashboard opens automatically alongside the CLI at `http://localhost:7862`. It shows everything happening in the session live:

- **Conversation timeline** — streaming responses with gaze/expression metadata, eval status, director decisions
- **World state** — static + dynamic facts with stable IDs, pending changes
- **Stage/script** — current scenario stage, goal, and beat script with current beat highlighted
- **People** — tracked people with scoped facts
- **Vision** — MJPEG feed + face/person counts when `--vision` is active
- **Timing** — TTFT, total response time, eval status in footer
- **Input** — text input bar and hotkeys (B=beat, 1-4=triggers, I=info) for when not using voice

The dashboard uses SSE for live streaming and `/state` for reconnect catchup — refresh the browser at any time to see the full session. Uses Gruvbox dark/light theme (toggle in header). No new dependencies (stdlib `http.server` + `threading`).

The HTML/CSS/JS is a single editable file (`character_eng/dashboard/index.html`) — edit and refresh the browser to see changes without restarting.

Disable with `--no-dashboard` or set `enabled = false` in `[dashboard]` config:

```toml
[dashboard]
enabled = true
port = 7862
```

## Remote Deployment (GCP / SSH tunnel)

Browser mode (`--browser`) enables remote access by routing mic and camera through a WebSocket bridge instead of local hardware. The supported remote path is the GCP VM plus SSH/HTTPS access.

### Two modes, one codebase

| Mode | Audio | Camera | Access |
|------|-------|--------|--------|
| **Local** (default) | sounddevice mic/speaker | cv2.VideoCapture | Terminal + browser :7862 |
| **SSH / Hosted** | Browser getUserMedia → WS / LiveKit | Browser camera → WS / LiveKit | `ssh -L 7862:localhost:7862 -L 7863:localhost:7863` or GCP hybrid scripts |

### Quick start

```bash
# Run locally with browser audio (e.g. for SSH tunnel access)
uv run -m character_eng --browser --character greg

# Run with browser audio + vision
uv run -m character_eng --browser --vision --character greg

# SSH tunnel from another machine
ssh -L 7862:localhost:7862 -L 7863:localhost:7863 user@server
# Then open http://localhost:7862 in your browser
```

The dashboard auto-detects the bridge — when connected, mic/camera controls appear in the sidebar. Click "Mic On" to start capturing audio, "Cam On" for camera (if `--vision` is active). Audio latency (RTT) is displayed.

### Docker / GCP

```bash
# Build container
docker build -t character-eng .

# Run locally with GPU
docker run --gpus all -p 7862:7862 -p 7863:7863 --env-file .env character-eng
```

For the parallel GCP VM path, see [`deploy/gcp/README.md`](deploy/gcp/README.md).

### CI/CD

Push to `main` should build and publish the container artifact. Deployment now targets the GCP VM path.

## Editing prompts

Characters live in `prompts/characters/<name>/` with these files:

| File | Purpose |
|------|---------|
| `prompt.txt` | Main template — uses `{{global_rules}}`, `{{character}}`, `{{scenario}}`, `{{world}}`, `{{people}}`, `{{vision}}` macros |
| `character.txt` | Personality, voice, and goals (optional `Long-term goal:` line) |
| `scenario_script.toml` | Preferred live scenario package: authored setup (`[setup]`) plus stage graph (`[scenario]`, `[[stage]]`) |
| `scenarios/*.toml` | Optional scenario variants used by QA/harness flows or alternate modes |
| `sims/*.sim.txt` | Sim scripts for `/sim` command — `time_offset \| description` per line (optional) |

Legacy `scenario.txt`, `world_static.txt`, and `world_dynamic.txt` are still supported as fallback for older characters, but the preferred authoring path is to keep setup and progression together inside `scenario_script.toml`.

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

**Live editing works** — edit any character prompt file in your editor, then type `/reload` in the chat to pick up changes without restarting. The conversation resets but you keep your place in the character menu. Reconciler, eval, plan, director, expression, vision-focus, and vision-synthesis system prompts are re-read on every call, so edits take effect immediately without `/reload`.

### Adding a new character

Create a directory under `prompts/characters/` with at least a `prompt.txt`. No code changes needed. The current built-in character is:

- **Greg** (`greg/`) — Robot head in the local live scenario, with mock walkup sim assets for testing.

See **[docs/guide.md](docs/guide.md)** for a detailed guide on creating characters, writing scenarios, tuning prompts, and testing — designed for creative collaborators who may not be familiar with the codebase.

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

# One-time browser install for Playwright UI regression tests
uv run playwright install chromium

# One-command validation: pytest + smoke + voice QA when keys are available
./scripts/validate.sh

# Extended validation: also runs qa_world + qa_chat against the active chat model
./scripts/validate.sh full

# E2e smoke test standalone (sends real messages, exercises full startup path)
uv run -m character_eng --smoke

# Integration tests (hit real LLM, default model: cerebras-llama)
uv run -m character_eng.qa_world                       # world state reconciler
uv run -m character_eng.qa_chat                        # end-to-end chat, world, eval
uv run -m character_eng.qa_world --model groq-llama    # test with Groq Llama
uv run -m character_eng.qa_chat --model groq-llama     # test with Groq Llama

# Voice connectivity test (Deepgram + ElevenLabs, no mic needed)
uv run -m character_eng.qa_voice

# Per-role LLM evaluation (objective + subjective, JSON + HTML reports)
uv run -m character_eng.qa_roles --model kimi-k2
uv run -m character_eng.qa_roles --merge log1.json log2.json  # side-by-side HTML

# Prompt simplification tests (full/medium/minimal variants)
uv run -m character_eng.qa_scaffold --model kimi-k2
uv run -m character_eng.qa_scaffold --merge log1.json log2.json

# Scaffold toggle tests (beats/thinker/director on/off × model)
uv run -m character_eng.qa_toggles --model kimi-k2
uv run -m character_eng.qa_toggles --models groq-llama-8b,kimi-k2
uv run -m character_eng.qa_toggles --merge logs/qa_toggles_*.json

# Persona QA (7 parallel personas, HTML report)
uv run -m character_eng.qa_personas                        # default model + character
uv run -m character_eng.qa_personas --model groq-llama     # test with Groq Llama
uv run -m character_eng.qa_personas --turns 5              # quick run with fewer turns
uv run -m character_eng.qa_personas --turns 5 --open       # run and open report in browser

# Open HTML reports in browser (starts local server for annotation save)
uv run -m character_eng.open_report                        # latest report in logs/
uv run -m character_eng.open_report logs/some_report.html  # specific file
uv run -m character_eng.open_report --test                 # generate + serve a spoofed test report

# Focused browser regressions for the dashboard + full-stack trace viewer
uv run pytest tests/test_dashboard_playwright.py tests/test_qa_full_stack_playwright.py
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
