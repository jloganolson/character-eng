# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Documentation Rule

**Always update `README.md` and `CLAUDE.md` when making changes** that affect any of the following: commands, architecture, modules, features, models, commands, testing, or configuration. Keep both files in sync with the actual codebase. This includes adding new modules, changing CLI behavior, adding/removing commands, changing model configs, or modifying test structure.

## Commands

```bash
# Run the app (API key loaded from .env via python-dotenv)
uv run -m character_eng

# Run unit tests
uv run pytest

# Run world reconciler integration test (hits real LLM)
uv run -m character_eng.qa_world [--model cerebras-llama|groq-llama|groq-gpt|gemini-3-flash|gemini-2.5-flash]

# Run chat QA integration test (hits real LLM)
uv run -m character_eng.qa_chat [--model cerebras-llama|groq-llama|groq-gpt|gemini-3-flash|gemini-2.5-flash]

# Run persona-based subjective QA (6 parallel personas, HTML report)
uv run -m character_eng.qa_personas [--model cerebras-llama] [--character greg] [--turns 10] [--open]

# Open latest HTML report in browser (or specific path)
uv run -m character_eng.open_report [path/to/report.html]

# Generate and serve a spoofed test report (for testing HTML/annotation UI)
uv run -m character_eng.open_report --test

# Start local vLLM server (interactive model picker)
uv run python -m character_eng.serve

# Start a specific local model
uv run python -m character_eng.serve lfm-2.6b

# Run model speed benchmark (all available models, 3 runs each)
uv run -m character_eng.benchmark

# Benchmark a specific model with custom run count
uv run -m character_eng.benchmark --model cerebras-llama --runs 5

# Run TTS backend speed benchmark (TTFA, RTF, total time)
uv run -m character_eng.tts_benchmark

# Benchmark a specific TTS backend with custom run count
uv run -m character_eng.tts_benchmark --backend chatterbox --runs 5

# Kill local vLLM server
lsof -ti:8000 | xargs kill -9

# Install/sync dependencies
uv sync

# Install with voice support (Deepgram STT + ElevenLabs TTS)
uv sync --extra voice

# Install with local GPU TTS (Qwen3-TTS, ~2GB VRAM)
uv sync --extra local-tts

# Run with voice mode
uv run -m character_eng --voice

# List audio devices (for config.toml input_device/output_device)
uv run -m character_eng.devices

# Run voice integration test (Deepgram + ElevenLabs connectivity)
uv run -m character_eng.qa_voice

# Add a dependency
uv add <package>
```

## Architecture

Interactive NPC character chat CLI with two-tier LLM backend. All use OpenAI-compatible endpoints. `CHAT_MODEL` (small/fast, default `cerebras-llama` 8B) handles streaming dialogue. `BIG_MODEL` (big/slow, default `groq-llama` 70B) handles eval, planning, and reconciliation — falls back to chat model if unavailable. Both are configured in `models.py`. No model selection menu — auto-selects on startup. Optional voice mode: Deepgram STT handles speech-to-text with turn detection, TTS is config-driven — five backends: ElevenLabs WebSocket (cloud, default), local Qwen3-TTS, Chatterbox-Turbo (in-process GPU), KaniTTS-2 (SSE streaming HTTP client), or MioTTS (HTTP client). All share the same 4-method interface (`send_text`, `flush`, `wait_for_done`, `close`). Two-speed character planning: fast eval (big model, every turn) tracks progress through a script, slow planner (big model) generates multi-beat scripts. Two-speed world updates: fast path stores changes as pending and lets the character react immediately, slow path (background reconciler using big model, falls back to chat model) reconciles pending changes into structured state mutations using stable fact IDs. User settings live in `config.toml` (gitignored); see `config.example.toml` for available options. Sixteen modules:

- **`character_eng/models.py`** — Model config registry. `MODELS` dict maps keys (`cerebras-llama`, `cerebras-gpt`, `groq-llama`, `groq-gpt`, `gemini-3-flash`, `gemini-2.5-flash`, `lfm-2.6b`, `lfm-1.2b`) to config dicts with `name`, `model`, `base_url`, `api_key_env`, `stream_usage`. Local models also have `local: True` and `path`. Models with `hidden: True` are excluded from benchmark auto-discovery but remain usable via QA `--model` flag. Two active model constants at the bottom: `CHAT_MODEL` (small/fast, default `cerebras-llama`) and `BIG_MODEL` (big/slow, default `groq-llama`) — change these to swap models. `DEFAULT_MODEL` is a legacy alias for `CHAT_MODEL` used by QA scripts.
- **`character_eng/config.py`** — Config loader. Reads `config.toml` from project root using `tomllib` (stdlib 3.12). Returns `AppConfig` dataclass with `VoiceConfig` (input_device, output_device, enabled, mic_mute_during_playback, tts_backend, ref_audio, tts_model, tts_device, tts_server_url). Device fields accept `int` (direct index), `str` (name substring for runtime lookup via `resolve_device()`), or `None` (system default). `tts_server_url` is used by kani/mio backends for the HTTP server endpoint. Gracefully returns defaults if file missing.
- **`character_eng/__main__.py`** — TUI entry point. Loads `config.toml` at startup via `load_config()`. `argparse` with `--voice` flag (overrides `config.voice.enabled`). Auto-selects `CHAT_MODEL` at startup (no model menu), then character selection. Voice device IDs from config are passed to `VoiceIO()` at both creation sites. Two nested loops: outer loop loads prompts, world state, and creates a chat session (breaks on `/reload`), inner loop handles user input and dispatches commands (`/reload`, `/trace`, `/back`, `/quit`, `/world`, `/beat`, `/plan`, `/goals`, `/voice`). Unknown commands (any `/` prefix not matching a known command) show an error with a `/help` hint. First user input always gets a natural LLM response (no beat delivery), then kicks off background eval + replan for visitor-aware script. Subsequent turns use LLM-guided beat delivery: when a beat exists, `stream_guided_beat()` injects the beat's intent and example line as guidance (via `load_beat_guide()`), then streams a real LLM response that reacts to the user's input while serving the beat's purpose. The `/beat` command (autonomous time-passing) still uses verbatim `deliver_beat()`. Beats with conditions are gated by `condition_check_call()` in the `/beat` path — if the condition isn't met, an idle line is displayed instead. If no beat (script empty/planner unavailable), falls back to LLM streaming. World changes (`/world <text>`) use two-speed updates: immediate pending + narrator injection + character reaction via LLM streaming, then background reconciliation + background eval. `/beat` delivers the next scripted beat if its condition is met, otherwise shows an in-character idle line. Non-blocking eval/plan: after each turn, a background eval thread (using `BIG_MODEL`, falling back to chat model) runs asynchronously — the user gets the prompt back immediately. Results are checked at the next turn start; stale results (where `_context_version` changed since the eval started) are discarded. When eval returns `advance` (script exhausted) or `off_book`, a background plan thread is started. Three background thread pools follow the same pattern: reconcile (`_reconcile_lock`), eval (`_eval_lock`), and plan (`_plan_lock`), each with version-based stale discard via `_context_version`. Only boot plan and `/beat` initial replan (when no script exists) remain synchronous. Uses `rich` for colored output and streaming display. Voice I/O integration: when voice mode is active, `voice_io` parameter is threaded through `stream_response`, `stream_guided_beat`, `deliver_beat`, `handle_beat`, and `handle_world_change` — LLM text chunks are simultaneously streamed to TTS, and barge-in (user starts speaking mid-response) cancels LLM+TTS+speaker. Barge-in awareness: when `voice_io._barged_in` is set, a system message is injected before the turn telling the character to acknowledge the interruption briefly, and auto-beat is suppressed after the response so the user gets silence until they speak again. After barge-in, `stream_response()` uses `speaker_playback_ratio` to truncate the recorded assistant message to approximate what was actually spoken (appends em dash). `deliver_beat()` defers `add_assistant()` until after TTS so it can record the truncated line on barge-in. Input comes from `voice_io.wait_for_input()` instead of console. `/voice` command toggles voice on/off mid-session. On exit (`/quit`, `/back`, Ctrl+C), stops voice I/O and saves both JSON log and an annotatable HTML report via `save_chat_html()` (imports `annotation_assets` from `qa_personas`).
- **`character_eng/chat.py`** — `ChatSession` wraps OpenAI client, configured via `model_config` dict (base URL, API key, model name). Manual message history with `system`/`user`/`assistant` roles. `add_assistant()` adds pre-rendered text to history without an LLM call (used for code-driven beat delivery). `replace_last_assistant(content)` modifies the most recent assistant message (used for barge-in truncation). `inject_system()` appends system-role messages mid-conversation. `get_history()` returns message list for eval context. Streams responses via generator with `try/finally` so partial responses from barge-in (generator closed mid-stream) still get recorded in message history. `stream_options` only sent when `model_config["stream_usage"]` is True (Cerebras doesn't support it).
- **`character_eng/prompts.py`** — Filesystem-based template engine. Scans `prompts/characters/` for subdirs containing `prompt.txt`. Resolves `{{key}}` macros via regex substitution (`global_rules`, `character`, `scenario`, `world`). Unknown macros left intact for debugging.
- **`character_eng/world.py`** — World state and script system. `WorldState` dataclass tracks static facts, mutable dynamic facts as `dict[str, str]` (ID → fact text, insertion-ordered) with stable IDs (`f1`, `f2`, ...), pending unreconciled changes, and an event log. `add_fact(text)` assigns next sequential ID. `add_pending(text)` stores unreconciled changes. `clear_pending()` drains pending list. `render_for_reconcile()` shows ID-tagged facts for reconciler context. `render()` shows facts without IDs (for character prompt) plus pending changes section. `Goals` dataclass holds a permanent `long_term` goal (parsed from `character.txt`). `Beat` holds a dialogue line, intent, optional `gaze`/`expression` fields for animation/TTS pre-rendering, and an optional `condition` (natural language precondition — empty means auto-deliver). `ConditionResult` holds `met` (bool), `idle` (in-character idle line when not met), `gaze`, and `expression`. `Script` is an ordered beat list with `current_beat`, `advance()`, `replace()`, `is_empty()`, `render()`, `show()`. `EvalResult` holds thought, gaze, expression, and `script_status` (advance/hold/off_book/bootstrap) with optional bootstrap fields. `PlanResult` holds a list of beats. `WorldUpdate` uses string fact IDs in `remove_facts`. `reconcile_call` sends a one-shot LLM call with JSON output using ID-tagged facts to produce a `WorldUpdate`. `eval_call` reads `eval_system.txt`, takes script context, returns script tracking status. `condition_check_call` reads `condition_system.txt`, checks a beat's condition against world state and conversation, returns a `ConditionResult`. `plan_call` reads `plan_system.txt`, uses the big model, returns a list of beats with gaze/expression/condition. `format_narrator_message` turns updates into `[bracketed stage directions]`. `format_pending_narrator` wraps a change description in brackets for fast-path injection. `load_beat_guide(intent, line)` reads `beat_guide.txt` and substitutes `{intent}` and `{line}` placeholders — used by guided beat delivery to inject intent context before the LLM response.
- **`character_eng/benchmark.py`** — Model speed benchmark. Runnable via `uv run -m character_eng.benchmark [--model KEY] [--runs N]`. Benchmarks three scenarios matching real usage: streaming chat (TTFT + total time), structured JSON reconcile call, and structured JSON eval call. Uses Greg's full prompts and world state for realistic payloads. Prints per-run details and a rich summary table, saves JSON logs to `logs/`.
- **`character_eng/serve.py`** — vLLM server launcher for local models. Scans `MODELS` for entries with `local: True`, kills any existing process on port 8000, then exec's into `vllm serve`. Accepts optional model key as CLI arg or shows interactive picker. Exports `build_vllm_cmd(key, cfg, port)`, `get_local_models()`, and `kill_port(port)`.
- **`character_eng/qa_personas.py`** — Persona-based subjective QA. `ConversationDriver` class encapsulates the full `__main__.py` background threading loop (reconcile/eval/plan) as instance state, so multiple personas run in parallel without interference. Six personas exercise stress scenarios: AFK Andy (minimal input), Chaos Carl (rapid world changes), Casual Player (normal engagement), Bored Player (redirections/dismissals), Beat Runner (scripted `/beat` every turn), Interrupter (alternates messages and beats). Persona actions generated by `BIG_MODEL` (Groq 70B) via structured JSON output. Runs all 6 in a `ThreadPoolExecutor(max_workers=6)` with 0.3s inter-turn sleep. Produces a self-contained HTML report (dark theme, per-persona cards, turn-by-turn conversation with action badges, eval details in collapsible `<details>`, stale discard markers) saved to `logs/qa_personas_{model}_{timestamp}.html`. Exports `annotation_assets(report_name, model_key, character)` which returns shared annotation CSS+JS+toolbar HTML — used by both persona reports and chat session HTML reports. Also exports `_esc()` (HTML escape) and `_action_badge()` for reuse by `__main__.py`'s `save_chat_html()`. Turn divs carry `data-persona`, `data-turn`, `data-action`, `data-input`, `data-response` attributes. Each turn has an "annotate" button at the bottom that toggles a textarea for notes. A sticky bottom toolbar shows annotation count and an "Export annotations" button. When served over HTTP (via `open_report`), annotations save directly to `logs/annotated/`; when opened as a file, falls back to browser download of a `.annotations.json` file. Export includes notes summary + full conversations for annotated personas only.
- **`character_eng/open_report.py`** — HTTP server for HTML reports with annotation save support. Runnable via `uv run -m character_eng.open_report [path]`. With no args, serves the latest HTML file in `logs/`. Starts a local HTTP server on a random port, serves the report at `/`, handles POST to `/save-annotations` which writes directly to `logs/annotated/`. Opens the browser automatically. Detects WSL (uses `cmd.exe /c start` for URLs), macOS (`open`), and Linux (`xdg-open`). `--test` flag generates a spoofed report with 4 fake personas (message, beat, world, error cases) using the real `generate_html_report()` codepath, then serves it — useful for testing HTML/annotation UI changes without LLM calls. Exports `serve_report(path)` for reuse by `qa_personas.py` (`--open` flag), `open_in_browser(url_or_path)` for general use, and `generate_test_report()` for programmatic test report creation.
- **`character_eng/voice.py`** — Optional voice I/O layer (install with `uv sync --extra voice`). Wraps around the existing text pipeline as an I/O layer — text mode continues to work unchanged. Top-level `resolve_device(spec, kind)` resolves device specs: `int` passes through, `str` does case-insensitive name substring matching against sounddevice devices filtered by I/O kind, `None` returns `None` (system default). Six classes: `MicStream` captures 16kHz/16-bit/mono PCM via sounddevice InputStream with callback queue + feeder daemon thread, accepts optional `device` ID. If the device doesn't support mono (e.g. XVF3800 2-ch mic array), opens with device's actual channel count and downmixes to mono (channel 0) in the callback. `SpeakerStream` plays PCM via sounddevice OutputStream with byte buffer — tries device default sample rate first, then ElevenLabs native 24kHz, then 48kHz (device-default-first avoids noisy failed probes on non-24kHz devices like the 16kHz XVF3800). PortAudio C-level stderr is suppressed during sample rate probing via fd redirect. `enqueue(pcm)`, `flush()` (barge-in), `wait_until_done()`, `wait_for_audio(timeout)` (blocks until first audio chunk arrives via `_audio_started` event), accepts optional `device` ID. Tracks `_bytes_played`/`_bytes_enqueued` for barge-in truncation — `playback_ratio` property returns fraction of enqueued audio that was actually played (0.0-1.0), `reset_counters()` zeros both at utterance start. `flush()` preserves counters so the ratio is available after barge-in. `DeepgramSTT` wraps Deepgram SDK live transcription (nova-3 model, endpointing, VAD events) — dispatches `on_transcript(text)` on speech_final/utterance_end, `on_turn_start()` on speech started (barge-in trigger). `ElevenLabsTTS` uses websocket-client to connect to ElevenLabs stream-input WebSocket (eleven_flash_v2_5 model, PCM 24kHz output) — `send_text(chunk)` for streaming, `flush()` for end of input, `wait_for_done(timeout)` blocks until ElevenLabs sends `isFinal` message (via `_generation_done` event), `close()` to reset (also sets `_generation_done` to unblock waiters). `KeyListener` reads keystrokes via `tty.setcbreak()` + `select` in a daemon thread, maps `w/b/p/g/t/Escape/Ctrl+C` to commands on the event queue. Saves terminal settings before entering cbreak mode and registers an `atexit` handler so settings are always restored on exit. `VoiceIO` orchestrator wires all components: `start()` resolves string device names via `resolve_device()` before creating streams. `start()`/`stop()` lifecycle, `wait_for_input()` blocks on unified event queue (transcripts, `/beat` from auto-beat timer, hotkey commands, sentinels), `speak(text_chunks)` feeds LLM streaming chunks to TTS then waits: flush → `tts.wait_for_done()` → `speaker.wait_for_audio()` → poll `speaker.is_done()` → `tts.close()`, `speak_text(text)` for one-shot beat TTS (same wait sequence), `cancel_speech()` for barge-in (sets cancelled event, clears `_is_speaking`, closes TTS, flushes speaker, cancels auto-beat timer). `_is_speaking` flag guards barge-in — `_on_turn_start()` only fires `cancel_speech()` when TTS is actively generating, preventing false triggers from ambient noise. `_barged_in` flag tracks whether the last cancellation was a barge-in — set by `cancel_speech()`, cleared by `speak()`/`speak_text()`. Used by `__main__.py` to inject barge-in context into the conversation and suppress auto-beat after the barge-in response. `speaker_playback_ratio` property delegates to `SpeakerStream.playback_ratio` — used after barge-in to truncate the recorded response to approximate what was actually spoken (text * ratio + em dash). `_on_turn_start()` always cancels the auto-beat timer (not just during active speech) so user speech during the countdown prevents stale `/beat` events from queuing. Uses `\n` prefix (not `\r\033[K`) when interrupting active speech to preserve partial response text in the terminal. Countdown visual: `_countdown_loop` prints dots to stdout (one per second, cleared on cancel/fire) guarded by `_started` so tests don't produce terminal output. Mic audio is muted (not forwarded to Deepgram) while `_is_speaking` is True to suppress echo from speaker pickup — controlled by `mic_mute_during_playback` flag (default True). Set to False for devices with hardware AEC (e.g. XVF3800) to enable voice barge-in during TTS playback. Hotkeys always work regardless. Accepts optional `input_device`/`output_device` IDs (int or str) passed to MicStream/SpeakerStream after resolution. Auto-beat timer: 4-second `threading.Timer` starts after TTS playback finishes, puts `/beat` on event queue, cancelled by user speech or `stop()`. `check_voice_available(tts_backend)` checks for required packages and API keys — accepts `tts_backend` param: `"local"` checks for `torch`/`transformers`, `"chatterbox"` checks for `chatterbox`/`torch`, `"kani"`/`"mio"` check for `requests`, `"elevenlabs"` checks `websocket-client`. All backends always require `DEEPGRAM_API_KEY` (STT is always Deepgram). Only `"elevenlabs"` needs `ELEVENLABS_API_KEY`. `VoiceIO.__init__` accepts `tts_backend`, `ref_audio`, `tts_model`, `tts_device`, `tts_server_url` params. `start()` resolves devices then branches on `_tts_backend`: `"local"` creates `LocalTTS`, `"chatterbox"` creates `ChatterboxTTS`, `"kani"` creates `KaniTTS`, `"mio"` creates `MioTTS`, `"elevenlabs"` creates `ElevenLabsTTS`. `list_audio_devices()` returns all sounddevice devices with index/name/channels/defaults. `get_default_devices()` returns current default input/output device info. Requires `DEEPGRAM_API_KEY` in `.env`, plus `ELEVENLABS_API_KEY` when using ElevenLabs backend.
- **`character_eng/local_tts.py`** — Local GPU TTS via vendored Qwen3-TTS (`thirdparty/qwen_tts/`). Drop-in replacement for `ElevenLabsTTS` with same 4-method interface (`send_text`, `flush`, `wait_for_done`, `close`). Key difference: ElevenLabs accepts streaming text chunks; Qwen3-TTS needs complete text per generation — `send_text()` buffers, `flush()` triggers background generation. Model is a module-level singleton loaded on first `flush()` via `_ensure_model()` (persists across utterances). Uses `attn_implementation="flash_attention_2"` for optimized attention (requires `flash-attn` package, included in `local-tts` extras). `_generate()` runs in a background thread: calls `model.generate_voice_clone_stream()`, iterates `AudioIteratorStreamer` chunks, converts float32→int16 PCM, delivers via `on_audio(pcm)` callback. Supports cancellation between chunks via `_cancelled` event. Constructor takes `on_audio`, `ref_audio_path` (WAV for voice cloning), `model_id` (HuggingFace model), `device` (GPU). Uses `x_vector_only_mode=True` for voice cloning (speaker embedding only, no ref text needed). Install with: `uv sync --extra local-tts`. Note: `torch` is pinned to `>=2.8,<2.9` for flash-attn prebuilt wheel compatibility.
- **`character_eng/chatterbox_tts.py`** — In-process GPU TTS via Chatterbox-Turbo (350M params, 24kHz output). Same 4-method interface as ElevenLabsTTS/LocalTTS. Module-level singleton via `load_model(device)`. `ChatterboxTTS` buffers text via `send_text()`, `flush()` spawns background thread calling `model.generate()` with zero-shot voice cloning from reference WAV. Converts torch tensor → int16 PCM, delivers in ~100ms chunks (2400 samples). 24kHz native — no resampling needed. Install with: `uv pip install chatterbox-tts` (separate from local-tts due to conflicting torch versions).
- **`character_eng/kani_tts.py`** — SSE streaming HTTP client for KaniTTS-2 OpenAI-compatible server. 4-method interface. `flush()` POSTs to `{server_url}/v1/audio/speech` with `stream_format: "sse"`, parses SSE events with base64-encoded PCM audio chunks, resamples 22050→24kHz via `np.interp`, delivers via `on_audio`. True streaming = lowest time-to-first-audio of local options. Only needs `requests` (no GPU deps client-side). Server must be started externally.
- **`character_eng/mio_tts.py`** — HTTP client for MioTTS inference server. 4-method interface. `flush()` POSTs to `{server_url}/v1/tts`, receives complete WAV response, extracts PCM via stdlib `wave`, resamples 44100→24kHz via `np.interp`, delivers in ~100ms chunks. Only needs `requests`. Server (vLLM + codec) must be started externally.
- **`character_eng/tts_benchmark.py`** — TTS backend speed benchmark. Runnable via `uv run -m character_eng.tts_benchmark [--backend ...] [--runs N]`. Measures TTFA (time to first audio), total generation time, audio duration, and RTF (real-time factor) across three sentence lengths (short/medium/long). Supports all five TTS backends. Prints rich summary table, saves JSON logs to `logs/tts_benchmark_{timestamp}.json`.
- **`character_eng/devices.py`** — Audio device lister. Runnable via `uv run -m character_eng.devices`. Prints all audio input/output devices with index, direction (IN/OUT), and name. Used to find device indices for `config.toml`.
- **`character_eng/qa_voice.py`** — Voice integration test. Runnable via `uv run -m character_eng.qa_voice`. Three tests: audio device enumeration (sounddevice), Deepgram STT connectivity (opens live WebSocket, sends sine wave PCM, verifies response), ElevenLabs TTS connectivity (opens stream-input WebSocket, sends test text, verifies audio bytes received). No microphone/speaker needed — tests API connectivity only.

## Prompt system

Characters are defined as directories under `prompts/characters/{snake_case_name}/` with these files:
- `prompt.txt` — template with `{{global_rules}}`, `{{character}}`, `{{scenario}}`, `{{world}}` macros
- `character.txt` — personality, voice, and optional goals (`Long-term goal:` line)
- `scenario.txt` — situation and context

Optional world state files (line-per-fact format):
- `world_static.txt` — permanent facts that never change
- `world_dynamic.txt` — initial mutable state (loaded with stable IDs `f1`, `f2`, ...)

`prompts/global_rules.txt` contains rules shared across all characters. Adding a new character requires no code changes — just create the directory with a `prompt.txt`.

System prompts for the reconciler, eval, condition, and plan systems are also file-based and live-editable:
- `prompts/reconcile_system.txt` — system prompt for the world-state reconciler LLM (processes pending changes using stable fact IDs)
- `prompts/eval_system.txt` — system prompt for the character eval LLM (script tracking + inner monologue)
- `prompts/condition_system.txt` — system prompt for the condition checker LLM (gates beat delivery)
- `prompts/plan_system.txt` — system prompt for the conversation planner LLM (beat generation with conditions)
- `prompts/beat_guide.txt` — template for beat guidance injected before LLM-guided delivery (has `{intent}` and `{line}` placeholders)

These are read from disk on every call, so edits take effect immediately (no `/reload` needed).

## Configuration

User settings live in `config.toml` (gitignored). Copy `config.example.toml` to get started. The app works without it — all settings have defaults.

```toml
[voice]
input_device = "reSpeaker"   # by name substring (case-insensitive), or integer index
output_device = "reSpeaker"  # route TTS through device for hardware AEC reference
enabled = true               # start in voice mode (--voice flag overrides)
mic_mute_during_playback = false  # disable for hardware AEC mics (e.g. XVF3800)
tts_backend = "local"        # "elevenlabs", "local", "chatterbox", "kani", or "mio"
ref_audio = "/path/to/reference.wav"  # reference audio for local/chatterbox voice cloning
tts_model = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"  # HuggingFace model ID
tts_device = "cuda:0"        # GPU device for local/chatterbox TTS
tts_server_url = "http://localhost:8000"  # server URL for kani/mio backends
```

Prefer `config.toml` over CLI flags for persistent settings.

## Models

Two-tier model system — change `CHAT_MODEL` and `BIG_MODEL` in `character_eng/models.py` to swap:

- **`CHAT_MODEL`** (`cerebras-llama`) — small/fast, used for streaming character dialogue. Requires `CEREBRAS_API_KEY`.
- **`BIG_MODEL`** (`groq-llama`) — big/slow, used for background eval, planning, and reconciliation. Requires `GROQ_API_KEY`. Falls back to chat model if unavailable.

No model selection menu — auto-selects on startup. All available models in the registry:

- **Llama 3.1 8B** (`cerebras-llama`) — `llama3.1-8b` via Cerebras. Requires `CEREBRAS_API_KEY`.
- **GPT-OSS 120B** (`cerebras-gpt`) — `gpt-oss-120b` via Cerebras. Requires `CEREBRAS_API_KEY`.
- **Llama 3.3 70B** (`groq-llama`) — `llama-3.3-70b-versatile` via Groq. Requires `GROQ_API_KEY`.
- **GPT-OSS 120B** (`groq-gpt`) — `openai/gpt-oss-120b` via Groq. Requires `GROQ_API_KEY`.
- **Gemini 3.0 Flash** (`gemini-3-flash`) — `gemini-3-flash-preview` via Google. Requires `GEMINI_API_KEY`.
- **Gemini 2.5 Flash** (`gemini-2.5-flash`) — `gemini-2.5-flash` via Google. Requires `GEMINI_API_KEY`.
- **LFM-2 2.6B** (`lfm-2.6b`) — Local via vLLM. Requires running `serve.py` first.
- **LFM-2.5 1.2B** (`lfm-1.2b`) — Local via vLLM. Requires running `serve.py` first.

Cloud models need their API key set in `.env`. Local models can be started manually via `serve.py`.

## Characters

- **Greg** (`prompts/characters/greg/`) — Robot head on a desk. Orb gazing scenario with world state. Long-term goal: experience the universe.

## In-chat commands

- `/world` — Show current world state (static facts with IDs, dynamic facts, pending changes, event log)
- `/world <text>` — Describe a change; character reacts immediately, background reconciler updates state with stable fact IDs. Reconcile results shown at start of next turn.
- `/beat` — Deliver next scripted beat (time passes). Checks beat condition first — if not met, shows an in-character idle line instead. If no script, triggers replanning.
- `/plan` — Show current script (beat list with current beat highlighted)
- `/goals` — Show character goals (long-term)
- `/voice` — Toggle voice mode on/off mid-session. Creates/starts or stops VoiceIO. Shows active mic/speaker device on enable.
- `/devices` — List all audio input/output devices with index, name, channel count, and default markers. Works even without voice mode active.
- `/reload` — Reload prompt files, restart conversation
- `/trace` — Show system prompt, model, token usage
- `/back` — Return to character selection
- `/quit` — Exit program
- Unknown `/` commands show an error message with a `/help` hint

### Voice mode hotkeys (when voice is active)

- `w` → `/world`, `b` → `/beat`, `p` → `/plan`, `g` → `/goals`, `t` → `/trace`
- `Escape` → Toggle voice off (switch to text mode)
- `Ctrl+C` → Exit

## Logging

All sessions save full JSON logs to `logs/` (gitignored). Log path is printed on session end. HTML reports include an annotation UI — click the pencil icon on any turn to leave a note, then "Export annotations" to download a `.annotations.json` file.

- **Model benchmarks**: `logs/benchmark_{timestamp}.json` — per-run timing (TTFT, total ms) and full response text for each scenario
- **TTS benchmarks**: `logs/tts_benchmark_{timestamp}.json` — per-sentence TTFA, total time, audio duration, and RTF for each TTS backend
- **Chat sessions (JSON)**: `logs/chat_{character}_{model}_{session}_{timestamp}.json` — every send, world change, reconcile, and eval with full untruncated responses
- **Chat sessions (HTML)**: `logs/chat_{character}_{model}_{session}_{timestamp}.html` — annotatable HTML report with same data, dark theme, collapsible eval/reconcile details, annotation export
- **QA chat**: `logs/qa_chat_{model}_{timestamp}.json` — same detail plus expect results
- **QA world**: `logs/qa_world_{model}_{timestamp}.json` — world state before/after each scenario plus reconciler updates
- **QA personas**: `logs/qa_personas_{model}_{timestamp}.html` — self-contained HTML report with per-persona conversation cards, eval details, stale discard markers, annotation export

## Testing

Three layers, from fast to slow:

**Unit tests** (`uv run pytest`) — Fast, mocked, no API calls. Run on every push via git pre-push hook.
- `tests/test_config.py` — load_config: missing file returns defaults, full config parses all fields, partial config fills defaults, invalid types handled, TTS backend defaults, local TTS config parsing, partial TTS defaults, string device names, tts_server_url defaults + parsing
- `tests/test_chat.py` — ChatSession: init, send (message history + streaming + partial response on generator close), add_assistant, replace_last_assistant (barge-in truncation), inject_system, get_history, trace_info, local model api_key config
- `tests/test_prompts.py` — Template loading, macro substitution, list_characters
- `tests/test_serve.py` — get_local_models, build_vllm_cmd, kill_port (missing lsof)
- `tests/test_local_tts.py` — LocalTTS: send_text buffers, flush clears buffer + starts gen thread, flush empty/whitespace sets done, close sets cancelled/generation_done/clears buffer, wait_for_done returns on set/timeout, float32→int16 PCM conversion, respects cancelled event, sets done on completion/error
- `tests/test_voice.py` — MicStream (callback queuing, stop drains queue, multichannel downmix to mono, actual_channels default), SpeakerStream (enqueue/flush/wait_until_done, callback buffer pull, silence padding, stop, audio_started event set/clear/wait_for_audio, resample flag default, upsample 2x, enqueue with/without resample, fallback to 48kHz, bytes_enqueued/bytes_played tracking, playback_ratio, reset_counters, flush preserves counters), DeepgramSTT (transcript dispatch, turn start, interim ignored, utterance end), ElevenLabsTTS (send_text, flush, close, generation_done event, wait_for_done timeout, close sets generation_done, recv_loop sets generation_done on isFinal), VoiceIO (event queue routing, auto-beat timer fire/cancel, cancel_speech sets event/flushes speaker/closes TTS, wait_for_input, barge-in guard with _is_speaking, mic muting during TTS playback, mic_mute_during_playback flag, _barged_in flag lifecycle: initially false, set by cancel_speech, cleared by speak/speak_text, _on_turn_start cancels auto-beat when not speaking, speaker_playback_ratio delegates to speaker/returns 1.0 without speaker, tts_backend storage/default, tts_server_url storage/default), KeyListener (key mapping), check_voice_available (all present, missing keys, local backend, local missing torch, chatterbox backend + missing deps, kani backend, mio backend, kani missing requests), sentinel string uniqueness, resolve_device (None passthrough, int passthrough, string name match, no match raises, wrong I/O kind raises)
- `tests/test_chatterbox_tts.py` — ChatterboxTTS: send_text buffers, flush clears buffer + starts gen thread, flush empty/whitespace sets done, close sets cancelled/generation_done/clears buffer, wait_for_done returns on set/timeout, tensor→int16 PCM conversion, chunked delivery (~100ms chunks), respects cancelled event, sets done on completion/error
- `tests/test_kani_tts.py` — KaniTTS: send_text buffers, flush clears buffer + starts gen thread, flush empty/whitespace sets done, close sets cancelled/generation_done/clears buffer, wait_for_done returns on set/timeout, resample ratio verification (22050→24000), SSE parsing with mock requests.post, respects cancelled event, sets done on completion/error, skips non-audio events, server URL trailing slash stripped
- `tests/test_mio_tts.py` — MioTTS: send_text buffers, flush clears buffer + starts gen thread, flush empty/whitespace sets done, close sets cancelled/generation_done/clears buffer, wait_for_done returns on set/timeout, resample ratio verification (44100→24000), WAV extraction + chunked delivery, respects cancelled event, sets done on completion/error, server URL trailing slash stripped, no-resample for 24kHz WAV
- `tests/test_world.py` — WorldState (dict-based dynamic with stable IDs, render/apply/show/load, add_fact, add_pending/clear_pending, render_for_reconcile, render_with_pending), Beat construction (with/without gaze/expression, with condition), ConditionResult (met/not met), Script (current_beat/advance/replace/is_empty/render/render_with_gaze_expression/render_with_condition/show), EvalResult (construction/bootstrap fields), Goals (render/load from character.txt), format_narrator, format_pending_narrator, load_beat_guide (substitutes placeholders, reads from disk each call, missing file raises), reconcile_call (6 scenarios incl. basic, no removals, empty state, multiple pending, reads reconcile_system.txt, IDs in context), eval_call (7 scenarios incl. script/goals in context, reads eval_system.txt), condition_check_call (4 scenarios incl. met/not met, condition in context, reads condition_system.txt), plan_call (6 scenarios incl. gaze/expression/condition parsing, plan_request in context, reads plan_system.txt)

**`qa_world`** (`uv run -m character_eng.qa_world [--model cerebras-llama|groq-llama|groq-gpt]`) — Integration test hitting real LLM. 4 cumulative reconcile_call scenarios with strict validation (schema, valid fact IDs, non-empty events). Uses stable fact IDs (`f1`-`f4`) instead of integer indices. Run manually before releases or after changes to world.py.

**`qa_chat`** (`uv run -m character_eng.qa_chat [--model cerebras-llama|groq-llama|groq-gpt]`) — End-to-end integration test hitting real LLM. Parses `test_plan.md` for test scenarios covering chat, world changes (two-speed: pending + synchronous reconcile), eval, script tracking with user pushback detection, and prompt injection. Run manually before releases or after changes to chat/world/eval pipeline.

**`qa_voice`** (`uv run -m character_eng.qa_voice`) — Voice connectivity test. Verifies audio device enumeration, Deepgram STT WebSocket connection + audio send, and ElevenLabs TTS WebSocket connection + text send + audio receive. No mic/speaker needed. Run after setting `DEEPGRAM_API_KEY` and `ELEVENLABS_API_KEY` in `.env`.

**`qa_personas`** (`uv run -m character_eng.qa_personas [--model cerebras-llama] [--character greg] [--turns 10]`) — Persona-based subjective QA. 6 LLM-driven personas chat with the character in parallel exercising the full async loop (background eval/plan/reconcile). Stress tests: rapid input causing stale discards (Chaos Carl), inaction (AFK Andy), overaction (Chaos Carl), script interruption (Interrupter), and pure beat delivery (Beat Runner). Produces an HTML report for human review. `--turns` overrides turn count for all personas.

`test_plan.md` is human-editable — add new sections to expand QA coverage without touching code. Supports these commands:
- `send: <text>` — send user message, store response
- `world: <text>` — describe a world change (fast-path + synchronous reconcile)
- `script: intent1 | line1, intent2 | line2` — load a script of beats for the eval to track
- `beat` — run eval against current conversation and script
- `expect: <check>` — assert a condition (`non_empty`, `max_length:<n>`, `world_updated`, `valid_thought`, `eval_status:<status>`, `in_character`)

**After implementing a plan or making significant changes**, always run the full test suite before reporting done:
1. `uv run pytest` — unit tests (must pass)
2. `uv run -m character_eng.qa_chat` — end-to-end QA (must pass)
3. `uv run -m character_eng.qa_world` — if world.py was touched (must pass)
