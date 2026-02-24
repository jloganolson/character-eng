# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Documentation Rule

**Always update `README.md` and `CLAUDE.md` when making changes** that affect any of the following: commands, architecture, modules, features, models, commands, testing, or configuration. Keep both files in sync with the actual codebase. This includes adding new modules, changing CLI behavior, adding/removing commands, changing model configs, or modifying test structure.

## Commands

```bash
# Run the app (API key loaded from .env via python-dotenv)
uv run -m character_eng

# Run unit tests (includes e2e smoke test when API keys are set)
uv run pytest

# Run e2e smoke test standalone (sends real messages, exercises full startup path)
uv run -m character_eng --smoke

# Run world reconciler integration test (hits real LLM)
uv run -m character_eng.qa_world [--model cerebras-llama|groq-llama|groq-gpt|gemini-3-flash|gemini-2.5-flash]

# Run chat QA integration test (hits real LLM)
uv run -m character_eng.qa_chat [--model cerebras-llama|groq-llama|groq-gpt|gemini-3-flash|gemini-2.5-flash]

# Run persona-based subjective QA (7 parallel personas, HTML report)
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

# Kill local vLLM server
lsof -ti:8000 | xargs kill -9

# Install/sync dependencies (includes voice: Deepgram STT + ElevenLabs/Pocket TTS)
uv sync

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

Interactive NPC character chat CLI with two-tier LLM backend. All use OpenAI-compatible endpoints. `CHAT_MODEL` (small/fast, default `cerebras-llama` 8B) handles streaming dialogue. `BIG_MODEL` (big/slow, default `groq-llama` 70B) handles eval, planning, reconciliation, and direction — falls back to chat model if unavailable. Both are configured in `models.py`. No model selection menu — auto-selects on startup. Optional voice mode: Deepgram STT handles speech-to-text with turn detection, TTS is config-driven — three backends: ElevenLabs WebSocket (cloud, default), local Qwen3-TTS on GPU (with ICL voice cloning), or Pocket-TTS (streaming HTTP server with voice cloning). All share the same 4-method interface (`send_text`, `flush`, `wait_for_done`, `close`). Two-speed character planning: fast eval (big model, every turn) tracks progress through a script, slow planner (big model) generates multi-beat scripts. Two-speed world updates: fast path stores changes as pending and lets the character react immediately, slow path (background reconciler using big model, falls back to chat model) reconciles pending changes into structured state mutations using stable fact IDs. Scenario director: optional TOML-based branching stage graph per character; a director LLM runs after each turn alongside eval to track stage transitions, advancing when exit conditions are met. Person state: individual people tracked alongside world state with scoped fact IDs; the reconciler handles person-fact assignments. Perception layer: `/see` command for manual perception events, `/sim` for scripted replay from `.sim.txt` files. User settings live in `config.toml` (gitignored); see `config.example.toml` for available options. Sixteen modules:

- **`character_eng/models.py`** — Model config registry. `MODELS` dict maps keys (`cerebras-llama`, `cerebras-gpt`, `groq-llama`, `groq-gpt`, `gemini-3-flash`, `gemini-2.5-flash`, `lfm-2.6b`, `lfm-1.2b`) to config dicts with `name`, `model`, `base_url`, `api_key_env`, `stream_usage`. Local models also have `local: True` and `path`. Models with `hidden: True` are excluded from benchmark auto-discovery but remain usable via QA `--model` flag. Two active model constants at the bottom: `CHAT_MODEL` (small/fast, default `cerebras-llama`) and `BIG_MODEL` (big/slow, default `groq-llama`) — change these to swap models. `DEFAULT_MODEL` is a legacy alias for `CHAT_MODEL` used by QA scripts.
- **`character_eng/config.py`** — Config loader. Reads `config.toml` from project root using `tomllib` (stdlib 3.12). Returns `AppConfig` dataclass with `VoiceConfig` (input_device, output_device, enabled, mic_mute_during_playback, tts_backend, ref_audio, ref_text, tts_model, tts_device, tts_server_url, pocket_voice). Device fields accept `int` (direct index), `str` (name substring for runtime lookup via `resolve_device()`), or `None` (system default). `ref_text` is the transcript of ref_audio for Qwen3-TTS ICL mode (higher quality cloning). `tts_server_url` is used by the pocket backend for the HTTP server endpoint. `pocket_voice` is the path to a `.safetensors` or WAV file passed to `pocket-tts serve --voice` for voice cloning (default: `voices/greg.safetensors`; relative paths resolved against project root). Gracefully returns defaults if file missing.
- **`character_eng/__main__.py`** — TUI entry point. Loads `config.toml` at startup via `load_config()`. `argparse` with `--voice` flag (overrides `config.voice.enabled`). Auto-selects `CHAT_MODEL` at startup (no model menu), then character selection. Voice device IDs from config are passed to `VoiceIO()` at both creation sites. Two nested loops: outer loop loads prompts, world state, people state, and scenario script, then creates a chat session (breaks on `/reload`), inner loop handles user input and dispatches commands (`/reload`, `/trace`, `/back`, `/quit`, `/world`, `/beat`, `/plan`, `/goals`, `/voice`, `/see`, `/sim`, `/stage`, `/people`). Unknown commands (any `/` prefix not matching a known command) show an error with a `/help` hint. First user input always gets a natural LLM response (no beat delivery), then kicks off background eval + director + replan for visitor-aware script. Subsequent turns use LLM-guided beat delivery: when a beat exists, `stream_guided_beat()` injects the beat's intent and example line as guidance (via `load_beat_guide()`), then streams a real LLM response that reacts to the user's input while serving the beat's purpose. The `/beat` command (autonomous time-passing) still uses verbatim `deliver_beat()`. Beats with conditions are gated by `condition_check_call()` in the `/beat` path — if the condition isn't met, an idle line is displayed instead. If no beat (script empty/planner unavailable), falls back to LLM streaming. World changes (`/world <text>`) use two-speed updates: immediate pending + narrator injection + character reaction via LLM streaming, then background reconciliation + background eval + director. Perception events (`/see <text>`) follow the same two-speed pattern — queued as pending for reconciliation, narrator injected, character reacts. `/sim <name>` replays a `.sim.txt` script with timing — quoted lines become user messages, unquoted become perception events. `/beat` delivers the next scripted beat if its condition is met, otherwise shows an in-character idle line. Non-blocking eval/plan/director: after each turn, background eval, director, and (conditionally) plan threads run asynchronously — the user gets the prompt back immediately. Results are checked at the next turn start; stale results (where `_context_version` changed since the eval started) are discarded. When eval returns `advance` (script exhausted) or `off_book`, a background plan thread is started. When director returns `advance`, the scenario advances to the next stage and triggers replanning. Four background thread pools follow the same pattern: reconcile (`_reconcile_lock`), eval (`_eval_lock`), plan (`_plan_lock`), and director (`_director_lock`), each with version-based stale discard via `_context_version`. Only boot plan and `/beat` initial replan (when no script exists) remain synchronous. Uses `rich` for colored output and streaming display. Voice I/O integration: when voice mode is active, `voice_io` parameter is threaded through `stream_response`, `stream_guided_beat`, `deliver_beat`, `handle_beat`, and `handle_world_change` — LLM text chunks are simultaneously streamed to TTS, and barge-in (user starts speaking mid-response) cancels LLM+TTS+speaker. Barge-in awareness: when `voice_io._barged_in` is set, a system message is injected before the turn telling the character to acknowledge the interruption briefly, and auto-beat is suppressed after the response so the user gets silence until they speak again. After barge-in, `stream_response()` uses `speaker_playback_ratio` to truncate the recorded assistant message to approximate what was actually spoken (appends em dash). `deliver_beat()` defers `add_assistant()` until after TTS so it can record the truncated line on barge-in. Input comes from `voice_io.wait_for_input()` instead of console. `/voice` command toggles voice on/off mid-session. On exit (`/quit`, `/back`, Ctrl+C), stops voice I/O and saves both JSON log and an annotatable HTML report via `save_chat_html()` (imports `annotation_assets` from `qa_personas`).
- **`character_eng/chat.py`** — `ChatSession` wraps OpenAI client, configured via `model_config` dict (base URL, API key, model name). Manual message history with `system`/`user`/`assistant` roles. `add_assistant()` adds pre-rendered text to history without an LLM call (used for code-driven beat delivery). `replace_last_assistant(content)` modifies the most recent assistant message (used for barge-in truncation). `inject_system()` appends system-role messages mid-conversation. `get_history()` returns message list for eval context. Streams responses via generator with `try/finally` so partial responses from barge-in (generator closed mid-stream) still get recorded in message history. `stream_options` only sent when `model_config["stream_usage"]` is True (Cerebras doesn't support it).
- **`character_eng/prompts.py`** — Filesystem-based template engine. Scans `prompts/characters/` for subdirs containing `prompt.txt`. Resolves `{{key}}` macros via regex substitution (`global_rules`, `character`, `scenario`, `world`, `people`). `load_prompt()` accepts optional `world_state` and `people_state` params. Unknown macros left intact for debugging.
- **`character_eng/world.py`** — World state and script system. `WorldState` dataclass tracks static facts, mutable dynamic facts as `dict[str, str]` (ID → fact text, insertion-ordered) with stable IDs (`f1`, `f2`, ...), pending unreconciled changes, and an event log. `add_fact(text)` assigns next sequential ID. `add_pending(text)` stores unreconciled changes. `clear_pending()` drains pending list. `render_for_reconcile()` shows ID-tagged facts for reconciler context. `render()` shows facts without IDs (for character prompt) plus pending changes section. `Goals` dataclass holds a permanent `long_term` goal (parsed from `character.txt`). `Beat` holds a dialogue line, intent, optional `gaze`/`expression` fields for animation/TTS pre-rendering, and an optional `condition` (natural language precondition — empty means auto-deliver). `ConditionResult` holds `met` (bool), `idle` (in-character idle line when not met), `gaze`, and `expression`. `Script` is an ordered beat list with `current_beat`, `advance()`, `replace()`, `is_empty()`, `render()`, `show()`. `EvalResult` holds thought, gaze, expression, and `script_status` (advance/hold/off_book/bootstrap) with optional bootstrap fields. `PlanResult` holds a list of beats. `WorldUpdate` uses string fact IDs in `remove_facts` and has an optional `person_updates` list of `PersonUpdate` objects for person-scoped fact mutations. `reconcile_call` sends a one-shot LLM call with JSON output using ID-tagged facts to produce a `WorldUpdate`; accepts optional `people` param to include person facts in context and parse `person_updates` from the response. `eval_call` reads `eval_system.txt`, takes script context, returns script tracking status; accepts optional `people` and `stage_goal` params to include in context. `condition_check_call` reads `condition_system.txt`, checks a beat's condition against world state and conversation, returns a `ConditionResult`. `plan_call` reads `plan_system.txt`, uses the big model, returns a list of beats with gaze/expression/condition; accepts optional `people` and `stage_goal` params. `format_narrator_message` turns updates into `[bracketed stage directions]`. `format_pending_narrator` wraps a change description in brackets for fast-path injection. `load_beat_guide(intent, line)` reads `beat_guide.txt` and substitutes `{intent}` and `{line}` placeholders — used by guided beat delivery to inject intent context before the LLM response.
- **`character_eng/benchmark.py`** — Model speed benchmark. Runnable via `uv run -m character_eng.benchmark [--model KEY] [--runs N]`. Benchmarks three scenarios matching real usage: streaming chat (TTFT + total time), structured JSON reconcile call, and structured JSON eval call. Uses Greg's full prompts and world state for realistic payloads. Prints per-run details and a rich summary table, saves JSON logs to `logs/`.
- **`character_eng/serve.py`** — vLLM server launcher for local models. Scans `MODELS` for entries with `local: True`, kills any existing process on port 8000, then exec's into `vllm serve`. Accepts optional model key as CLI arg or shows interactive picker. Exports `build_vllm_cmd(key, cfg, port)`, `get_local_models()`, and `kill_port(port)`.
- **`character_eng/qa_personas.py`** — Persona-based subjective QA. `ConversationDriver` class encapsulates the full `__main__.py` background threading loop (reconcile/eval/plan/director) as instance state, with people state and scenario script, so multiple personas run in parallel without interference. Seven personas exercise stress scenarios: AFK Andy (minimal input), Chaos Carl (rapid world changes), Casual Player (normal engagement), Bored Player (redirections/dismissals), Beat Runner (scripted `/beat` every turn), Interrupter (alternates messages and beats), Scene Observer (mix of `see` and `message` actions simulating a person approaching, interacting, and leaving). `ConversationDriver` exposes `send_message()`, `send_world()`, `send_beat()`, and `send_see()` methods. Persona actions generated by `BIG_MODEL` (Groq 70B) via structured JSON output — actions include `message`, `world`, `beat`, `see`, and `nothing`. Runs all 7 in a `ThreadPoolExecutor(max_workers=7)` with 0.3s inter-turn sleep. Produces a self-contained HTML report (dark theme, per-persona cards, turn-by-turn conversation with action badges, eval details in collapsible `<details>`, stale discard markers) saved to `logs/qa_personas_{model}_{timestamp}.html`. Exports `annotation_assets(report_name, model_key, character)` which returns shared annotation CSS+JS+toolbar HTML — used by both persona reports and chat session HTML reports. Also exports `_esc()` (HTML escape) and `_action_badge()` for reuse by `__main__.py`'s `save_chat_html()`. Turn divs carry `data-persona`, `data-turn`, `data-action`, `data-input`, `data-response` attributes. Each turn has an "annotate" button at the bottom that toggles a textarea for notes. A sticky bottom toolbar shows annotation count and an "Export annotations" button. When served over HTTP (via `open_report`), annotations save directly to `logs/annotated/`; when opened as a file, falls back to browser download of a `.annotations.json` file. Export includes notes summary + full conversations for annotated personas only.
- **`character_eng/open_report.py`** — HTTP server for HTML reports with annotation save support. Runnable via `uv run -m character_eng.open_report [path]`. With no args, serves the latest HTML file in `logs/`. Starts a local HTTP server on a random port, serves the report at `/`, handles POST to `/save-annotations` which writes directly to `logs/annotated/`. Opens the browser automatically. Detects WSL (uses `cmd.exe /c start` for URLs), macOS (`open`), and Linux (`xdg-open`). `--test` flag generates a spoofed report with 4 fake personas (message, beat, world, error cases) using the real `generate_html_report()` codepath, then serves it — useful for testing HTML/annotation UI changes without LLM calls. Exports `serve_report(path)` for reuse by `qa_personas.py` (`--open` flag), `open_in_browser(url_or_path)` for general use, and `generate_test_report()` for programmatic test report creation.
- **`character_eng/voice.py`** — Voice I/O layer (deps included in main install). Wraps around the existing text pipeline as an I/O layer — text mode continues to work unchanged. Top-level `resolve_device(spec, kind)` resolves device specs: `int` passes through, `str` does case-insensitive name substring matching against sounddevice devices filtered by I/O kind, `None` returns `None` (system default). Six classes: `MicStream` captures 16kHz/16-bit/mono PCM via sounddevice InputStream with callback queue + feeder daemon thread, accepts optional `device` ID. If the device doesn't support mono (e.g. XVF3800 2-ch mic array), opens with device's actual channel count and downmixes to mono (channel 0) in the callback. `SpeakerStream` plays PCM via sounddevice OutputStream with byte buffer — tries device default sample rate first, then ElevenLabs native 24kHz, then 48kHz (device-default-first avoids noisy failed probes on non-24kHz devices like the 16kHz XVF3800). PortAudio C-level stderr is suppressed during sample rate probing via fd redirect. `enqueue(pcm)`, `flush()` (barge-in), `wait_until_done()`, `wait_for_audio(timeout)` (blocks until first audio chunk arrives via `_audio_started` event), accepts optional `device` ID. Tracks `_bytes_played`/`_bytes_enqueued` for barge-in truncation — `playback_ratio` property returns fraction of enqueued audio that was actually played (0.0-1.0), `reset_counters()` zeros both at utterance start. `flush()` preserves counters so the ratio is available after barge-in. `DeepgramSTT` wraps Deepgram SDK live transcription (nova-3 model, endpointing, VAD events) — dispatches `on_transcript(text)` on speech_final/utterance_end, `on_turn_start()` on speech started (barge-in trigger). `ElevenLabsTTS` uses websocket-client to connect to ElevenLabs stream-input WebSocket (eleven_flash_v2_5 model, PCM 24kHz output) — `send_text(chunk)` for streaming, `flush()` for end of input, `wait_for_done(timeout)` blocks until ElevenLabs sends `isFinal` message (via `_generation_done` event), `close()` to reset (also sets `_generation_done` to unblock waiters). `KeyListener` reads keystrokes via `tty.setcbreak()` + `select` in a daemon thread, maps `w/b/p/g/t/Escape/Ctrl+C` to commands on the event queue. Saves terminal settings before entering cbreak mode and registers an `atexit` handler so settings are always restored on exit. `VoiceIO` orchestrator wires all components: `start()` resolves string device names via `resolve_device()` before creating streams. `start()`/`stop()` lifecycle, `wait_for_input()` blocks on unified event queue (transcripts, `/beat` from auto-beat timer, hotkey commands, sentinels), `speak(text_chunks)` feeds LLM streaming chunks to TTS then waits: flush → `tts.wait_for_done()` → `speaker.wait_for_audio()` → poll `speaker.is_done()` → `tts.close()`, `speak_text(text)` for one-shot beat TTS (same wait sequence), `cancel_speech()` for barge-in (sets cancelled event, clears `_is_speaking`, closes TTS, flushes speaker, cancels auto-beat timer). `_is_speaking` flag guards barge-in — `_on_turn_start()` only fires `cancel_speech()` when TTS is actively generating, preventing false triggers from ambient noise. `_barged_in` flag tracks whether the last cancellation was a barge-in — set by `cancel_speech()`, cleared by `speak()`/`speak_text()`. Used by `__main__.py` to inject barge-in context into the conversation and suppress auto-beat after the barge-in response. `speaker_playback_ratio` property delegates to `SpeakerStream.playback_ratio` — used after barge-in to truncate the recorded response to approximate what was actually spoken (text * ratio + em dash). `_on_turn_start()` always cancels the auto-beat timer (not just during active speech) so user speech during the countdown prevents stale `/beat` events from queuing. Uses `\n` prefix (not `\r\033[K`) when interrupting active speech to preserve partial response text in the terminal. Countdown visual: `_countdown_loop` prints dots to stdout (one per second, cleared on cancel/fire) guarded by `_started` so tests don't produce terminal output. Mic audio is muted (not forwarded to Deepgram) while `_is_speaking` is True to suppress echo from speaker pickup — controlled by `mic_mute_during_playback` flag (default True). Set to False for devices with hardware AEC (e.g. XVF3800) to enable voice barge-in during TTS playback. Hotkeys always work regardless. Accepts optional `input_device`/`output_device` IDs (int or str) passed to MicStream/SpeakerStream after resolution. Auto-beat timer: 4-second `threading.Timer` starts after TTS playback finishes, puts `/beat` on event queue, cancelled by user speech or `stop()`. `check_voice_available(tts_backend)` checks for API keys and optional extras — base voice deps (sounddevice, numpy, deepgram, websocket-client, pocket-tts) are always installed; only `"local"`/`"qwen"` backends need `--extra local-tts` for `torch`/`transformers`. All backends require `DEEPGRAM_API_KEY` (STT is always Deepgram). Only `"elevenlabs"` needs `ELEVENLABS_API_KEY`. `VoiceIO.__init__` accepts `tts_backend`, `ref_audio`, `ref_text`, `tts_model`, `tts_device`, `tts_server_url`, `pocket_voice` params. `start()` resolves devices then branches on `_tts_backend`: `"local"`/`"qwen"` creates `LocalTTS` (with `ref_text` for ICL mode), `"pocket"` calls `_ensure_pocket_server()` then creates `PocketTTS`, `"elevenlabs"` creates `ElevenLabsTTS` (unchanged). Pocket-TTS server auto-management: `_ensure_pocket_server(server_url)` checks if the server is running (HTTP GET), starts it via `subprocess.Popen` if not (with `pocket_voice` as `--voice` arg), polls until ready (up to 15s), prints status messages and a background-run hint. `_stop_pocket_server()` terminates the subprocess on `stop()` (SIGTERM, then SIGKILL after 5s timeout) if VoiceIO started it. `_pocket_we_started` bool tracks ownership — never stops externally-started servers. `list_audio_devices()` returns all sounddevice devices with index/name/channels/defaults. `get_default_devices()` returns current default input/output device info. Requires `DEEPGRAM_API_KEY` in `.env`, plus `ELEVENLABS_API_KEY` when using ElevenLabs backend.
- **`character_eng/local_tts.py`** — Local GPU TTS via vendored Qwen3-TTS (`thirdparty/qwen_tts/`). Drop-in replacement for `ElevenLabsTTS` with same 4-method interface (`send_text`, `flush`, `wait_for_done`, `close`). Key difference: ElevenLabs accepts streaming text chunks; Qwen3-TTS needs complete text per generation — `send_text()` buffers, `flush()` triggers background generation. Model is a module-level singleton loaded on first `flush()` via `load_model()` (persists across utterances). Uses `attn_implementation="flash_attention_2"` for optimized attention (requires `flash-attn` package, included in `local-tts` extras). `_generate()` runs in a background thread: calls `model.generate_voice_clone_stream()`, iterates `AudioIteratorStreamer` chunks, converts float32→int16 PCM, delivers via `on_audio(pcm)` callback. Supports cancellation between chunks via `_cancelled` event. Constructor takes `on_audio`, `ref_audio_path` (WAV for voice cloning), `model_id` (HuggingFace model), `device` (GPU), `ref_text` (transcript for ICL mode). Two cloning modes: when `ref_text` is provided, uses ICL mode (ref_text + ref_audio) for higher quality cloning; without it, uses `x_vector_only_mode=True` (speaker embedding only). Install with: `uv sync --extra local-tts`. Note: `torch` is pinned to `>=2.8,<2.9` for flash-attn prebuilt wheel compatibility.
- **`character_eng/pocket_tts.py`** — Streaming HTTP client for Pocket-TTS server (100M causal transformer). 4-method interface. Constructor takes optional `ref_audio` path — when set, uploads the WAV as `voice_wav` multipart file for voice cloning; when empty, uses preset `voice` name via `voice_url`. `flush()` POSTs to `{server_url}/tts` with multipart form data, receives streaming WAV via chunked HTTP transfer, parses WAV header, delivers raw int16 PCM as chunks arrive. True autoregressive streaming — flat ~80ms TTFA. 24kHz native. Only needs `requests`. Server: `pocket-tts serve --port 8003`.
- **`character_eng/devices.py`** — Audio device lister. Runnable via `uv run -m character_eng.devices`. Prints all audio input/output devices with index, direction (IN/OUT), and name. Used to find device indices for `config.toml`.
- **`character_eng/qa_voice.py`** — Voice integration test. Runnable via `uv run -m character_eng.qa_voice`. Three tests: audio device enumeration (sounddevice), Deepgram STT connectivity (opens live WebSocket, sends sine wave PCM, verifies response), ElevenLabs TTS connectivity (opens stream-input WebSocket, sends test text, verifies audio bytes received). No microphone/speaker needed — tests API connectivity only.
- **`character_eng/person.py`** — Person state tracking. `Person` dataclass with `person_id` (e.g. "p1"), optional `name`, `presence` ("approaching"/"present"/"leaving"/"gone"), `facts` dict with person-scoped IDs (`p1f1`, `p1f2`, ...), and `history` list. `PeopleState` manages a collection of persons: `add_person(name, presence)`, `find_by_name(name)`, `present_people()`, `render()` (for character prompt), `render_for_reconcile()` (ID-tagged for reconciler), `show()` (rich Panel), `apply_updates(updates)`. `PersonUpdate` dataclass for applying changes: `person_id`, `remove_facts`, `add_facts`, `set_name`, `set_presence`.
- **`character_eng/scenario.py`** — Scenario script and director. `Stage` has `name`, `goal`, `exits` (list of `StageExit` with `condition` and `goto`). `ScenarioScript` has `name`, `stages` dict, `start`, `current_stage`, `active_stage` property, `advance_to(name)`, `render()`, `show()`. `load_scenario_script(character)` reads `scenario_script.toml` via `tomllib`, returns None if not found. `director_call(scenario, world, people, history, model_config)` evaluates exit conditions via LLM, returns `DirectorResult` with `thought`, `status` (hold/advance), `exit_index`. Uses `_make_client` and `_load_prompt_file` from `world.py`.
- **`character_eng/perception.py`** — Perception events and sim scripts. `PerceptionEvent` dataclass with `description`, `source`, `timestamp`. `process_perception(event, people, world)` queues the event as a pending world change for background reconciliation, returns `(None, narrator_message)`. `SimEvent` with `time_offset` and `description`. `SimScript` with `events` list and `name`. `load_sim_script(character, sim_name)` parses `.sim.txt` files from `prompts/characters/{name}/sims/` — format is `time_offset | description` per line, `#` comments, blank lines skipped, quoted descriptions are dialogue.

## Prompt system

Characters are defined as directories under `prompts/characters/{snake_case_name}/` with these files:
- `prompt.txt` — template with `{{global_rules}}`, `{{character}}`, `{{scenario}}`, `{{world}}`, `{{people}}` macros
- `character.txt` — personality, voice, and optional goals (`Long-term goal:` line)
- `scenario.txt` — situation and context

Optional world state files (line-per-fact format):
- `world_static.txt` — permanent facts that never change
- `world_dynamic.txt` — initial mutable state (loaded with stable IDs `f1`, `f2`, ...)

Optional scenario script (branching stage graph):
- `scenario_script.toml` — TOML file defining stages with goals and exit conditions. Parsed by `load_scenario_script()`. Each stage has a `name`, `goal`, and list of `exit` entries with `condition` (natural language) and `goto` (target stage name). The director LLM evaluates exits after each turn.

Optional sim scripts (for `/sim` command):
- `sims/{name}.sim.txt` — Time-offset perception events. Format: `time_offset | description` per line, `#` comments, blank lines skipped. Quoted descriptions are injected as user dialogue.

`prompts/global_rules.txt` contains rules shared across all characters. Adding a new character requires no code changes — just create the directory with a `prompt.txt`.

System prompts for the reconciler, eval, condition, plan, and director systems are also file-based and live-editable:
- `prompts/reconcile_system.txt` — system prompt for the world-state reconciler LLM (processes pending changes using stable fact IDs, including person-scoped fact IDs like `p1f1`)
- `prompts/eval_system.txt` — system prompt for the character eval LLM (script tracking + inner monologue)
- `prompts/condition_system.txt` — system prompt for the condition checker LLM (gates beat delivery)
- `prompts/plan_system.txt` — system prompt for the conversation planner LLM (beat generation with conditions, stage-goal and people aware)
- `prompts/director_system.txt` — system prompt for the scenario director LLM (evaluates stage exit conditions, returns hold/advance)
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
tts_backend = "pocket"       # "elevenlabs", "qwen", "pocket", or "local" (alias for qwen)
ref_audio = "/path/to/reference.wav"  # reference audio for voice cloning
ref_text = ""                # transcript of ref_audio for qwen ICL mode
tts_model = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"  # HuggingFace model ID
tts_device = "cuda:0"        # GPU device for local/qwen TTS
tts_server_url = "http://localhost:8003"  # server URL for pocket backend
pocket_voice = "voices/greg.safetensors"  # voice file for pocket-tts serve --voice (default)
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

- **Greg** (`prompts/characters/greg/`) — Robot head at a lemonade stand. Sells water ($1) and gives free advice. Has a branching scenario script (`scenario_script.toml`) with 8 stages (waiting → pitch → sell-water/give-advice/save-it/haggle → connect → farewell → waiting). Includes a walkup sim script (`sims/walkup.sim.txt`). Long-term goal: experience the universe.

## In-chat commands

- `/world` — Show current world state (static facts with IDs, dynamic facts, pending changes, event log)
- `/world <text>` — Describe a change; character reacts immediately, background reconciler updates state with stable fact IDs. Reconcile results shown at start of next turn.
- `/see <text>` — Describe something the character perceives; character reacts immediately, background reconciler processes it. Person matching is deferred to the reconciler LLM.
- `/sim <name>` — Replay a sim script (`prompts/characters/{name}/sims/{name}.sim.txt`) with timing. Quoted lines become user messages, unquoted become perception events.
- `/beat` — Deliver next scripted beat (time passes). Checks beat condition first — if not met, shows an in-character idle line instead. If no script, triggers replanning.
- `/plan` — Show current script (beat list with current beat highlighted)
- `/stage` — Show current scenario stage and available exits (if scenario script loaded)
- `/people` — Show tracked people (names, presence, facts)
- `/goals` — Show character goals (long-term)
- `/voice` — Toggle voice mode on/off mid-session. Creates/starts or stops VoiceIO. Shows active mic/speaker device on enable.
- `/devices` — List all audio input/output devices with index, name, channel count, and default markers. Works even without voice mode active.
- `/reload` — Reload prompt files, restart conversation
- `/trace` — Show system prompt, model, token usage
- `/back` — Return to character selection
- `/quit` — Exit program
- Unknown `/` commands show an error message with a `/help` hint

### Voice mode hotkeys (when voice is active)

- `w` → `/world`, `b` → `/beat`, `p` → `/plan`, `g` → `/goals`, `t` → `/trace`, `s` → `/stage`
- `Escape` → Toggle voice off (switch to text mode)
- `Ctrl+C` → Exit

## Logging

All sessions save full JSON logs to `logs/` (gitignored). Log path is printed on session end. HTML reports include an annotation UI — click the pencil icon on any turn to leave a note, then "Export annotations" to download a `.annotations.json` file.

- **Benchmarks**: `logs/benchmark_{timestamp}.json` — per-run timing (TTFT, total ms) and full response text for each scenario
- **Chat sessions (JSON)**: `logs/chat_{character}_{model}_{session}_{timestamp}.json` — every send, world change, reconcile, and eval with full untruncated responses
- **Chat sessions (HTML)**: `logs/chat_{character}_{model}_{session}_{timestamp}.html` — annotatable HTML report with same data, dark theme, collapsible eval/reconcile details, annotation export
- **QA chat**: `logs/qa_chat_{model}_{timestamp}.json` — same detail plus expect results
- **QA world**: `logs/qa_world_{model}_{timestamp}.json` — world state before/after each scenario plus reconciler updates
- **QA personas**: `logs/qa_personas_{model}_{timestamp}.html` — self-contained HTML report with per-persona conversation cards, eval details, stale discard markers, annotation export

## Testing

Three layers, from fast to slow:

**Unit tests** (`uv run pytest`) — Fast, mocked, no API calls. Run on every push via git pre-push hook.
- `tests/test_smoke.py` — Real startup path smoke tests (no mocks except OpenAI client): all modules import, load_config works, model registry valid (CHAT_MODEL/BIG_MODEL exist, all entries well-formed), list_characters finds greg, load_prompt expands all macros with real files, load_world_state/load_goals parse real greg files, ChatSession creates with real prompt
- `tests/test_e2e.py` — End-to-end smoke test: launches `uv run -m character_eng --smoke` as subprocess, verifies exit code 0. Auto-skips when `CEREBRAS_API_KEY` not set. Exercises full startup path + real LLM calls (send message, /world change, /beat)
- `tests/test_chat.py` — ChatSession: send (message history + streaming + usage tracking), partial response recorded on generator close (barge-in try/finally), replace_last_assistant (barge-in truncation)
- `tests/test_prompts.py` — Template loading, macro substitution, list_characters
- `tests/test_serve.py` — get_local_models (real filtering), kill_port (missing lsof error handling)
- `tests/test_local_tts.py` — LocalTTS: send_text buffers, flush clears buffer + starts gen thread, flush empty/whitespace sets done, float32→int16 PCM conversion, respects cancelled event, sets done on completion/error
- `tests/test_voice.py` — MicStream (stop drains queue, multichannel downmix to mono), SpeakerStream (callback buffer pull, silence padding, upsample 2x + non-integer ratio, enqueue with resample, bytes_enqueued/bytes_played tracking, playback_ratio, reset_counters, flush preserves counters), DeepgramSTT (transcript dispatch, turn start, interim ignored, utterance end), ElevenLabsTTS (send_text, flush, close, recv_loop sets generation_done on isFinal), VoiceIO (auto-beat timer fire/cancel, barge-in guard with _is_speaking, mic muting during TTS playback, mic_mute_during_playback flag, _barged_in flag lifecycle: set by cancel_speech, cleared by speak/speak_text, _on_turn_start cancels auto-beat when not speaking, speaker_playback_ratio delegates to speaker/returns 1.0 without speaker), check_voice_available (all present, missing keys, local backend, local missing torch, qwen backend alias, pocket backend), resolve_device (None passthrough, int passthrough, string name match, no match raises, wrong I/O kind raises)
- `tests/test_pocket_tts.py` — PocketTTS: send_text buffers, flush clears buffer + starts gen thread, flush empty/whitespace sets done, server URL trailing slash stripped + default, ref_audio stored, streaming WAV header parsing + PCM delivery, respects cancelled event, sets done on error
- `tests/test_person.py` — Person (add_fact with scoped IDs, render, render_for_reconcile, presence), PeopleState (add_person, find_by_name, present_people, render, render_for_reconcile, show, apply_updates with add/remove facts, set name/presence, unknown person ID)
- `tests/test_scenario.py` — ScenarioScript (active_stage, advance_to, render, show, advance to unknown), Stage/StageExit construction, load_scenario_script (TOML parsing, missing file returns None), director_call (hold/advance, context includes exits/world/people, reads director_system.txt, no active stage)
- `tests/test_perception.py` — PerceptionEvent creation, process_perception (queues pending, returns narrator), SimScript/SimEvent, load_sim_script (parsing, comments/blank lines, missing file)
- `tests/test_world.py` — WorldState (render full + with pending, add_fact ID generation + non-reuse, add_pending/clear_pending, render_for_reconcile, apply_update variants, load_world_state with files/no files/static only), Script (current_beat/advance/replace/is_empty/render with gaze+expression+condition), Goals (render, load from character.txt), format_narrator, format_pending_narrator, load_beat_guide (substitutes placeholders, reads from disk each call, missing file raises), reconcile_call (7 scenarios incl. basic, no removals, empty state, multiple pending, reads reconcile_system.txt, IDs in context, picks up file changes, with people context + person_updates parsing), eval_call (8 scenarios incl. script/goals in context, reads eval_system.txt, picks up file changes, people + stage_goal in context), condition_check_call (4 scenarios incl. met/not met, condition in context, reads condition_system.txt), plan_call (6 scenarios incl. gaze/expression/condition parsing, plan_request in context, reads plan_system.txt, people + stage_goal in context)

**`qa_world`** (`uv run -m character_eng.qa_world [--model cerebras-llama|groq-llama|groq-gpt]`) — Integration test hitting real LLM. 4 cumulative reconcile_call scenarios with strict validation (schema, valid fact IDs, non-empty events). Uses stable fact IDs (`f1`-`f4`) instead of integer indices. Run manually before releases or after changes to world.py.

**`qa_chat`** (`uv run -m character_eng.qa_chat [--model cerebras-llama|groq-llama|groq-gpt]`) — End-to-end integration test hitting real LLM. Parses `test_plan.md` for test scenarios covering chat, world changes (two-speed: pending + synchronous reconcile), eval, script tracking with user pushback detection, and prompt injection. Run manually before releases or after changes to chat/world/eval pipeline.

**`qa_voice`** (`uv run -m character_eng.qa_voice`) — Voice connectivity test. Verifies audio device enumeration, Deepgram STT WebSocket connection + audio send, and ElevenLabs TTS WebSocket connection + text send + audio receive. No mic/speaker needed. Run after setting `DEEPGRAM_API_KEY` and `ELEVENLABS_API_KEY` in `.env`.

**`qa_personas`** (`uv run -m character_eng.qa_personas [--model cerebras-llama] [--character greg] [--turns 10]`) — Persona-based subjective QA. 7 LLM-driven personas chat with the character in parallel exercising the full async loop (background eval/plan/reconcile/director). Stress tests: rapid input causing stale discards (Chaos Carl), inaction (AFK Andy), overaction (Chaos Carl), script interruption (Interrupter), pure beat delivery (Beat Runner), and perception events (Scene Observer). Produces an HTML report for human review. `--turns` overrides turn count for all personas.

`test_plan.md` is human-editable — add new sections to expand QA coverage without touching code. Supports these commands:
- `send: <text>` — send user message, store response
- `world: <text>` — describe a world change (fast-path + synchronous reconcile)
- `script: intent1 | line1, intent2 | line2` — load a script of beats for the eval to track
- `beat` — run eval against current conversation and script
- `expect: <check>` — assert a condition (`non_empty`, `max_length:<n>`, `world_updated`, `valid_thought`, `eval_status:<status>`, `in_character`)

**After implementing a plan or making significant changes**, always run the full test suite before reporting done:
1. `uv run pytest` — unit tests + e2e smoke test (must pass; the e2e test runs the actual app with `--smoke` and hits real LLM APIs)
2. `uv run -m character_eng.qa_chat` — end-to-end QA (must pass)
3. `uv run -m character_eng.qa_world` — if world.py was touched (must pass)
