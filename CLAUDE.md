# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Documentation Rule

**Always update `README.md` and `CLAUDE.md` when making changes** that affect commands, architecture, modules, features, models, testing, or configuration.

## Commands

```bash
uv run -m character_eng              # Run the app
uv run -m character_eng --vision     # Run with vision (auto-starts vision service)
uv run -m character_eng --vision-mock walkup.json  # Vision with mock replay (no camera)
uv run -m character_eng --character mara           # Auto-select character
uv run -m character_eng --character mara --sim curious  # Run sim non-interactively
uv run pytest                        # Unit tests (+ e2e smoke when API keys set)
uv run -m character_eng --smoke      # E2e smoke test (real LLM calls)
uv run -m character_eng.qa_world     # World reconciler integration test
uv run -m character_eng.qa_chat      # Chat QA integration test
uv run -m character_eng.qa_personas  # 7 parallel persona stress test (HTML report)
uv run -m character_eng.qa_voice     # Voice connectivity test
uv run -m character_eng.benchmark    # Model speed benchmark
uv run -m character_eng.serve        # Start local vLLM server
uv run -m character_eng.devices      # List audio devices
uv run -m character_eng.open_report  # Open latest HTML report
uv sync                              # Install deps
uv sync --extra local-tts            # Install with local GPU TTS
uv add <package>                     # Add a dependency
lsof -ti:8000 | xargs kill -9       # Kill local vLLM server

# Vision service (separate venv, separate process)
cd services/vision && uv sync        # Install vision deps (first time)
uv run --project services/vision python services/vision/app.py  # Manual start
```

QA scripts accept `--model cerebras-llama|groq-llama|groq-gpt|gemini-3-flash|gemini-2.5-flash`. Persona QA accepts `--character greg --turns 10 --open`.

## Architecture

Interactive NPC character chat CLI with single-tier 8B LLM backend (OpenAI-compatible endpoints):
- **CHAT_MODEL** (small/fast, default `groq-llama-8b` 8B) — all LLM calls: streaming dialogue, expression, eval, director, planning, reconciliation
- **BIG_MODEL** (kept for API compat, no longer required) — `get_big_model_config()` still exists but nothing depends on 70B

Configured in `models.py`. No model selection menu — auto-selects on startup.

### Core systems
- **Two-speed world updates**: fast path stores pending changes + character reacts immediately; slow path (background reconciler, 8B) reconciles into structured state with stable fact IDs (`f1`, `f2`, ...)
- **Parallel post-response microservices**: `script_check_call` + `thought_call` + `director_call` run concurrently via `ThreadPoolExecutor` after each response (~350ms total vs ~1050ms sequential). Expression also parallelized for `/beat` path.
- **Single-beat planning**: `single_beat_call` (8B) generates one beat at a time synchronously — no background plan thread, no stale discard. Called at boot, after script completion, and on off-book events.
- **Person state**: individual people tracked with scoped fact IDs (`p1f1`, `p1f2`, ...); reconciler handles assignments
- **Perception**: `/see` for manual events, `/sim` for scripted replay from `.sim.txt` files
- **Expression post-processing**: After each character response, `expression_call` (8B) derives gaze target + facial expression from the dialogue. Gaze picks from scene targets (static list, or live vision targets when `--vision` is active); expression is one of 12 emotions. Social gaze modifiers handled downstream in robot control, not by the LLM.
- **Vision mode** (`--vision`): Live visual intelligence via standalone vision service (`services/vision/`). Camera capture, face tracking (InsightFace), person tracking (SAM3 + torchreid ReID), VLM questioning (LFM2-VL-3B via vLLM). Vision service has its own venv/deps — zero conflict with main app. character-eng polls via HTTP, runs synthesis LLM call (8B) to distill visual data into perception events, and feeds dynamic gaze targets to expression system.
- **Voice mode**: Deepgram STT + configurable TTS (ElevenLabs, Qwen3-TTS local, Pocket-TTS). WebRTC AEC3 (via LiveKit) keeps mic live during playback for barge-in. All TTS backends share 4-method interface (`send_text`, `flush`, `wait_for_done`, `close`)

### Modules (22 total)
| Module | Purpose |
|--------|---------|
| `__main__.py` | TUI entry point, command dispatch, parallel microservice orchestration |
| `chat.py` | ChatSession — OpenAI client wrapper, message history, streaming |
| `models.py` | Model config registry, CHAT_MODEL/BIG_MODEL constants |
| `config.py` | `config.toml` loader → AppConfig/VoiceConfig/VisionConfig dataclasses |
| `prompts.py` | Filesystem template engine, `{{macro}}` substitution (incl. `{{vision}}`) |
| `world.py` | WorldState, Script, Goals, Beat, reconcile/eval/plan/single_beat/condition/expression/script_check/thought LLM calls |
| `person.py` | Person/PeopleState tracking with scoped fact IDs |
| `scenario.py` | ScenarioScript (TOML stage graph), director_call |
| `perception.py` | PerceptionEvent, SimScript, `/see` and `/sim` support |
| `aec.py` | LiveKitAEC (WebRTC AEC3) + resample_to_16k helper |
| `voice.py` | VoiceIO orchestrator, MicStream, SpeakerStream, DeepgramSTT, ElevenLabsTTS, KeyListener |
| `local_tts.py` | Qwen3-TTS local GPU TTS (drop-in ElevenLabs replacement) |
| `pocket_tts.py` | Pocket-TTS streaming HTTP client |
| `serve.py` | vLLM server launcher for local models |
| `benchmark.py` | Model speed benchmark (TTFT, structured JSON) |
| `qa_personas.py` | 7-persona parallel stress test with HTML report |
| `open_report.py` | HTTP server for HTML reports with annotation save |
| `vision/__init__.py` | Vision module package |
| `vision/context.py` | Visual dataclasses (FaceObservation, PersonObservation, etc.) + VisualContext |
| `vision/client.py` | HTTP client for vision service (stdlib urllib, zero deps) |
| `vision/synthesis.py` | `vision_synthesis_call` — distills raw visual data into perception events (8B) |
| `vision/focus.py` | `visual_focus_call` — generates constant/ephemeral VLM questions and SAM targets (8B) |
| `vision/manager.py` | VisionManager — orchestrates polling, synthesis, focus, people resolution |

### Post-response microservices (parallel, 8B)
After each character response, `run_post_response` fires three 8B calls concurrently via `ThreadPoolExecutor`: `script_check_call` + `thought_call` (eval), `director_call` (stage transitions). For `/beat`, `expression_call` is also included in the parallel batch. ~350ms total instead of ~1050ms sequential. Results are immediate — no stale discard needed.

### Background threading
Only reconcile uses a background thread. Planning is fully synchronous via `single_beat_call` (8B, ~350ms). When `--vision` is active, VisionManager runs a continuous background thread (poll + synthesis, min 750ms between cycles).

### Vision service (`services/vision/`)
Standalone Flask service with its own `pyproject.toml` and `.venv`. Runs camera capture, InsightFace face tracking, SAM3 + torchreid person tracking, and VLM questioning. Exposes `/snapshot`, `/set_questions`, `/set_sam_targets`, `/health` endpoints. Auto-launched by `--vision` flag. Debug UI at `:7860`. ~10.5GB VRAM (SAM3 3.4GB + InsightFace 0.8GB + ReID 0.25GB + LFM2-VL-3B 6GB).

## Prompt system

Characters live in `prompts/characters/{name}/` with: `prompt.txt` (template), `character.txt`, `scenario.txt`, optional `world_static.txt`, `world_dynamic.txt`, `scenario_script.toml`, `sims/*.sim.txt`.

System prompts (`prompts/*.txt`) for reconciler, eval, condition, plan, director, expression, beat_guide, vision_synthesis, and visual_focus are read from disk on every call — edits take effect immediately.

## Configuration

User settings in `config.toml` (gitignored). See `config.example.toml`. App works without it.

## Characters

- **Greg** (`prompts/characters/greg/`) — Robot head at a lemonade stand. Scenario script with 6 stages. Walkup sim script.
- **Mara** (`prompts/characters/mara/`) — Fortune teller at a carnival booth. Scenario script with 5 stages. Curious visitor sim script.

## In-chat commands

`/info`, `/world <text>`, `/beat`, `/see <text>`, `/sim <name>`, `1`-`4` (stage triggers), `/vision`, `/voice`, `/devices`, `/reload`, `/trace`, `/back`, `/quit`. Stage HUD shows `[stage] 1: label 2: label` before each prompt.

Voice hotkeys: `i`=info, `b`=beat, `t`=trace, `1-4`=triggers, `q`=quit, `Escape`=toggle voice, `Ctrl+C`=exit.

## Testing

Three layers, fast to slow:

**Unit tests** (`uv run pytest`) — Mocked, no API calls. Pre-push hook. Test files: `test_smoke`, `test_e2e`, `test_chat`, `test_prompts`, `test_serve`, `test_local_tts`, `test_voice`, `test_pocket_tts`, `test_person`, `test_scenario`, `test_perception`, `test_world`, `test_vision`.

**Integration QA** — Hit real LLMs. `qa_world` (4 reconcile scenarios), `qa_chat` (parses `test_plan.md`), `qa_voice` (API connectivity), `qa_personas` (7 parallel personas, HTML report).

`test_plan.md` commands: `send:`, `world:`, `script:`, `beat`, `expect:` (`non_empty`, `max_length:<n>`, `world_updated`, `valid_thought`, `eval_status:<status>`, `in_character`).

**After significant changes**, run before reporting done:
1. `uv run pytest` (must pass)
2. `uv run -m character_eng.qa_chat` (must pass)
3. `uv run -m character_eng.qa_world` — if world.py was touched

## Logging

All sessions save to `logs/` (gitignored). Formats: JSON (chat, QA), HTML (chat, personas — with annotation UI).
