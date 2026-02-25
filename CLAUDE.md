# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Documentation Rule

**Always update `README.md` and `CLAUDE.md` when making changes** that affect commands, architecture, modules, features, models, testing, or configuration.

## Commands

```bash
uv run -m character_eng              # Run the app
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
```

QA scripts accept `--model cerebras-llama|groq-llama|groq-gpt|gemini-3-flash|gemini-2.5-flash`. Persona QA accepts `--character greg --turns 10 --open`.

## Architecture

Interactive NPC character chat CLI with two-tier LLM backend (OpenAI-compatible endpoints):
- **CHAT_MODEL** (small/fast, default `cerebras-llama` 8B) — streaming dialogue
- **BIG_MODEL** (big/slow, default `groq-llama` 70B) — eval, planning, reconciliation, direction. Falls back to chat model.

Both configured in `models.py`. No model selection menu — auto-selects on startup.

### Core systems
- **Two-speed world updates**: fast path stores pending changes + character reacts immediately; slow path (background reconciler) reconciles into structured state with stable fact IDs (`f1`, `f2`, ...)
- **Two-speed planning**: fast eval (every turn) tracks script progress; slow planner generates multi-beat scripts
- **Scenario director**: TOML-based branching stage graph; director LLM evaluates exit conditions after each turn
- **Person state**: individual people tracked with scoped fact IDs (`p1f1`, `p1f2`, ...); reconciler handles assignments
- **Perception**: `/see` for manual events, `/sim` for scripted replay from `.sim.txt` files
- **Voice mode**: Deepgram STT + configurable TTS (ElevenLabs, Qwen3-TTS local, Pocket-TTS). All TTS backends share 4-method interface (`send_text`, `flush`, `wait_for_done`, `close`)

### Modules (16 total)
| Module | Purpose |
|--------|---------|
| `__main__.py` | TUI entry point, command dispatch, background thread orchestration |
| `chat.py` | ChatSession — OpenAI client wrapper, message history, streaming |
| `models.py` | Model config registry, CHAT_MODEL/BIG_MODEL constants |
| `config.py` | `config.toml` loader → AppConfig/VoiceConfig dataclasses |
| `prompts.py` | Filesystem template engine, `{{macro}}` substitution |
| `world.py` | WorldState, Script, Goals, Beat, reconcile/eval/plan/condition LLM calls |
| `person.py` | Person/PeopleState tracking with scoped fact IDs |
| `scenario.py` | ScenarioScript (TOML stage graph), director_call |
| `perception.py` | PerceptionEvent, SimScript, `/see` and `/sim` support |
| `voice.py` | VoiceIO orchestrator, MicStream, SpeakerStream, DeepgramSTT, ElevenLabsTTS, KeyListener |
| `local_tts.py` | Qwen3-TTS local GPU TTS (drop-in ElevenLabs replacement) |
| `pocket_tts.py` | Pocket-TTS streaming HTTP client |
| `serve.py` | vLLM server launcher for local models |
| `benchmark.py` | Model speed benchmark (TTFT, structured JSON) |
| `qa_personas.py` | 7-persona parallel stress test with HTML report |
| `open_report.py` | HTTP server for HTML reports with annotation save |

### Background threading
Four async thread pools (reconcile, eval, plan, director) — each with a lock and version-based stale discard via `_context_version`. Only boot plan and `/beat` initial replan are synchronous.

## Prompt system

Characters live in `prompts/characters/{name}/` with: `prompt.txt` (template), `character.txt`, `scenario.txt`, optional `world_static.txt`, `world_dynamic.txt`, `scenario_script.toml`, `sims/*.sim.txt`.

System prompts (`prompts/*.txt`) for reconciler, eval, condition, plan, director, and beat_guide are read from disk on every call — edits take effect immediately.

## Configuration

User settings in `config.toml` (gitignored). See `config.example.toml`. App works without it.

## Characters

- **Greg** (`prompts/characters/greg/`) — Robot head at a lemonade stand. Scenario script with 6 stages. Walkup sim script.

## In-chat commands

`/info`, `/world <text>`, `/beat`, `/see <text>`, `/sim <name>`, `1`-`4` (stage triggers), `/voice`, `/devices`, `/reload`, `/trace`, `/back`, `/quit`. Stage HUD shows `[stage] 1: label 2: label` before each prompt.

Voice hotkeys: `i`=info, `b`=beat, `t`=trace, `1-4`=triggers, `q`=quit, `Escape`=toggle voice, `Ctrl+C`=exit.

## Testing

Three layers, fast to slow:

**Unit tests** (`uv run pytest`) — Mocked, no API calls. Pre-push hook. Test files: `test_smoke`, `test_e2e`, `test_chat`, `test_prompts`, `test_serve`, `test_local_tts`, `test_voice`, `test_pocket_tts`, `test_person`, `test_scenario`, `test_perception`, `test_world`.

**Integration QA** — Hit real LLMs. `qa_world` (4 reconcile scenarios), `qa_chat` (parses `test_plan.md`), `qa_voice` (API connectivity), `qa_personas` (7 parallel personas, HTML report).

`test_plan.md` commands: `send:`, `world:`, `script:`, `beat`, `expect:` (`non_empty`, `max_length:<n>`, `world_updated`, `valid_thought`, `eval_status:<status>`, `in_character`).

**After significant changes**, run before reporting done:
1. `uv run pytest` (must pass)
2. `uv run -m character_eng.qa_chat` (must pass)
3. `uv run -m character_eng.qa_world` — if world.py was touched

## Logging

All sessions save to `logs/` (gitignored). Formats: JSON (chat, QA), HTML (chat, personas — with annotation UI).
