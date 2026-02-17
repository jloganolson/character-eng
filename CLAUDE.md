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

# Run world director integration test (hits real LLM)
uv run -m character_eng.qa_world [--model cerebras-llama|groq-llama|groq-gpt]

# Run chat QA integration test (hits real LLM)
uv run -m character_eng.qa_chat [--model cerebras-llama|groq-llama|groq-gpt]

# Start local vLLM server (interactive model picker)
uv run python -m character_eng.serve

# Start a specific local model
uv run python -m character_eng.serve lfm-2.6b

# Kill local vLLM server
lsof -ti:8000 | xargs kill -9

# Install/sync dependencies
uv sync

# Add a dependency
uv add <package>
```

## Architecture

Interactive NPC character chat CLI with selectable LLM backend (Cerebras, Groq, or local vLLM models). All use OpenAI-compatible endpoints. Six modules:

- **`character_eng/models.py`** — Model config registry. `MODELS` dict maps keys (`cerebras-llama`, `groq-llama`, `groq-gpt`, `lfm-2.6b`, `lfm-1.2b`) to config dicts with `name`, `model`, `base_url`, `api_key_env`, `stream_usage`. Local models also have `local: True` and `path`. `DEFAULT_MODEL` is `cerebras-llama`.
- **`character_eng/__main__.py`** — TUI entry point. Model selection at startup (`pick_model()`), then character selection. Two nested loops: outer loop loads prompts, world state, and creates a chat session (breaks on `/reload`), inner loop handles user input and dispatches commands (`/reload`, `/trace`, `/back`, `/quit`, `/world`, `/think`). Uses `rich` for colored output and streaming display. Manages integrated vLLM lifecycle: detects unavailable local models, offers to start vLLM via `s` menu option with live output streaming and health polling, auto-cleans up the vLLM process on exit via `atexit` and `finally`.
- **`character_eng/chat.py`** — `ChatSession` wraps OpenAI client, configured via `model_config` dict (base URL, API key, model name). Manual message history with `system`/`user`/`assistant` roles. `inject_system()` appends system-role messages mid-conversation. `get_history()` returns message list for `/think` context. Streams responses via generator. `stream_options` only sent when `model_config["stream_usage"]` is True (Cerebras doesn't support it).
- **`character_eng/prompts.py`** — Filesystem-based template engine. Scans `prompts/characters/` for subdirs containing `prompt.txt`. Resolves `{{key}}` macros via regex substitution (`global_rules`, `character`, `scenario`, `world`). Unknown macros left intact for debugging.
- **`character_eng/world.py`** — World state system. `WorldState` dataclass tracks static facts, mutable dynamic facts, and an event log. `director_call` sends a one-shot LLM call with JSON output to produce a `WorldUpdate` (remove/add facts, events). `think_call` produces a `ThinkResult` (inner monologue + optional action). Both accept `model_config`. `format_narrator_message` turns updates into `[bracketed stage directions]` injected as system-role messages.
- **`character_eng/serve.py`** — vLLM server launcher for local models. Scans `MODELS` for entries with `local: True`, kills any existing process on port 8000, then exec's into `vllm serve`. Accepts optional model key as CLI arg or shows interactive picker. Exports `build_vllm_cmd(key, cfg, port)`, `get_local_models()`, and `kill_port(port)` for reuse by `__main__.py`.

## Prompt system

Characters are defined as directories under `prompts/characters/{snake_case_name}/` with these files:
- `prompt.txt` — template with `{{global_rules}}`, `{{character}}`, `{{scenario}}`, `{{world}}` macros
- `character.txt` — personality and voice
- `scenario.txt` — situation and context

Optional world state files (line-per-fact format):
- `world_static.txt` — permanent facts that never change
- `world_dynamic.txt` — initial mutable state

`prompts/global_rules.txt` contains rules shared across all characters. Adding a new character requires no code changes — just create the directory with a `prompt.txt`.

System prompts for the director and think systems are also file-based and live-editable:
- `prompts/director_system.txt` — system prompt for the world-state director LLM
- `prompts/think_system.txt` — system prompt for the character inner monologue LLM

These are read from disk on every call, so edits take effect immediately (no `/reload` needed).

## Models

Model is chosen once at startup. All systems (chat, director, think) use the same model. Available models:

- **Llama 3.1 8B** (`cerebras-llama`) — `llama3.1-8b` via Cerebras. Fast inference. Does not support `stream_options`. Requires `CEREBRAS_API_KEY`.
- **Llama 3.3 70B** (`groq-llama`) — `llama-3.3-70b-versatile` via Groq. Does not support `stream_options`. Requires `GROQ_API_KEY`.
- **GPT-OSS 120B** (`groq-gpt`) — `openai/gpt-oss-120b` via Groq. Does not support `stream_options`. Requires `GROQ_API_KEY`.
- **LFM-2 2.6B** (`lfm-2.6b`) — Local via vLLM. Requires running `serve.py` first.
- **LFM-2.5 1.2B** (`lfm-1.2b`) — Local via vLLM. Requires running `serve.py` first.

Cloud models need their API key set in `.env`. Local models can be started directly from the app's model menu (option `s`) or manually via `serve.py`. When started from the app, vLLM output streams live and the process is auto-cleaned on exit. If only one model is available, it's auto-selected. If multiple are available, a menu appears at startup.

## Characters

- **Greg** (`prompts/characters/greg/`) — Robot head on a desk. Orb gazing scenario with world state.

## In-chat commands

- `/world` — Show current world state (static facts, dynamic facts, event log)
- `/world <text>` — Describe a change; director LLM updates state, narrator message injected as system role, character reacts
- `/think` — Character inner monologue + optional autonomous action (talk/emote)
- `/reload` — Reload prompt files, restart conversation
- `/trace` — Show system prompt, model, token usage
- `/back` — Return to character selection
- `/quit` — Exit program

## Logging

All sessions save full JSON logs to `logs/` (gitignored). Log path is printed on session end.

- **Chat sessions**: `logs/chat_{character}_{model}_{timestamp}.json` — every send, world change, and think with full untruncated responses
- **QA chat**: `logs/qa_chat_{model}_{timestamp}.json` — same detail plus expect results
- **QA world**: `logs/qa_world_{model}_{timestamp}.json` — world state before/after each scenario plus director updates

## Testing

Three layers, from fast to slow:

**Unit tests** (`uv run pytest`) — Fast, mocked, no API calls. Run on every push via git pre-push hook.
- `tests/test_chat.py` — ChatSession: init, send (message history + streaming), inject_system, get_history, trace_info, local model api_key config
- `tests/test_prompts.py` — Template loading, macro substitution, list_characters
- `tests/test_serve.py` — get_local_models, build_vllm_cmd, kill_port (missing lsof)
- `tests/test_world.py` — WorldState (render/apply/show/load), format_narrator, director_call (3 scenarios), think_call (4 scenarios)

**`qa_world`** (`uv run -m character_eng.qa_world [--model cerebras-llama|groq-llama|groq-gpt]`) — Integration test hitting real LLM. 4 cumulative director_call scenarios with strict validation (schema, index ranges, non-empty events). Run manually before releases or after changes to world.py.

**`qa_chat`** (`uv run -m character_eng.qa_chat [--model cerebras-llama|groq-llama|groq-gpt]`) — End-to-end integration test hitting real LLM. Parses `test_plan.md` for test scenarios covering chat, world changes, think, and prompt injection. Run manually before releases or after changes to chat/world/think pipeline.

`test_plan.md` is human-editable — add new sections to expand QA coverage without touching code.

**After implementing a plan or making significant changes**, always run the full test suite before reporting done:
1. `uv run pytest` — unit tests (must pass)
2. `uv run -m character_eng.qa_chat` — end-to-end QA (must pass)
3. `uv run -m character_eng.qa_world` — if world.py was touched (must pass)
