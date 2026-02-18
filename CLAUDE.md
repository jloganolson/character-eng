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

# Install/sync dependencies
uv sync

# Add a dependency
uv add <package>
```

## Architecture

Interactive NPC character chat CLI with selectable LLM backend (Cerebras, Groq, Google Gemini, or local vLLM models). All use OpenAI-compatible endpoints. Two-speed character planning: fast eval (chat model, every turn) tracks progress through a script, slow planner (Gemini Flash 3) generates multi-beat scripts. Two-speed world updates: fast path stores changes as pending and lets the character react immediately, slow path (background reconciler using PLAN_MODEL, falls back to chat model) reconciles pending changes into structured state mutations using stable fact IDs. Seven modules:

- **`character_eng/models.py`** — Model config registry. `MODELS` dict maps keys (`cerebras-llama`, `cerebras-gpt`, `groq-llama`, `groq-gpt`, `gemini-3-flash`, `gemini-2.5-flash`, `lfm-2.6b`, `lfm-1.2b`) to config dicts with `name`, `model`, `base_url`, `api_key_env`, `stream_usage`. Local models also have `local: True` and `path`. Models with `hidden: True` are excluded from the startup menu and benchmark auto-discovery but remain usable via `--model` flag. `DEFAULT_MODEL` is `cerebras-llama`. `PLAN_MODEL` is `gemini-3-flash` (used by the planner and reconciler). `THINK_MODEL` is `groq-llama` (used for background eval; falls back to chat model if `GROQ_API_KEY` not set).
- **`character_eng/__main__.py`** — TUI entry point. Model selection at startup (`pick_model()`), then character selection. Two nested loops: outer loop loads prompts, world state, and creates a chat session (breaks on `/reload`), inner loop handles user input and dispatches commands (`/reload`, `/trace`, `/back`, `/quit`, `/world`, `/beat`, `/plan`, `/goals`). Unknown commands (any `/` prefix not matching a known command) show an error with a `/help` hint. First user input always gets a natural LLM response (no beat delivery), then kicks off background eval + replan for visitor-aware script. Subsequent turns use LLM-guided beat delivery: when a beat exists, `stream_guided_beat()` injects the beat's intent and example line as guidance (via `load_beat_guide()`), then streams a real LLM response that reacts to the user's input while serving the beat's purpose. The `/beat` command (autonomous time-passing) still uses verbatim `deliver_beat()`. Beats with conditions are gated by `condition_check_call()` in the `/beat` path — if the condition isn't met, an idle line is displayed instead. If no beat (script empty/planner unavailable), falls back to LLM streaming. World changes (`/world <text>`) use two-speed updates: immediate pending + narrator injection + character reaction via LLM streaming, then background reconciliation + background eval. `/beat` delivers the next scripted beat if its condition is met, otherwise shows an in-character idle line. Non-blocking eval/plan: after each turn, a background eval thread (using `THINK_MODEL` via Groq, falling back to chat model) runs asynchronously — the user gets the prompt back immediately. Results are checked at the next turn start; stale results (where `_context_version` changed since the eval started) are discarded. When eval returns `advance` (script exhausted) or `off_book`, a background plan thread is started. Three background thread pools follow the same pattern: reconcile (`_reconcile_lock`), eval (`_eval_lock`), and plan (`_plan_lock`), each with version-based stale discard via `_context_version`. Only boot plan and `/beat` initial replan (when no script exists) remain synchronous. Uses `rich` for colored output and streaming display. Manages integrated vLLM lifecycle.
- **`character_eng/chat.py`** — `ChatSession` wraps OpenAI client, configured via `model_config` dict (base URL, API key, model name). Manual message history with `system`/`user`/`assistant` roles. `add_assistant()` adds pre-rendered text to history without an LLM call (used for code-driven beat delivery). `inject_system()` appends system-role messages mid-conversation. `get_history()` returns message list for eval context. Streams responses via generator. `stream_options` only sent when `model_config["stream_usage"]` is True (Cerebras doesn't support it).
- **`character_eng/prompts.py`** — Filesystem-based template engine. Scans `prompts/characters/` for subdirs containing `prompt.txt`. Resolves `{{key}}` macros via regex substitution (`global_rules`, `character`, `scenario`, `world`). Unknown macros left intact for debugging.
- **`character_eng/world.py`** — World state and script system. `WorldState` dataclass tracks static facts, mutable dynamic facts as `dict[str, str]` (ID → fact text, insertion-ordered) with stable IDs (`f1`, `f2`, ...), pending unreconciled changes, and an event log. `add_fact(text)` assigns next sequential ID. `add_pending(text)` stores unreconciled changes. `clear_pending()` drains pending list. `render_for_reconcile()` shows ID-tagged facts for reconciler context. `render()` shows facts without IDs (for character prompt) plus pending changes section. `Goals` dataclass holds a permanent `long_term` goal (parsed from `character.txt`). `Beat` holds a dialogue line, intent, optional `gaze`/`expression` fields for animation/TTS pre-rendering, and an optional `condition` (natural language precondition — empty means auto-deliver). `ConditionResult` holds `met` (bool), `idle` (in-character idle line when not met), `gaze`, and `expression`. `Script` is an ordered beat list with `current_beat`, `advance()`, `replace()`, `is_empty()`, `render()`, `show()`. `EvalResult` holds thought, gaze, expression, and `script_status` (advance/hold/off_book/bootstrap) with optional bootstrap fields. `PlanResult` holds a list of beats. `WorldUpdate` uses string fact IDs in `remove_facts`. `reconcile_call` sends a one-shot LLM call with JSON output using ID-tagged facts to produce a `WorldUpdate`. `eval_call` reads `eval_system.txt`, takes script context, returns script tracking status. `condition_check_call` reads `condition_system.txt`, checks a beat's condition against world state and conversation, returns a `ConditionResult`. `plan_call` reads `plan_system.txt`, uses the plan model (Gemini Flash 3), returns a list of beats with gaze/expression/condition. `format_narrator_message` turns updates into `[bracketed stage directions]`. `format_pending_narrator` wraps a change description in brackets for fast-path injection. `load_beat_guide(intent, line)` reads `beat_guide.txt` and substitutes `{intent}` and `{line}` placeholders — used by guided beat delivery to inject intent context before the LLM response.
- **`character_eng/benchmark.py`** — Model speed benchmark. Runnable via `uv run -m character_eng.benchmark [--model KEY] [--runs N]`. Benchmarks three scenarios matching real usage: streaming chat (TTFT + total time), structured JSON reconcile call, and structured JSON eval call. Uses Greg's full prompts and world state for realistic payloads. Prints per-run details and a rich summary table, saves JSON logs to `logs/`.
- **`character_eng/serve.py`** — vLLM server launcher for local models. Scans `MODELS` for entries with `local: True`, kills any existing process on port 8000, then exec's into `vllm serve`. Accepts optional model key as CLI arg or shows interactive picker. Exports `build_vllm_cmd(key, cfg, port)`, `get_local_models()`, and `kill_port(port)` for reuse by `__main__.py`.

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

## Models

Model is chosen once at startup for chat (streaming responses). Background eval uses `THINK_MODEL` (`groq-llama`) for faster thinking — requires `GROQ_API_KEY`; falls back to the chat model if unavailable. The planner and reconciler use `PLAN_MODEL` (`gemini-3-flash`) separately — requires `GEMINI_API_KEY`. If the plan model's API key is not set, the planner is unavailable and the script system runs in eval-only mode; the reconciler falls back to the chat model. Available models:

- **Llama 3.1 8B** (`cerebras-llama`) — `llama3.1-8b` via Cerebras. Fast inference. Does not support `stream_options`. Requires `CEREBRAS_API_KEY`.
- **GPT-OSS 120B** (`cerebras-gpt`) — `gpt-oss-120b` via Cerebras. Does not support `stream_options`. Requires `CEREBRAS_API_KEY`.
- **Llama 3.3 70B** (`groq-llama`) — `llama-3.3-70b-versatile` via Groq. Does not support `stream_options`. Requires `GROQ_API_KEY`. Also used as the think model for background eval.
- **GPT-OSS 120B** (`groq-gpt`) — `openai/gpt-oss-120b` via Groq. Does not support `stream_options`. Requires `GROQ_API_KEY`.
- **Gemini 3.0 Flash** (`gemini-3-flash`) — `gemini-3-flash-preview` via Google. Does not support `stream_options`. Requires `GEMINI_API_KEY`. Also used as the plan/reconcile model.
- **Gemini 2.5 Flash** (`gemini-2.5-flash`) — `gemini-2.5-flash` via Google. Does not support `stream_options`. Requires `GEMINI_API_KEY`.
- **LFM-2 2.6B** (`lfm-2.6b`) — Local via vLLM. Requires running `serve.py` first.
- **LFM-2.5 1.2B** (`lfm-1.2b`) — Local via vLLM. Requires running `serve.py` first.

Cloud models need their API key set in `.env`. Local models can be started directly from the app's model menu (option `s`) or manually via `serve.py`. When started from the app, vLLM output streams live and the process is auto-cleaned on exit. If only one model is available, it's auto-selected. If multiple are available, a menu appears at startup.

## Characters

- **Greg** (`prompts/characters/greg/`) — Robot head on a desk. Orb gazing scenario with world state. Long-term goal: experience the universe.

## In-chat commands

- `/world` — Show current world state (static facts with IDs, dynamic facts, pending changes, event log)
- `/world <text>` — Describe a change; character reacts immediately, background reconciler updates state with stable fact IDs. Reconcile results shown at start of next turn.
- `/beat` — Deliver next scripted beat (time passes). Checks beat condition first — if not met, shows an in-character idle line instead. If no script, triggers replanning.
- `/plan` — Show current script (beat list with current beat highlighted)
- `/goals` — Show character goals (long-term)
- `/reload` — Reload prompt files, restart conversation
- `/trace` — Show system prompt, model, token usage
- `/back` — Return to character selection
- `/quit` — Exit program
- Unknown `/` commands show an error message with a `/help` hint

## Logging

All sessions save full JSON logs to `logs/` (gitignored). Log path is printed on session end.

- **Benchmarks**: `logs/benchmark_{timestamp}.json` — per-run timing (TTFT, total ms) and full response text for each scenario
- **Chat sessions**: `logs/chat_{character}_{model}_{timestamp}.json` — every send, world change, reconcile, and eval with full untruncated responses
- **QA chat**: `logs/qa_chat_{model}_{timestamp}.json` — same detail plus expect results
- **QA world**: `logs/qa_world_{model}_{timestamp}.json` — world state before/after each scenario plus reconciler updates

## Testing

Three layers, from fast to slow:

**Unit tests** (`uv run pytest`) — Fast, mocked, no API calls. Run on every push via git pre-push hook.
- `tests/test_chat.py` — ChatSession: init, send (message history + streaming), add_assistant, inject_system, get_history, trace_info, local model api_key config
- `tests/test_prompts.py` — Template loading, macro substitution, list_characters
- `tests/test_serve.py` — get_local_models, build_vllm_cmd, kill_port (missing lsof)
- `tests/test_world.py` — WorldState (dict-based dynamic with stable IDs, render/apply/show/load, add_fact, add_pending/clear_pending, render_for_reconcile, render_with_pending), Beat construction (with/without gaze/expression, with condition), ConditionResult (met/not met), Script (current_beat/advance/replace/is_empty/render/render_with_gaze_expression/render_with_condition/show), EvalResult (construction/bootstrap fields), Goals (render/load from character.txt), format_narrator, format_pending_narrator, load_beat_guide (substitutes placeholders, reads from disk each call, missing file raises), reconcile_call (6 scenarios incl. basic, no removals, empty state, multiple pending, reads reconcile_system.txt, IDs in context), eval_call (7 scenarios incl. script/goals in context, reads eval_system.txt), condition_check_call (4 scenarios incl. met/not met, condition in context, reads condition_system.txt), plan_call (6 scenarios incl. gaze/expression/condition parsing, plan_request in context, reads plan_system.txt)

**`qa_world`** (`uv run -m character_eng.qa_world [--model cerebras-llama|groq-llama|groq-gpt]`) — Integration test hitting real LLM. 4 cumulative reconcile_call scenarios with strict validation (schema, valid fact IDs, non-empty events). Uses stable fact IDs (`f1`-`f4`) instead of integer indices. Run manually before releases or after changes to world.py.

**`qa_chat`** (`uv run -m character_eng.qa_chat [--model cerebras-llama|groq-llama|groq-gpt]`) — End-to-end integration test hitting real LLM. Parses `test_plan.md` for test scenarios covering chat, world changes (two-speed: pending + synchronous reconcile), eval, script tracking with user pushback detection, and prompt injection. Run manually before releases or after changes to chat/world/eval pipeline.

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
