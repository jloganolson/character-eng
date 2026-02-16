# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the app (API key loaded from .env via python-dotenv)
uv run -m character_eng

# Run unit tests
uv run pytest

# Run world director integration test (hits real LLM)
uv run -m character_eng.qa_world

# Run chat QA integration test (hits real LLM)
uv run -m character_eng.qa_chat

# Install/sync dependencies
uv sync

# Add a dependency
uv add <package>
```

## Architecture

Interactive NPC character chat CLI powered by Google Gemini via OpenAI-compatible endpoint. Four modules:

- **`character_eng/__main__.py`** — TUI entry point. Two nested loops: outer loop loads prompts, world state, and creates a chat session (breaks on `/reload`), inner loop handles user input and dispatches commands (`/reload`, `/trace`, `/back`, `/quit`, `/world`, `/think`). Uses `rich` for colored output and streaming display.
- **`character_eng/chat.py`** — `ChatSession` wraps OpenAI client pointed at Gemini's OpenAI-compatible endpoint. Manual message history with `system`/`user`/`assistant` roles. `inject_system()` appends system-role messages mid-conversation. `get_history()` returns message list for `/think` context. Streams responses via generator. Model is `gemini-2.0-flash`.
- **`character_eng/prompts.py`** — Filesystem-based template engine. Scans `prompts/characters/` for subdirs containing `prompt.txt`. Resolves `{{key}}` macros via regex substitution (`global_rules`, `character`, `scenario`, `world`). Unknown macros left intact for debugging.
- **`character_eng/world.py`** — World state system. `WorldState` dataclass tracks static facts, mutable dynamic facts, and an event log. `director_call` sends a one-shot LLM call with JSON output to produce a `WorldUpdate` (remove/add facts, events). `think_call` produces a `ThinkResult` (inner monologue + optional action). `format_narrator_message` turns updates into `[bracketed stage directions]` injected as system-role messages.

## Prompt system

Characters are defined as directories under `prompts/characters/{snake_case_name}/` with these files:
- `prompt.txt` — template with `{{global_rules}}`, `{{character}}`, `{{scenario}}`, `{{world}}` macros
- `character.txt` — personality and voice
- `scenario.txt` — situation and context

Optional world state files (line-per-fact format):
- `world_static.txt` — permanent facts that never change
- `world_dynamic.txt` — initial mutable state

`prompts/global_rules.txt` contains rules shared across all characters. Adding a new character requires no code changes — just create the directory with a `prompt.txt`.

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
