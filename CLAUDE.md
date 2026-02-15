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

# Install/sync dependencies
uv sync

# Add a dependency
uv add <package>
```

## Architecture

Interactive NPC character chat CLI powered by Google Gemini. Four modules:

- **`character_eng/__main__.py`** — TUI entry point. Two nested loops: outer loop loads prompts, world state, and creates a chat session (breaks on `/reload`), inner loop handles user input and dispatches commands (`/reload`, `/trace`, `/back`, `/quit`, `/world`). Uses `rich` for colored output and streaming display.
- **`character_eng/chat.py`** — `ChatSession` wraps `google-genai` client. Streams responses via generator, stores last response for `/trace` token metadata. Model is `gemini-2.0-flash`.
- **`character_eng/prompts.py`** — Filesystem-based template engine. Scans `prompts/characters/` for subdirs containing `prompt.txt`. Resolves `{{key}}` macros via regex substitution (`global_rules`, `character`, `scenario`, `world`). Unknown macros left intact for debugging.
- **`character_eng/world.py`** — World state system. `WorldState` dataclass tracks static facts, mutable dynamic facts, and an event log. `director_call` sends a one-shot Gemini call with structured JSON output to produce a `WorldUpdate` (remove/add facts, events). `format_narrator_message` turns updates into `[bracketed stage directions]` injected into chat.

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
- `/world <text>` — Describe a change; director LLM updates state, narrator message injected, character reacts
- `/reload` — Reload prompt files, restart conversation
- `/trace` — Show system prompt, model, token usage
- `/back` — Return to character selection
- `/quit` — Exit program
