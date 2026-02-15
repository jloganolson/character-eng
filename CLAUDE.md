# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the app
export GEMINI_API_KEY="..."
uv run -m character_eng

# Install/sync dependencies
uv sync

# Add a dependency
uv add <package>
```

There is no test suite yet.

## Architecture

Interactive NPC character chat CLI powered by Google Gemini. Three modules:

- **`character_eng/__main__.py`** — TUI entry point. Two nested loops: outer loop loads prompts and creates a chat session (breaks on `/reload`), inner loop handles user input and dispatches commands (`/reload`, `/trace`, `/back`, `/quit`). Uses `rich` for colored output and streaming display.
- **`character_eng/chat.py`** — `ChatSession` wraps `google-genai` client. Streams responses via generator, stores last response for `/trace` token metadata. Model is `gemini-2.0-flash`.
- **`character_eng/prompts.py`** — Filesystem-based template engine. Scans `prompts/characters/` for subdirs containing `prompt.txt`. Resolves `{{key}}` macros via regex substitution (`global_rules`, `character`, `scenario`). Unknown macros left intact for debugging.

## Prompt system

Characters are defined as directories under `prompts/characters/{snake_case_name}/` with three plain text files:
- `prompt.txt` — template with `{{global_rules}}`, `{{character}}`, `{{scenario}}` macros
- `character.txt` — personality and voice
- `scenario.txt` — world state and situation

`prompts/global_rules.txt` contains rules shared across all characters. Adding a new character requires no code changes — just create the directory with a `prompt.txt`.
