# Character Engine

Interactive NPC chat CLI with selectable LLM backend (Cerebras, Groq, or local vLLM models). Characters have personalities, world state that evolves during conversation, and an inner monologue system that lets them act autonomously.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Add API keys to `.env` (at least one required):

```
CEREBRAS_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
```

If multiple keys are set, you'll choose a model at startup. If only one is set, it's auto-selected.

## Run

```bash
uv run -m character_eng
```

Pick a character from the menu, then chat. In-session commands:

| Command | What it does |
|---------|-------------|
| `/world` | Show current world state |
| `/world <text>` | Describe a change — a director LLM updates the world, the character reacts |
| `/think` | Trigger the character's inner monologue — they may speak or act on their own |
| `/reload` | Reload all prompt files from disk (for live editing) |
| `/trace` | Show system prompt, model, and token usage |
| `/back` | Return to character select |
| `/quit` | Exit |

## Editing prompts

Characters live in `prompts/characters/<name>/` with these files:

| File | Purpose |
|------|---------|
| `prompt.txt` | Main template — uses `{{global_rules}}`, `{{character}}`, `{{scenario}}`, `{{world}}` macros |
| `character.txt` | Personality and voice |
| `scenario.txt` | Situation and context |
| `world_static.txt` | Permanent facts (one per line, optional) |
| `world_dynamic.txt` | Initial mutable state (one per line, optional) |

`prompts/global_rules.txt` contains rules shared across all characters.

System-level prompts are also editable files in `prompts/`:

| File | Purpose |
|------|---------|
| `director_system.txt` | System prompt for the world-state director LLM |
| `think_system.txt` | System prompt for the character inner monologue LLM |

**Live editing works** — edit any character prompt file in your editor, then type `/reload` in the chat to pick up changes without restarting. The conversation resets but you keep your place in the character menu. Director and think system prompts are re-read on every call, so edits take effect immediately without `/reload`.

### Adding a new character

Create a directory under `prompts/characters/` with at least a `prompt.txt`. No code changes needed. Example:

```
prompts/characters/my_npc/
  prompt.txt
  character.txt
  scenario.txt
```

## Local models

Local models run via vLLM on your GPU. Use `serve.py` to start the server:

```bash
# Start vLLM server (interactive model picker)
uv run python -m character_eng.serve

# Start a specific model directly
uv run python -m character_eng.serve lfm-2.6b

# Kill the server
lsof -ti:8000 | xargs kill -9
```

Available local models: LFM-2 2.6B (`lfm-2.6b`), LFM-2.5 1.2B (`lfm-1.2b`). The server auto-kills any existing process on port 8000 before starting.

Local models appear in the model menu automatically when the vLLM server is running.

## Testing

```bash
# Unit tests (mocked, fast)
uv run pytest

# Integration tests (hit real LLM, default model: cerebras-llama)
uv run -m character_eng.qa_world                       # world state director
uv run -m character_eng.qa_chat                        # end-to-end chat, world, think
uv run -m character_eng.qa_world --model groq-llama    # test with Groq Llama
uv run -m character_eng.qa_chat --model groq-llama     # test with Groq Llama
```

`test_plan.md` defines the QA chat scenarios in a human-editable format — add new test sections without touching code.

## Logs

Every chat session and QA run saves a full JSON log to `logs/` (gitignored). The log path is printed when the session ends. Logs include complete untruncated responses — useful for reviewing conversation quality across models.
