# Character Engine

Interactive NPC chat CLI with selectable LLM backend (Cerebras, Groq, Google Gemini, or local vLLM models). Characters have personalities, world state that evolves during conversation, and a two-speed script system — a fast eval (every turn) tracks progress through premeditated dialogue beats, while a slow planner (Gemini Flash 3) generates multi-beat conversation scripts. Beats are delivered via code-driven paste (no LLM regeneration), enabling TTS pre-rendering. Each beat includes gaze and expression data for animation.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Add API keys to `.env` (at least one required):

```
CEREBRAS_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
```

If multiple keys are set, you'll choose a model at startup. If only one is set, it's auto-selected. The planner requires `GEMINI_API_KEY` — if not set, the script system runs in eval-only mode.

## Run

```bash
uv run -m character_eng
```

Pick a character from the menu, then chat. The character evaluates each turn, tracks a script of dialogue beats, and adapts naturally. In-session commands:

| Command | What it does |
|---------|-------------|
| `/world` | Show current world state |
| `/world <text>` | Describe a change — a director LLM updates the world, the character reacts |
| `/think` | Trigger character eval — may deliver a beat, bootstrap a script, or trigger replanning |
| `/plan` | Show current script (beat list with current beat highlighted) |
| `/goals` | Show character goals (long-term) |
| `/reload` | Reload all prompt files from disk (for live editing) |
| `/trace` | Show system prompt, model, and token usage |
| `/back` | Return to character select |
| `/quit` | Exit |

## How the script system works

Each conversation turn follows one of two paths:

**Beat exists**: The pre-rendered beat line is pasted directly as the character's response — no LLM call needed. This enables TTS pre-rendering since the exact text is known ahead of time. After delivery, the eval decides: `advance` (move to next beat), `hold` (stay on current beat), or `off_book` (conversation diverged, trigger replanning). Each beat also carries `gaze` and `expression` data for animation.

**No beat** (script empty or planner unavailable): Falls back to LLM streaming for the response. World changes (`/world <text>`) always use LLM streaming since they need dynamic reactions.

## Editing prompts

Characters live in `prompts/characters/<name>/` with these files:

| File | Purpose |
|------|---------|
| `prompt.txt` | Main template — uses `{{global_rules}}`, `{{character}}`, `{{scenario}}`, `{{world}}` macros |
| `character.txt` | Personality, voice, and goals (optional `Long-term goal:` line) |
| `scenario.txt` | Situation and context |
| `world_static.txt` | Permanent facts (one per line, optional) |
| `world_dynamic.txt` | Initial mutable state (one per line, optional) |

`prompts/global_rules.txt` contains rules shared across all characters.

System-level prompts are also editable files in `prompts/`:

| File | Purpose |
|------|---------|
| `director_system.txt` | System prompt for the world-state director LLM |
| `eval_system.txt` | System prompt for the character eval LLM (script tracking + inner monologue) |
| `plan_system.txt` | System prompt for the conversation planner LLM (beat generation) |

**Live editing works** — edit any character prompt file in your editor, then type `/reload` in the chat to pick up changes without restarting. The conversation resets but you keep your place in the character menu. Director, eval, and plan system prompts are re-read on every call, so edits take effect immediately without `/reload`.

### Adding a new character

Create a directory under `prompts/characters/` with at least a `prompt.txt`. No code changes needed. Example:

```
prompts/characters/my_npc/
  prompt.txt
  character.txt
  scenario.txt
```

## Local models

Local models run via vLLM on your GPU. You can start vLLM two ways:

**From the app** (recommended) — when local models are configured but vLLM isn't running, the model menu shows a hint and an `s` option to start a local model. Pick a model, and vLLM launches with live output streaming. Once healthy, the model is auto-selected. The vLLM process is cleaned up automatically when you quit the app.

**Standalone** — use `serve.py` directly:

```bash
# Start vLLM server (interactive model picker)
uv run python -m character_eng.serve

# Start a specific model directly
uv run python -m character_eng.serve lfm-2.6b

# Kill the server
lsof -ti:8000 | xargs kill -9
```

Available local models: LFM-2 2.6B (`lfm-2.6b`), LFM-2.5 1.2B (`lfm-1.2b`). The server auto-kills any existing process on port 8000 before starting.

## Testing

```bash
# Unit tests (mocked, fast)
uv run pytest

# Integration tests (hit real LLM, default model: cerebras-llama)
uv run -m character_eng.qa_world                       # world state director
uv run -m character_eng.qa_chat                        # end-to-end chat, world, eval
uv run -m character_eng.qa_world --model groq-llama    # test with Groq Llama
uv run -m character_eng.qa_chat --model groq-llama     # test with Groq Llama
```

`test_plan.md` defines the QA chat scenarios in a human-editable format — add new test sections without touching code.

## Benchmarking

Measure real-world latency across all available models for the three LLM call patterns: streaming chat, structured JSON director calls, and structured JSON eval calls.

```bash
# Benchmark all available models (3 runs per scenario)
uv run -m character_eng.benchmark

# Benchmark a specific model with more runs
uv run -m character_eng.benchmark --model cerebras-llama --runs 5
```

Prints per-run details (TTFT, total time, response text) and a summary table with averages and standard deviations. Full results are saved to `logs/benchmark_*.json`.

## Logs

Every chat session and QA run saves a full JSON log to `logs/` (gitignored). The log path is printed when the session ends. Logs include complete untruncated responses — useful for reviewing conversation quality across models.
