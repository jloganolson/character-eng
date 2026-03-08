You are continuing work in `/home/logan/Projects/character-eng` on branch `codex/full-automation-tinker`.

Start by reading:

1. `CURRENT_STATE.md`
2. `character_eng/qa_full_stack.py`
3. `logs/qa_full_stack_greg_groq-llama-8b_20260307_204917.html`

Context:

- The committed work from tonight is the full-stack QA/viewer/runtime tracing path, not the whole dirty worktree.
- Relevant commits:
  - `868b38f` Clarify TTS start timing in QA handoff
  - `e75e2b2` Prewarm filler clips and unify chat audio trace
  - `fa3a6c9` Trace prompts and real voice path in QA viewer
  - `d262700` Improve vision boot feedback
  - `2ddf1ed` Split trace timing and tighten compact mode
  - `697b285` Add compact trace mode and browser regression tests
  - `2559cee` Add event inspector to full-stack trace viewer
  - `2494d01` Improve full-stack trace viewer controls

What is already working:

- The strict live harness can boot the real local vision stack.
- The QA trace viewer is stream-centric and browser-tested with Playwright.
- Prompt/input/output blocks are visible in the inspector.
- Vision question/SAM-target configuration is surfaced via `vision_focus`.
- Real live assistant audio is traced.
- Filler clips are prewarmed and visible in the live trace.
- Latest validated live command:

```bash
uv run -m character_eng.qa_full_stack \
  --model groq-llama-8b \
  --vision-source live \
  --assistant-audio-path live \
  --filler-lead-ms 1
```

Latest validated focused tests:

```bash
uv run pytest tests/test_filler_audio.py tests/test_voice.py tests/test_qa_full_stack.py tests/test_qa_full_stack_playwright.py
```

What not to do first:

- Do not treat the entire dirty worktree as one feature.
- Do not start with the browser/deploy stack unless you first classify the local-core files.
- Do not commit `.claude/` or vision log files.

Primary goal for tomorrow:

- Triage the remaining dirty files into:
  - commit soon
  - redesign/re-validate
  - drop/ignore

That triage has now been sketched in `CURRENT_STATE.md`; do not repeat it from scratch unless the worktree changes. Use it as the starting hypothesis and validate it file-by-file as you touch things.

Then execute this burndown:

1. Stabilize and commit the local core bucket:
   - `character_eng/person.py`
   - `character_eng/qa_chat.py`
   - `character_eng/qa_world.py`
   - `character_eng/scenario.py`
   - `prompts/reconcile_system.txt`
   - `prompts/global_rules.txt`
   - `prompts/characters/greg/character.txt`
   - `tests/test_e2e.py`
   - `tests/test_world.py`
   - `tests/test_perception.py`

The expected shape of that first commit is:

- reconcile/person sanitization
- stricter QA checks
- prompt de-meta cleanup
- scenario filename support
- related tests

2. Re-evaluate the local dashboard/runtime visibility bucket:
   - `character_eng/dashboard/index.html`
   - `character_eng/dashboard/server.py`
   - `character_eng/config.py`
   - `config.example.toml`
   - `tests/test_dashboard.py`
   - `tests/test_config.py`
   - `tests/test_runtime_controls.py`
   - `character_eng/dashboard/system_map.html`
   - `scripts/validate.sh`
   - `scripts/vision_smoke.sh`

Important: this bucket is mixed. Some parts are valuable local-runtime work, but the patch also contains unfinished bridge/remote logic. Split those concerns before committing anything here.

3. Only then decide whether the browser/remote stack is worth finishing now:
   - `character_eng/bridge.py`
   - `character_eng/browser_voice.py`
   - `Dockerfile`
   - `.github/workflows/deploy.yml`
   - `deploy/runpod.py`
   - `deploy/runpod.toml`
   - `tests/test_bridge.py`

Desired output tomorrow:

- A short review of the dirty tree with specific keep/redesign/drop calls.
- At least one focused commit for the local-core bucket.
- Updated `CURRENT_STATE.md` with what was actually kept and what was intentionally deferred.
