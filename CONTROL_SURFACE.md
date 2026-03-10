# Control Surface

This is the concrete control map for the local Greg experience.

## Central Authoring Files

- [prompts/prompt_registry.toml](/home/logan/Projects/character-eng/prompts/prompt_registry.toml)
  - Central registry for subsystem prompt assets.
- [prompts/characters/greg/character_manifest.toml](/home/logan/Projects/character-eng/prompts/characters/greg/character_manifest.toml)
  - Greg’s central asset map.
- [prompts/characters/mara/character_manifest.toml](/home/logan/Projects/character-eng/prompts/characters/mara/character_manifest.toml)
  - Mara’s central asset map.

## Prompt Count

- Total prompt files in [prompts](/home/logan/Projects/character-eng/prompts): `30`
- Not all `30` are active in a normal live Greg run.
- Some prompt files exist as assets or QA helpers.
- Some subsystems use inline system strings in code instead of prompt files.

## Active Greg Live Path: Main Prompt Knobs

### Base chat persona

- [prompts/characters/greg/prompt.txt](/home/logan/Projects/character-eng/prompts/characters/greg/prompt.txt)
  - Top-level template.
  - Pulls in `global_rules`, `character`, `scenario`, `world`, `people`, `vision`.
- [prompts/global_rules.txt](/home/logan/Projects/character-eng/prompts/global_rules.txt)
  - Global behavior rules.
- [prompts/characters/greg/character.txt](/home/logan/Projects/character-eng/prompts/characters/greg/character.txt)
  - Greg’s persona and speaking style.
- [prompts/characters/greg/scenario.txt](/home/logan/Projects/character-eng/prompts/characters/greg/scenario.txt)
  - Scenario framing.
- [prompts/characters/greg/world_static.txt](/home/logan/Projects/character-eng/prompts/characters/greg/world_static.txt)
  - Static world facts.
- [prompts/characters/greg/world_dynamic.txt](/home/logan/Projects/character-eng/prompts/characters/greg/world_dynamic.txt)
  - Dynamic world facts starter.

### Scenario flow

- [prompts/characters/greg/scenario_script.toml](/home/logan/Projects/character-eng/prompts/characters/greg/scenario_script.toml)
- [prompts/characters/greg/scenario_script_full.toml](/home/logan/Projects/character-eng/prompts/characters/greg/scenario_script_full.toml)
- [prompts/characters/greg/scenario_script_office_pov.toml](/home/logan/Projects/character-eng/prompts/characters/greg/scenario_script_office_pov.toml)
- [prompts/characters/greg/scenario_script_punchy.toml](/home/logan/Projects/character-eng/prompts/characters/greg/scenario_script_punchy.toml)
  - Stage graph, exits, and beat scaffolding.
  - One file is selected at runtime.

### Specialist prompt files

- [prompts/reconcile_system.txt](/home/logan/Projects/character-eng/prompts/reconcile_system.txt)
  - Converts pending changes into world updates.
- [prompts/eval_system.txt](/home/logan/Projects/character-eng/prompts/eval_system.txt)
  - Broader eval path.
  - Exists, but the live path often uses `script_check_call()` instead.
- [prompts/director_system.txt](/home/logan/Projects/character-eng/prompts/director_system.txt)
  - Decides whether to advance scenario stage.
- [prompts/plan_system.txt](/home/logan/Projects/character-eng/prompts/plan_system.txt)
  - Multi-beat planning.
- [prompts/expression_system.txt](/home/logan/Projects/character-eng/prompts/expression_system.txt)
  - Gaze target + expression selection.
- [prompts/condition_system.txt](/home/logan/Projects/character-eng/prompts/condition_system.txt)
  - Beat precondition checks.
- [prompts/beat_guide.txt](/home/logan/Projects/character-eng/prompts/beat_guide.txt)
  - Beat-specific guidance injected into chat before Greg speaks.
- [prompts/visual_focus.txt](/home/logan/Projects/character-eng/prompts/visual_focus.txt)
  - Chooses questions and SAM targets for the vision service.
- [prompts/vision_synthesis.txt](/home/logan/Projects/character-eng/prompts/vision_synthesis.txt)
  - Turns raw vision bundle into compact perception events.

## Prompt Files That Exist But Are Not The Main Live Source Of Truth

- [prompts/automation_short_turn.txt](/home/logan/Projects/character-eng/prompts/automation_short_turn.txt)
  - QA harness asset.
- [prompts/characters/greg/sims/walkup.sim.txt](/home/logan/Projects/character-eng/prompts/characters/greg/sims/walkup.sim.txt)
- [prompts/characters/greg/sims/walkup_dialogue.sim.txt](/home/logan/Projects/character-eng/prompts/characters/greg/sims/walkup_dialogue.sim.txt)
  - QA/sim assets.

## Code Knobs Outside Prompts

These matter just as much as prompt text.

### Prompt assembly

- [character_eng/prompts.py](/home/logan/Projects/character-eng/character_eng/prompts.py)
  - `load_prompt()` builds the main Greg system prompt.
  - Uses [prompts/characters/greg/character_manifest.toml](/home/logan/Projects/character-eng/prompts/characters/greg/character_manifest.toml) and [prompts/prompt_registry.toml](/home/logan/Projects/character-eng/prompts/prompt_registry.toml).
  - It decides how `world`, `people`, and `vision` are rendered into the prompt.
  - If you want to change what the base chat model sees, this file matters as much as any prompt file.

### Runtime injections

- [character_eng/chat.py](/home/logan/Projects/character-eng/character_eng/chat.py)
  - `inject_system()` appends dynamic system messages.
  - `upsert_system()` replaces tagged runtime messages.
  - `remove_tagged_system()` clears transient runtime context cleanly.
- [character_eng/__main__.py](/home/logan/Projects/character-eng/character_eng/__main__.py)
  - Adds per-turn guardrails with `upsert_system("runtime_turn_guardrails", ...)`.
  - Injects narrator/perception/system guidance.
  - Adds beat guidance before Greg speaks.
  - Controls pause/resume/restart semantics.

### Triggering, cadence, and parallelism

- [character_eng/__main__.py](/home/logan/Projects/character-eng/character_eng/__main__.py)
  - Turn boundaries.
  - When vision events are drained.
  - When post-response calls run.
  - When auto-beat fires.
  - When filler starts.
- [character_eng/vision/manager.py](/home/logan/Projects/character-eng/character_eng/vision/manager.py)
  - Vision poll cadence.
  - Focus update cadence.
  - Synthesis cadence.
  - Event dedupe and batching.
- [character_eng/voice.py](/home/logan/Projects/character-eng/character_eng/voice.py)
  - STT/TTS timing.
  - Barge-in.
  - When filler is inserted.

### Model routing

- [character_eng/models.py](/home/logan/Projects/character-eng/character_eng/models.py)
  - Decides which provider/model serves each subsystem.
  - Changing prompts without knowing the model/provider split is incomplete control.

### Scenario mechanics

- [character_eng/scenario.py](/home/logan/Projects/character-eng/character_eng/scenario.py)
  - Loads stage graph.
  - Resolves stage exits.
- [character_eng/world.py](/home/logan/Projects/character-eng/character_eng/world.py)
  - World reconciliation.
  - Beat checking.
  - Planning.
  - Expression.

### World and people state

- [character_eng/world.py](/home/logan/Projects/character-eng/character_eng/world.py)
  - `WorldState` is the durable world memory.
- [character_eng/person.py](/home/logan/Projects/character-eng/character_eng/person.py)
  - `PeopleState` is the durable person memory.
- [character_eng/perception.py](/home/logan/Projects/character-eng/character_eng/perception.py)
  - Converts perception events into pending state changes.

### Voice and filler

- [character_eng/voice.py](/home/logan/Projects/character-eng/character_eng/voice.py)
  - Live mic/STT/TTS loop.
  - Speech timing traces.
  - Filler trigger timing.
- [character_eng/filler_audio.py](/home/logan/Projects/character-eng/character_eng/filler_audio.py)
  - Prewarmed filler clip bank.
  - Sentiment-to-phrase selection.
- [character_eng/config.py](/home/logan/Projects/character-eng/character_eng/config.py)
  - `filler_enabled`
  - `filler_lead_ms`

### Vision control

- [character_eng/vision/focus.py](/home/logan/Projects/character-eng/character_eng/vision/focus.py)
  - Chooses what the vision system should look for.
- [character_eng/vision/synthesis.py](/home/logan/Projects/character-eng/character_eng/vision/synthesis.py)
  - Chooses what to emit from the raw visual bundle.
- [character_eng/vision/manager.py](/home/logan/Projects/character-eng/character_eng/vision/manager.py)
  - Owns live polling, event queueing, evidence tagging, suppression, and drain behavior.

## Highest-Leverage Files By Behavior

### Greg sounds wrong

- [prompts/characters/greg/character.txt](/home/logan/Projects/character-eng/prompts/characters/greg/character.txt)
- [prompts/global_rules.txt](/home/logan/Projects/character-eng/prompts/global_rules.txt)
- [character_eng/__main__.py](/home/logan/Projects/character-eng/character_eng/__main__.py)

### Greg advances scenario wrong

- [prompts/characters/greg/scenario_script.toml](/home/logan/Projects/character-eng/prompts/characters/greg/scenario_script.toml)
- [character_eng/scenario.py](/home/logan/Projects/character-eng/character_eng/scenario.py)
- [prompts/director_system.txt](/home/logan/Projects/character-eng/prompts/director_system.txt)
- [character_eng/world.py](/home/logan/Projects/character-eng/character_eng/world.py)

### Greg remembers the world wrong

- [prompts/reconcile_system.txt](/home/logan/Projects/character-eng/prompts/reconcile_system.txt)
- [character_eng/world.py](/home/logan/Projects/character-eng/character_eng/world.py)
- [character_eng/person.py](/home/logan/Projects/character-eng/character_eng/person.py)

### Vision notices the wrong things

- [prompts/visual_focus.txt](/home/logan/Projects/character-eng/prompts/visual_focus.txt)
- [prompts/vision_synthesis.txt](/home/logan/Projects/character-eng/prompts/vision_synthesis.txt)
- [character_eng/vision/manager.py](/home/logan/Projects/character-eng/character_eng/vision/manager.py)

### Turn pacing feels wrong

- [character_eng/__main__.py](/home/logan/Projects/character-eng/character_eng/__main__.py)
- [character_eng/voice.py](/home/logan/Projects/character-eng/character_eng/voice.py)
- [character_eng/filler_audio.py](/home/logan/Projects/character-eng/character_eng/filler_audio.py)
- [character_eng/config.py](/home/logan/Projects/character-eng/character_eng/config.py)

## Short Answers To Common “Where Is The Knob?” Questions

### “Where do I change Greg’s actual line behavior?”

- [prompts/characters/greg/character.txt](/home/logan/Projects/character-eng/prompts/characters/greg/character.txt)
- [prompts/global_rules.txt](/home/logan/Projects/character-eng/prompts/global_rules.txt)
- [character_eng/__main__.py](/home/logan/Projects/character-eng/character_eng/__main__.py) for runtime guardrails

### “Where do I change what Greg notices?”

- [prompts/visual_focus.txt](/home/logan/Projects/character-eng/prompts/visual_focus.txt)
- [prompts/vision_synthesis.txt](/home/logan/Projects/character-eng/prompts/vision_synthesis.txt)
- [character_eng/vision/manager.py](/home/logan/Projects/character-eng/character_eng/vision/manager.py)

### “Where do I change how world facts get updated?”

- [prompts/reconcile_system.txt](/home/logan/Projects/character-eng/prompts/reconcile_system.txt)
- [character_eng/world.py](/home/logan/Projects/character-eng/character_eng/world.py)
- [character_eng/person.py](/home/logan/Projects/character-eng/character_eng/person.py)

### “Where do I change filler?”

- [character_eng/filler_audio.py](/home/logan/Projects/character-eng/character_eng/filler_audio.py)
- [character_eng/voice.py](/home/logan/Projects/character-eng/character_eng/voice.py)
- [character_eng/config.py](/home/logan/Projects/character-eng/character_eng/config.py)
