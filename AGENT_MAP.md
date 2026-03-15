# Agent Map

This is the concrete subsystem map for the live Greg experience.

## Core Rule

Not every subsystem is an LLM call.

Some are:
- LLM-based agents

Some are:
- orchestration/state machines
- service wrappers
- event transformers

## Dashboard Panel Registry

- The dashboard right rail is registry-driven in `character_eng/dashboard/index.html`.
- Inspector panels are defined in `INSPECTOR_PANELS`.
- Sidebar panels are defined in `SIDEBAR_PANELS`.
- Sidebar grouping, ordering, collapse state, visibility, and empty/error states flow through the registry helpers near those definitions.
- For UX work on the dashboard, edit the registry first, then the specific renderer/reset path if needed. Do not add one-off panel toggles or reorder panels directly in scattered DOM code.
- `Vision` and `Plan` are permanent panels now. Missing data should usually render as explicit empty/error state, not by hiding the panel.

## Primary Runtime Agents

### Base Chat

- Purpose: generate Greg’s spoken line
- Files:
  - [character_eng/chat.py](/home/logan/Projects/character-eng/character_eng/chat.py)
  - [character_eng/prompts.py](/home/logan/Projects/character-eng/character_eng/prompts.py)
- Takes in:
  - assembled base system prompt
  - chat history
  - injected runtime system messages
  - current user input
- Triggered by:
  - user turn
  - scripted reaction turn
- Outputs:
  - streamed assistant text
  - message history updates

### Runtime Turn Guardrails

- Purpose: constrain the next Greg line at the last moment
- Files:
  - [character_eng/__main__.py](/home/logan/Projects/character-eng/character_eng/__main__.py)
  - [character_eng/chat.py](/home/logan/Projects/character-eng/character_eng/chat.py)
- Takes in:
  - current stage
  - current beat
  - runtime rules like “keep it short”
  - live local context
- Triggered by:
  - immediately before base chat call
- Outputs:
  - tagged system message inserted with `upsert_system()`

### Beat Guidance

- Purpose: inject the current scripted beat intent as transient guidance for the next line
- Files:
  - [character_eng/__main__.py](/home/logan/Projects/character-eng/character_eng/__main__.py)
  - [prompts/beat_guide.txt](/home/logan/Projects/character-eng/prompts/beat_guide.txt)
- Takes in:
  - current beat intent
  - current example line
- Triggered by:
  - guided beat delivery
- Outputs:
  - tagged runtime system guidance

### World State

- Purpose: durable memory of world facts and world events
- Files:
  - [character_eng/world.py](/home/logan/Projects/character-eng/character_eng/world.py)
- Takes in:
  - static facts
  - dynamic facts
  - pending change descriptions
  - reconcile results
- Triggered by:
  - startup
  - perception/world-change processing
  - reconcile application
- Outputs:
  - rendered world prompt text
  - current fact/event state for other agents

### People State

- Purpose: durable memory of who is present and what is true about them
- Files:
  - [character_eng/person.py](/home/logan/Projects/character-eng/character_eng/person.py)
- Takes in:
  - person updates from reconcile
  - visual person presence changes
- Triggered by:
  - perception processing
  - reconcile application
- Outputs:
  - rendered people prompt text
  - current people state for other agents

### Reconciler

- Purpose: convert pending changes into durable world/people updates
- Files:
  - [character_eng/world.py](/home/logan/Projects/character-eng/character_eng/world.py)
  - [prompts/reconcile_system.txt](/home/logan/Projects/character-eng/prompts/reconcile_system.txt)
- Takes in:
  - current world facts
  - recent world events
  - current people state
  - pending change descriptions
- Triggered by:
  - when reconcile is enabled and pending changes exist
- Outputs:
  - `remove_facts`
  - `add_facts`
  - `events`
  - `person_updates`

### Script Checker

- Purpose: decide whether the current beat intent has landed
- Files:
  - [character_eng/world.py](/home/logan/Projects/character-eng/character_eng/world.py)
- Takes in:
  - beat intent
  - latest line
  - recent conversation
  - world/people context
- Triggered by:
  - post-response
- Outputs:
  - `advance`
  - `hold`
  - `off_book`
  - optional redirect/request text

### Thinker

- Purpose: produce short inner monologue about the current interaction
- Files:
  - [character_eng/world.py](/home/logan/Projects/character-eng/character_eng/world.py)
  - [prompts/think_system.txt](/home/logan/Projects/character-eng/prompts/think_system.txt)
- Takes in:
  - character system prompt
  - world state
  - goals
  - recent conversation
- Triggered by:
  - post-response
- Outputs:
  - 1-2 sentence thought
  - thought can be injected back into the session

### Director

- Purpose: decide whether the scenario stage should change
- Files:
  - [character_eng/scenario.py](/home/logan/Projects/character-eng/character_eng/scenario.py)
  - [prompts/director_system.txt](/home/logan/Projects/character-eng/prompts/director_system.txt)
- Takes in:
  - current stage
  - stage exits
  - world state
  - people state
  - recent conversation
- Triggered by:
  - post-response
- Outputs:
  - `advance` or `hold`
  - exit index
  - director thought

### Planner

- Purpose: choose the next beat or multi-beat path
- Files:
  - [character_eng/world.py](/home/logan/Projects/character-eng/character_eng/world.py)
  - [prompts/plan_system.txt](/home/logan/Projects/character-eng/prompts/plan_system.txt)
- Takes in:
  - system prompt
  - current goals
  - world state
  - people state
  - recent conversation
  - stage goal
- Triggered by:
  - when post-response says a new beat is needed
  - when scenario/off-book redirect requires replanning
- Outputs:
  - next beat
  - or multi-beat plan

### Condition Checker

- Purpose: evaluate beat preconditions
- Files:
  - [character_eng/world.py](/home/logan/Projects/character-eng/character_eng/world.py)
  - [prompts/condition_system.txt](/home/logan/Projects/character-eng/prompts/condition_system.txt)
- Takes in:
  - condition text
  - world state
  - people state
- Triggered by:
  - when a condition check is needed
- Outputs:
  - pass/fail result and rationale

### Expression / Gaze

- Purpose: choose face expression and gaze target for Greg’s current line
- Files:
  - [character_eng/world.py](/home/logan/Projects/character-eng/character_eng/world.py)
  - [prompts/expression_system.txt](/home/logan/Projects/character-eng/prompts/expression_system.txt)
- Takes in:
  - latest assistant line
  - gaze targets
- Triggered by:
  - during/after line generation
- Outputs:
  - `gaze`
  - `gaze_type`
  - `expression`

## Vision Agents

### Vision Service

- Purpose: produce raw visual snapshot data
- Files:
  - [services/vision/app.py](/home/logan/Projects/character-eng/services/vision/app.py)
- Takes in:
  - camera frames
  - current question set
  - current SAM target set
- Triggered by:
  - continuous service loop
  - HTTP requests from the app
- Outputs:
  - snapshot JSON
  - VLM answers
  - tracker/object/face/person data
  - annotated video feed

### Visual Focus Planner

- Purpose: decide what the vision service should ask/look for next
- Files:
  - [character_eng/vision/focus.py](/home/logan/Projects/character-eng/character_eng/vision/focus.py)
  - [prompts/visual_focus.txt](/home/logan/Projects/character-eng/prompts/visual_focus.txt)
- Takes in:
  - current beat
  - stage goal
  - current thought
  - world state
  - people state
- Triggered by:
  - periodic focus refresh in the vision manager
  - beat/stage changes
- Outputs:
  - constant questions
  - ephemeral questions
  - constant SAM targets
  - ephemeral SAM targets

### Vision Synthesizer

- Purpose: compress raw visual bundle into useful perception events
- Files:
  - [character_eng/vision/synthesis.py](/home/logan/Projects/character-eng/character_eng/vision/synthesis.py)
  - [prompts/vision_synthesis.txt](/home/logan/Projects/character-eng/prompts/vision_synthesis.txt)
- Takes in:
  - raw visual snapshot
  - world state
  - people state
  - beat
  - stage goal
  - previous synthesis summary
- Triggered by:
  - periodic synthesis pass in the vision manager
- Outputs:
  - compact perception event lines
  - gaze candidates
  - short thought/summary

### Vision Manager

- Purpose: orchestrate live vision and bridge it into the main runtime
- Files:
  - [character_eng/vision/manager.py](/home/logan/Projects/character-eng/character_eng/vision/manager.py)
- Takes in:
  - vision service snapshots
  - focus planner output
  - synthesis output
  - current beat/stage/world/people state
- Triggered by:
  - startup
  - internal cadence timers
  - pause/resume
- Outputs:
  - queued perception events
  - vision trace payloads for dashboard
  - current rendered visual context
  - gaze targets

### Perception Processor

- Purpose: turn a perception event into pending world/people changes
- Files:
  - [character_eng/perception.py](/home/logan/Projects/character-eng/character_eng/perception.py)
- Takes in:
  - one synthesized perception event
  - current people state
  - current world state
- Triggered by:
  - when `VisionManager.drain_events()` is consumed by the main loop
- Outputs:
  - pending change descriptions
  - narrator-style text summary

## Voice Agents

### STT

- Purpose: detect user speech and produce final transcript
- Files:
  - [character_eng/voice.py](/home/logan/Projects/character-eng/character_eng/voice.py)
- Takes in:
  - live mic audio
- Triggered by:
  - continuous voice loop
- Outputs:
  - speech-start events
  - transcript events
  - timing metadata

### Primary TTS

- Purpose: synthesize Greg’s spoken line
- Files:
  - [character_eng/voice.py](/home/logan/Projects/character-eng/character_eng/voice.py)
- Takes in:
  - final assistant text
  - selected voice backend
- Triggered by:
  - after line generation begins
- Outputs:
  - audio playback
  - timing events

### Filler

- Purpose: hide TTS startup latency with short prewarmed clips
- Files:
  - [character_eng/filler_audio.py](/home/logan/Projects/character-eng/character_eng/filler_audio.py)
  - [character_eng/voice.py](/home/logan/Projects/character-eng/character_eng/voice.py)
- Takes in:
  - response text
  - expression/sentiment hint
  - filler lead time
- Triggered by:
  - after Greg’s response starts if real TTS audio has not started quickly enough
- Outputs:
  - short filler clip like `Hmm.` or `Okay.`
  - filler trace event

## Orchestration Agents

### Post-Response Coordinator

- Purpose: run the “after Greg speaks” logic fanout
- Files:
  - [character_eng/__main__.py](/home/logan/Projects/character-eng/character_eng/__main__.py)
- Takes in:
  - latest Greg line
  - world state
  - people state
  - scenario state
- Triggered by:
  - after a response completes
- Outputs:
  - script check result
  - thought
  - director result
  - expression result
  - replan signal

### Auto-Beat

- Purpose: trigger idle/scripted beats without user speech
- Files:
  - [character_eng/__main__.py](/home/logan/Projects/character-eng/character_eng/__main__.py)
  - [character_eng/voice.py](/home/logan/Projects/character-eng/character_eng/voice.py)
- Takes in:
  - runtime state
  - timer state
- Triggered by:
  - resume/startup
  - end of turns
- Outputs:
  - `/beat` style trigger into the main loop

## The Divisions That Are Easy To Confuse

### Planner vs Thinker vs Director

- Thinker:
  - “What am I thinking right now?”
  - output is inner monologue
  - does not directly move stage
- Director:
  - “Should the scenario stage change?”
  - output is stage transition decision
- Planner:
  - “What beat should happen next?”
  - output is beat content/intention

### World vs Reconciler

- World:
  - the durable store
  - facts, recent events, renderers
- Reconciler:
  - the transformation step that decides how pending text changes modify world/people state

### Raw Vision vs Synthesized Perception

- Raw vision:
  - VLM answers
  - counts
  - detections
  - face/person/object data
- Synthesized perception:
  - compact claims like `A person is now visible in the scene`
  - these are what mostly flow into world/chat/planner

### Chat vs Expression

- Chat:
  - chooses what Greg says
- Expression:
  - chooses how Greg looks while saying it

## What The Main Runtime Actually Consumes

- Base chat sees:
  - assembled system prompt
  - history
  - runtime guardrails
  - injected narrator/system messages
- World/planner/director mostly see:
  - durable world state
  - durable people state
  - synthesized perception events after they are drained and processed
- Expression sees:
  - latest line
  - available gaze targets
- Filler sees:
  - only timing and sentiment hints around TTS latency
