# Full-Stack QA / Viewer Handoff — 2026-03-07

## What Changed Tonight

This branch now has a real full-stack QA path for the local experience, not just unit tests or bridge-only work.

Key commits from tonight:

| Commit | Summary |
|------|---------|
| `2494d01` | Full-width trace viewer controls |
| `2559cee` | Event inspector + note bug fix |
| `697b285` | Compact mode + Playwright regression test |
| `3ef5652` | Wait for vision stack before sampling |
| `2ddf1ed` | Split response timing and tighten compact mode |
| `d262700` | Ported `vlm-qa` style vision boot feedback |
| `fa3a6c9` | Prompt tracing + real live assistant-audio path in QA viewer |
| `e75e2b2` | Prewarmed filler library + unified audio trace into chat lane |

Main files touched for this pass:

| File | Purpose |
|------|---------|
| `character_eng/qa_full_stack.py` | Full-stack live/generated harness, viewer HTML, prompt inspector, unified reply timing, vision focus tracing |
| `character_eng/world.py` | LLM trace hook support for reconcile/planner/eval calls |
| `character_eng/chat.py` | Chat prompt tracing |
| `character_eng/voice.py` | Output-only live TTS harness path, filler tracing, filler prewarm |
| `character_eng/filler_audio.py` | Cached/prewarmed filler clip bank |
| `tests/test_qa_full_stack.py` | Viewer/report assertions |
| `tests/test_qa_full_stack_playwright.py` | Browser regression test for report UI |
| `tests/test_voice.py` | Live timing/filler trace tests |
| `tests/test_filler_audio.py` | Filler cache/prewarm tests |

## Current State

- The trace artifact is now stream-centric, not turn-centric.
- Greg reply text + TTS markers live together in the `chat` lane.
- Prompt/input/output blocks are visible in the inspector for LLM-backed events.
- Vision question lists and SAM target lists are surfaced via a `vision_focus` event.
- The strict live harness can exercise:
  - actual webcam bootstrap
  - actual local vision stack boot
  - actual assistant TTS path
  - actual filler path
- Playwright is in place for the trace viewer UI.

## Latest Verified Run

Command:

```bash
uv run -m character_eng.qa_full_stack \
  --model groq-llama-8b \
  --vision-source live \
  --assistant-audio-path live \
  --filler-lead-ms 1
```

Latest artifact:

- `logs/qa_full_stack_greg_groq-llama-8b_20260307_204917.html`
- `logs/qa_full_stack_greg_groq-llama-8b_20260307_204917.json`

What that run proved:

- local vision stack cold-booted fully (`vllm`, `sam3`, `face`, `person`)
- prompt traces were captured
- filler fired in the real path
- `assistant_tts_start` is now separate from `response_done`

Important timing from that run:

- `response_done` at `20:50:06.975`
- `assistant_tts_start` at `20:50:07.686`
- `assistant_filler` at `20:50:07.688`
- `assistant_tts_first_audio` at `20:50:08.538`

So filler is now effectively immediate relative to real TTS start, not after audio begins.

## Focused Validation

```bash
uv run pytest tests/test_filler_audio.py tests/test_voice.py tests/test_qa_full_stack.py tests/test_qa_full_stack_playwright.py
```

Result at end of session:

- `68 passed`

## Things Still Worth Doing Next

1. Extend Playwright coverage from the report viewer into the live dashboard/runtime controls.
2. Make compact mode even denser, closer to an NLE/gantt strip.
3. Attach each `vision_poll` to the exact active question/target set in force at that moment, not just the earlier `vision_focus` event.
4. Improve live-scene grounding for the dark-office case so content is more useful once the room is visible again.

## Branch Hygiene Note

There are many unrelated modified/untracked files on this branch from earlier work. The commits above are the reliable ones to reason from for the QA/viewer/runtime path.

## Dirty Tree Triage

The remaining uncommitted files are not one coherent change. After reviewing the actual diffs, the right split is more specific than last night’s rough bucket list.

### 1. Keep Soon

These are coherent, local-first, and directly support the validated path from tonight:

| File | Call | Why |
|------|------|-----|
| `character_eng/person.py` | keep soon | Sanitizes generated person-fact IDs and rejects invalid presence values; aligns with the real reconcile corruption we saw. |
| `character_eng/qa_chat.py` | keep soon | Makes QA match the current split eval path and fails on leaked fact IDs / stale events instead of silently passing. |
| `character_eng/qa_world.py` | keep soon | Upgrades world QA from schema smoke to actual drift detection (duplicate events, stale history, leaked IDs). |
| `character_eng/scenario.py` | keep soon | Small, useful change: lets callers pick a scenario TOML filename explicitly. |
| `prompts/reconcile_system.txt` | keep soon | Tightens reconciler instructions around plain fact text and event freshness; matches real bugs seen in logs. |
| `prompts/global_rules.txt` | keep soon | Removes AI/meta self-description drift and strengthens the in-character rule. |
| `prompts/characters/greg/character.txt` | keep soon | Cleans up Greg’s “large language models” self-reference. |
| `tests/test_world.py` | keep soon | Encodes the sanitize/dedupe behavior and explicit redirect handling into tests. |
| `tests/test_e2e.py` | keep soon | Fixes provider gating and adds a real local vision mock dialogue path instead of only the old smoke route. |
| `tests/test_perception.py` | keep soon | Adds coverage for the dialogue sim actually being present/loadable. |
| `scripts/validate.sh` | keep soon | Good local/CI validation entrypoint; current shape is practical and aligned with the repo. |
| `scripts/vision_smoke.sh` | keep soon | Valuable live contract check for the real vision service endpoints. |
| `character_eng/dashboard/system_map.html` | keep soon | Useful artifact for onboarding/debugging; low risk. |

### 2. Keep But Redesign / Re-validate

These are real work, but still tangled or incomplete enough that they should not just be committed as-is:

| File | Call | Why |
|------|------|-----|
| `character_eng/bridge.py` | redesign / re-validate | Useful direction, but earlier review already found live-event delivery and command fallback issues. |
| `character_eng/browser_voice.py` | redesign / re-validate | Incomplete relative to the live `VoiceIO` path: no filler, less instrumentation, and it diverges from the local runtime we just hardened. |
| `character_eng/dashboard/index.html` | redesign / re-validate | Contains both good local runtime UI ideas and unfinished bridge/remote logic in one huge patch; should be separated before landing. |
| `character_eng/dashboard/server.py` | redesign / re-validate | Mostly fine, but it is coupled to the dashboard patch above and should land with a cleaner local-runtime story. |
| `character_eng/config.py` | redesign / re-validate | The filler persistence pieces are good; the bridge config additions are tied to the unfinished browser path. Split before committing. |
| `config.example.toml` | redesign / re-validate | Same issue as `config.py`: mixed local-good and bridge-specific config in one patch. |
| `tests/test_dashboard.py` | redesign / re-validate | Some useful local coverage, but it is bundled with the mixed dashboard patch. |
| `tests/test_bridge.py` | redesign / re-validate | Helpful if the bridge path stays alive, but not enough reason to commit the bridge stack unchanged. |
| `tests/test_config.py` | redesign / re-validate | Good once config changes are split into local vs bridge concerns. |
| `tests/test_runtime_controls.py` | redesign / re-validate | Valuable, but should land with a more deliberate local-runtime dashboard patch. |
| `Dockerfile` | redesign / re-validate | Useful base, but deploy should follow stable local/browser behavior, not lead it. |
| `.dockerignore` | redesign / re-validate | Fine mechanically, but part of the same deploy bucket. |
| `.github/workflows/deploy.yml` | redesign / re-validate | Better than nothing, but should not be trusted until the deploy/runtime story is intentionally finished. |
| `deploy/runpod.py` | redesign / re-validate | Potentially useful ops tooling, but outside the current local-first stabilization path. |
| `deploy/runpod.toml` | redesign / re-validate | Same as above. |
| `CLAUDE.md` | redesign / re-validate | Documentation drifted toward the unfinished bridge/deploy path; should be updated only after decisions are made. |

### 3. Drop / Ignore / Do Not Commit

These should not drive decisions unless deliberately mined for context:

| File | Call | Why |
|------|------|-----|
| `.claude/` | ignore / do not commit | Local workspace residue, not repo product code. |
| `services/vision/app.log` | ignore / do not commit | Local runtime log. |
| `services/vision/vllm.log` | ignore / do not commit | Local runtime log. |
| `vision_plan.md` | likely stale | Planning residue; useful only if someone explicitly wants historical notes. |

## Recommended Burndown Order

Tomorrow’s work should not start by “finishing everything in the dirty tree.” It should proceed in this order:

1. Local core path
   Files: `qa_chat.py`, `qa_world.py`, `person.py`, `scenario.py`, prompt files, related tests.
   Goal: lock down the real local Greg + vision route first.

2. Local dashboard/runtime visibility
   Files: `dashboard/index.html`, `dashboard/server.py`, `config.py`, `system_map.html`.
   Goal: split the local-runtime UI pieces from the unfinished bridge logic, then land the local pieces cleanly.

3. Browser/remote bridge
   Files: `bridge.py`, `browser_voice.py`, deploy stack, related tests.
   Goal: either finish it properly or explicitly defer it.

4. Deploy automation
   Files: `Dockerfile`, workflow, RunPod helpers.
   Goal: only after local behavior and browser behavior are actually stable.

## Practical Recommendation

Push the branch if you want tonight’s committed QA/viewer work safely remote. The dirty worktree will not be pushed unless it is committed.

The next agent should:

1. Read this file.
2. Read `AGENT_START.md`.
3. Review the latest committed QA/viewer commits, not the whole dirty tree at once.
4. Convert the “Keep Soon” bucket into one or two focused commits before touching the bridge/deploy stack again.
5. For the dashboard/config bucket, separate `local-runtime controls` from `browser bridge` before making commit decisions.

# Browser Bridge Handoff — 2026-03-06

## What Was Done Today

Implemented remote deployment infrastructure for character-eng: browser-based mic/camera via WebSocket bridge, so people without a local GPU can use the app over RunPod or SSH tunnel. Everything is on the `lemonade-stand` branch, **nothing committed yet** — all changes are unstaged.

### New Files (untracked)

| File | Purpose |
|------|---------|
| `character_eng/bridge.py` | BridgeServer (HTTP dashboard + WS on single port), BrowserMicStream, BrowserSpeakerStream |
| `character_eng/browser_voice.py` | BrowserVoiceIO — drop-in VoiceIO replacement using bridge instead of sounddevice |
| `tests/test_bridge.py` | 14 unit tests for bridge + browser_voice (all pass) |
| `Dockerfile` | CUDA 12.8 + Ubuntu 24.04 + uv, syncs main + vision deps |
| `.dockerignore` | Standard exclusions |
| `deploy/runpod.py` | RunPod CLI (up/down/status/deploy/restart/ssh/logs/destroy) via GraphQL API |
| `deploy/runpod.toml` | RunPod pod config (GPU type, image, ports) |
| `.github/workflows/deploy.yml` | CI/CD: build Docker image, push to GHCR, restart RunPod |

### Modified Files

| File | Changes |
|------|---------|
| `character_eng/__main__.py` | `--browser` flag, BrowserVoiceIO wiring, bridge.start(), `/pause` `/resume` `/restart` commands, dashboard forwarder removed (bridge routes directly now) |
| `character_eng/config.py` | Added `BridgeConfig(enabled, port=7862)` to AppConfig |
| `character_eng/dashboard/index.html` | Bridge panel (mic/cam/pause/restart/beat/trigger buttons, debug stats), AudioWorklet refs changed from Blob URLs to served JS endpoints, WS URL uses `wss://host/bridge` (protocol-aware), `sendCommand()` routes through WS, `sendInput()` uses WS when available |
| `character_eng/vision/client.py` | Added `inject_frame(jpeg_bytes)` method |
| `services/vision/app.py` | Added `WebcamCapture.inject()`, `/inject_frame` endpoint, `--no-camera` flag |
| `config.example.toml` | Added `[bridge]` section |
| `pyproject.toml` | Added `websockets>=14.0` dependency |
| `CLAUDE.md` | Updated module count, added bridge/browser_voice docs |
| `README.md` | Added remote deployment section |

## Architecture

```
Browser (HTTPS via tailscale serve)
  |
  +-- GET / .................. Dashboard HTML (served by bridge)
  +-- GET /state ............. Dashboard event snapshot (JSON)
  +-- GET /events ............ SSE stream (snapshot-only, not live push)
  +-- GET /capture-worklet.js  AudioWorklet for mic capture
  +-- GET /playback-worklet.js AudioWorklet for TTS playback
  +-- GET /debug ............. Bridge stats (mic frames, speaker bytes, WS conns)
  +-- WS  /bridge ............ Multiplexed WebSocket:
        0x01 + PCM ........... Audio in (browser mic -> Deepgram STT)
        0x02 + PCM ........... Audio out (TTS -> browser speakers)
        0x03 + JPEG .......... Video in (browser cam -> vision inject)
        JSON {type:"ping"}... Latency measurement
        JSON {type:"command"} Dashboard commands (/pause, /resume, /restart, /beat, etc.)
```

Single port (:7862) serves everything. `tailscale serve --bg 7862` proxies it as HTTPS (required for `getUserMedia`).

## What Was Tested (Programmatic)

All of these pass via Python `websockets` client — NOT via real browser:

- **243 unit tests pass** (`uv run pytest`)
- HTML endpoint serves correctly (200, correct content-type)
- `/state`, `/events` (SSE), `/debug` endpoints work
- `/capture-worklet.js` and `/playback-worklet.js` serve valid JS with `registerProcessor`
- WS `/bridge` connects, ping/pong works
- Audio frames (0x01 prefix) arrive and increment `mic_frames`/`mic_bytes` in debug
- Video frames (0x03 prefix) arrive without crash
- Speaker bytes enqueue/send tracking works
- `/pause` command stops auto-beat (0 new log lines in 6-8 seconds)
- `/resume` command resumes auto-beat
- `/restart` creates new session (new session ID in logs)
- `/pause` works after `/restart` (the forwarder-thread bug was fixed)
- Bridge.start() is idempotent (doesn't crash on re-entry after restart)

## KNOWN BUGS — Not Yet Validated in Real Browser

### BUG 1: Pause/Restart/Resume don't work from browser (CRITICAL)

**Symptom**: User clicks Pause button in dashboard, Greg keeps going. Restart also fails.

**What we know**:
- Programmatic Python WS tests show the command flow works: `bridge receives → voice_event_queue → chat_loop handles`
- The user confirmed it does NOT work from their actual browser via Tailscale HTTPS
- Server logs DO show `[Bridge] Command: /pause` arriving in at least some cases
- The bug may be timing-related: by the time `/pause` lands, auto-beat has already queued several `/beat`s

**What was tried**:
1. Original: forwarder thread draining `_dashboard_input_queue` → `voice_io._event_queue` — broke on restart because thread held stale closure reference
2. Current: bridge routes commands directly to `bridge.voice_event_queue` which is set to `voice_io._event_queue` by BrowserVoiceIO.start() — programmatic tests pass but browser still broken

**Debugging approach**:
1. Start server: `uv run -m character_eng --browser --character greg`
2. Set up HTTPS: `sudo tailscale serve --bg 7862`
3. Open browser to `https://logan-desktop.[tailnet].ts.net/`
4. Open browser DevTools console
5. Click Pause, check:
   - Is the WS connected? (bridge panel should show "Connected")
   - Does `sendCommand('/pause')` actually send? Add `console.log` to `sendCommand`
   - Check server stderr: `tail -f /tmp/character-eng-browser.log | grep -E "Bridge|Forwarder|Paused"`
   - Check `/debug` endpoint: does `ws_connections` show the browser?

**Likely root causes to investigate**:
- WS might not be connecting over Tailscale HTTPS (wss:// upgrade may fail through proxy)
- If WS isn't connected, `sendCommand` falls back to `fetch('/send')` which was REMOVED from bridge.py (websockets can't read POST bodies)
- The `_bridgeWs` global set on connect may be stale or the `ws` closure variable in `sendCommand` may reference a different connection

### BUG 2: SSE is snapshot-only (minor)

The bridge's `/events` endpoint returns a one-shot snapshot of current events, not a long-lived SSE stream. This means the dashboard won't get live updates after initial load when served by the bridge. The stdlib dashboard server (`dashboard/server.py`) does proper SSE streaming but is bypassed in bridge mode.

**Impact**: Dashboard shows events from page load but won't update live. User must refresh to see new events. The `/state` endpoint works for manual refresh.

**Fix options**:
- Add SSE fan-out support to bridge (complex — websockets process_request is synchronous)
- Use WS for live event push instead of SSE (add a new message type)
- Poll `/state` periodically from JS as a fallback when in bridge mode

### BUG 3: Scroll behavior

User reported "crazy scroll behavior" with each auto-beat. The `scrollToBottom()` fires on every event. We added `autoScroll = false` when paused, but since pause itself may not be working (Bug 1), this hasn't been validated.

## What Needs Human/Browser Testing

### Test 1: WS connectivity over Tailscale
```
1. uv run -m character_eng --browser --character greg
2. sudo tailscale serve --bg 7862
3. Open https://logan-desktop.[tailnet].ts.net/
4. Open DevTools console
5. Check: does bridge panel say "Connected"?
6. Run in console: _bridgeWs  (should be a WebSocket object, not null)
7. Check /debug endpoint: ws_connections should be >= 1
```

### Test 2: Commands reaching server
```
1. With server running and browser connected
2. In browser console: sendCommand('/pause')  (call it directly)
3. Check server log: grep "Bridge.*Command" /tmp/character-eng-browser.log
4. If no command appears → WS isn't connected, commands going to dead fetch('/send')
```

### Test 3: Mic audio flow
```
1. Pause the session first (once that works)
2. Click Mic On
3. Speak
4. Check /debug endpoint: mic_frames should increment
5. If mic_frames stays 0 → AudioWorklet not sending, or WS not connected
```

### Test 4: Full conversation via browser
```
1. Start fresh: uv run -m character_eng --browser --character greg
2. Open in browser
3. Click Mic On
4. Say something to Greg
5. Greg should respond via TTS (audio plays in browser)
6. Check: speaker_bytes_sent in /debug should be > 0
```

## Files to Read First

If starting fresh, read in this order:
1. This file
2. `CLAUDE.md` — project conventions, commands, architecture
3. `character_eng/bridge.py` — the core of the problem
4. `character_eng/browser_voice.py` — voice orchestrator for bridge mode
5. `character_eng/dashboard/index.html` lines 296-320 (bridge panel HTML) and lines 714-940 (bridge JS)
6. `character_eng/__main__.py` lines 854-900 (bridge voice setup) and lines 1103-1144 (pause/restart handlers)

## Environment Notes

- Tailscale is set up: `tailscale status` shows `logan-desktop` at `100.85.170.123`, laptop `jlo-win-laptop` at `100.111.57.16`
- `sudo tailscale serve --bg 7862` persists across reboots (one-time setup, already done)
- `getUserMedia` requires HTTPS — that's why everything must go through tailscale serve on a single port
- The websockets library v14+ `process_request` hook can serve HTTP responses but CANNOT read POST request bodies (only path + headers). This is why `/send` was removed from bridge HTTP and commands route through WS JSON frames instead.
- 243 unit tests pass as of this session
