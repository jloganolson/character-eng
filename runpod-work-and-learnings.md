# RunPod Work And Learnings

Date: March 18, 2026
Repo: `/home/logan/Projects/character-eng`
Branch: `runpod-test`

## Goal

Get the browser-oriented `character-eng` stack running on RunPod with:

- a public session-manager URL
- real per-session browser runtime creation
- working shared vision service
- visual end-to-end validation against the live remote pod

## Where We Started

The repo already had:

- a session manager and bridge architecture
- a RunPod deploy scaffold
- a full GPU image path
- a manager-lite image path for infrastructure validation

The initial handoff commit was `2e1a8c9` (`Add browser session manager and remote media bridge`).

## What Worked

### 1. Lite control-plane validation succeeded

The manager-lite path ended up being the right first proving ground. It validated:

- pod creation
- registry auth handling
- RunPod proxy URL usage
- manager health and session creation

Important result:

- the correct public access pattern for the manager is the RunPod proxy URL, not direct IP substitution
- returned session URLs must use `https://<pod-id>-7870.proxy.runpod.net/...`

### 2. Full pod eventually worked on a 4090

After fixing image compatibility and startup ordering, the full path worked on RunPod:

- 4090-only placement
- manager health `200`
- real session creation
- session URL loaded successfully
- external vision status reported ready for:
  - vLLM
  - SAM3
  - InsightFace
  - ReID

We also exercised the remote session through Chrome MCP and visually confirmed:

- dashboard rendered
- world state populated
- session reached `watching`
- runtime panel showed the shared vision service healthy

### 3. Local heavy-runtime debugging improved substantially

We added a local-first heavy smoke path so we could stop using RunPod as the first debugger:

- `scripts/vision_smoke.sh`
- `scripts/run_heavy.sh --once --json --teardown`

That gave us a deterministic local check for:

- vLLM startup
- vision app startup
- model tracker readiness
- SAM3 auth/runtime failures

This was one of the highest-value changes in the whole effort.

## What Failed Or Slowed Us Down

### 1. Private GHCR auth was broken initially

Early pods failed with:

- `unauthorized to use image`

Root cause:

- RunPod needed container registry auth attached explicitly

Fix:

- added registry-auth creation/reuse to `deploy/runpod.py`
- updated deploy config/examples

### 2. We were reading the public hostname wrong

Early assumptions treated the pod as if manager URLs should use public IP substitution.

Reality:

- RunPod exposes the HTTP service via proxy domain
- the manager is available at `https://<pod-id>-7870.proxy.runpod.net`

This changed both deploy behavior and returned session URLs.

### 3. Pod placement was availability-driven, not latency-aware

We initially let RunPod place wherever capacity existed.

That was wrong for this use case because latency matters.

Fix:

- added data-center preferences and GPU selection support
- configured California-first / US-west-first fallback
- later constrained deployment to `RTX 4090`

### 4. The original full image was too large for a healthy cold-start loop

The first full-image workflow was dominated by:

- push time
- pull time
- extract time
- pre-runtime wait time

That made destroy/recreate iteration too slow and too expensive.

This was the central operational problem of the RunPod effort.

### 5. CUDA compatibility on RunPod nodes broke one deploy

One pod failed before container start with a CUDA requirement mismatch.

Fix:

- lowered the base image from CUDA `12.8.1` to CUDA `12.6.3`

That restored compatibility with the available RunPod 4090 nodes.

### 6. `MODE=full` originally blocked manager startup behind heavy startup

The full container originally held the manager hostage behind heavy service startup.

Fix:

- changed the entrypoint so heavy services start in the background
- manager comes up immediately

This was necessary for usable manager health checks and session creation.

### 7. SAM3 had a real Hugging Face gated-access issue

The stack was not just failing randomly.

Actual problem:

- `facebook/sam3` is gated
- the environment lacked authorized HF credentials

Fix:

- added `HF_TOKEN` support through local and RunPod paths
- confirmed token-authenticated access

This unblocked SAM3.

### 8. SSH is only partially real right now

Current state:

- RunPod shows an SSH/TCP mapping for port `22`
- the container image does not actually install or run `sshd`

Consequence:

- direct TCP SSH to the pod IP/port is misleading and unreliable
- RunPod relay SSH may still work if key injection is correct, but that is a separate path

This means the current `22/tcp` exposure is not an honest contract.

### 9. Pocket-TTS is the last major startup tax

The current heavy-start flow can still spend a large amount of time bootstrapping Pocket-TTS on first run.

What we observed:

- Pocket-TTS runtime bootstrap prepared a large dependency set
- this consumed roughly `1m 44s` in one observed pod boot
- the existing warmup timeout was too short for that path

This is now one of the main remaining cold-start bottlenecks.

## Major Code/Deploy Improvements Landed

Relevant commits:

- `6ce5f29` `Improve RunPod deploy flow and slim heavy runtime`
- `b4ec87c` `Stabilize RunPod full-runtime deploy path`

Key changes made during this effort:

- RunPod registry-auth support in deploy tooling
- environment placeholder resolution for deploy config
- manager/proxy URL handling for RunPod
- California-first and 4090-aware placement controls
- manager-lite validation path
- local heavy smoke workflow
- improved heavy-start diagnostics and model-status reporting
- SAM3 fallback and native `sam3` path support
- HF-token wiring for local and RunPod
- CUDA base-image compatibility fix

## Current Live State

As of March 18, 2026, the tracked pod state is:

- pod id: `j60hw5i3ircwic`
- desired status: `RUNNING`
- cost per hour: `$0.59`
- public IP: `203.57.40.227`
- proxy manager URL: `https://j60hw5i3ircwic-7870.proxy.runpod.net`

Known-good image baseline:

- `ghcr.io/jloganolson/character-eng:runpod-full-slim-v9`

Known-good result on that baseline:

- manager healthy
- real remote session created
- remote browser session loaded
- external vision service healthy

## What We Learned

### 1. The platform path is proven

We are no longer asking whether RunPod can host this app at all.

It can.

The deploy path, manager path, session path, proxy path, and heavy vision path have all been proven.

### 2. The real problem is iteration economics

The remaining challenge is not basic correctness.

It is:

- cold-start time
- image size
- expensive destroy/recreate loops
- slow feedback on runtime issues

### 3. Local-first debugging is mandatory

Using RunPod as the first place to discover runtime failures was a bad loop.

The local heavy smoke path changed that.

We should continue to treat RunPod as confirmation, not primary exploration.

### 4. Warm-pod reuse matters more than another round of image shaving

Once the full path worked on RunPod, the next bottleneck stopped being code correctness.

It became repeated cold starts.

For this workload, reusing a warm pod is more important than chasing every last GB out of the image.

### 5. Pocket-TTS should follow the same separation pattern as local

Locally, Pocket and vision are not sharing one Python environment.

They work because they are isolated:

- app env for Pocket
- vision env for vLLM/SAM3/etc.

That separation is part of why local behavior is stable.

## Reflection On The Current Plan

The current broad plan still makes sense, but one part of it should change.

### What still makes sense

- keep the image-first RunPod architecture
- keep the manager-lite path for infrastructure validation
- keep local-first heavy debugging
- keep California/US-west and 4090-only placement

### What should change

We should stop treating cold-start optimization as the main immediate objective.

That is not the highest-value next step now that:

- the full pod already worked
- the browser flow already worked
- the shared vision service already worked

The current highest-value objective is fast, reliable iteration on the known-good full path.

### Recommended plan from here

1. Preserve `v9` as the known-good RunPod baseline.
2. Revert the incomplete CPU-Pocket experiment rather than compounding it.
3. Keep Pocket GPU-backed if local parity is the goal.
4. Mirror local separation on RunPod:
   - one env for app/Pocket
   - one env for vision
5. Prefer persistent volume reuse and warm-pod reuse over repeated full pod destruction.
6. Treat image rebuilds as deliberate milestones, not the default debug loop.
7. Decide explicitly whether container-level SSH is a real requirement:
   - if yes, implement real `sshd`
   - if no, remove `22/tcp`

## Best Next Step

The best next step is not another speculative image rewrite.

It is:

- cleanly restore the known-good baseline
- choose the intended Pocket strategy
- optimize the iteration loop around warm pod reuse and persistent runtime state

If local parity is the priority, the most coherent plan is:

- keep Pocket GPU-backed
- keep it isolated from the vision env
- reuse the mounted `/workspace` runtime state so that Pocket and model assets do not have to be rebuilt or re-fetched on every pod lifecycle event

That is the closest match to what already works locally, and it avoids turning the RunPod effort back into a cold-start science project.
