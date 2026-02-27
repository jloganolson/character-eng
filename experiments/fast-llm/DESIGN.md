# Fast LLM Microservices — System Design & Experiments

## Current LLM System Map

### Model Tiers

| Tier | Key | Model | Provider | Role |
|------|-----|-------|----------|------|
| **CHAT** | `cerebras-llama` | Llama 3.1 8B | Cerebras (→ Groq fallback) | Streaming dialogue, expression, condition |
| **BIG** | `groq-llama` | Llama 3.3 70B | Groq | Eval, plan, reconcile, director |

### All LLM Calls

| # | Call | Model | Sync/Async | Trigger | Output | Latency |
|---|------|-------|------------|---------|--------|---------|
| 1 | `ChatSession.send()` | 8B | **Sync streaming** | Every user msg | Dialogue text | ~100-300ms TTFT |
| 2 | `expression_call()` | 8B | **Sync** (overlaps TTS) | After every response | gaze + emotion | ~50-150ms |
| 3 | `condition_check_call()` | 8B | **Sync** (gates /beat) | /beat with condition | met + idle line | ~100-200ms |
| 4 | `eval_call()` | 70B | **Async bg thread** | After every turn | thought + advance/hold/off_book | ~500-2000ms |
| 5 | `plan_call()` | 70B | **Both** | Boot/stage sync, eval-triggered async | List of beats | ~1-5s |
| 6 | `reconcile_call()` | 70B | **Async bg thread** | /world, /see, sim | Fact mutations | ~500-2000ms |
| 7 | `director_call()` | 70B | **Async bg thread** | After every turn | hold/advance + exit_index | ~500-2000ms |

### Turn Lifecycle

```
User input arrives
  ├─ _check_reconcile()   ← apply stale bg result (always applies, no version check)
  ├─ _check_eval()        ← apply if version matches, may trigger plan
  ├─ _check_director()    ← apply if version matches, may trigger plan
  ├─ _check_plan()        ← apply if version matches
  │
  ├─ stream_response()    ← 8B sync (user sees this)
  ├─ expression_call()    ← 8B sync (overlaps TTS in voice mode)
  │
  ├─ _bump_version()
  ├─ _start_eval()        ← 70B fire-and-forget bg thread
  └─ _start_director()    ← 70B fire-and-forget bg thread
```

### Threading Model

Four single-slot thread pools. Each has:
- A `threading.Lock` protecting a result global
- A `threading.Thread` tracking the in-flight call
- Version-based stale discard (eval, plan, director — **not** reconcile)

Results **only surface at the start of the next turn**. `_context_version` increments on user messages, /world, /see, /beat.

### What Works Well: The Expression Pattern

`expression_call` is the model for fast microservices:
- **Small model** (8B) → fast (~50-150ms)
- **Minimal context** (just the last dialogue line) → fast
- **Focused output** (gaze + emotion, 2-3 fields) → reliable
- **Overlaps TTS** → zero perceived cost in voice mode
- **No background thread** → no race conditions, no stale discard

## Problems With the Current Big-Model Background System

### 1. Delayed effect (the foot gun)

Background results only take effect at the **next turn boundary**. If the user sends 3 rapid messages, earlier eval/plan/director results get version-discarded. If the user pauses, a slow result arrives and gets applied to a context that's already moved on.

### 2. Eval → Plan cascade = 2-turn delay

```
Turn N: eval fires in background
Turn N+1: eval result checked → "off_book" → plan fires in background
Turn N+2: plan result checked → new script applied
Turn N+3: character finally uses new script
```

The character is flying blind for 2-3 turns after going off-book.

### 3. Three 70B calls compete for quota

After each turn: eval + director + (sometimes reconcile) all hit the Groq 70B endpoint simultaneously. Rate limits mean they slow each other down.

### 4. Sync plan at boot/stage = biggest latency spike

`plan_call()` blocks for 1-5s at boot and stage transitions. The user stares at nothing.

### 5. No stale discard on reconcile

Minor — reconcile always applies. If world state changed significantly since the reconcile started, the result could be slightly wrong.

## Proposal: Fast LLM Microservices

Decompose the 70B background calls into smaller 8B calls where the decision is tractable. Keep 70B only for tasks that genuinely need it (multi-beat creative planning, complex fact reconciliation).

### The key insight

Most of the "big" calls are really answering **simple classification questions**:

| Call | Core question | Tractable at 8B? |
|------|--------------|-------------------|
| eval | "Did the last exchange satisfy this beat's intent?" | **Yes** — it's a yes/no |
| director | "Does the conversation match any of these exit conditions?" | **Yes** — it's a yes/no per condition |
| reconcile | "Merge these text changes into structured facts with IDs" | **Maybe** — structured output is harder |
| plan | "Generate 1-6 creative dialogue beats" | **No** — genuinely needs reasoning |

### Target architecture

```
User input arrives
  ├─ stream_response()       ← 8B sync (user sees this)
  │
  ├─ expression_call()       ← 8B sync "microservice" (overlaps TTS)
  ├─ fast_eval_call()        ← 8B sync "microservice" (after response)
  ├─ fast_director_call()    ← 8B sync "microservice" (after response)
  │
  ├─ [if eval says advance & script empty] → plan_call() 70B async
  └─ [if world changed] → reconcile_call() 70B async
```

Benefits:
- Eval and director results are **immediate** — no 2-turn delay
- Only plan and reconcile run in the background (genuinely slow tasks)
- Fewer 70B calls → less rate-limit contention
- Simpler threading — two bg slots instead of four

## Experiments

### Experiment 1: Fast Eval (`eval_ab.py`)

**Hypothesis:** 8B can match 70B on the advance/hold/off_book decision.

Method: Build test scenarios with known-correct eval outcomes. Run each through both 8B and 70B. Compare agreement rate and latency.

Input types (matching real system inputs):
- Dialogue exchanges (user said X, character said Y)
- World events ("A customer approaches the stand")
- Yes/no answers to proactive questions ("yes" / "no")

### Experiment 2: Fast Director (`director_ab.py`)

**Hypothesis:** 8B can match 70B on "does this conversation match exit condition X?"

Method: Build test scenarios with known stage + exit conditions. Run through both models. Compare.

This is essentially the same decision as `condition_check_call` (which already uses 8B successfully).

### Experiment 3: Beat-at-a-Time Planning (`plan_single_ab.py`)

**Hypothesis:** Instead of generating 1-6 beats at once (70B), generate one beat at a time with 8B after each advance.

Method: Compare the quality of single-beat 8B plans vs multi-beat 70B plans by running both on the same conversation states and judging beat quality.

### Experiment 4: Fast Reconcile (`reconcile_ab.py`)

**Hypothesis:** 8B can handle fact merging for simple world changes.

Method: Run the same pending changes through both models. Compare structured output quality (correct IDs, sensible facts, no hallucinated removals).

## Results

### Experiment 1: Eval A/B (Run 1)

| Metric | 8B (Cerebras) | 70B (Groq) |
|--------|---------------|------------|
| Correct | **3/10 (30%)** | **6/10 (60%)** |
| Avg latency | 327ms | 676ms |
| Speedup | 2.1x faster | — |
| Agreement | 7/10 (70%) |

**Key finding:** 8B is heavily biased toward "hold" — it doesn't engage with the script tracking task. It generates vaguely relevant inner monologue but doesn't actually check if the current beat's intent was satisfied. Even 70B only gets 6/10, missing several "advance" cases where the beat was clearly completed.

**Analysis of failures:**
- 8B returned "hold" for almost every scenario (7/10), only getting "advance" for the most obvious case (user said "Yes please!" to "want water?")
- 70B correctly detected "off_book" cases (user rejects/redirects) but also missed some "advance" cases
- Both models failed on `advance_direct_match` where the character literally offered a flier (the beat's intent) — suggests the eval prompt's context is too heavy for either model to focus on the actual script check

**Conclusion:** Naively running the same eval prompt on 8B doesn't work. The eval task is too complex as-is (inner monologue + script tracking in one call). **Next step: split eval into two focused calls** — one for inner monologue (8B), one for script classification (8B with a stripped-down prompt that just asks "did the beat's intent get accomplished?").

### Experiment 2: Director A/B (Run 1)

| Metric | 8B (Cerebras) | 70B (Groq) |
|--------|---------------|------------|
| Correct | **11/12 (92%)** | **12/12 (100%)** |
| Avg latency | 438ms | 448ms |
| Agreement | 11/12 (92%) |

**Key finding:** 8B matches 70B nearly perfectly on the director task. The only disagreement was a borderline case (`watching_someone_walks_past` where 8B picked exit[1] instead of exit[0] — both "advance" results, just different exit selection when the scenario is ambiguous).

**Run 2 variability:**
- 8B: 9/12 (75%), 70B: 10/12 (83%)
- Non-deterministic, with some latency spikes on Cerebras (cold start)
- 70B also made mistakes in run 2 that it got right in run 1

**Conclusion:** Director is a strong candidate for 8B. It's essentially the same task as `condition_check_call` (which already uses 8B). The exit conditions are explicit text, and the model just needs to pattern-match against the conversation. Consider running director synchronously as a fast microservice post-response.

### Experiment 3: Split Eval A/B (Run 1)

Comparing three approaches: original eval on 8B, focused script_check on 8B, original eval on 70B.

| Metric | eval-8b | check-8b | eval-70b |
|--------|---------|----------|----------|
| Correct | **3/12 (25%)** | **7/12 (58%)** | **8/12 (67%)** |
| Avg latency | 3077ms* | 3790ms* | 588ms |

*Cerebras latency was elevated this run (rate limiting from parallel experiments). Typical Cerebras 8B latency is 200-500ms.

**Key finding:** The focused script_check prompt nearly doubles 8B accuracy (25% → 58%) and approaches 70B (67%). Stripping out the monologue generation lets 8B focus on the classification.

**Breakdown by category:**
- Advances: check-8b got 3/6 vs eval-8b's 0/6 (70B got 4/6)
- Holds: check-8b got 2/3 vs eval-8b's 3/3 (70B got 3/3) — one false advance
- Off-book: check-8b got 2/3 vs eval-8b's 0/3 (70B got 2/3)

**Remaining failures:**
- `advance_world_event`: All three said "hold" — the greeting was delivered but the system message format confused all models
- `advance_yes_answer`: check-8b said "off_book" (user said "yes please" to water question — should advance)
- `advance_no_answer`: Both 8B variants and 70B said "hold" — genuinely hard case (question was asked, answer received, but answer was "no")
- `hold_beat_not_yet`: check-8b false-advanced — the focused prompt lost the context that the conversation is still mid-topic
- `off_book_topic_change`: check-8b said "hold" — didn't pick up on "I don't really care about lemonade" as a redirect

**Conclusion:** The split approach works. check-8b is competitive with eval-70b, especially on the advance/off_book decisions that matter most for responsiveness. The remaining failures suggest further prompt refinement (especially around the "no answer = still accomplished" pattern) could close the gap. Next steps: refine the script_check prompt and test with more scenarios.

### Experiments 4-8 (Pending — scripts built, not yet run)

| # | Script | What it tests |
|---|--------|---------------|
| 4 | `script_check_refine_ab.py` | 3 prompt variants (v1 baseline, v2 hints, v3 upcoming-beat context) on 20 scenarios targeting known failure modes |
| 5 | `reconcile_ab.py` | 8B vs 70B on structured world-state reconciliation (8 scenarios, weighted scoring on removals/additions/events/people) |
| 6 | `plan_single_ab.py` | Single-beat 8B planning vs multi-beat 70B, quality judged by 70B (6 scenarios) |
| 7 | `combined_ab.py` | Merging expression + script_check + director into one 8B call (8 scenarios, per-field agreement + correctness) |
| 8 | `parallel_bench.py` | Sequential vs concurrent execution of post-response microservices (latency benchmark, 5 runs per mode) |

## Running

```bash
uv run python experiments/fast-llm/eval_ab.py                    # Exp 1: Eval A/B
uv run python experiments/fast-llm/director_ab.py                 # Exp 2: Director A/B
uv run python experiments/fast-llm/eval_split_ab.py               # Exp 3: Split Eval A/B
uv run python experiments/fast-llm/script_check_refine_ab.py      # Exp 4: Script Check Refinement
uv run python experiments/fast-llm/reconcile_ab.py                # Exp 5: Reconcile A/B
uv run python experiments/fast-llm/plan_single_ab.py              # Exp 6: Single-Beat Planning
uv run python experiments/fast-llm/combined_ab.py                 # Exp 7: Combined Post-Response
uv run python experiments/fast-llm/parallel_bench.py              # Exp 8: Parallel Benchmark
```

All scripts accept `--open` to launch the HTML report in a browser. Some accept `--model` to override the default model.
