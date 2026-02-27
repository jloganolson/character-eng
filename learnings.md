# Learnings

Hard-won knowledge from development. Things that took real debugging time or weren't obvious from docs.

## ReSpeaker XVF3800 AEC Integration (2026-02-21)

### The core insight

The XVF3800's acoustic echo cancellation only works when TTS audio is routed **through the ReSpeaker's own output**. The DSP uses its output as the reference signal to subtract from the mic input. If audio goes directly to speakers (e.g. via USB to Audioengine), the ReSpeaker has no reference and picks up the full echo.

- **Through ReSpeaker out**: mic records ~-71 dB (near silence)
- **Through speakers directly**: mic records ~-30 to -40 dB (clear echo)
- **Delta**: 30-40 dB of cancellation, confirmed across multiple samples and voices

### Config that works

```toml
[voice]
input_device = "reSpeaker"
output_device = "reSpeaker"       # THIS IS THE KEY - routes TTS through ReSpeaker for AEC reference
mic_mute_during_playback = false  # hardware AEC handles echo, no need for software mute
```

### Device enumeration is unstable

USB device indices change across reboots, suspend/resume cycles, and when devices are plugged/unplugged. Integer device IDs in config break silently. Solution: name-based matching (`"reSpeaker"` instead of `14`), resolved at runtime via `resolve_device()` in `voice.py`.

### SpeakerStream sample rate fallback

The ReSpeaker only supports 16kHz output. ElevenLabs and the `SpeakerStream` default to 24kHz PCM. The fallback chain matters:

- **Bad order** (24kHz -> 48kHz -> device default): PortAudio prints noisy C-level error spam to stderr for each failed rate probe, even though the fallback works correctly.
- **Good order** (device default -> 24kHz -> 48kHz): Hits 16kHz on the first try, no noise.
- The stderr noise is from PortAudio's C layer, not Python — suppressing it requires fd-level redirect (`os.dup2` to `/dev/null` during the probe).

### Validation approach

We built a standalone test suite (`experiments/aec/`) that doesn't need a human in the loop:

1. **dB comparison** (`aec_db_test.py`): Play WAVs through both paths, record from mic, compare RMS levels. Clear 30-40 dB delta = AEC working.
2. **VAD test** (`aec_vad_test.py`): Play segments through alternating paths, run Silero VAD on the recording. Speech should only be detected during no-AEC segments. Uses 512-sample windows at 16kHz (Silero's required chunk size — not 512ms, which was our first bug).
3. **Integration test** (`integration_test.py`): Runs TTS through the real `VoiceIO` pipeline, records from mic, checks VAD. Tests the full stack end-to-end.

### Silero VAD gotcha

Silero VAD requires **exactly 512 samples per chunk at 16kHz** (32ms windows). The window size is in samples, not milliseconds. Passing 512ms worth of samples (8192) crashes with a cryptic TorchScript error about "Provided number of samples is 8192 (Supported values: 256 for 8000 sample rate, 512 for 16000)".

### Post-suspend AEC behavior

After a laptop suspend/resume cycle, the AEC still works but RMS levels are slightly higher (~-43 dB vs ~-71 dB in a fresh session). VAD still doesn't detect speech — the higher RMS is ambient noise, not echo leakage. The AEC DSP may need a few seconds to recalibrate after wake.

---

## Downshifting Background LLM Calls from 70B to 8B (2026-02-26)

### The problem

The system has four background LLM calls (eval, plan, reconcile, director) that all run on 70B in fire-and-forget threads. Results only surface at the next turn boundary. This creates:

- **2-turn delay** on eval→plan cascades (eval detects off-book → plan fires → plan result applied → character finally adapts)
- **Rate-limit contention** from 3 simultaneous 70B calls after every turn
- **Stale results** that land after the conversation has moved on
- **Race conditions** from the version-based discard system (results silently dropped)

The `expression_call` pattern (8B, sync, minimal context, overlaps TTS) works great. Question: can we push more calls to that pattern?

### What we tested

Three A/B experiments in `experiments/fast-llm/`. Each feeds the same scenarios through 8B (Cerebras Llama 3.1 8B) and 70B (Groq Llama 3.3 70B) and compares accuracy + latency.

### Finding 1: Director works at 8B (92% accuracy)

The director's job is "does the conversation match any of these exit conditions?" — it's structurally identical to `condition_check_call`, which already runs on 8B.

- 8B: 11/12 correct (92%), avg 438ms
- 70B: 12/12 correct (100%), avg 448ms
- The one 8B miss was a borderline case (picked the right action "advance" but wrong exit index)

**Takeaway:** Director can move to 8B sync as a post-response microservice. No background thread needed.

### Finding 2: Eval fails at 8B with the original prompt (25% accuracy)

The eval prompt asks for two things at once: inner monologue + script status classification. The 8B model generates plausible-sounding monologue but completely ignores the script tracking.

- eval-8b: 3/12 correct (25%) — returned "hold" for almost everything
- eval-70b: 8/12 correct (67%)

**Root cause:** Creative generation drowns out classification. The 8B model "fills the slot" with monologue and defaults to "hold" for the status.

### Finding 3: Splitting eval into focused calls recovers most of the accuracy

A stripped-down "script_check" prompt that only asks "did the conversation accomplish this beat's intent?" — no monologue, no world state preamble, just the beat intent + recent conversation:

- eval-8b (original): 3/12 (25%)
- check-8b (focused): 7/12 (58%)
- eval-70b (baseline): 8/12 (67%)

The focused prompt nearly doubles 8B accuracy and approaches 70B.

### The general principle

8B handles **focused classification with explicit criteria**:
- "Does conversation match this exit condition?" → yes (director, condition_check)
- "What gaze target and expression fit this line?" → yes (expression_call)
- "Did the conversation accomplish this beat's intent?" → mostly yes (script_check)

8B fails when **bundling creative generation with classification**:
- "Write inner monologue AND classify script status" → no (eval_call)

### Remaining weaknesses in script_check at 8B

1. **"No" answers still count as accomplished.** Beat intent "ask if they want water", user says "no thanks" — the question was asked (advance), but 8B reads "no" as rejection (hold). Both 8B and 70B struggle here. Likely needs a prompt hint: "If the beat's intent was to ASK a question, receiving any answer (yes or no) means the intent is accomplished."

2. **Off-book detection needs explicit rejection language.** "I don't really care about lemonade" should trigger off_book, but the 8B check-prompt missed it. The focused prompt stripped too much context — it lost the signal that the user is dismissing the topic vs. just not engaging with it yet. May need to include the beat's remaining context (next 1-2 beats) so the model can see the user is rejecting the *direction*, not just one beat.

3. **World events in system messages confuse all models.** `[A person walks up to the stand]` as a system message isn't clearly parsed as a world event by either model tier. The narrator bracket format may need a more explicit label.

4. **One false advance on a hold case.** The focused prompt lost the signal that conversation was still mid-topic. Including a one-line conversation summary or "turns since beat started" counter could help.

### Proposed next steps

**High confidence (do first):**
1. Move director to 8B sync — data strongly supports it. Run as microservice after each response, like expression_call. Eliminates one background thread entirely.
2. Split eval into `thought_call` (8B, inner monologue only) + `script_check_call` (8B, classification only). Run both synchronously after each response.

**Medium confidence (test more):**
3. Refine script_check prompt:
   - Add hint about "asking a question = accomplished when answered"
   - Include next 1-2 beat intents so the model can detect divergence (off_book)
   - Add conversation turn count since beat started
   - Test with 20+ scenarios to get statistical significance
4. Test reconcile at 8B — hardest call to downshift because of structured JSON output with fact IDs, but worth trying with few-shot examples in the prompt.

**Lower confidence (experiment later):**
5. Beat-at-a-time planning: generate one beat with 8B after each advance instead of a multi-beat script with 70B. Would eliminate the plan background thread entirely but quality is unknown.
6. Combine expression + script_check + director into a single 8B "post-response" call with a multi-output JSON schema. One call, three classifications, ~200ms total.

### Test infrastructure

All experiments live in `experiments/fast-llm/`. Each script:
- Defines test scenarios as dataclasses with expected outcomes
- Runs each scenario through multiple model variants
- Prints per-scenario results to stdout
- Saves JSON + HTML report to `experiments/fast-llm/results/`
- HTML report has color-coded badges (green=correct, red=wrong) and latency

Run: `uv run python experiments/fast-llm/eval_split_ab.py --open`
