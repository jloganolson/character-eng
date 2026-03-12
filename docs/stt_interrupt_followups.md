# STT Interrupt Follow-Ups

## What Changed

- Pre-audio user interruptions now trigger the keep-listening path before assistant audio starts.
- Deferred user fragments can be merged into the next accepted utterance.
- Queued stale transcript fragments already covered by the merged utterance are pruned.
- Aborted partial assistant replies no longer linger as ghost cards in the live dashboard.
- Tiny interrupted assistant fragments no longer pollute conversation history.

## What We Did Not Change

- Deepgram endpoint timing was intentionally left alone.
- `endpointing` and `utterance_end_ms` remain tuned for responsiveness.

## Current Read

- The main remaining uncertainty is not threshold tuning; it is event ordering in real interactions.
- The likely failure classes are:
  - one utterance split into two transcript turns,
  - stale queued transcript text surviving after a resumed utterance,
  - continue-listening arbitration racing with partial assistant output.

## Recommended Next Step

- If this topic is revisited, prefer better instrumentation over more behavioral changes.
- The most useful events to capture together are:
  - `SpeechStarted`
  - `Results.is_final`
  - `Results.speech_final`
  - `UtteranceEnd`
  - pre-audio cancel
  - continue-listening decision
  - deferred-prefix merge
  - stale-transcript prune
  - final `turn_start`

## Regression Coverage Added

- Voice queue/deferred merge coverage in `tests/test_voice.py`
- Runtime guardrail/history handling in `tests/test_runtime_controls.py` and `tests/test_chat.py`
- Live dashboard orphan-reply cleanup in `tests/test_dashboard_playwright.py`
