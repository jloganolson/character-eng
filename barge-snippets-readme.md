# Barge-In Snippets

This is the workflow for capturing and re-testing false barge-ins and other interruption bugs.

## What To Capture

Use this when Greg:
- interrupts himself
- gets falsely barged in by his own output
- cuts off too early
- ignores a real interruption

Typical examples:
- STT fires on Greg's own speech
- AEC/echo suppression is not strong enough
- barge-in timing is too aggressive
- transcript-triggered interruption is wrong

## Capture Flow In The Dashboard

1. Find the bad moment in the debug timeline.
2. Click the interrupted `assistant_reply` or the `response_interrupted` event.
3. Click `Capture As Snippet`.
4. Leave `Barge-In Turn` selected when the event is an interruption case.
5. Give it a useful title.

Recommended title style:
- `false barge-in on nice scarf`
- `greg interrupts on speech-start before first audio`

Recommended tags:
- `barge-in`
- `false-barge-in` or `real-barge-in`
- `aec`
- `echo`
- `stt`
- `timing`

6. Save the snippet.

The snippet captures:
- checkpoint/context around the turn
- raw user audio clip
- raw assistant output audio clip
- nearby event timeline bundle
- derived barge-in timing metadata

## What The Snippet Stores

For `Barge-In Turn`, the snippet manifest includes:
- `speech_started_s`
- `transcript_final_s`
- `assistant_first_audio_s`
- `reason`
- `transcript_text`
- `assistant_text`
- `interrupted_text`
- `tags`

That makes it possible to re-run the current barge-in policy against the recorded timing without needing a full live session.

## Running The Offline Check

Run all saved barge-in snippets:

```bash
./.venv/bin/python -m character_eng.qa_barge_in --tag barge-in
```

Run only known false positives:

```bash
./.venv/bin/python -m character_eng.qa_barge_in --tag false-barge-in
```

Run only one policy mode:

```bash
./.venv/bin/python -m character_eng.qa_barge_in --tag barge-in --mode aec
./.venv/bin/python -m character_eng.qa_barge_in --tag barge-in --mode legacy
```

## How To Read The Output

Each line reports:
- mode: `aec` or `legacy`
- status: `ok`, `mismatch`, or `n/a`
- snippet title
- expected behavior
- actual behavior
- interruption reason
- transcript text
- user/assistant audio clip duration

Example:

```text
aec | ok | false barge-in on nice scarf | expected continue | actual continue | reason barge_in | transcript oh hey man | user_audio 1.20s | assistant_audio 2.80s
```

If a snippet is tagged:
- `false-barge-in` -> expected result is `continue`
- `real-barge-in` or `expected-barge-in` -> expected result is `cancel`

The runner exits nonzero if any snippet with an explicit expectation mismatches.

## Important Limitation

This first pass does not re-run Deepgram end to end from the raw audio.

It replays the recorded timing/events into the current `VoiceIO` barge-in logic, which is still useful because that is the main decision layer we control in code.

The raw mic clip and assistant-output clip are still preserved, so we can later build a deeper STT/AEC round-trip harness on top of the same snippets.

## Suggested Dev Loop

1. Capture a few false positives.
2. Tag them consistently.
3. Change barge-in logic.
4. Run:

```bash
./.venv/bin/python -m character_eng.qa_barge_in --tag false-barge-in
```

5. Check whether mismatches dropped.
6. Re-test live once the offline cases look good.
