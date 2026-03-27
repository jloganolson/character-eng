# QA TODO

Use this once you are back at the computer and can do a real interactive pass.

## Before the live run

1. Run the offline pack first:
   - `./scripts/robot_sim_eval.sh`
2. Start the runtime with the dashboard the same way you normally do for Greg.
3. Keep the Robot Sim panel visible in `column` mode first, then switch to `fullscreen` once the basics look right.

## Panel and layout checks

1. Toggle the panel through `Min`, `Col`, and `Max`.
2. In `column` mode, make sure it docks cleanly in the sidebar and does not block resizing or inspector clicks.
3. In `fullscreen` mode, make sure it feels like a user-facing view rather than a debug overlay.
4. Refresh the page once and confirm the panel restores its last mode and repopulates state.

## Face and gaze checks

1. Have Greg produce a few different tones:
   - neutral / focused
   - curious
   - happy or excited
   - surprised or suspicious if you can get them naturally
2. Confirm the displayed face state matches the behavior event labels and feels directionally right.
3. Check both `hold` and `glance` gaze behavior.
4. Make sure gaze direction feels believable in the panel and does not jitter between polls.

## Wave checks

1. Walk up or greet Greg and see whether a wave happens at a natural social beat.
2. Verify the wave reads clearly in both `column` and `fullscreen`.
3. Make sure repeated greetings do not cause constant spammy waving.
4. Trigger a manual wave from the panel once to confirm the control path still works.

## Fist-bump checks

1. Introduce yourself by name and watch for an automatic fist-bump offer.
2. Try an intro plus a task pivot:
   - `"Hi, I'm Logan. Do you know what time it is?"`
3. Confirm the arm extension is readable and the panel enables `Bump now`, `Timeout`, and `Cancel` only when appropriate.
4. Test all three outcomes:
   - successful bump
   - timeout
   - cancel
5. Confirm the robot retracts cleanly after each path and returns to idle.

## Negative checks

1. Ask plain factual questions and make sure Greg stays on `none`.
2. Let Greg see you without speaking and confirm that a wave can happen from the visual beat alone.
3. While a fist bump is already active, make sure a second offer does not stack on top of it.

## What to note down

- moments where the chosen action felt socially wrong
- moments where the action was right but the visual readability felt off
- any expression names that do not map well to the eventual graphical direction
- any layout glitches when switching between `column` and `fullscreen`
- whether the current panel is good enough as a temporary user-view proxy

## If something looks wrong

1. Save the rough scenario that triggered it.
2. Re-run the closest canned case in the sim harness.
3. If needed, add a new canned case to `character_eng/sim_eval.py` and a regression to `tests/test_sim_eval.py` before changing prompts or frontend code.
