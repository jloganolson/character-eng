"""VisionManager — orchestrates polling, synthesis, focus, and people resolution."""

from __future__ import annotations

import collections
import threading
import time

from character_eng.perception import PerceptionEvent
from character_eng.vision.client import VisionClient
from character_eng.vision.context import RawVisualSnapshot, VisualContext
from character_eng.vision.synthesis import SynthesisResult, vision_synthesis_call
from character_eng.vision.focus import VisualFocusResult, visual_focus_call


class VisionManager:
    """Orchestrates vision polling, synthesis, focus planning, and people resolution.

    One background thread runs a continuous loop: poll /snapshot, run synthesis,
    push events. Min 750ms between cycles.
    """

    def __init__(self, service_url: str = "http://localhost:7860", min_interval: float = 0.75):
        self._client = VisionClient(service_url)
        self._context = VisualContext()
        self._min_interval = min_interval
        self._events: collections.deque[PerceptionEvent] = collections.deque(maxlen=50)
        self._running = False
        self._thread: threading.Thread | None = None
        self._prev_synthesis: SynthesisResult | None = None
        self._model_config: dict | None = None
        self._world = None
        self._people = None
        self._beat = None
        self._stage_goal = ""
        # People resolver state: visual identity -> people_state person_id
        self._visual_to_person: dict[str, str] = {}
        self._known_persons: set[str] = set()  # visual identities we've seen

    def start(self, model_config: dict, world=None, people=None) -> None:
        if self._running:
            return
        self._model_config = model_config
        self._world = world
        self._people = people
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None

    def drain_events(self) -> list[PerceptionEvent]:
        events = []
        while self._events:
            try:
                events.append(self._events.popleft())
            except IndexError:
                break
        return events

    def update_context(self, world=None, people=None, beat=None, stage_goal: str | None = None) -> None:
        if world is not None:
            self._world = world
        if people is not None:
            self._people = people
        if beat is not None:
            self._beat = beat
        if stage_goal is not None:
            self._stage_goal = stage_goal

    def update_focus(self, beat, stage_goal: str, thought: str, world, people, model_config: dict) -> None:
        """Run visual_focus_call and update vision service questions/targets."""
        try:
            result = visual_focus_call(beat, stage_goal, thought, world, people, model_config)
        except Exception:
            return

        try:
            self._client.set_questions(result.constant_questions, result.ephemeral_questions)
        except Exception:
            pass
        try:
            self._client.set_sam_targets(result.constant_sam_targets, result.ephemeral_sam_targets)
        except Exception:
            pass

    def get_gaze_targets(self) -> list[str]:
        targets = self._context.get_gaze_targets()
        return targets if targets else None

    @property
    def context(self) -> VisualContext:
        return self._context

    def _loop(self) -> None:
        while self._running:
            t0 = time.perf_counter()
            try:
                self._tick()
            except Exception as e:
                print(f"[vision] loop error: {e}", flush=True)
            elapsed = time.perf_counter() - t0
            wait = max(0, self._min_interval - elapsed)
            if wait > 0:
                time.sleep(wait)

    def _tick(self) -> None:
        # 1. Poll snapshot
        try:
            snapshot = self._client.snapshot()
        except Exception:
            return
        self._context.update(snapshot)

        # 2. People resolution: emit presence-change events
        self._resolve_people(snapshot)

        # 3. Vision synthesis
        if self._model_config is None:
            return
        result = vision_synthesis_call(
            visual_context=self._context,
            world=self._world,
            people=self._people,
            beat=self._beat,
            stage_goal=self._stage_goal,
            model_config=self._model_config,
            previous_synthesis=self._prev_synthesis,
        )
        self._prev_synthesis = result

        # 4. Push events
        for event_text in result.events:
            self._events.append(PerceptionEvent(description=event_text, source="visual"))

    def _resolve_people(self, snapshot: RawVisualSnapshot) -> None:
        """Map visual identities to PeopleState, emit presence-change events."""
        current_ids = set()
        for p in snapshot.persons:
            current_ids.add(p.identity)

        # New arrivals
        for vid in current_ids:
            if vid not in self._known_persons:
                self._known_persons.add(vid)
                self._events.append(PerceptionEvent(
                    description=f"{vid} appeared in view",
                    source="visual",
                ))
                # Map to people state
                if self._people is not None:
                    pid = self._people.get_or_create(vid)
                    self._visual_to_person[vid] = pid

        # Departures (only if previously known)
        disappeared = self._known_persons - current_ids
        for vid in disappeared:
            self._known_persons.discard(vid)
            self._events.append(PerceptionEvent(
                description=f"{vid} is no longer visible",
                source="visual",
            ))
