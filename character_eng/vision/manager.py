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
        self._presence_visible = False
        # Recent event dedup: normalized text -> timestamp
        self._recent_events: dict[str, float] = {}
        self._event_cooldown = 15.0  # seconds before a similar event can fire again
        # Rolling event history for synthesis context (raw text, last N events)
        self._event_history: collections.deque[str] = collections.deque(maxlen=10)
        # Dashboard integration
        self._collector = None
        self._last_dashboard_push = 0.0
        self._last_focus: VisualFocusResult | None = None
        self._paused = False

    def start(self, model_config: dict, world=None, people=None, collector=None) -> None:
        if self._running:
            return
        self._model_config = model_config
        self._world = world
        self._people = people
        if collector is not None:
            self._collector = collector
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
                event = self._events.popleft()
                events.append(event)
                self._event_history.append(event.description)
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

    def set_paused(self, paused: bool) -> None:
        self._paused = paused

    def update_focus(self, beat, stage_goal: str, thought: str, world, people, model_config: dict, scenario=None) -> None:
        """Run visual_focus_call and update vision service questions/targets."""
        try:
            result = visual_focus_call(beat, stage_goal, thought, world, people, model_config, scenario=scenario)
        except Exception:
            return
        self._last_focus = result

        try:
            self._client.set_questions(result.constant_questions, result.ephemeral_questions)
        except Exception:
            pass
        try:
            self._client.set_sam_targets(result.constant_sam_targets, result.ephemeral_sam_targets)
        except Exception:
            pass
        if self._collector is not None:
            self._collector.push("vision_focus", {
                "stage_goal": stage_goal,
                "thought": thought,
                "constant_questions": list(result.constant_questions),
                "ephemeral_questions": list(result.ephemeral_questions),
                "constant_sam_targets": list(result.constant_sam_targets),
                "ephemeral_sam_targets": list(result.ephemeral_sam_targets),
            })

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
        if self._paused:
            return
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
        # Build previous synthesis with full event history for better dedup
        prev = SynthesisResult(events=list(self._event_history))
        result = vision_synthesis_call(
            visual_context=self._context,
            world=self._world,
            people=self._people,
            beat=self._beat,
            stage_goal=self._stage_goal,
            model_config=self._model_config,
            previous_synthesis=prev,
        )

        # 3b. Push vision snapshot to dashboard (every 5s)
        now = time.time()
        if self._collector and now - self._last_dashboard_push > 5.0:
            self._last_dashboard_push = now
            self._collector.push("vision_snapshot", {
                "faces": len(snapshot.faces),
                "persons": len(snapshot.persons),
                "objects": len(snapshot.objects),
                "object_labels": [obj.label for obj in snapshot.objects],
                "vlm_answers": [
                    {
                        "question": answer.question,
                        "answer": answer.answer,
                        "slot_type": answer.slot_type,
                    }
                    for answer in snapshot.vlm_answers
                ] if snapshot.vlm_answers else [],
                "focus": {
                    "constant_questions": list(self._last_focus.constant_questions),
                    "ephemeral_questions": list(self._last_focus.ephemeral_questions),
                    "constant_sam_targets": list(self._last_focus.constant_sam_targets),
                    "ephemeral_sam_targets": list(self._last_focus.ephemeral_sam_targets),
                } if self._last_focus is not None else None,
            })

        # 4. Push events (with dedup)
        # Expire old entries
        self._recent_events = {k: v for k, v in self._recent_events.items()
                               if now - v < self._event_cooldown}
        for event_text in result.events:
            key = self._normalize_event(event_text)
            if key not in self._recent_events:
                trace = self._trace_payload(snapshot, kind="vision_synthesis", signals=list(result.signals))
                self._recent_events[key] = now
                self._events.append(PerceptionEvent(
                    description=event_text,
                    source="visual",
                    trace=trace,
                    kind="visual_claim",
                    payload={"signals": list(result.signals)},
                ))

    @staticmethod
    def _normalize_event(text: str) -> str:
        """Normalize event text for dedup. Strips articles, lowercases, collapses whitespace."""
        return " ".join(text.lower().strip().split())

    @staticmethod
    def _has_human_presence(snapshot: RawVisualSnapshot) -> bool:
        if snapshot.persons or snapshot.faces:
            return True
        return any((obj.label or "").strip().lower() == "person" for obj in snapshot.objects)

    @staticmethod
    def _visible_identity_candidates(snapshot: RawVisualSnapshot) -> list[str]:
        ids: list[str] = []
        for person in snapshot.persons:
            identity = str(person.identity or "").strip()
            if identity and identity not in ids:
                ids.append(identity)
        for face in snapshot.faces:
            identity = str(face.identity or "").strip()
            if identity and identity.lower() != "unknown" and identity not in ids:
                ids.append(identity)
        return ids

    def _resolve_people(self, snapshot: RawVisualSnapshot) -> None:
        """Map visual presence to PeopleState, emit stable presence-change events."""
        current_ids = self._visible_identity_candidates(snapshot)
        visible_now = self._has_human_presence(snapshot)

        if self._people is not None and visible_now:
            present_people = self._people.present_people()
            reusable_person_id = present_people[0].person_id if len(present_people) == 1 else None
            for vid in current_ids:
                person_id = self._visual_to_person.get(vid)
                if person_id is None:
                    if reusable_person_id is not None:
                        person_id = reusable_person_id
                    else:
                        person_id = self._people.get_or_create(vid)
                    self._visual_to_person[vid] = person_id
                person = self._people.people.get(person_id)
                if person is not None:
                    person.presence = "present"
                    if not person.name:
                        person.name = vid

        if self._people is not None and not visible_now:
            for person in self._people.present_people():
                person.presence = "gone"

        self._known_persons = set(current_ids)

        if visible_now and not self._presence_visible:
            identity = current_ids[0] if current_ids else ""
            person_id = self._visual_to_person.get(identity) if identity else None
            self._events.append(PerceptionEvent(
                description="A person appeared in view",
                source="visual",
                trace=self._trace_payload(snapshot, kind="person_presence", identity=identity),
                kind="person_presence",
                payload={
                    "identity": identity,
                    "person_id": person_id,
                    "presence": "present",
                    "change": "appeared",
                    "signals": ["person_visible"],
                },
            ))
        elif not visible_now and self._presence_visible:
            self._events.append(PerceptionEvent(
                description="A person is no longer visible",
                source="visual",
                trace=self._trace_payload(snapshot, kind="person_presence"),
                kind="person_presence",
                payload={
                    "identity": "",
                    "person_id": None,
                    "presence": "gone",
                    "change": "disappeared",
                    "signals": ["person_departed"],
                },
            ))

        self._presence_visible = visible_now

    def _trace_payload(self, snapshot: RawVisualSnapshot, **extra) -> dict:
        payload = {
            "faces": len(snapshot.faces),
            "persons": len(snapshot.persons),
            "objects": len(snapshot.objects),
            "object_labels": [obj.label for obj in snapshot.objects],
            "vlm_answers": [
                {
                    "question": answer.question,
                    "answer": answer.answer,
                    "slot_type": answer.slot_type,
                }
                for answer in (snapshot.vlm_answers or [])
            ],
            "focus": {
                "constant_questions": list(self._last_focus.constant_questions),
                "ephemeral_questions": list(self._last_focus.ephemeral_questions),
                "constant_sam_targets": list(self._last_focus.constant_sam_targets),
                "ephemeral_sam_targets": list(self._last_focus.ephemeral_sam_targets),
            } if self._last_focus is not None else None,
        }
        payload.update(extra)
        return payload
