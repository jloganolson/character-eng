"""VisionManager — orchestrates polling, synthesis, focus, and people resolution."""

from __future__ import annotations

import collections
import re
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

    def set_paused(self, paused: bool) -> None:
        self._paused = paused

    def update_focus(self, beat, stage_goal: str, thought: str, world, people, model_config: dict) -> None:
        """Run visual_focus_call and update vision service questions/targets."""
        try:
            result = visual_focus_call(beat, stage_goal, thought, world, people, model_config)
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
                trace = self._trace_payload(snapshot, kind="vision_synthesis")
                evidence = self._classify_event_evidence(snapshot, event_text)
                trace.update(evidence)
                if trace["contradicting_answers"] and not trace["supporting_answers"]:
                    continue
                self._recent_events[key] = now
                self._event_history.append(event_text)
                self._events.append(PerceptionEvent(
                    description=event_text,
                    source="visual",
                    trace=trace,
                ))

    @staticmethod
    def _normalize_event(text: str) -> str:
        """Normalize event text for dedup. Strips articles, lowercases, collapses whitespace."""
        t = text.lower().strip()
        t = re.sub(r'\b(a|an|the|is|are|now|still|just|has|have)\b', '', t)
        t = re.sub(r'\s+', ' ', t).strip()
        return t

    @staticmethod
    def _answer_polarity(answer_text: str) -> str:
        text = (answer_text or "").strip().lower()
        if not text:
            return "unknown"
        negative_patterns = [
            r"\bno\b",
            r"\bnot\b",
            r"\bdoes not\b",
            r"\bdo not\b",
            r"\bthere (?:does|do) not appear\b",
            r"\bthere is no\b",
            r"\bthere are no\b",
            r"\bcannot be determined\b",
            r"\bimpossible to\b",
            r"\bnot possible to determine\b",
        ]
        if any(re.search(pattern, text) for pattern in negative_patterns):
            return "no"
        positive_patterns = [
            r"\byes\b",
            r"\bthere is\b",
            r"\bthere are\b",
            r"\bone person\b",
            r"\ba person\b",
            r"\bsomeone is\b",
            r"\bsomeone appears to be\b",
        ]
        if any(re.search(pattern, text) for pattern in positive_patterns):
            return "yes"
        return "unknown"

    @staticmethod
    def _event_keywords(text: str) -> set[str]:
        lowered = re.sub(r"[^a-z0-9\s]+", " ", (text or "").lower())
        words = {
            word.rstrip("s")
            for word in lowered.split()
            if len(word) >= 4 and word not in {
                "there", "with", "from", "that", "this", "into", "near",
                "someone", "person", "people", "stand", "greg", "scene",
            }
        }
        return words

    @classmethod
    def _question_matches_event(cls, question_text: str, event_text: str) -> bool:
        question_keywords = cls._event_keywords(question_text)
        event_keywords = cls._event_keywords(event_text)
        if not question_keywords or not event_keywords:
            return False
        overlap = question_keywords & event_keywords
        if overlap:
            return True
        # Handle small wording shifts like "approaching" vs "approaches".
        softened = {word.rstrip("ing").rstrip("ed") for word in question_keywords}
        event_softened = {word.rstrip("ing").rstrip("ed") for word in event_keywords}
        return bool(softened & event_softened)

    @classmethod
    def _classify_event_evidence(cls, snapshot: RawVisualSnapshot, event_text: str) -> dict:
        supporting_answers = []
        contradicting_answers = []
        for answer in snapshot.vlm_answers or []:
            if not cls._question_matches_event(answer.question, event_text):
                continue
            answer_payload = {
                "question": answer.question,
                "answer": answer.answer,
                "slot_type": answer.slot_type,
            }
            polarity = cls._answer_polarity(answer.answer)
            if polarity == "yes":
                supporting_answers.append(answer_payload)
            elif polarity == "no":
                contradicting_answers.append(answer_payload)
        return {
            "supporting_answers": supporting_answers,
            "contradicting_answers": contradicting_answers,
        }

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
                    trace=self._trace_payload(snapshot, kind="person_presence", identity=vid),
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
                trace=self._trace_payload(snapshot, kind="person_presence", identity=vid),
            ))

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
