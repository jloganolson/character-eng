"""VisionManager — orchestrates polling, synthesis, focus, and people resolution."""

from __future__ import annotations

import collections
import threading
import time

from character_eng.perception import PerceptionEvent
from character_eng.scenario import VisionTrigger
from character_eng.vision.client import VisionClient
from character_eng.vision.context import RawVisualSnapshot, VisualContext
from character_eng.vision.interpret import vision_state_update_call
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
        self._scenario = None
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
        self._emitted_vlm_answer_ids: set[str] = set()
        self._trigger_streaks: dict[str, int] = {}
        self._active_trigger_signals: set[str] = set()
        self._system_task_signatures: dict[str, str] = {}

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

    def update_context(self, world=None, people=None, beat=None, stage_goal: str | None = None, scenario=None) -> None:
        if world is not None:
            self._world = world
        if people is not None:
            self._people = people
        if beat is not None:
            self._beat = beat
        if stage_goal is not None:
            self._stage_goal = stage_goal
        if scenario is not None:
            self._scenario = scenario

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
            self._client.set_questions(result.constant_vlm_specs or result.constant_questions, result.ephemeral_vlm_specs or result.ephemeral_questions)
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
                "constant_vlm_specs": [spec.to_payload() for spec in result.constant_vlm_specs],
                "ephemeral_vlm_specs": [spec.to_payload() for spec in result.ephemeral_vlm_specs],
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
        cycle_event_ids = self._emit_raw_dashboard_events(snapshot)

        # 2. People resolution: emit presence-change events
        self._resolve_people(snapshot)

        # 2b. Deterministic stage-level trigger rules
        self._evaluate_trigger_rules(snapshot, cycle_event_ids)

        # 2c. System vision tasks -> direct world/person updates
        self._emit_system_task_updates(snapshot, cycle_event_ids)

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
                        "task_id": answer.task_id,
                        "label": answer.label,
                        "slot_type": answer.slot_type,
                        "target": answer.target,
                        "cadence_s": answer.cadence_s,
                        "target_bbox": answer.target_bbox,
                        "target_identity": answer.target_identity,
                        "interpret_as": answer.interpret_as,
                    }
                    for answer in snapshot.vlm_answers
                ] if snapshot.vlm_answers else [],
                "focus": {
                    "constant_questions": list(self._last_focus.constant_questions),
                    "ephemeral_questions": list(self._last_focus.ephemeral_questions),
                    "constant_sam_targets": list(self._last_focus.constant_sam_targets),
                    "ephemeral_sam_targets": list(self._last_focus.ephemeral_sam_targets),
                    "constant_vlm_specs": [spec.to_payload() for spec in self._last_focus.constant_vlm_specs],
                    "ephemeral_vlm_specs": [spec.to_payload() for spec in self._last_focus.ephemeral_vlm_specs],
                } if self._last_focus is not None else None,
            })

        # 4. Push events (with dedup)
        # Expire old entries
        self._recent_events = {k: v for k, v in self._recent_events.items()
                               if now - v < self._event_cooldown}
        for event_text in result.events:
            key = self._normalize_event(event_text)
            if key not in self._recent_events:
                trace = self._trace_payload(
                    snapshot,
                    kind="vision_synthesis",
                    input_event_ids=list(cycle_event_ids),
                    signals=list(result.signals),
                    provenance={
                        "primary": "vision_synthesis_llm",
                        "label": "Vision synthesis LLM",
                        "components": self._component_sources(snapshot),
                        "inference": "llm synthesis over visual snapshot",
                        "thought": getattr(result, "thought", ""),
                    },
                )
                self._recent_events[key] = now
                self._events.append(PerceptionEvent(
                    description=event_text,
                    source="visual",
                    trace=trace,
                    kind="visual_claim",
                    payload={
                        "signals": list(result.signals),
                        "event_id": f"{snapshot.cycle_id}:claim:{len(self._events)}",
                        "input_event_ids": list(cycle_event_ids),
                        "vision_cycle_id": snapshot.cycle_id,
                    },
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
            cycle_id = snapshot.cycle_id or "vision-cycle"
            identity = current_ids[0] if current_ids else ""
            person_id = self._visual_to_person.get(identity) if identity else None
            self._events.append(PerceptionEvent(
                description="A person appeared in view",
                source="visual",
                trace=self._trace_payload(
                    snapshot,
                    kind="person_presence",
                    input_event_ids=self._snapshot_input_event_ids(snapshot),
                    identity=identity,
                    provenance=self._presence_provenance(snapshot, change="appeared"),
                ),
                kind="person_presence",
                payload={
                    "identity": identity,
                    "person_id": person_id,
                    "presence": "present",
                    "change": "appeared",
                    "signals": ["person_visible"],
                    "event_id": f"{cycle_id}:presence:appeared",
                    "input_event_ids": self._snapshot_input_event_ids(snapshot),
                    "vision_cycle_id": cycle_id,
                },
            ))
        elif not visible_now and self._presence_visible:
            cycle_id = snapshot.cycle_id or "vision-cycle"
            self._events.append(PerceptionEvent(
                description="A person is no longer visible",
                source="visual",
                trace=self._trace_payload(
                    snapshot,
                    kind="person_presence",
                    input_event_ids=self._snapshot_input_event_ids(snapshot),
                    provenance=self._presence_provenance(snapshot, change="disappeared"),
                ),
                kind="person_presence",
                payload={
                    "identity": "",
                    "person_id": None,
                    "presence": "gone",
                    "change": "disappeared",
                    "signals": ["person_departed"],
                    "event_id": f"{cycle_id}:presence:disappeared",
                    "input_event_ids": self._snapshot_input_event_ids(snapshot),
                    "vision_cycle_id": cycle_id,
                },
            ))

        self._presence_visible = visible_now

    def _trace_payload(self, snapshot: RawVisualSnapshot, **extra) -> dict:
        payload = {
            "vision_cycle_id": snapshot.cycle_id,
            "faces": len(snapshot.faces),
            "persons": len(snapshot.persons),
            "objects": len(snapshot.objects),
            "object_labels": [obj.label for obj in snapshot.objects],
            "vlm_answers": [
                {
                    "task_id": answer.task_id,
                    "label": answer.label,
                    "question": answer.question,
                    "answer": answer.answer,
                    "slot_type": answer.slot_type,
                    "answer_id": answer.answer_id,
                    "target": answer.target,
                    "cadence_s": answer.cadence_s,
                    "target_bbox": answer.target_bbox,
                    "target_identity": answer.target_identity,
                    "interpret_as": answer.interpret_as,
                }
                for answer in (snapshot.vlm_answers or [])
            ],
            "focus": {
                "constant_questions": list(self._last_focus.constant_questions),
                "ephemeral_questions": list(self._last_focus.ephemeral_questions),
                "constant_sam_targets": list(self._last_focus.constant_sam_targets),
                "ephemeral_sam_targets": list(self._last_focus.ephemeral_sam_targets),
                "constant_vlm_specs": [spec.to_payload() for spec in self._last_focus.constant_vlm_specs],
                "ephemeral_vlm_specs": [spec.to_payload() for spec in self._last_focus.ephemeral_vlm_specs],
            } if self._last_focus is not None else None,
        }
        payload.update(extra)
        return payload

    def _snapshot_input_event_ids(self, snapshot: RawVisualSnapshot) -> list[str]:
        cycle_id = snapshot.cycle_id or "vision-cycle"
        event_ids = [f"{cycle_id}:snapshot"]
        if snapshot.faces:
            event_ids.append(f"{cycle_id}:face")
        if snapshot.objects or snapshot.persons:
            event_ids.append(f"{cycle_id}:sam3")
        if snapshot.persons:
            for person in snapshot.persons:
                track_id = getattr(person, "track_id", None)
                if track_id is not None:
                    event_ids.append(f"{cycle_id}:reid:{track_id}")
        for answer in snapshot.vlm_answers:
            answer_id = str(answer.answer_id or "").strip()
            if answer_id:
                event_ids.append(answer_id)
        return event_ids

    def _emit_raw_dashboard_events(self, snapshot: RawVisualSnapshot) -> list[str]:
        if self._collector is None:
            return self._snapshot_input_event_ids(snapshot)

        cycle_id = snapshot.cycle_id or f"vision-cycle-{int(time.time() * 1000)}"
        trace = dict(snapshot.trace or {})
        event_ids: list[str] = []

        snapshot_event_id = f"{cycle_id}:snapshot"
        event_ids.append(snapshot_event_id)
        self._collector.push("vision_snapshot_read", {
            "event_id": snapshot_event_id,
            "vision_cycle_id": cycle_id,
            "faces": len(snapshot.faces),
            "persons": len(snapshot.persons),
            "objects": len(snapshot.objects),
            "object_labels": [obj.label for obj in snapshot.objects],
            "vlm_answer_count": len(snapshot.vlm_answers),
            "input_event_ids": [],
        })

        face_trace = dict(trace.get("face_tracking", {}) or {})
        if snapshot.faces or face_trace.get("timing"):
            face_event_id = f"{cycle_id}:face"
            event_ids.append(face_event_id)
            self._collector.push("face_tracking", {
                "event_id": face_event_id,
                "vision_cycle_id": cycle_id,
                "input_event_ids": [snapshot_event_id],
                "faces": [
                    {
                        "identity": getattr(face, "identity", "") if not isinstance(face, dict) else face.get("identity", ""),
                        "bbox": getattr(face, "bbox", None) if not isinstance(face, dict) else face.get("bbox"),
                        "confidence": getattr(face, "confidence", 0.0) if not isinstance(face, dict) else face.get("confidence", 0.0),
                        "gaze_direction": getattr(face, "gaze_direction", "") if not isinstance(face, dict) else face.get("gaze_direction", ""),
                        "looking_at_camera": getattr(face, "looking_at_camera", False) if not isinstance(face, dict) else face.get("looking_at_camera", False),
                        "track_id": getattr(face, "track_id", None) if not isinstance(face, dict) else face.get("track_id"),
                    }
                    for face in (face_trace.get("faces") or snapshot.faces)
                ],
                "timing": dict(face_trace.get("timing", {}) or {}),
            })

        sam3_trace = dict(trace.get("sam3_detection", {}) or {})
        if snapshot.persons or snapshot.objects or sam3_trace.get("timing"):
            sam3_event_id = f"{cycle_id}:sam3"
            event_ids.append(sam3_event_id)
            self._collector.push("sam3_detection", {
                "event_id": sam3_event_id,
                "vision_cycle_id": cycle_id,
                "input_event_ids": [snapshot_event_id],
                "persons": [
                    {
                        "identity": getattr(person, "identity", "") if not isinstance(person, dict) else person.get("identity", ""),
                        "bbox": getattr(person, "bbox", None) if not isinstance(person, dict) else person.get("bbox"),
                        "confidence": getattr(person, "confidence", 0.0) if not isinstance(person, dict) else person.get("confidence", 0.0),
                        "track_id": getattr(person, "track_id", None) if not isinstance(person, dict) else person.get("track_id"),
                    }
                    for person in (sam3_trace.get("persons") or snapshot.persons)
                ],
                "objects": [
                    {
                        "label": getattr(obj, "label", "") if not isinstance(obj, dict) else obj.get("label", ""),
                        "bbox": getattr(obj, "bbox", None) if not isinstance(obj, dict) else obj.get("bbox"),
                        "confidence": getattr(obj, "confidence", 0.0) if not isinstance(obj, dict) else obj.get("confidence", 0.0),
                    }
                    for obj in (sam3_trace.get("objects") or snapshot.objects)
                ],
                "timing": dict(sam3_trace.get("timing", {}) or {}),
            })

        reid_trace = dict(trace.get("reid_tracking", {}) or {})
        reid_input_ids = [snapshot_event_id]
        if snapshot.persons or sam3_trace.get("persons") or sam3_trace.get("objects"):
            reid_input_ids.append(f"{cycle_id}:sam3")
        for person in reid_trace.get("persons", []) or []:
            track_id = person.get("track_id")
            event_id = f"{cycle_id}:reid:{track_id if track_id is not None else len(event_ids)}"
            event_ids.append(event_id)
            self._collector.push("reid_track", {
                "event_id": event_id,
                "vision_cycle_id": cycle_id,
                "input_event_ids": list(reid_input_ids),
                "track_id": track_id,
                "identity": person.get("identity", ""),
                "identity_source": person.get("identity_source", ""),
                "confidence": person.get("confidence", 0.0),
                "bbox": person.get("bbox"),
                "timing": dict(reid_trace.get("timing", {}) or {}),
            })

        for answer in snapshot.vlm_answers:
            answer_id = str(answer.answer_id or "").strip()
            if not answer_id or answer_id in self._emitted_vlm_answer_ids:
                continue
            self._emitted_vlm_answer_ids.add(answer_id)
            event_ids.append(answer_id)
            self._collector.push("vlm_answer", {
                "event_id": answer_id,
                "vision_cycle_id": cycle_id,
                "input_event_ids": [snapshot_event_id],
                "task_id": answer.task_id,
                "label": answer.label,
                "question": answer.question,
                "answer": answer.answer,
                "slot_type": answer.slot_type,
                "elapsed": answer.elapsed,
                "answer_ts": answer.timestamp,
                "target": answer.target,
                "cadence_s": answer.cadence_s,
                "target_bbox": answer.target_bbox,
                "target_identity": answer.target_identity,
                "interpret_as": answer.interpret_as,
            })

        return event_ids

    @staticmethod
    def _component_sources(snapshot: RawVisualSnapshot) -> list[str]:
        sources: list[str] = []
        if snapshot.persons:
            sources.append("person_tracker")
        if snapshot.faces:
            sources.append("face_tracker")
        if snapshot.objects:
            sources.append("sam3_objects")
        if snapshot.vlm_answers:
            sources.append("vlm_answers")
        return sources

    def _presence_provenance(self, snapshot: RawVisualSnapshot, change: str) -> dict:
        if snapshot.persons:
            primary = "person_tracker"
        elif snapshot.faces:
            primary = "face_tracker"
        else:
            primary = "sam3_person_detection"
        return {
            "primary": primary,
            "label": "Presence tracker",
            "change": change,
            "components": self._component_sources(snapshot) or [primary],
            "inference": "direct presence heuristic",
        }

    def _emit_system_task_updates(self, snapshot: RawVisualSnapshot, input_event_ids: list[str]) -> None:
        if self._model_config is None:
            return

        changed: list[dict] = []
        active_keys: set[str] = set()
        for answer in snapshot.vlm_answers:
            interpret_as = str(answer.interpret_as or "").strip()
            if interpret_as not in {"world_state", "person_description_static", "person_description_dynamic"}:
                continue
            signature_key = str(answer.task_id or answer.question or "").strip()
            if not signature_key:
                continue
            active_keys.add(signature_key)
            normalized_answer = " ".join(str(answer.answer or "").strip().lower().split())
            signature = "|".join((
                normalized_answer,
                str(answer.target_identity or "").strip().lower(),
                ",".join(str(v) for v in (answer.target_bbox or ())),
            ))
            if not normalized_answer or normalized_answer.startswith("(error:"):
                self._system_task_signatures[signature_key] = signature
                continue
            if self._system_task_signatures.get(signature_key) == signature:
                continue
            self._system_task_signatures[signature_key] = signature
            changed.append({
                "task_id": answer.task_id,
                "label": answer.label or answer.task_id,
                "question": answer.question,
                "answer": answer.answer,
                "interpret_as": interpret_as,
                "target": answer.target,
                "target_identity": answer.target_identity,
                "target_person_id": self._resolve_target_person_id(answer),
            })

        for key in list(self._system_task_signatures):
            if key not in active_keys:
                self._system_task_signatures.pop(key, None)

        if not changed:
            return

        result = vision_state_update_call(
            world=self._world,
            people=self._people,
            task_answers=changed,
            model_config=self._model_config,
        )
        update = result.update
        if not (update.remove_facts or update.add_facts or update.events or update.person_updates):
            return

        cycle_id = snapshot.cycle_id or "vision-cycle"
        payload = {
            "event_id": f"{cycle_id}:vision_state_update",
            "input_event_ids": list(input_event_ids),
            "vision_cycle_id": cycle_id,
            "task_answers": list(changed),
            "remove_facts": list(update.remove_facts),
            "add_facts": list(update.add_facts),
            "events": list(update.events),
            "person_updates": [
                {
                    "person_id": item.person_id,
                    "remove_facts": list(item.remove_facts),
                    "add_facts": list(item.add_facts),
                    "set_name": item.set_name,
                    "set_presence": item.set_presence,
                }
                for item in update.person_updates
            ],
        }
        summary = result.summary or self._summarize_world_update(update)
        trace = self._trace_payload(
            snapshot,
            kind="vision_state_update",
            input_event_ids=list(input_event_ids),
            task_answers=list(changed),
            world_update={
                "remove_facts": list(update.remove_facts),
                "add_facts": list(update.add_facts),
                "events": list(update.events),
                "person_updates": payload["person_updates"],
            },
            provenance={
                "primary": "vision_state_update_llm",
                "label": "Vision state interpreter",
                "components": self._component_sources(snapshot),
                "inference": "llm interpretation of system vision tasks",
                "thought": result.thought,
            },
        )
        self._events.append(PerceptionEvent(
            description=summary or "Vision state updated",
            source="visual",
            trace=trace,
            kind="vision_state_update",
            payload=payload,
        ))
        if self._collector is not None:
            self._collector.push("vision_state_update", {
                "event_id": payload["event_id"],
                "vision_cycle_id": cycle_id,
                "input_event_ids": list(input_event_ids),
                "summary": summary,
                "task_answers": list(changed),
                "remove_facts": list(update.remove_facts),
                "add_facts": list(update.add_facts),
                "events": list(update.events),
                "person_updates": payload["person_updates"],
            })

    def _resolve_target_person_id(self, answer) -> str:
        identity = str(answer.target_identity or "").strip()
        if identity:
            person_id = self._visual_to_person.get(identity)
            if person_id:
                return person_id
        if self._people is not None:
            present = self._people.present_people()
            if len(present) == 1:
                return present[0].person_id
        return ""

    @staticmethod
    def _summarize_world_update(update) -> str:
        if update.events:
            return update.events[0]
        if update.add_facts:
            return update.add_facts[0]
        if update.person_updates:
            first = update.person_updates[0]
            if first.add_facts:
                return first.add_facts[0]
            if first.set_presence:
                return f"{first.person_id} is now {first.set_presence}"
        return ""

    def _evaluate_trigger_rules(self, snapshot: RawVisualSnapshot, input_event_ids: list[str]) -> None:
        if self._scenario is None:
            self._trigger_streaks.clear()
            self._active_trigger_signals.clear()
            return

        triggers = list(self._scenario.active_vision_triggers())
        active_signals = {trigger.signal for trigger in triggers if trigger.signal}
        for signal in list(self._trigger_streaks):
            if signal not in active_signals:
                self._trigger_streaks.pop(signal, None)
        self._active_trigger_signals.intersection_update(active_signals)

        for trigger in triggers:
            matched, evidence = self._trigger_matches(trigger, snapshot)
            signal = trigger.signal
            streak = self._trigger_streaks.get(signal, 0)
            streak = streak + 1 if matched else 0
            self._trigger_streaks[signal] = streak
            if not matched:
                self._active_trigger_signals.discard(signal)
                continue
            if streak < max(1, trigger.frames) or signal in self._active_trigger_signals:
                continue
            self._active_trigger_signals.add(signal)
            cycle_id = snapshot.cycle_id or "vision-cycle"
            description = trigger.description or self._describe_trigger(trigger, evidence)
            payload = {
                "signals": [signal],
                "trigger_signal": signal,
                "event_id": f"{cycle_id}:trigger:{signal}",
                "input_event_ids": list(input_event_ids),
                "vision_cycle_id": cycle_id,
                "frames": max(1, trigger.frames),
                "streak": streak,
                "rule": {
                    "signal": trigger.signal,
                    "source": trigger.source,
                    "label": trigger.label,
                    "task_id": trigger.task_id,
                    "answer_contains": trigger.answer_contains,
                    "frames": trigger.frames,
                },
            }
            trace = self._trace_payload(
                snapshot,
                kind="vision_trigger",
                input_event_ids=list(input_event_ids),
                trigger=payload["rule"],
                evidence=evidence,
                provenance={
                    "primary": f"{trigger.source}_trigger",
                    "label": "Deterministic trigger",
                    "components": self._component_sources(snapshot),
                    "inference": "rule evaluation over raw vision outputs",
                },
            )
            self._events.append(PerceptionEvent(
                description=description,
                source="visual",
                trace=trace,
                kind="vision_trigger",
                payload=payload,
            ))
            if self._collector is not None:
                self._collector.push("vision_trigger", {
                    "event_id": payload["event_id"],
                    "vision_cycle_id": cycle_id,
                    "input_event_ids": list(input_event_ids),
                    "signal": signal,
                    "description": description,
                    "rule": payload["rule"],
                    "streak": streak,
                    "evidence": evidence,
                })

    @staticmethod
    def _describe_trigger(trigger: VisionTrigger, evidence: dict) -> str:
        if trigger.source == "sam3":
            label = trigger.label or "object"
            return f"Vision trigger fired: {label} detected"
        if trigger.source == "vlm":
            label = evidence.get("task_label") or trigger.task_id or "vlm task"
            return f"Vision trigger fired: {label}"
        return f"Vision trigger fired: {trigger.signal}"

    @staticmethod
    def _label_match(candidate: str, target: str) -> bool:
        return str(candidate or "").strip().lower() == str(target or "").strip().lower()

    def _trigger_matches(self, trigger: VisionTrigger, snapshot: RawVisualSnapshot) -> tuple[bool, dict]:
        source = trigger.source.lower()
        label = trigger.label.strip()
        if source == "presence":
            matched = self._has_human_presence(snapshot)
            return matched, {"presence": matched}
        if source == "sam3":
            if label.lower() in {"person", "people"}:
                if snapshot.persons:
                    return True, {"persons": len(snapshot.persons)}
                for obj in snapshot.objects:
                    if self._label_match(obj.label, "person"):
                        return True, {"object_label": obj.label, "bbox": obj.bbox, "confidence": obj.confidence}
                return False, {}
            for obj in snapshot.objects:
                if self._label_match(obj.label, label):
                    return True, {
                        "object_label": obj.label,
                        "bbox": obj.bbox,
                        "confidence": obj.confidence,
                    }
            return False, {}
        if source == "insight":
            if not label:
                return bool(snapshot.faces), {"faces": len(snapshot.faces)}
            for face in snapshot.faces:
                if self._label_match(face.identity, label):
                    return True, {
                        "identity": face.identity,
                        "bbox": face.bbox,
                        "confidence": face.confidence,
                    }
            return False, {}
        if source == "reid":
            if not label:
                return bool(snapshot.persons), {"persons": len(snapshot.persons)}
            for person in snapshot.persons:
                if self._label_match(person.identity, label):
                    return True, {
                        "identity": person.identity,
                        "bbox": person.bbox,
                        "confidence": person.confidence,
                    }
            return False, {}
        if source == "vlm":
            for answer in snapshot.vlm_answers:
                if trigger.task_id and str(answer.task_id).strip() != trigger.task_id:
                    continue
                if trigger.answer_contains and trigger.answer_contains.lower() not in str(answer.answer or "").lower():
                    continue
                return True, {
                    "task_id": answer.task_id,
                    "task_label": answer.label or answer.task_id,
                    "answer_id": answer.answer_id,
                    "answer": answer.answer,
                    "target_bbox": answer.target_bbox,
                    "target_identity": answer.target_identity,
                }
            return False, {}
        return False, {}
