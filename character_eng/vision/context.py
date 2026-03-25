"""Visual context dataclasses and thread-safe container."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class FaceObservation:
    identity: str
    bbox: tuple[int, int, int, int]  # x, y, w, h
    age: int | None = None
    gender: str | None = None
    confidence: float = 0.0
    gaze_direction: str = "unknown"
    looking_at_camera: bool = False


@dataclass
class PersonObservation:
    identity: str
    bbox: tuple[int, int, int, int]  # x, y, w, h
    confidence: float = 0.0


@dataclass
class ObjectDetection:
    label: str
    bbox: tuple[int, int, int, int] | None = None
    confidence: float = 0.0


@dataclass
class VLMAnswer:
    question: str
    answer: str
    task_id: str = ""
    label: str = ""
    elapsed: float = 0.0
    slot_type: str = "constant"  # "constant" or "ephemeral"
    timestamp: float = 0.0
    answer_id: str = ""
    target: str = "scene"
    cadence_s: float = 0.0
    interpret_as: str = "general"
    target_bbox: tuple[int, int, int, int] | None = None
    target_identity: str = ""


@dataclass
class RawVisualSnapshot:
    faces: list[FaceObservation] = field(default_factory=list)
    persons: list[PersonObservation] = field(default_factory=list)
    objects: list[ObjectDetection] = field(default_factory=list)
    vlm_answers: list[VLMAnswer] = field(default_factory=list)
    timestamp: float = 0.0
    cycle_id: str = ""
    trace: dict = field(default_factory=dict)

    @classmethod
    def from_json(cls, data: dict) -> RawVisualSnapshot:
        faces = [
            FaceObservation(
                identity=f.get("identity", "Unknown"),
                bbox=tuple(f["bbox"]) if "bbox" in f else (0, 0, 0, 0),
                age=f.get("age"),
                gender=f.get("gender"),
                confidence=f.get("confidence", 0.0),
                gaze_direction=f.get("gaze_direction", "unknown"),
                looking_at_camera=f.get("looking_at_camera", False),
            )
            for f in data.get("faces", [])
        ]
        persons = [
            PersonObservation(
                identity=p.get("identity", "Unknown"),
                bbox=tuple(p["bbox"]) if "bbox" in p else (0, 0, 0, 0),
                confidence=p.get("confidence", 0.0),
            )
            for p in data.get("persons", [])
        ]
        objects = [
            ObjectDetection(
                label=o.get("label", "unknown"),
                bbox=tuple(o["bbox"]) if o.get("bbox") else None,
                confidence=o.get("confidence", 0.0),
            )
            for o in data.get("objects", [])
        ]
        vlm_answers = [
            VLMAnswer(
                task_id=v.get("task_id", ""),
                label=v.get("label", ""),
                question=v.get("question", ""),
                answer=v.get("answer", ""),
                elapsed=v.get("elapsed", 0.0),
                slot_type=v.get("slot_type", "constant"),
                timestamp=v.get("timestamp", 0.0),
                answer_id=v.get("answer_id", ""),
                target=v.get("target", "scene"),
                cadence_s=v.get("cadence_s", 0.0),
                interpret_as=v.get("interpret_as", "general"),
                target_bbox=tuple(v["target_bbox"]) if v.get("target_bbox") else None,
                target_identity=v.get("target_identity", ""),
            )
            for v in data.get("vlm_answers", [])
        ]
        return cls(
            faces=faces,
            persons=persons,
            objects=objects,
            vlm_answers=vlm_answers,
            timestamp=data.get("timestamp", 0.0),
            cycle_id=str(data.get("cycle_id", "") or ""),
            trace=dict(data.get("trace", {}) or {}),
        )


class VisualContext:
    """Thread-safe container for the latest visual snapshot."""

    def __init__(self):
        self._lock = threading.Lock()
        self._snapshot: RawVisualSnapshot | None = None

    def update(self, snapshot: RawVisualSnapshot) -> None:
        with self._lock:
            self._snapshot = snapshot

    @property
    def snapshot(self) -> RawVisualSnapshot | None:
        with self._lock:
            return self._snapshot

    def render_for_prompt(self) -> str:
        """Render current visual state as text for character prompt injection."""
        with self._lock:
            snap = self._snapshot
        if snap is None:
            return ""

        parts = []

        if snap.faces:
            face_descs = []
            for f in snap.faces:
                desc = f.identity
                if f.age:
                    desc += f", ~{f.age}{f.gender or ''}"
                face_descs.append(desc)
            parts.append(f"Visible faces: {'; '.join(face_descs)}")

        if snap.persons:
            person_counts: dict[str, int] = {}
            for person in snap.persons:
                label = str(person.identity or "unknown person").strip() or "unknown person"
                person_counts[label] = person_counts.get(label, 0) + 1
            person_descs = [
                f"{label} x{count}" if count > 1 else label
                for label, count in person_counts.items()
            ]
            parts.append(f"Visible people: {'; '.join(person_descs)}")

        if snap.objects:
            object_counts: dict[str, int] = {}
            for obj in snap.objects:
                label = str(obj.label or "unknown object").strip() or "unknown object"
                object_counts[label] = object_counts.get(label, 0) + 1
            obj_descs = [
                f"{label} x{count}" if count > 1 else label
                for label, count in object_counts.items()
            ]
            parts.append(f"Visible objects: {'; '.join(obj_descs)}")

        if snap.vlm_answers:
            for v in snap.vlm_answers:
                label = v.label or v.task_id or v.question
                target = f" [{v.target}]" if v.target and v.target != "scene" else ""
                parts.append(f"VLM {label}{target}: {v.answer}")

        return "\n".join(parts) if parts else ""

    def get_gaze_targets(self) -> list[str]:
        """Return gaze target labels derived from current visual state."""
        with self._lock:
            snap = self._snapshot
        if snap is None:
            return []

        targets = []
        for p in snap.persons:
            if p.identity not in targets:
                targets.append(p.identity)
        for f in snap.faces:
            if f.identity not in targets:
                targets.append(f.identity)
        for o in snap.objects:
            if o.label not in targets:
                targets.append(o.label)
        return targets
