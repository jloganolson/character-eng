from __future__ import annotations

import re
from dataclasses import dataclass, field


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(text: str, fallback: str = "vision_task") -> str:
    slug = _SLUG_RE.sub("_", str(text or "").strip().lower()).strip("_")
    return slug or fallback


@dataclass
class VLMTaskSpec:
    task_id: str
    question: str
    label: str = ""
    target: str = "scene"  # "scene" | "nearest_person"
    cadence_s: float = 0.0
    interpret_as: str = "general"
    system_role: str = ""
    metadata: dict = field(default_factory=dict)

    def to_payload(self) -> dict:
        payload = {
            "task_id": self.task_id,
            "question": self.question,
            "label": self.label or self.task_id,
            "target": self.target,
            "cadence_s": float(self.cadence_s or 0.0),
            "interpret_as": self.interpret_as or "general",
            "system_role": self.system_role or "",
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    @classmethod
    def from_payload(cls, payload: dict | str, *, default_id: str = "vision_task") -> "VLMTaskSpec":
        if hasattr(payload, "to_payload"):
            payload = payload.to_payload()
        if isinstance(payload, str):
            question = payload.strip()
            task_id = _slugify(question, fallback=default_id)
            return cls(task_id=task_id, question=question, label=question)

        question = str(payload.get("question", "")).strip()
        label = str(payload.get("label", "")).strip()
        task_id = str(payload.get("task_id", "")).strip() or _slugify(label or question, fallback=default_id)
        target = str(payload.get("target", "scene")).strip().lower() or "scene"
        if target not in {"scene", "nearest_person"}:
            target = "scene"
        try:
            cadence_s = float(payload.get("cadence_s", 0.0) or 0.0)
        except (TypeError, ValueError):
            cadence_s = 0.0
        return cls(
            task_id=task_id,
            question=question,
            label=label or question or task_id,
            target=target,
            cadence_s=max(0.0, cadence_s),
            interpret_as=str(payload.get("interpret_as", "general")).strip() or "general",
            system_role=str(payload.get("system_role", "")).strip(),
            metadata=dict(payload.get("metadata", {}) or {}),
        )


def generic_task_specs(
    questions: list[str],
    *,
    prefix: str,
    cadence_s: float,
) -> list[VLMTaskSpec]:
    specs: list[VLMTaskSpec] = []
    seen_ids: set[str] = set()
    for index, question in enumerate(questions, start=1):
        cleaned = str(question).strip()
        if not cleaned:
            continue
        target = "nearest_person" if "person" in cleaned.lower() else "scene"
        task_id = _slugify(f"{prefix}_{cleaned}", fallback=f"{prefix}_{index}")
        suffix = 2
        base = task_id
        while task_id in seen_ids:
            task_id = f"{base}_{suffix}"
            suffix += 1
        seen_ids.add(task_id)
        specs.append(VLMTaskSpec(
            task_id=task_id,
            question=cleaned,
            label=cleaned,
            target=target,
            cadence_s=cadence_s,
        ))
    return specs
