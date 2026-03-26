from __future__ import annotations

import argparse
import html
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

from character_eng.history import HISTORY_ROOT, deserialize_people, deserialize_world, load_checkpoint
from character_eng.models import MICRO_MODEL, MODELS
from character_eng.vision.interpret import vision_state_update_call
from character_eng.world import PersonUpdate, WorldUpdate


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _fact_match(expected: str, actual: str) -> bool:
    e = _normalize_text(expected)
    a = _normalize_text(actual)
    if not e or not a:
        return False
    return e == a or e in a or a in e


def _consume_matches(expected: list[str], actual: list[str]) -> tuple[list[str], list[str]]:
    remaining_actual = list(actual)
    missing: list[str] = []
    matched_actual: set[int] = set()
    for item in expected:
        match_index = next((idx for idx, candidate in enumerate(remaining_actual) if idx not in matched_actual and _fact_match(item, candidate)), None)
        if match_index is None:
            missing.append(item)
        else:
            matched_actual.add(match_index)
    extras = [candidate for idx, candidate in enumerate(remaining_actual) if idx not in matched_actual]
    return missing, extras


@dataclass
class VisionStateSnippet:
    manifest_path: Path
    title: str
    session_id: str
    tags: list[str]
    capture_mode: str
    source_event_type: str
    task_answers: list[dict]
    expected_update: dict


@dataclass
class PersonUpdateDiff:
    person_id: str
    missing_remove_facts: list[str] = field(default_factory=list)
    extra_remove_facts: list[str] = field(default_factory=list)
    missing_facts: list[str] = field(default_factory=list)
    extra_facts: list[str] = field(default_factory=list)
    expected_presence: str = ""
    actual_presence: str = ""
    expected_name: str = ""
    actual_name: str = ""
    unexpected_presence: bool = False
    unexpected_name: bool = False

    @property
    def ok(self) -> bool:
        return (
            not self.missing_remove_facts
            and not self.extra_remove_facts
            and not self.missing_facts
            and not self.extra_facts
            and (not self.expected_presence or self.expected_presence == self.actual_presence)
            and (not self.expected_name or self.expected_name == self.actual_name)
            and not self.unexpected_presence
            and not self.unexpected_name
        )


@dataclass
class VisionStateEvalResult:
    snippet: VisionStateSnippet
    actual_update: WorldUpdate
    missing_removed_world_facts: list[str] = field(default_factory=list)
    extra_removed_world_facts: list[str] = field(default_factory=list)
    missing_world_facts: list[str] = field(default_factory=list)
    extra_world_facts: list[str] = field(default_factory=list)
    missing_events: list[str] = field(default_factory=list)
    extra_events: list[str] = field(default_factory=list)
    person_diffs: list[PersonUpdateDiff] = field(default_factory=list)
    unexpected_person_updates: list[str] = field(default_factory=list)
    summary: str = ""
    thought: str = ""

    @property
    def ok(self) -> bool:
        return (
            not self.missing_removed_world_facts
            and not self.extra_removed_world_facts
            and not self.missing_world_facts
            and not self.extra_world_facts
            and not self.missing_events
            and not self.extra_events
            and not self.unexpected_person_updates
            and all(item.ok for item in self.person_diffs)
        )


def _expected_person_updates(payload: dict) -> list[PersonUpdate]:
    updates: list[PersonUpdate] = []
    for raw in payload.get("person_updates", []) or []:
        if not isinstance(raw, dict):
            continue
        updates.append(PersonUpdate(
            person_id=str(raw.get("person_id", "")).strip(),
            remove_facts=[str(item).strip() for item in raw.get("remove_facts", []) if str(item).strip()],
            add_facts=[str(item).strip() for item in raw.get("add_facts", []) if str(item).strip()],
            set_name=str(raw.get("set_name", "")).strip() or None,
            set_presence=str(raw.get("set_presence", "")).strip() or None,
        ))
    return updates


def iter_vision_state_snippets(root: Path = HISTORY_ROOT, *, required_tags: set[str] | None = None) -> Iterable[VisionStateSnippet]:
    for bucket in ("moments", "pinned", "sessions"):
        bucket_dir = root / bucket
        if not bucket_dir.exists():
            continue
        for candidate in sorted(bucket_dir.iterdir()):
            manifest_path = candidate / "manifest.json"
            if not manifest_path.exists():
                continue
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            capture_mode = str(payload.get("capture_mode") or payload.get("bundle", {}).get("capture_mode") or "")
            if capture_mode != "vision_state_turn":
                continue
            tags = [str(item).strip() for item in payload.get("tags", []) if str(item).strip()]
            lowered = {item.lower() for item in tags}
            if required_tags and not required_tags.issubset(lowered):
                continue
            bundle = dict(payload.get("bundle") or {})
            vision_state = dict(bundle.get("vision_state") or {})
            task_answers = list(vision_state.get("task_answers") or [])
            expected_update = dict(vision_state.get("expected_update") or {})
            if not task_answers or not expected_update:
                continue
            yield VisionStateSnippet(
                manifest_path=manifest_path,
                title=str(payload.get("title") or candidate.name),
                session_id=str(payload.get("session_id") or candidate.name),
                tags=tags,
                capture_mode=capture_mode,
                source_event_type=str(vision_state.get("source_event_type") or bundle.get("event", {}).get("type", "")),
                task_answers=task_answers,
                expected_update=expected_update,
            )


def evaluate_vision_state_snippet(snippet: VisionStateSnippet, *, model_config: dict) -> VisionStateEvalResult:
    checkpoint = load_checkpoint(snippet.manifest_path.parent)
    world = deserialize_world(checkpoint.get("world"))
    people = deserialize_people(checkpoint.get("people"))
    result = vision_state_update_call(
        world=world,
        people=people,
        task_answers=snippet.task_answers,
        model_config=model_config,
    )

    expected_remove_facts = [str(item).strip() for item in snippet.expected_update.get("remove_facts", []) if str(item).strip()]
    expected_world_facts = [str(item).strip() for item in snippet.expected_update.get("add_facts", []) if str(item).strip()]
    expected_events = [str(item).strip() for item in snippet.expected_update.get("events", []) if str(item).strip()]
    missing_removed_world_facts, extra_removed_world_facts = _consume_matches(expected_remove_facts, list(result.update.remove_facts))
    missing_world_facts, extra_world_facts = _consume_matches(expected_world_facts, list(result.update.add_facts))
    missing_events, extra_events = _consume_matches(expected_events, list(result.update.events))

    actual_person_by_id = {item.person_id: item for item in result.update.person_updates}
    person_diffs: list[PersonUpdateDiff] = []
    expected_person_ids: set[str] = set()
    for expected in _expected_person_updates(snippet.expected_update):
        expected_person_ids.add(expected.person_id)
        actual = actual_person_by_id.get(expected.person_id, PersonUpdate(person_id=expected.person_id))
        missing_remove_facts, extra_remove_facts = _consume_matches(list(expected.remove_facts), list(actual.remove_facts))
        missing_facts, extra_facts = _consume_matches(list(expected.add_facts), list(actual.add_facts))
        person_diffs.append(PersonUpdateDiff(
            person_id=expected.person_id,
            missing_remove_facts=missing_remove_facts,
            extra_remove_facts=extra_remove_facts,
            missing_facts=missing_facts,
            extra_facts=extra_facts,
            expected_presence=expected.set_presence or "",
            actual_presence=actual.set_presence or "",
            expected_name=expected.set_name or "",
            actual_name=actual.set_name or "",
            unexpected_presence=not (expected.set_presence or "") and bool(actual.set_presence or ""),
            unexpected_name=not (expected.set_name or "") and bool(actual.set_name or ""),
        ))
    unexpected_person_updates = sorted(person_id for person_id in actual_person_by_id if person_id not in expected_person_ids)

    return VisionStateEvalResult(
        snippet=snippet,
        actual_update=result.update,
        missing_removed_world_facts=missing_removed_world_facts,
        extra_removed_world_facts=extra_removed_world_facts,
        missing_world_facts=missing_world_facts,
        extra_world_facts=extra_world_facts,
        missing_events=missing_events,
        extra_events=extra_events,
        person_diffs=person_diffs,
        unexpected_person_updates=unexpected_person_updates,
        summary=result.summary,
        thought=result.thought,
    )


def _render_result_card(result: VisionStateEvalResult) -> str:
    status = "PASS" if result.ok else "REVIEW"
    status_class = "pass" if result.ok else "review"
    expected = result.snippet.expected_update
    actual = result.actual_update
    person_lines = []
    for diff in result.person_diffs:
        person_lines.append(
            f"<div><b>{html.escape(diff.person_id)}</b> · expected presence: {html.escape(diff.expected_presence or '-')} · actual presence: {html.escape(diff.actual_presence or '-')}</div>"
        )
        if diff.missing_remove_facts:
            person_lines.append(f"<div class='miss'>missing person removals: {html.escape(', '.join(diff.missing_remove_facts))}</div>")
        if diff.extra_remove_facts:
            person_lines.append(f"<div class='extra'>extra person removals: {html.escape(', '.join(diff.extra_remove_facts))}</div>")
        if diff.missing_facts:
            person_lines.append(f"<div class='miss'>missing person facts: {html.escape(', '.join(diff.missing_facts))}</div>")
        if diff.extra_facts:
            person_lines.append(f"<div class='extra'>extra person facts: {html.escape(', '.join(diff.extra_facts))}</div>")
        if diff.unexpected_presence:
            person_lines.append(f"<div class='extra'>unexpected presence: {html.escape(diff.actual_presence)}</div>")
        if diff.unexpected_name:
            person_lines.append(f"<div class='extra'>unexpected name: {html.escape(diff.actual_name)}</div>")
    return f"""
    <article class="card">
      <div class="card-head">
        <h2>{html.escape(result.snippet.title)}</h2>
        <span class="status {status_class}">{status}</span>
      </div>
      <div class="meta">source event: {html.escape(result.snippet.source_event_type or '-')} · tags: {html.escape(', '.join(result.snippet.tags) or '-')}</div>
      <div class="grid">
        <section>
          <h3>Task Answers</h3>
          <pre>{html.escape(json.dumps(result.snippet.task_answers, indent=2))}</pre>
        </section>
        <section>
          <h3>Expected Update</h3>
          <pre>{html.escape(json.dumps(expected, indent=2))}</pre>
        </section>
        <section>
          <h3>Actual Update</h3>
          <pre>{html.escape(json.dumps({
              "remove_facts": actual.remove_facts,
              "add_facts": actual.add_facts,
              "events": actual.events,
              "person_updates": [
                  {
                      "person_id": item.person_id,
                      "remove_facts": item.remove_facts,
                      "add_facts": item.add_facts,
                      "set_name": item.set_name,
                      "set_presence": item.set_presence,
                  }
                  for item in actual.person_updates
              ],
          }, indent=2))}</pre>
        </section>
        <section>
          <h3>Diff</h3>
          <div>summary: {html.escape(result.summary or '-')}</div>
          <div>thought: {html.escape(result.thought or '-')}</div>
          {f"<div class='miss'>missing world removals: {html.escape(', '.join(result.missing_removed_world_facts))}</div>" if result.missing_removed_world_facts else ""}
          {f"<div class='extra'>extra world removals: {html.escape(', '.join(result.extra_removed_world_facts))}</div>" if result.extra_removed_world_facts else ""}
          {f"<div class='miss'>missing world facts: {html.escape(', '.join(result.missing_world_facts))}</div>" if result.missing_world_facts else ""}
          {f"<div class='extra'>extra world facts: {html.escape(', '.join(result.extra_world_facts))}</div>" if result.extra_world_facts else ""}
          {f"<div class='miss'>missing events: {html.escape(', '.join(result.missing_events))}</div>" if result.missing_events else ""}
          {f"<div class='extra'>extra events: {html.escape(', '.join(result.extra_events))}</div>" if result.extra_events else ""}
          {f"<div class='extra'>unexpected person updates: {html.escape(', '.join(result.unexpected_person_updates))}</div>" if result.unexpected_person_updates else ""}
          {''.join(person_lines) or '<div>No person diffs.</div>'}
        </section>
      </div>
    </article>
    """


def write_vision_state_report(results: list[VisionStateEvalResult], html_path: Path) -> None:
    html_path.parent.mkdir(parents=True, exist_ok=True)
    passed = sum(1 for item in results if item.ok)
    total = len(results)
    body = "\n".join(_render_result_card(item) for item in results)
    html_path.write_text(
        f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Vision State Snippet Report</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, sans-serif; background: #f5f1e8; color: #1d1a16; margin: 0; padding: 24px; }}
    h1 {{ margin: 0 0 8px; }}
    .summary {{ margin-bottom: 20px; color: #5b5349; }}
    .card {{ background: #fffdf8; border: 1px solid #d9cfbf; border-radius: 14px; padding: 16px; margin-bottom: 16px; box-shadow: 0 8px 24px rgba(65, 50, 30, 0.08); }}
    .card-head {{ display: flex; justify-content: space-between; align-items: baseline; gap: 12px; }}
    .status {{ font-size: 12px; font-weight: 700; letter-spacing: 0.08em; }}
    .status.pass {{ color: #116149; }}
    .status.review {{ color: #8b3f1f; }}
    .meta {{ color: #6c6358; margin-bottom: 12px; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }}
    section {{ background: #f8f3ea; border-radius: 10px; padding: 12px; }}
    pre {{ white-space: pre-wrap; font-size: 12px; }}
    .miss {{ color: #8b1e3f; margin-top: 8px; }}
    .extra {{ color: #8b5a1e; margin-top: 8px; }}
  </style>
</head>
<body>
  <h1>Vision State Snippet Report</h1>
  <div class="summary">generated {html.escape(datetime.now().isoformat())} · {passed}/{total} passing</div>
  {body or '<p>No matching snippets found.</p>'}
</body>
</html>""",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate vision-state snippets against the current interpreter.")
    parser.add_argument("--root", type=Path, default=HISTORY_ROOT)
    parser.add_argument("--tag", action="append", default=[], help="Require snippet tag (can be repeated)")
    parser.add_argument("--model", default=MICRO_MODEL, choices=MODELS.keys())
    parser.add_argument("--report", type=Path, default=None, help="Optional HTML report path")
    args = parser.parse_args()

    required_tags = {item.strip().lower() for item in args.tag if item.strip()}
    snippets = list(iter_vision_state_snippets(args.root, required_tags=required_tags or None))
    if not snippets:
        print("No vision-state snippets found.")
        return 1

    model_config = MODELS[args.model]
    results = [evaluate_vision_state_snippet(item, model_config=model_config) for item in snippets]
    report_path = args.report
    if report_path is None:
        logs_dir = Path(__file__).resolve().parent.parent / "logs"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = logs_dir / f"qa_vision_snippets_{args.model}_{timestamp}.html"
    write_vision_state_report(results, report_path)

    for item in results:
        status = "PASS" if item.ok else "REVIEW"
        print(f"{status} | {item.snippet.title} | summary {item.summary or '-'}")
        if item.missing_removed_world_facts:
            print(f"  missing world removals: {item.missing_removed_world_facts}")
        if item.extra_removed_world_facts:
            print(f"  extra world removals: {item.extra_removed_world_facts}")
        if item.missing_world_facts:
            print(f"  missing world facts: {item.missing_world_facts}")
        if item.extra_world_facts:
            print(f"  extra world facts: {item.extra_world_facts}")
        if item.missing_events:
            print(f"  missing events: {item.missing_events}")
        if item.extra_events:
            print(f"  extra events: {item.extra_events}")
        for diff in item.person_diffs:
            if diff.missing_remove_facts:
                print(f"  missing {diff.person_id} removals: {diff.missing_remove_facts}")
            if diff.extra_remove_facts:
                print(f"  extra {diff.person_id} removals: {diff.extra_remove_facts}")
            if diff.missing_facts:
                print(f"  missing {diff.person_id} facts: {diff.missing_facts}")
            if diff.extra_facts:
                print(f"  extra {diff.person_id} facts: {diff.extra_facts}")
        if item.unexpected_person_updates:
            print(f"  unexpected person updates: {item.unexpected_person_updates}")
    print(f"Report: {report_path}")
    return 0 if all(item.ok for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
