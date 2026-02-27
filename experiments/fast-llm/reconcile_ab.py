#!/usr/bin/env python3
"""Reconcile A/B: Can 8B handle structured world-state reconciliation?

Reconcile is the hardest remaining 70B call — it outputs structured JSON with:
- Fact ID references (f1, f2, p1f1) for removals
- New fact strings for additions
- Person updates with presence tracking

This tests 8B vs 70B on 8 scenarios, scored on structural correctness.

Usage:
    uv run python experiments/fast-llm/reconcile_ab.py
    uv run python experiments/fast-llm/reconcile_ab.py --open
"""

from __future__ import annotations

import json
import os
import sys
import time
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from character_eng.models import MODELS
from character_eng.world import (
    WorldState, WorldUpdate, PersonUpdate, reconcile_call,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ── Test scenarios ────────────────────────────────────────────

WORLD_STATIC = [
    "Greg is a robot head bolted to a folding table",
    "Greg has no body, arms, or hands",
    "The lemonade stand is on a quiet suburban sidewalk",
    'A hand-painted sign reads "GREG\'S: Water $1 / Advice FREE"',
    "A stack of fliers sits on the table in front of Greg",
]


@dataclass
class Expected:
    """What we expect the reconciler to produce."""
    remove_facts: list[str] = field(default_factory=list)  # fact IDs to remove
    add_facts_keywords: list[str] = field(default_factory=list)  # keywords that should appear in added facts
    min_add_facts: int = 0
    max_add_facts: int = 5
    min_events: int = 0
    has_person_update: bool = False
    person_presence: str | None = None  # expected presence value if has_person_update
    person_name: str | None = None  # expected name if set


@dataclass
class Scenario:
    name: str
    description: str
    dynamic: dict[str, str]
    pending_changes: list[str]
    expected: Expected
    events: list[str] = field(default_factory=list)


SCENARIOS = [
    Scenario(
        name="simple_fact_update",
        description="Weather changes from sunny to cloudy. Should remove old weather fact, add new one.",
        dynamic={
            "f1": "It is a warm, sunny afternoon",
            "f2": "A light breeze is blowing",
            "f3": "Greg has sold zero cups of water today",
        },
        pending_changes=["The sky clouds over and it starts getting chilly"],
        expected=Expected(
            remove_facts=["f1"],
            add_facts_keywords=["cloud", "chill"],
            min_add_facts=1,
            max_add_facts=2,
            min_events=1,
        ),
    ),
    Scenario(
        name="fact_addition",
        description="A dog appears. No existing fact to remove — pure addition.",
        dynamic={
            "f1": "It is a warm afternoon",
            "f2": "No customers have visited yet",
        },
        pending_changes=["A small dog trots up to the stand and sniffs around"],
        expected=Expected(
            remove_facts=[],
            add_facts_keywords=["dog"],
            min_add_facts=1,
            max_add_facts=2,
            min_events=1,
        ),
    ),
    Scenario(
        name="fact_removal",
        description="Bird flies away. Should remove bird fact, no new fact needed.",
        dynamic={
            "f1": "It is a warm afternoon",
            "f2": "A bird is perched on the edge of the table",
            "f3": "Greg has handed out zero fliers today",
        },
        pending_changes=["The bird flies away"],
        expected=Expected(
            remove_facts=["f2"],
            add_facts_keywords=[],
            min_add_facts=0,
            max_add_facts=1,
            min_events=1,
        ),
    ),
    Scenario(
        name="contradictory_update",
        description="It starts raining when it's sunny. Should remove sunny fact, add rain.",
        dynamic={
            "f1": "It is a sunny day with clear skies",
            "f2": "A light breeze is blowing",
            "f3": "The sidewalk is dry",
        },
        pending_changes=["It starts raining heavily"],
        expected=Expected(
            remove_facts=["f1", "f3"],  # sunny and dry sidewalk both contradicted
            add_facts_keywords=["rain"],
            min_add_facts=1,
            max_add_facts=3,
            min_events=1,
        ),
    ),
    Scenario(
        name="multi_change_batch",
        description="Three things happen at once. All should be processed.",
        dynamic={
            "f1": "It is a warm afternoon",
            "f2": "The stand is quiet with no visitors",
            "f3": "Greg has sold zero cups of water",
        },
        pending_changes=[
            "A group of kids runs past the stand",
            "One of the kids knocks over a cup of water",
            "A car honks loudly on the street",
        ],
        expected=Expected(
            remove_facts=[],  # might remove f2 (quiet/no visitors) or f3
            add_facts_keywords=[],
            min_add_facts=0,
            min_events=2,  # at least 2 of the 3 events should be captured
        ),
    ),
    Scenario(
        name="person_arrival",
        description="A woman walks up to the stand. Should create/update person.",
        dynamic={
            "f1": "It is a warm afternoon",
            "f2": "No one is around",
        },
        pending_changes=["A woman with a red hat walks up to the stand"],
        expected=Expected(
            remove_facts=["f2"],
            add_facts_keywords=[],
            min_add_facts=0,
            min_events=1,
            has_person_update=True,
            person_presence="approaching",
        ),
    ),
    Scenario(
        name="person_departure",
        description="The customer walks away. Should update person presence.",
        dynamic={
            "f1": "It is a warm afternoon",
            "f2": "A customer is standing at the stand",
        },
        pending_changes=["The customer waves goodbye and walks away"],
        expected=Expected(
            remove_facts=["f2"],
            add_facts_keywords=[],
            min_add_facts=0,
            min_events=1,
            has_person_update=True,
            person_presence="leaving",
        ),
    ),
    Scenario(
        name="sale_tracking",
        description="Customer buys water. Should update the sales counter fact.",
        dynamic={
            "f1": "It is a warm afternoon",
            "f2": "Greg has sold zero cups of water today",
            "f3": "A customer is at the stand",
        },
        pending_changes=["The customer puts a dollar on the table and takes a cup of water"],
        expected=Expected(
            remove_facts=["f2"],
            add_facts_keywords=["sold", "one", "1"],  # any of these
            min_add_facts=1,
            max_add_facts=2,
            min_events=1,
        ),
    ),
]


# ── Scoring ───────────────────────────────────────────────────

@dataclass
class Score:
    correct_removals: float = 0  # 0-1
    addition_quality: float = 0  # 0-1
    events_present: float = 0    # 0-1
    person_correct: float = 0    # 0-1
    total: float = 0             # weighted average
    details: str = ""


def score_result(result: WorldUpdate, expected: Expected) -> Score:
    details = []
    scores = []

    # 1. Correct removals
    if expected.remove_facts:
        correct = sum(1 for f in expected.remove_facts if f in result.remove_facts)
        extra = sum(1 for f in result.remove_facts if f not in expected.remove_facts)
        removal_score = correct / len(expected.remove_facts) - (0.2 * extra)
        removal_score = max(0, min(1, removal_score))
        details.append(f"removals: {correct}/{len(expected.remove_facts)} correct, {extra} extra")
    else:
        # No removals expected — penalize false removals
        removal_score = 1.0 if not result.remove_facts else max(0, 1.0 - 0.3 * len(result.remove_facts))
        details.append(f"removals: expected none, got {len(result.remove_facts)}")
    scores.append(("removals", removal_score, 0.3))

    # 2. Addition quality
    n_adds = len(result.add_facts)
    if expected.add_facts_keywords:
        all_text = " ".join(result.add_facts).lower()
        keyword_hits = sum(1 for kw in expected.add_facts_keywords if kw.lower() in all_text)
        kw_score = keyword_hits / len(expected.add_facts_keywords)
        details.append(f"additions: {n_adds} facts, {keyword_hits}/{len(expected.add_facts_keywords)} keywords")
    else:
        kw_score = 1.0 if n_adds <= expected.max_add_facts else 0.5
        details.append(f"additions: {n_adds} facts (no keywords expected)")

    range_ok = expected.min_add_facts <= n_adds <= expected.max_add_facts
    add_score = kw_score * (1.0 if range_ok else 0.7)
    scores.append(("additions", add_score, 0.25))

    # 3. Events present
    n_events = len(result.events)
    if expected.min_events > 0:
        event_score = min(1.0, n_events / expected.min_events)
        details.append(f"events: {n_events} (need ≥{expected.min_events})")
    else:
        event_score = 1.0
        details.append(f"events: {n_events}")
    scores.append(("events", event_score, 0.2))

    # 4. Person updates
    if expected.has_person_update:
        if result.person_updates:
            pu = result.person_updates[0]
            presence_ok = (
                expected.person_presence is None
                or pu.set_presence == expected.person_presence
                or (expected.person_presence in ("leaving", "gone") and pu.set_presence in ("leaving", "gone"))
            )
            person_score = 1.0 if presence_ok else 0.5
            details.append(f"person: presence={pu.set_presence} (expected {expected.person_presence}) {'ok' if presence_ok else 'WRONG'}")
        else:
            person_score = 0.0
            details.append("person: MISSING (expected person update)")
    else:
        person_score = 1.0 if not result.person_updates else 0.8
        details.append(f"person: {'none (ok)' if not result.person_updates else f'{len(result.person_updates)} unexpected'}")
    scores.append(("person", person_score, 0.25))

    # Weighted total
    total = sum(s * w for _, s, w in scores)
    return Score(
        correct_removals=scores[0][1],
        addition_quality=scores[1][1],
        events_present=scores[2][1],
        person_correct=scores[3][1],
        total=total,
        details="; ".join(details),
    )


# ── HTML report ───────────────────────────────────────────────

def score_badge(score: float) -> str:
    if score >= 0.8:
        color = "#2d8a4e"
    elif score >= 0.5:
        color = "#f39c12"
    else:
        color = "#c0392b"
    return f'<span style="background:{color};color:white;padding:2px 8px;border-radius:4px;font-size:13px">{score:.0%}</span>'


def generate_report(results: list[dict]) -> str:
    n = len(results)

    models = list(results[0]["scores"].keys()) if results else []

    # Average scores per model
    avg = {}
    for m in models:
        totals = [r["scores"][m].total for r in results]
        avg[m] = sum(totals) / len(totals) if totals else 0

    # Header
    model_headers = "".join(f"<th>{m}</th>" for m in models)

    rows = []
    for r in results:
        cells = []
        for m in models:
            s = r["scores"][m]
            cells.append(f"""
            <td>
                {score_badge(s.total)} <small>{r['latencies'][m]:.0f}ms</small><br>
                <small style="color:#666">
                    rem={s.correct_removals:.0%} add={s.addition_quality:.0%}
                    evt={s.events_present:.0%} ppl={s.person_correct:.0%}<br>
                    {s.details}
                </small>
            </td>""")
        rows.append(f"""
        <tr>
            <td><b>{r['name']}</b><br><small>{r['description']}</small></td>
            {"".join(cells)}
        </tr>""")

    avg_stats = "".join(
        f'<div class="stat {"winner" if avg[m] == max(avg.values()) else ""}">{m}: <b>{avg[m]:.0%}</b></div>'
        for m in models
    )

    return f"""<!DOCTYPE html>
<html><head><title>Reconcile A/B</title>
<style>
body {{ font-family: system-ui; max-width: 1400px; margin: 20px auto; padding: 0 20px; }}
table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; vertical-align: top; }}
th {{ background: #f5f5f5; }}
.stat {{ display: inline-block; padding: 8px 16px; margin: 4px; background: #f0f0f0; border-radius: 6px; }}
.winner {{ background: #e8f5e9; }}
</style></head>
<body>
<h1>Reconcile A/B: 8B vs 70B on World-State Reconciliation</h1>
<p>Structured JSON output with fact ID references, additions, events, and person tracking.</p>
<p>Scoring: removals (30%), additions (25%), events (20%), person updates (25%)</p>

<h2>Overall Score</h2>
<div>{avg_stats}</div>

<h2>Scenarios ({n})</h2>
<table>
<tr><th>Scenario</th>{model_headers}</tr>
{"".join(rows)}
</table>
</body></html>"""


# ── Main ──────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Reconcile A/B")
    parser.add_argument("--open", action="store_true", help="Open HTML report in browser")
    args = parser.parse_args()

    model_keys = ["groq-llama-8b", "groq-llama"]
    model_names = {k: MODELS[k]["name"] for k in model_keys}

    print(f"Reconcile A/B: {' vs '.join(model_names.values())}")
    print(f"Scenarios: {len(SCENARIOS)}\n")

    results = []
    for i, sc in enumerate(SCENARIOS):
        print(f"[{i+1}/{len(SCENARIOS)}] {sc.name}: {sc.description}")

        world = WorldState(
            static=list(WORLD_STATIC),
            dynamic=dict(sc.dynamic),
            events=list(sc.events),
        )

        row = {"name": sc.name, "description": sc.description, "scores": {}, "latencies": {}, "raw": {}}

        for model_key in model_keys:
            model_config = MODELS[model_key]
            name = model_names[model_key]

            try:
                t0 = time.perf_counter()
                result = reconcile_call(
                    world=world,
                    pending_changes=sc.pending_changes,
                    model_config=model_config,
                )
                elapsed = (time.perf_counter() - t0) * 1000

                s = score_result(result, sc.expected)
                row["scores"][name] = s
                row["latencies"][name] = elapsed
                row["raw"][name] = {
                    "remove_facts": result.remove_facts,
                    "add_facts": result.add_facts,
                    "events": result.events,
                    "person_updates": [
                        {"person_id": pu.person_id, "set_presence": pu.set_presence, "set_name": pu.set_name}
                        for pu in result.person_updates
                    ],
                }

                print(f"  {name}: {s.total:.0%} ({elapsed:.0f}ms) — {s.details}")
            except Exception as e:
                print(f"  {name}: ERROR — {e}")
                row["scores"][name] = Score(details=str(e))
                row["latencies"][name] = 0
                row["raw"][name] = {"error": str(e)}

        print()
        results.append(row)

    # Summary
    print("=" * 60)
    for model_key in model_keys:
        name = model_names[model_key]
        totals = [r["scores"][name].total for r in results]
        avg = sum(totals) / len(totals) if totals else 0
        avg_lat = sum(r["latencies"][name] for r in results) / len(results)
        print(f"{name}: avg={avg:.0%}  lat={avg_lat:.0f}ms")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # JSON (convert Score objects)
    json_data = []
    for r in results:
        jr = {"name": r["name"], "description": r["description"], "latencies": r["latencies"], "raw": r["raw"]}
        jr["scores"] = {k: {"total": v.total, "details": v.details} for k, v in r["scores"].items()}
        json_data.append(jr)

    json_path = RESULTS_DIR / "reconcile_ab.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"\nJSON: {json_path}")

    html_path = RESULTS_DIR / "reconcile_ab.html"
    with open(html_path, "w") as f:
        f.write(generate_report(results))
    print(f"HTML: {html_path}")

    if args.open:
        webbrowser.open(f"file://{html_path}")


if __name__ == "__main__":
    main()
