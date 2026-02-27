#!/usr/bin/env python3
"""Parallel Microservice Benchmark: Sequential vs concurrent post-response calls.

The post-response chain currently runs sequentially:
  expression → script_check → thought → director

These four calls are independent — they don't depend on each other's results.
This benchmark tests whether running them concurrently via ThreadPoolExecutor
reduces wall-clock time without hitting Groq rate limits.

Usage:
    uv run python experiments/fast-llm/parallel_bench.py
    uv run python experiments/fast-llm/parallel_bench.py --runs 10
    uv run python experiments/fast-llm/parallel_bench.py --open
"""

from __future__ import annotations

import json
import os
import sys
import time
import webbrowser
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from character_eng.models import MODELS
from character_eng.world import (
    Beat, WorldState, Script, Goals, EvalResult,
    expression_call, script_check_call, thought_call,
    _llm_call,
)
from character_eng.scenario import ScenarioScript, Stage, StageExit, director_call

RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ── Test fixture ──────────────────────────────────────────────

SYSTEM_PROMPT = """You are Greg, a robot head sitting on a folding table. You have no body and no hands — just a head (eyes and a speaker for a mouth), bolted to the table.
You are aware of being a robot. You are very curious."""

HISTORY = [
    {"role": "system", "content": "[A person walks up to the stand]"},
    {"role": "assistant", "content": "Oh hey! Welcome to Greg's! Best water on the block. Want a flier?"},
    {"role": "user", "content": "Sure, what's on the flier?"},
    {"role": "assistant", "content": "It's got a coupon for a free cup of water! Plus all the info about the stand. Pretty cool, right?"},
]

LAST_LINE = "It's got a coupon for a free cup of water! Plus all the info about the stand. Pretty cool, right?"

BEAT = Beat(
    line="So what brings you out today?",
    intent="Ask about their day and make small talk",
    condition="",
)

WORLD = WorldState(
    static=["Greg is a robot head on a folding table", "A stack of fliers sits on the table"],
    dynamic={"f1": "It is a warm afternoon", "f2": "A person is at the stand"},
)

GAZE_TARGETS = ["person", "stand", "fliers", "sign", "street"]

# Minimal scenario for director
SCENARIO = ScenarioScript(
    name="Test Scenario",
    stages={
        "chatting": Stage(
            name="chatting",
            goal="Have a friendly conversation",
            exits=[
                StageExit(condition="The person says goodbye", goto="goodbye", label="leaves"),
                StageExit(condition="The person asks to buy water", goto="selling", label="buys"),
            ],
        ),
    },
    start="chatting",
    current_stage="chatting",
)

PEOPLE = None  # No people state for simplicity
SCRIPT = Script()
SCRIPT._beats = [BEAT]
SCRIPT._index = 0
GOALS = Goals(long_term="Experience the universe")


# ── Call wrappers ─────────────────────────────────────────────

def call_expression(model_config: dict) -> tuple[str, float]:
    t0 = time.perf_counter()
    result = expression_call(LAST_LINE, model_config, gaze_targets=GAZE_TARGETS)
    elapsed = (time.perf_counter() - t0) * 1000
    return "expression", elapsed


def call_script_check(model_config: dict) -> tuple[str, float]:
    t0 = time.perf_counter()
    result = script_check_call(
        beat=BEAT, history=HISTORY, model_config=model_config, world=WORLD,
    )
    elapsed = (time.perf_counter() - t0) * 1000
    return "script_check", elapsed


def call_thought(model_config: dict) -> tuple[str, float]:
    t0 = time.perf_counter()
    result = thought_call(
        system_prompt=SYSTEM_PROMPT, history=HISTORY, model_config=model_config,
    )
    elapsed = (time.perf_counter() - t0) * 1000
    return "thought", elapsed


def call_director(model_config: dict) -> tuple[str, float]:
    t0 = time.perf_counter()
    result = director_call(
        scenario=SCENARIO, world=WORLD, people=PEOPLE,
        history=HISTORY, model_config=model_config,
    )
    elapsed = (time.perf_counter() - t0) * 1000
    return "director", elapsed


# ── Benchmark modes ───────────────────────────────────────────

def run_sequential(model_config: dict) -> dict:
    """Run all four calls sequentially. Returns per-call latencies + total."""
    t0 = time.perf_counter()
    results = {}

    for call_fn in [call_expression, call_script_check, call_thought, call_director]:
        name, elapsed = call_fn(model_config)
        results[name] = elapsed

    total = (time.perf_counter() - t0) * 1000
    results["total"] = total
    return results


def run_parallel(model_config: dict) -> dict:
    """Run all four calls concurrently. Returns per-call latencies + total."""
    t0 = time.perf_counter()
    results = {}

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(fn, model_config): fn.__name__
            for fn in [call_expression, call_script_check, call_thought, call_director]
        }
        for future in as_completed(futures):
            name, elapsed = future.result()
            results[name] = elapsed

    total = (time.perf_counter() - t0) * 1000
    results["total"] = total
    return results


# ── HTML report ───────────────────────────────────────────────

def generate_report(all_runs: list[dict], model_name: str) -> str:
    seq_runs = [r for r in all_runs if r["mode"] == "sequential"]
    par_runs = [r for r in all_runs if r["mode"] == "parallel"]

    def stat(runs, key):
        vals = [r[key] for r in runs]
        if len(vals) < 2:
            return {"mean": vals[0] if vals else 0, "std": 0, "min": vals[0] if vals else 0, "max": vals[0] if vals else 0}
        return {"mean": statistics.mean(vals), "std": statistics.stdev(vals), "min": min(vals), "max": max(vals)}

    calls = ["expression", "script_check", "thought", "director", "total"]

    rows = []
    for call in calls:
        s = stat(seq_runs, call)
        p = stat(par_runs, call)
        is_total = call == "total"
        speedup = s["mean"] / p["mean"] if p["mean"] > 0 else 0
        style = "font-weight:bold; background:#f5f5f5" if is_total else ""
        winner = "background:#e8f5e9" if p["mean"] < s["mean"] * 0.8 else ""
        rows.append(f"""
        <tr style="{style}">
            <td>{'<b>TOTAL</b>' if is_total else call}</td>
            <td>{s['mean']:.0f}ms (±{s['std']:.0f})</td>
            <td style="{winner}">{p['mean']:.0f}ms (±{p['std']:.0f})</td>
            <td><b>{speedup:.2f}x</b></td>
        </tr>""")

    # Per-run detail
    run_rows = []
    for i, r in enumerate(all_runs):
        mode_badge = f'<span style="background:{"#2196f3" if r["mode"] == "parallel" else "#9e9e9e"};color:white;padding:2px 6px;border-radius:4px">{r["mode"]}</span>'
        run_rows.append(f"""
        <tr>
            <td>{i+1}</td>
            <td>{mode_badge}</td>
            <td>{r.get('expression', 0):.0f}</td>
            <td>{r.get('script_check', 0):.0f}</td>
            <td>{r.get('thought', 0):.0f}</td>
            <td>{r.get('director', 0):.0f}</td>
            <td><b>{r['total']:.0f}</b></td>
        </tr>""")

    n_runs = len(seq_runs)
    seq_total = stat(seq_runs, "total")
    par_total = stat(par_runs, "total")
    speedup = seq_total["mean"] / par_total["mean"] if par_total["mean"] > 0 else 0

    return f"""<!DOCTYPE html>
<html><head><title>Parallel Microservice Benchmark</title>
<style>
body {{ font-family: system-ui; max-width: 1200px; margin: 20px auto; padding: 0 20px; }}
table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
th {{ background: #f5f5f5; }}
.big {{ font-size: 2em; font-weight: bold; }}
.stat {{ display: inline-block; padding: 12px 20px; margin: 4px; background: #f0f0f0; border-radius: 6px; }}
</style></head>
<body>
<h1>Parallel Microservice Benchmark</h1>
<p>Model: <b>{model_name}</b> | Runs: <b>{n_runs}</b> per mode</p>
<p>Tests sequential vs concurrent execution of the 4 post-response microservices.</p>

<div>
    <div class="stat">Sequential: <span class="big">{seq_total['mean']:.0f}ms</span> avg</div>
    <div class="stat">Parallel: <span class="big">{par_total['mean']:.0f}ms</span> avg</div>
    <div class="stat" style="background:#e8f5e9">Speedup: <span class="big">{speedup:.2f}x</span></div>
</div>

<h2>Average by Call</h2>
<table>
<tr><th>Call</th><th>Sequential</th><th>Parallel</th><th>Speedup</th></tr>
{"".join(rows)}
</table>

<h2>All Runs (ms)</h2>
<table>
<tr><th>#</th><th>Mode</th><th>expression</th><th>script_check</th><th>thought</th><th>director</th><th>Total</th></tr>
{"".join(run_rows)}
</table>
</body></html>"""


# ── Main ──────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Parallel Microservice Benchmark")
    parser.add_argument("--runs", type=int, default=5, help="Runs per mode (default: 5)")
    parser.add_argument("--model", default="groq-llama-8b", help="Model key (default: groq-llama-8b)")
    parser.add_argument("--open", action="store_true", help="Open HTML report in browser")
    args = parser.parse_args()

    model_key = args.model
    model_config = MODELS[model_key]
    model_name = model_config["name"]
    n_runs = args.runs

    print(f"Parallel Microservice Benchmark: {model_name}")
    print(f"Runs per mode: {n_runs}\n")

    all_runs = []

    # Warmup
    print("Warmup...")
    call_expression(model_config)
    print()

    # Alternate sequential/parallel to avoid systematic bias
    for i in range(n_runs):
        print(f"[{i+1}/{n_runs}] Sequential...")
        seq = run_sequential(model_config)
        seq["mode"] = "sequential"
        all_runs.append(seq)
        print(f"  total: {seq['total']:.0f}ms  (expr={seq['expression']:.0f} check={seq['script_check']:.0f} thought={seq['thought']:.0f} dir={seq['director']:.0f})")

        print(f"[{i+1}/{n_runs}] Parallel...")
        par = run_parallel(model_config)
        par["mode"] = "parallel"
        all_runs.append(par)
        print(f"  total: {par['total']:.0f}ms  (expr={par['expression']:.0f} check={par['script_check']:.0f} thought={par['thought']:.0f} dir={par['director']:.0f})")
        print()

    # Summary
    seq_totals = [r["total"] for r in all_runs if r["mode"] == "sequential"]
    par_totals = [r["total"] for r in all_runs if r["mode"] == "parallel"]
    seq_mean = statistics.mean(seq_totals)
    par_mean = statistics.mean(par_totals)
    speedup = seq_mean / par_mean if par_mean > 0 else 0

    print("=" * 60)
    print(f"Sequential: {seq_mean:.0f}ms avg (±{statistics.stdev(seq_totals):.0f})" if len(seq_totals) > 1 else f"Sequential: {seq_mean:.0f}ms")
    print(f"Parallel:   {par_mean:.0f}ms avg (±{statistics.stdev(par_totals):.0f})" if len(par_totals) > 1 else f"Parallel:   {par_mean:.0f}ms")
    print(f"Speedup:    {speedup:.2f}x")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "parallel_bench.json"
    with open(json_path, "w") as f:
        json.dump(all_runs, f, indent=2)
    print(f"\nJSON: {json_path}")

    html_path = RESULTS_DIR / "parallel_bench.html"
    with open(html_path, "w") as f:
        f.write(generate_report(all_runs, model_name))
    print(f"HTML: {html_path}")

    if args.open:
        webbrowser.open(f"file://{html_path}")


if __name__ == "__main__":
    main()
