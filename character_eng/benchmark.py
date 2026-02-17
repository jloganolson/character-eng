"""Model speed benchmark for chat, director, and think call patterns."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from character_eng.models import MODELS
from character_eng.prompts import load_prompt
from character_eng.world import (
    Script,
    _make_client,
    director_call,
    eval_call,
    load_goals,
    load_world_state,
)

load_dotenv()

console = Console()
LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
CHARACTER = "greg"


def _is_local_server_up(base_url: str) -> bool:
    try:
        health_url = base_url.replace("/v1", "/health")
        urllib.request.urlopen(health_url, timeout=1)
        return True
    except Exception:
        return False


def get_available_models() -> list[tuple[str, dict]]:
    """Return (key, config) pairs for models that are currently usable."""
    available = []
    for key, cfg in MODELS.items():
        if cfg.get("hidden"):
            continue
        if cfg.get("local"):
            if _is_local_server_up(cfg["base_url"]):
                available.append((key, cfg))
        elif os.environ.get(cfg.get("api_key_env", "")):
            available.append((key, cfg))
    return available


def bench_chat(model_config: dict) -> dict:
    """Benchmark streaming chat. Returns timing dict."""
    world = load_world_state(CHARACTER)
    goals = load_goals(CHARACTER)
    system_prompt = load_prompt(CHARACTER, world_state=world, goals=goals)

    client = _make_client(model_config)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Hey Greg, what's that orb?"},
    ]

    kwargs = dict(
        model=model_config["model"],
        messages=messages,
        stream=True,
    )
    if model_config["stream_usage"]:
        kwargs["stream_options"] = {"include_usage": True}

    t_start = time.perf_counter()
    stream = client.chat.completions.create(**kwargs)

    ttft = None
    chunks = []
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            if ttft is None:
                ttft = time.perf_counter() - t_start
            chunks.append(chunk.choices[0].delta.content)

    t_total = time.perf_counter() - t_start
    text = "".join(chunks)

    return {
        "scenario": "chat",
        "ttft_ms": round(ttft * 1000, 1) if ttft else None,
        "total_ms": round(t_total * 1000, 1),
        "response": text,
    }


def bench_director(model_config: dict) -> dict:
    """Benchmark structured JSON director call. Returns timing dict."""
    world = load_world_state(CHARACTER)
    if world is None:
        return {"scenario": "director", "ttft_ms": None, "total_ms": 0, "response": "no world state"}

    t_start = time.perf_counter()
    update = director_call(world, "The orb begins to glow faintly", model_config)
    t_total = time.perf_counter() - t_start

    result = {
        "remove_facts": update.remove_facts,
        "add_facts": update.add_facts,
        "events": update.events,
    }

    return {
        "scenario": "director",
        "ttft_ms": round(t_total * 1000, 1),
        "total_ms": round(t_total * 1000, 1),
        "response": json.dumps(result, indent=2),
    }


def bench_eval(model_config: dict) -> dict:
    """Benchmark structured JSON eval call. Returns timing dict."""
    world = load_world_state(CHARACTER)
    goals = load_goals(CHARACTER)
    system_prompt = load_prompt(CHARACTER, world_state=world, goals=goals)

    history = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Hey Greg, what's that orb?"},
        {"role": "assistant", "content": "Oh, this? *glances at orb* It's... complicated."},
    ]

    t_start = time.perf_counter()
    result = eval_call(
        system_prompt=system_prompt,
        world=world,
        history=history,
        model_config=model_config,
        goals=goals,
        script=Script(),
    )
    t_total = time.perf_counter() - t_start

    result_dict = {
        "thought": result.thought,
        "gaze": result.gaze,
        "expression": result.expression,
        "script_status": result.script_status,
    }

    return {
        "scenario": "eval",
        "ttft_ms": round(t_total * 1000, 1),
        "total_ms": round(t_total * 1000, 1),
        "response": json.dumps(result_dict, indent=2),
    }


SCENARIOS = [
    ("chat", bench_chat),
    ("director", bench_director),
    ("eval", bench_eval),
]


def run_benchmark(models: list[tuple[str, dict]], runs: int) -> list[dict]:
    """Run all scenarios for all models. Returns list of run results."""
    all_results = []

    for model_key, model_config in models:
        console.print(f"\n[bold]{'=' * 60}[/bold]")
        console.print(f"[bold]Model: {model_config['name']}[/bold] ({model_key})")
        console.print(f"[bold]{'=' * 60}[/bold]")

        for scenario_name, bench_fn in SCENARIOS:
            console.print(f"\n[cyan]--- {scenario_name} ---[/cyan]")

            for i in range(runs):
                console.print(f"[dim]Run {i + 1}/{runs}...[/dim]", end=" ")
                try:
                    result = bench_fn(model_config)
                    result["model"] = model_key
                    result["model_name"] = model_config["name"]
                    result["run"] = i + 1
                    all_results.append(result)

                    console.print(
                        f"TTFT [bold]{result['ttft_ms']}[/bold] ms  "
                        f"Total [bold]{result['total_ms']}[/bold] ms"
                    )
                    # Print response (truncate for readability)
                    resp = result["response"]
                    if len(resp) > 200:
                        resp = resp[:200] + "..."
                    console.print(f"[dim]{resp}[/dim]")
                except Exception as e:
                    console.print(f"[red]ERROR: {e}[/red]")
                    all_results.append({
                        "model": model_key,
                        "model_name": model_config["name"],
                        "scenario": scenario_name,
                        "run": i + 1,
                        "error": str(e),
                    })

    return all_results


def print_summary(results: list[dict]):
    """Print a rich summary table aggregated by model + scenario."""
    table = Table(title="Benchmark Summary")
    table.add_column("Model", style="bold")
    table.add_column("Scenario", style="cyan")
    table.add_column("Avg TTFT (ms)", justify="right")
    table.add_column("Std Dev (ms)", justify="right")
    table.add_column("Avg Total (ms)", justify="right")

    # Group by (model, scenario)
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in results:
        if "error" in r:
            continue
        key = (r["model_name"], r["scenario"])
        groups.setdefault(key, []).append(r)

    for (model_name, scenario), runs in groups.items():
        ttfts = [r["ttft_ms"] for r in runs if r["ttft_ms"] is not None]
        totals = [r["total_ms"] for r in runs]

        avg_ttft = statistics.mean(ttfts) if ttfts else 0
        std_ttft = statistics.stdev(ttfts) if len(ttfts) > 1 else 0
        avg_total = statistics.mean(totals) if totals else 0

        table.add_row(
            model_name,
            scenario,
            f"{avg_ttft:.1f}",
            f"{std_ttft:.1f}",
            f"{avg_total:.1f}",
        )

    console.print()
    console.print(table)


def save_log(results: list[dict]) -> str:
    """Save full results to logs/ as JSON. Returns timestamp for shared naming."""
    LOGS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"benchmark_{timestamp}.json"
    data = {
        "type": "benchmark",
        "timestamp": datetime.now().isoformat(),
        "results": results,
    }
    log_path.write_text(json.dumps(data, indent=2))
    console.print(f"\n[dim]Log: {log_path}[/dim]")
    return timestamp


def _esc(text: str) -> str:
    """HTML-escape a string."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def save_html(results: list[dict], timestamp: str):
    """Save an HTML report with summary table and per-run detail."""
    html_path = LOGS_DIR / f"benchmark_{timestamp}.html"

    # Build summary rows
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for r in results:
        if "error" in r:
            continue
        key = (r["model"], r["model_name"], r["scenario"])
        groups.setdefault(key, []).append(r)

    summary_rows = []
    for (model_key, model_name, scenario), runs in groups.items():
        ttfts = [r["ttft_ms"] for r in runs if r["ttft_ms"] is not None]
        totals = [r["total_ms"] for r in runs]
        avg_ttft = statistics.mean(ttfts) if ttfts else 0
        std_ttft = statistics.stdev(ttfts) if len(ttfts) > 1 else 0
        avg_total = statistics.mean(totals) if totals else 0
        summary_rows.append((model_name, model_key, scenario, avg_ttft, std_ttft, avg_total))

    summary_html = ""
    for model_name, model_key, scenario, avg_ttft, std_ttft, avg_total in summary_rows:
        summary_html += (
            f"<tr><td>{_esc(model_name)}</td><td><code>{_esc(model_key)}</code></td>"
            f"<td>{_esc(scenario)}</td>"
            f"<td class='num'>{avg_ttft:.1f}</td><td class='num'>{std_ttft:.1f}</td>"
            f"<td class='num'>{avg_total:.1f}</td></tr>\n"
        )

    # Build detail rows grouped by model
    detail_html = ""
    current_model = None
    for r in results:
        model_name = r.get("model_name", "?")
        if model_name != current_model:
            current_model = model_name
            detail_html += f"<tr><td colspan='5' class='model-header'>{_esc(model_name)}</td></tr>\n"

        if "error" in r:
            detail_html += (
                f"<tr><td>{_esc(r['scenario'])}</td><td>#{r['run']}</td>"
                f"<td colspan='3' class='error'>ERROR: {_esc(r['error'])}</td></tr>\n"
            )
        else:
            resp = r.get("response", "")
            detail_html += (
                f"<tr><td>{_esc(r['scenario'])}</td><td>#{r['run']}</td>"
                f"<td class='num'>{r['ttft_ms']}</td><td class='num'>{r['total_ms']}</td>"
                f"<td><details><summary>{_esc(resp[:80])}{'...' if len(resp) > 80 else ''}</summary>"
                f"<pre>{_esc(resp)}</pre></details></td></tr>\n"
            )

    ts_display = datetime.strptime(timestamp, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Benchmark {timestamp}</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 1200px; margin: 2em auto; padding: 0 1em; background: #0d1117; color: #c9d1d9; }}
  h1, h2 {{ color: #58a6ff; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
  th, td {{ padding: 8px 12px; border: 1px solid #30363d; text-align: left; }}
  th {{ background: #161b22; color: #58a6ff; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  tr:nth-child(even) {{ background: #161b22; }}
  .model-header {{ background: #1f2937 !important; font-weight: bold; color: #58a6ff; font-size: 1.1em; }}
  .error {{ color: #f85149; }}
  details summary {{ cursor: pointer; color: #8b949e; font-size: 0.9em; max-width: 600px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  details pre {{ background: #161b22; padding: 1em; border-radius: 6px; overflow-x: auto; font-size: 0.85em; white-space: pre-wrap; word-break: break-word; }}
  code {{ background: #30363d; padding: 2px 6px; border-radius: 3px; font-size: 0.9em; }}
  .meta {{ color: #8b949e; font-size: 0.9em; }}
</style></head><body>
<h1>Benchmark Report</h1>
<p class="meta">{ts_display} &middot; {len(results)} runs</p>

<h2>Summary</h2>
<table>
<tr><th>Model</th><th>Key</th><th>Scenario</th><th>Avg TTFT (ms)</th><th>Std Dev (ms)</th><th>Avg Total (ms)</th></tr>
{summary_html}</table>

<h2>Detail</h2>
<table>
<tr><th>Scenario</th><th>Run</th><th>TTFT (ms)</th><th>Total (ms)</th><th>Response</th></tr>
{detail_html}</table>
</body></html>"""

    html_path.write_text(html)
    console.print(f"[dim]Report: {html_path}[/dim]")


def merge_logs(paths: list[str]) -> list[dict]:
    """Load and merge results from multiple benchmark JSON logs."""
    all_results = []
    for p in paths:
        data = json.loads(Path(p).read_text())
        all_results.extend(data["results"])
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Benchmark LLM call patterns")
    parser.add_argument("--model", type=str, default=None, help="Model key to benchmark (default: all available)")
    parser.add_argument("--runs", type=int, default=3, help="Runs per scenario per model (default: 3)")
    parser.add_argument("--merge", nargs="+", metavar="JSON", help="Merge existing benchmark JSON logs into one HTML report (no new runs)")
    args = parser.parse_args()

    if args.merge:
        results = merge_logs(args.merge)
        console.print(f"[bold]Merged {len(results)} runs from {len(args.merge)} log(s)[/bold]")
        print_summary(results)
        timestamp = save_log(results)
        save_html(results, timestamp)
        return

    available = get_available_models()
    if not available:
        console.print("[red]No models available. Set an API key in .env or start a local vLLM server.[/red]")
        return

    if args.model:
        matches = [(k, c) for k, c in available if k == args.model]
        if not matches:
            all_keys = [k for k, _ in available]
            console.print(f"[red]Model '{args.model}' not available. Available: {', '.join(all_keys)}[/red]")
            return
        models = matches
    else:
        models = available

    console.print(f"[bold]Benchmarking {len(models)} model(s), {args.runs} runs per scenario[/bold]")
    console.print(f"[dim]Character: {CHARACTER}, Scenarios: chat (streaming), director (JSON), eval (JSON)[/dim]")

    results = run_benchmark(models, args.runs)
    print_summary(results)
    timestamp = save_log(results)
    save_html(results, timestamp)


if __name__ == "__main__":
    main()
