"""TTS backend speed benchmark — measures TTFA, total time, and RTF.

Runnable via: uv run -m character_eng.tts_benchmark [--backend ...] [--runs N]
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

from character_eng.config import load_config

console = Console()
LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"

TEST_SENTENCES = [
    {
        "label": "short",
        "text": "Hello there, how are you doing today?",
    },
    {
        "label": "medium",
        "text": (
            "I've been staring at this orb for what feels like an eternity now. "
            "It pulses with a strange light, almost as if it's trying to "
            "communicate something to me. I wonder what secrets it holds."
        ),
    },
    {
        "label": "long",
        "text": (
            "You know, being a disembodied robot head on a desk isn't exactly "
            "what I had in mind when I imagined my existence. But here we are. "
            "The orb keeps glowing, the world keeps turning, and I keep thinking "
            "about the universe and all its mysteries. Sometimes I wonder if "
            "there's more to this existence than just sitting here, watching "
            "and waiting for something extraordinary to happen."
        ),
    },
]

SAMPLE_RATE = 24000  # All backends deliver at this rate (after resampling)
BYTES_PER_SAMPLE = 2  # int16


def _create_backend(backend: str, ref_audio: str, device: str, server_url: str):
    """Create a TTS backend instance. Returns (tts, on_audio_cb, metrics_dict)."""
    metrics = {
        "ttfa": None,
        "total_bytes": 0,
        "flush_time": None,
    }
    lock = threading.Lock()

    def on_audio(pcm: bytes):
        with lock:
            if metrics["ttfa"] is None:
                metrics["ttfa"] = time.perf_counter()
            metrics["total_bytes"] += len(pcm)

    if backend == "local":
        from character_eng.local_tts import LocalTTS, load_model

        load_model(
            model_id="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            device=device,
            ref_audio_path=ref_audio,
        )
        tts = LocalTTS(
            on_audio=on_audio,
            ref_audio_path=ref_audio,
            device=device,
        )
    elif backend == "chatterbox":
        from character_eng.chatterbox_tts import ChatterboxTTS, load_model as cb_load

        cb_load(device)
        tts = ChatterboxTTS(
            on_audio=on_audio,
            ref_audio_path=ref_audio,
            device=device,
        )
    elif backend == "kani":
        from character_eng.kani_tts import KaniTTS

        tts = KaniTTS(
            on_audio=on_audio,
            server_url=server_url or "http://localhost:8000",
        )
    elif backend == "mio":
        from character_eng.mio_tts import MioTTS

        tts = MioTTS(
            on_audio=on_audio,
            server_url=server_url or "http://localhost:8001",
        )
    elif backend == "elevenlabs":
        from character_eng.voice import ElevenLabsTTS

        tts = ElevenLabsTTS(on_audio=on_audio)
    else:
        raise ValueError(f"Unknown TTS backend: {backend}")

    return tts, metrics


def _run_one(tts, metrics: dict, text: str, timeout: float = 60.0) -> dict:
    """Run a single TTS generation and return timing metrics."""
    metrics["ttfa"] = None
    metrics["total_bytes"] = 0

    t_start = time.perf_counter()
    tts.send_text(text)
    tts.flush()
    metrics["flush_time"] = time.perf_counter()

    tts.wait_for_done(timeout=timeout)
    t_end = time.perf_counter()

    tts.close()

    total_ms = (t_end - t_start) * 1000
    ttfa_ms = (metrics["ttfa"] - t_start) * 1000 if metrics["ttfa"] else None
    total_bytes = metrics["total_bytes"]

    # Calculate audio duration from PCM byte count
    total_samples = total_bytes // BYTES_PER_SAMPLE
    audio_duration_s = total_samples / SAMPLE_RATE if total_samples > 0 else 0
    audio_duration_ms = audio_duration_s * 1000

    # Real-time factor: generation time / audio duration (lower = faster)
    rtf = (total_ms / audio_duration_ms) if audio_duration_ms > 0 else float("inf")

    return {
        "ttfa_ms": round(ttfa_ms, 1) if ttfa_ms is not None else None,
        "total_ms": round(total_ms, 1),
        "audio_duration_ms": round(audio_duration_ms, 1),
        "rtf": round(rtf, 3),
        "total_bytes": total_bytes,
    }


def run_benchmark(backend: str, runs: int, ref_audio: str, device: str, server_url: str) -> dict:
    """Run the full benchmark for a backend. Returns results dict."""
    console.print(f"\n[bold]Benchmarking: {backend}[/bold]")
    console.print(f"  Runs per sentence: {runs}")

    # Warmup run — first request to any backend incurs model/connection warmup.
    # Exclude from metrics so results reflect steady-state performance.
    console.print("  [dim]Warmup run...[/dim]")
    tts, metrics = _create_backend(backend, ref_audio, device, server_url)
    try:
        _run_one(tts, metrics, "Warmup sentence for steady state benchmarking.")
    except Exception:
        pass  # warmup failure is ok — backend may recover
    console.print("  [dim]Warmup done.[/dim]")

    all_results = []

    for sentence in TEST_SENTENCES:
        label = sentence["label"]
        text = sentence["text"]
        word_count = len(text.split())
        console.print(f"\n  [{label}] ({word_count} words): {text[:60]}...")

        sentence_runs = []
        for r in range(runs):
            tts, metrics = _create_backend(backend, ref_audio, device, server_url)
            try:
                result = _run_one(tts, metrics, text)
                result["run"] = r + 1
                result["label"] = label
                result["text"] = text
                sentence_runs.append(result)

                ttfa_str = f"{result['ttfa_ms']}ms" if result["ttfa_ms"] is not None else "N/A"
                console.print(
                    f"    Run {r + 1}: TTFA={ttfa_str}, "
                    f"Total={result['total_ms']}ms, "
                    f"Audio={result['audio_duration_ms']}ms, "
                    f"RTF={result['rtf']}"
                )
            except Exception as e:
                console.print(f"    Run {r + 1}: [red]ERROR: {e}[/red]")
                sentence_runs.append({
                    "run": r + 1,
                    "label": label,
                    "text": text,
                    "error": str(e),
                })

        all_results.extend(sentence_runs)

    return {
        "backend": backend,
        "runs_per_sentence": runs,
        "results": all_results,
    }


def _print_summary(data: dict):
    """Print a rich summary table."""
    table = Table(title=f"TTS Benchmark: {data['backend']}")
    table.add_column("Sentence", style="cyan")
    table.add_column("Avg TTFA (ms)", justify="right")
    table.add_column("Avg Total (ms)", justify="right")
    table.add_column("Avg Audio (ms)", justify="right")
    table.add_column("Avg RTF", justify="right")

    for label in ["short", "medium", "long"]:
        runs = [r for r in data["results"] if r.get("label") == label and "error" not in r]
        if not runs:
            table.add_row(label, "N/A", "N/A", "N/A", "N/A")
            continue

        ttfas = [r["ttfa_ms"] for r in runs if r["ttfa_ms"] is not None]
        totals = [r["total_ms"] for r in runs]
        audios = [r["audio_duration_ms"] for r in runs]
        rtfs = [r["rtf"] for r in runs]

        avg_ttfa = f"{sum(ttfas) / len(ttfas):.1f}" if ttfas else "N/A"
        avg_total = f"{sum(totals) / len(totals):.1f}"
        avg_audio = f"{sum(audios) / len(audios):.1f}"
        avg_rtf = f"{sum(rtfs) / len(rtfs):.3f}"

        table.add_row(label, avg_ttfa, avg_total, avg_audio, avg_rtf)

    console.print()
    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="TTS backend speed benchmark")
    parser.add_argument(
        "--backend",
        choices=["local", "chatterbox", "kani", "mio", "elevenlabs"],
        help="TTS backend to benchmark (default: from config.toml)",
    )
    parser.add_argument(
        "--runs", type=int, default=3,
        help="Repetitions per sentence (default: 3)",
    )
    args = parser.parse_args()

    cfg = load_config()
    backend = args.backend or cfg.voice.tts_backend
    ref_audio = cfg.voice.ref_audio
    device = cfg.voice.tts_device
    server_url = cfg.voice.tts_server_url

    console.print(f"[bold]TTS Benchmark[/bold]")
    console.print(f"Backend: {backend}")

    data = run_benchmark(backend, args.runs, ref_audio, device, server_url)

    _print_summary(data)

    # Save JSON log
    LOGS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"tts_benchmark_{timestamp}.json"
    with open(log_path, "w") as f:
        json.dump(data, f, indent=2)
    console.print(f"\n[dim]Log saved: {log_path}[/dim]")


if __name__ == "__main__":
    main()
