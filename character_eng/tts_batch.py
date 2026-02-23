"""Batch TTS comparison — run lines from a text file through multiple backends, save WAVs + report.

Runnable via: uv run -m character_eng.tts_batch tts_lines.txt [--backends kokoro,pocket,kani]

Output: one WAV per backend (all lines concatenated with beep separators) + report.md.
"""

from __future__ import annotations

import argparse
import math
import struct
import sys
import threading
import time
import wave
from datetime import datetime
from pathlib import Path

from rich.console import Console

from character_eng.config import load_config

console = Console()
LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"

SAMPLE_RATE = 24000
BYTES_PER_SAMPLE = 2  # int16
CHANNELS = 1

# Separator between lines: short beep + silence
BEEP_FREQ_HZ = 880  # A5
BEEP_DURATION_S = 0.05
BEEP_AMPLITUDE = 3000  # low volume (max int16 = 32767)
SILENCE_DURATION_S = 0.1


def _make_separator() -> bytes:
    """Generate a low-volume beep followed by silence as int16 PCM."""
    # Beep
    beep_samples = int(SAMPLE_RATE * BEEP_DURATION_S)
    beep = bytearray()
    for i in range(beep_samples):
        val = int(BEEP_AMPLITUDE * math.sin(2 * math.pi * BEEP_FREQ_HZ * i / SAMPLE_RATE))
        beep += struct.pack("<h", val)
    # Silence
    silence_samples = int(SAMPLE_RATE * SILENCE_DURATION_S)
    silence = b"\x00\x00" * silence_samples
    return bytes(beep) + silence

# Default backends in order, with their default server URLs/ports
DEFAULT_BACKENDS = ["kokoro", "pocket", "kani", "local", "elevenlabs"]

BACKEND_PORTS = {
    "kokoro": "http://localhost:8880",
    "pocket": "http://localhost:8003",
    "kani": "http://localhost:8000",
    "mio": "http://localhost:8001",
    "kitten": "http://localhost:8005",
}

# Hardcoded voice ID for ElevenLabs batch tool
ELEVENLABS_VOICE_ID = "qXpMhyvQqiRxWQs4qSSB"


def _create_backend(backend: str, ref_audio: str, device: str, server_url: str):
    """Create a TTS backend instance. Returns (tts, on_audio_cb, metrics_dict, pcm_buf)."""
    metrics = {
        "ttfa": None,
        "total_bytes": 0,
        "flush_time": None,
    }
    pcm_buf = bytearray()
    lock = threading.Lock()

    def on_audio(pcm: bytes):
        with lock:
            if metrics["ttfa"] is None:
                metrics["ttfa"] = time.perf_counter()
            metrics["total_bytes"] += len(pcm)
            pcm_buf.extend(pcm)

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
    elif backend == "kani":
        from character_eng.kani_tts import KaniTTS

        tts = KaniTTS(
            on_audio=on_audio,
            server_url=server_url or BACKEND_PORTS["kani"],
        )
    elif backend == "kokoro":
        from character_eng.kokoro_tts import KokoroTTS

        tts = KokoroTTS(
            on_audio=on_audio,
            server_url=server_url or BACKEND_PORTS["kokoro"],
        )
    elif backend == "pocket":
        from character_eng.pocket_tts import PocketTTS

        tts = PocketTTS(
            on_audio=on_audio,
            server_url=server_url or BACKEND_PORTS["pocket"],
        )
    elif backend == "mio":
        from character_eng.mio_tts import MioTTS

        tts = MioTTS(
            on_audio=on_audio,
            server_url=server_url or BACKEND_PORTS["mio"],
        )
    elif backend == "kitten":
        from character_eng.kitten_tts import KittenTTS

        tts = KittenTTS(
            on_audio=on_audio,
            server_url=server_url or BACKEND_PORTS["kitten"],
        )
    elif backend == "elevenlabs":
        from character_eng.voice import ElevenLabsTTS

        tts = ElevenLabsTTS(on_audio=on_audio, voice_id=ELEVENLABS_VOICE_ID)
    else:
        raise ValueError(f"Unknown TTS backend: {backend}")

    return tts, metrics, pcm_buf


def _run_one(tts, metrics: dict, text: str, timeout: float = 60.0) -> dict:
    """Run a single TTS generation and return timing metrics."""
    metrics["ttfa"] = None
    metrics["total_bytes"] = 0

    t_start = time.perf_counter()
    tts.send_text(text)
    tts.flush()

    tts.wait_for_done(timeout=timeout)
    t_end = time.perf_counter()

    tts.close()

    total_ms = (t_end - t_start) * 1000
    ttfa_ms = (metrics["ttfa"] - t_start) * 1000 if metrics["ttfa"] else None
    total_bytes = metrics["total_bytes"]

    total_samples = total_bytes // BYTES_PER_SAMPLE
    audio_duration_ms = (total_samples / SAMPLE_RATE) * 1000 if total_samples > 0 else 0
    rtf = (total_ms / audio_duration_ms) if audio_duration_ms > 0 else float("inf")

    return {
        "ttfa_ms": round(ttfa_ms, 1) if ttfa_ms is not None else None,
        "total_ms": round(total_ms, 1),
        "audio_duration_ms": round(audio_duration_ms, 1),
        "rtf": round(rtf, 3),
        "total_bytes": total_bytes,
    }


def _save_wav(path: Path, pcm: bytes):
    """Write raw int16 PCM as a 24kHz/16-bit/mono WAV."""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(BYTES_PER_SAMPLE)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm)


def _check_backend(backend: str, ref_audio: str, device: str, server_url: str) -> bool:
    """Check if a backend is reachable by doing a quick test generation."""
    try:
        tts, metrics, pcm_buf = _create_backend(backend, ref_audio, device, server_url)
        _run_one(tts, metrics, "Test.")
        return metrics["total_bytes"] > 0
    except Exception:
        return False


def _write_report(
    out_dir: Path,
    input_file: str,
    lines: list[str],
    results: list[dict],
):
    """Write report.md with a metrics table."""
    now = datetime.now().strftime("%Y-%m-%d")
    report = f"# TTS Batch Comparison\n"
    report += f"**Lines:** {input_file} ({len(lines)} lines)\n"
    report += f"**Date:** {now}\n\n"
    report += "| # | Text | Backend | TTFA (ms) | Total (ms) | Audio (ms) | RTF |\n"
    report += "|---|------|---------|-----------|------------|------------|-----|\n"

    for r in results:
        text_preview = r["text"][:40] + ("..." if len(r["text"]) > 40 else "")
        ttfa = str(round(r["ttfa_ms"])) if r["ttfa_ms"] is not None else "N/A"
        report += (
            f"| {r['line_num']:02d} "
            f"| {text_preview} "
            f"| {r['backend']} "
            f"| {ttfa} "
            f"| {round(r['total_ms'])} "
            f"| {round(r['audio_duration_ms'])} "
            f"| {r['rtf']:.3f} |\n"
        )

    report_path = out_dir / "report.md"
    report_path.write_text(report)
    return report_path


def main():
    parser = argparse.ArgumentParser(description="Batch TTS comparison — generate WAVs + metrics report")
    parser.add_argument("input_file", help="Text file with one line per utterance")
    parser.add_argument(
        "--backends",
        help=f"Comma-separated backends (default: {','.join(DEFAULT_BACKENDS)})",
    )
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        console.print(f"[red]File not found: {input_path}[/red]")
        sys.exit(1)

    lines = [l.strip() for l in input_path.read_text().splitlines() if l.strip()]
    if not lines:
        console.print("[red]No non-empty lines in input file.[/red]")
        sys.exit(1)

    backends = args.backends.split(",") if args.backends else DEFAULT_BACKENDS

    cfg = load_config()
    ref_audio = cfg.voice.ref_audio
    device = cfg.voice.tts_device
    server_url = cfg.voice.tts_server_url

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = LOGS_DIR / f"tts_batch_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold]TTS Batch Comparison[/bold]")
    console.print(f"Input: {input_path} ({len(lines)} lines)")
    console.print(f"Backends: {', '.join(backends)}")
    console.print(f"Output: {out_dir}\n")

    # Probe backends
    reachable = []
    for b in backends:
        console.print(f"  Checking {b}...", end=" ")
        if _check_backend(b, ref_audio, device, server_url):
            console.print("[green]OK[/green]")
            reachable.append(b)
        else:
            console.print("[red]unreachable, skipping[/red]")

    if not reachable:
        console.print("[red]No reachable backends. Exiting.[/red]")
        sys.exit(1)

    console.print(f"\nUsing backends: {', '.join(reachable)}\n")

    all_results = []
    separator = _make_separator()

    for b in reachable:
        console.print(f"[bold cyan]{b}[/bold cyan]")

        # Warmup
        console.print("  [dim]Warmup...[/dim]")
        try:
            tts, metrics, pcm_buf = _create_backend(b, ref_audio, device, server_url)
            _run_one(tts, metrics, "Warmup sentence.")
        except Exception:
            pass

        combined_pcm = bytearray()

        for i, line in enumerate(lines, 1):
            tts, metrics, pcm_buf = _create_backend(b, ref_audio, device, server_url)
            try:
                result = _run_one(tts, metrics, line)
                result["line_num"] = i
                result["text"] = line
                result["backend"] = b

                # Append to combined WAV (with separator between lines)
                if combined_pcm:
                    combined_pcm.extend(separator)
                combined_pcm.extend(pcm_buf)

                ttfa_str = f"{result['ttfa_ms']}ms" if result["ttfa_ms"] is not None else "N/A"
                console.print(
                    f"  {i:02d}. TTFA={ttfa_str}  "
                    f"Total={result['total_ms']}ms  "
                    f"Audio={result['audio_duration_ms']}ms  "
                    f"RTF={result['rtf']:.3f}  "
                    f"[dim]{line[:50]}[/dim]"
                )
                all_results.append(result)

            except Exception as e:
                console.print(f"  {i:02d}. [red]ERROR: {e}[/red]  [dim]{line[:50]}[/dim]")
                all_results.append({
                    "line_num": i,
                    "text": line,
                    "backend": b,
                    "ttfa_ms": None,
                    "total_ms": 0,
                    "audio_duration_ms": 0,
                    "rtf": float("inf"),
                    "error": str(e),
                })

        # Save one combined WAV per backend
        if combined_pcm:
            wav_path = out_dir / f"{b}.wav"
            _save_wav(wav_path, bytes(combined_pcm))
            dur_s = len(combined_pcm) / BYTES_PER_SAMPLE / SAMPLE_RATE
            console.print(f"  [green]Saved {wav_path.name} ({dur_s:.1f}s)[/green]")

        console.print()

    # Write report
    report_path = _write_report(out_dir, str(input_path), lines, all_results)
    console.print(f"[bold green]Done.[/bold green] Output: {out_dir}")
    console.print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
