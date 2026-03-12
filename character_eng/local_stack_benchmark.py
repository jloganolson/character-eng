"""Repeatable local-stack benchmark for the GPU-heavy vision + TTS services.

The benchmark assumes the heavy stack is already running, ideally via:

    VISION_NO_CAMERA=1 ./scripts/run_heavy.sh

It then injects the same prerecorded vision frames on every machine so the
vision workload is identical across runs. The report focuses on latency for:

- vision polling (`/snapshot`, `/frame.jpg`, annotated frame)
- vision VLM question answering (`/ask` + `/result/<slot>`)
- Pocket-TTS first-audio and full synthesis
- overlapped vision + TTS + polling load to expose contention
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from character_eng.pocket_tts import PocketTTS
from character_eng.vision.client import VisionClient

console = Console()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
DEFAULT_FRAME_DIR = PROJECT_ROOT / "services" / "vision" / "test_set"
DEFAULT_QUESTION = "Describe the person in frame and what they are doing. Keep it to 1-2 sentences."
DEFAULT_TTS_TEXT = (
    "Free water, free advice. If you want the short version, take a flyer and ask me one question."
)
DEFAULT_OUTPUT_PREFIX = "local_stack_benchmark"


@dataclass
class PCMCollector:
    parts: list[bytes]
    start_at: float
    first_chunk_at: float | None = None

    def __init__(self) -> None:
        self.parts = []
        self.start_at = time.perf_counter()
        self.first_chunk_at = None

    def __call__(self, data: bytes) -> None:
        if self.first_chunk_at is None:
            self.first_chunk_at = time.perf_counter()
        self.parts.append(bytes(data))

    @property
    def pcm(self) -> bytes:
        return b"".join(self.parts)


class FrameFeeder:
    """Continuously inject prerecorded frames into the vision service."""

    def __init__(self, client: VisionClient, frame_paths: list[Path], fps: float) -> None:
        if not frame_paths:
            raise ValueError("frame_paths cannot be empty")
        self._client = client
        self._frames = [(path, path.read_bytes()) for path in frame_paths]
        self._interval = 1.0 / max(fps, 0.1)
        self._index = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def inject_now(self, index: int) -> str:
        with self._lock:
            self._index = index % len(self._frames)
            frame_path, payload = self._frames[self._index]
        self._client.inject_frame(payload)
        return frame_path.name

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                frame_path, payload = self._frames[self._index]
                self._index = (self._index + 1) % len(self._frames)
            try:
                self._client.inject_frame(payload)
            except Exception:
                pass
            self._stop.wait(self._interval)


class VisionPoller:
    """Adds background vision polling load during overlap scenarios."""

    def __init__(self, client: VisionClient, interval: float) -> None:
        self._client = client
        self._interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[dict[str, float]] = []
        self._errors: list[str] = []
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, float | int]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        with self._lock:
            samples = list(self._samples)
            errors = list(self._errors)
        metrics: dict[str, float | int] = {
            "poll_count": len(samples),
            "poll_errors": len(errors),
        }
        for key in ("snapshot_ms", "raw_frame_ms", "annotated_frame_ms", "total_ms"):
            values = [sample[key] for sample in samples if key in sample]
            if values:
                metrics[f"{key}_mean"] = round(statistics.mean(values), 1)
                metrics[f"{key}_p95"] = round(percentile(values, 95), 1)
        return metrics

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                sample = measure_vision_poll_once(self._client)
                with self._lock:
                    self._samples.append(sample)
            except Exception as exc:
                with self._lock:
                    self._errors.append(str(exc))
            self._stop.wait(self._interval)


def json_request(
    url: str,
    *,
    method: str = "GET",
    data: dict[str, Any] | None = None,
    timeout: float = 10.0,
) -> dict[str, Any]:
    payload = None
    headers = {}
    if data is not None:
        payload = json.dumps(data).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=payload, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def percentile(values: list[float], pct: float) -> float:
    if not values:
        raise ValueError("percentile() requires at least one value")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * (pct / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def summarize_results(results: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float | int]]]:
    grouped: dict[str, dict[str, list[float]]] = {}
    for result in results:
        scenario = str(result.get("scenario", "unknown"))
        metrics = result.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        for key, value in metrics.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            grouped.setdefault(scenario, {}).setdefault(key, []).append(float(value))

    summary: dict[str, dict[str, dict[str, float | int]]] = {}
    for scenario, metrics in grouped.items():
        summary[scenario] = {}
        for key, values in metrics.items():
            summary[scenario][key] = {
                "count": len(values),
                "mean": round(statistics.mean(values), 1),
                "median": round(statistics.median(values), 1),
                "p95": round(percentile(values, 95), 1),
                "max": round(max(values), 1),
            }
    return summary


def build_comparison_rows(
    left_label: str,
    left_summary: dict[str, dict[str, dict[str, float | int]]],
    right_label: str,
    right_summary: dict[str, dict[str, dict[str, float | int]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario in sorted(set(left_summary) & set(right_summary)):
        left_metrics = left_summary[scenario]
        right_metrics = right_summary[scenario]
        for metric in sorted(set(left_metrics) & set(right_metrics)):
            left_mean = float(left_metrics[metric].get("mean", 0.0))
            right_mean = float(right_metrics[metric].get("mean", 0.0))
            delta = right_mean - left_mean
            delta_pct = 0.0 if left_mean == 0 else (delta / left_mean) * 100.0
            rows.append({
                "scenario": scenario,
                "metric": metric,
                left_label: round(left_mean, 1),
                right_label: round(right_mean, 1),
                "delta": round(delta, 1),
                "delta_pct": round(delta_pct, 1),
            })
    return rows


def load_frame_paths(frame_dir: Path) -> list[Path]:
    frames = sorted(path for path in frame_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not frames:
        raise RuntimeError(f"No image frames found in {frame_dir}")
    return frames


def check_heavy_stack(vision_client: VisionClient, pocket_url: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if not vision_client.health():
        raise RuntimeError(
            "Vision service is not healthy. Start the heavy stack first, ideally with "
            "`VISION_NO_CAMERA=1 ./scripts/run_heavy.sh`."
        )
    model_status = vision_client.model_status()
    if model_status.get("vllm") != "ready":
        raise RuntimeError(f"vLLM is not ready: {model_status}")

    try:
        with urllib.request.urlopen(pocket_url, timeout=3) as resp:
            body = resp.read().decode("utf-8", errors="ignore").strip()
    except Exception as exc:
        raise RuntimeError(
            "Pocket-TTS is not reachable. Start the heavy stack first, ideally with "
            "`VISION_NO_CAMERA=1 ./scripts/run_heavy.sh`."
        ) from exc

    return model_status, {"status": "ok", "body": body[:120]}


def maybe_warn_about_app() -> None:
    sock = socket.socket()
    sock.settimeout(0.25)
    try:
        if sock.connect_ex(("127.0.0.1", 7862)) == 0:
            console.print(
                "[yellow]Warning:[/yellow] dashboard/app port :7862 is open. "
                "For cleaner numbers, run the benchmark against `run_heavy.sh` only."
            )
    finally:
        sock.close()


def ensure_trackers_active(base_url: str, vision_client: VisionClient, timeout: float) -> dict[str, Any]:
    status = vision_client.model_status()
    if isinstance(status.get("face"), dict) and status["face"].get("status") not in {"unavailable", "disabled"}:
        if not status["face"].get("enabled", False):
            json_request(f"{base_url.rstrip('/')}/toggle_face_tracking", method="POST", data={"enabled": True})
    if isinstance(status.get("person"), dict) and status["person"].get("status") not in {"unavailable", "disabled"}:
        if not status["person"].get("enabled", False):
            json_request(f"{base_url.rstrip('/')}/toggle_person_tracking", method="POST", data={"enabled": True})

    deadline = time.time() + timeout
    last_status = status
    while time.time() < deadline:
        last_status = vision_client.model_status()
        sam3 = last_status.get("sam3", {})
        face = last_status.get("face", {})
        person = last_status.get("person", {})

        if isinstance(sam3, dict) and sam3.get("status") == "error":
            raise RuntimeError(f"SAM3 failed to load: {sam3.get('error', '')}")
        if isinstance(face, dict) and face.get("status") == "error":
            raise RuntimeError(f"Face tracker failed to load: {face.get('error', '')}")
        if isinstance(person, dict) and person.get("status") == "error":
            raise RuntimeError(f"Person tracker failed to load: {person.get('error', '')}")

        face_ready = not isinstance(face, dict) or face.get("status") in {"ready", "unavailable", "disabled"}
        person_ready = not isinstance(person, dict) or person.get("status") in {"ready", "unavailable", "disabled"}
        sam3_ready = not isinstance(sam3, dict) or sam3.get("status") in {"ready", "unavailable", "disabled"}
        if face_ready and person_ready and sam3_ready:
            return last_status
        time.sleep(0.5)
    raise RuntimeError(f"Trackers did not become ready: {last_status}")


def wait_for_repeatable_activity(vision_client: VisionClient, timeout: float) -> float | None:
    """Wait until the injected frame stream produces at least one visible detection."""
    deadline = time.time() + timeout
    start = time.perf_counter()
    while time.time() < deadline:
        snapshot = vision_client.snapshot()
        if snapshot.persons or snapshot.faces or snapshot.objects:
            return round((time.perf_counter() - start) * 1000, 1)
        time.sleep(0.25)
    return None


def clear_managed_questions(base_url: str) -> None:
    try:
        json_request(
            f"{base_url.rstrip('/')}/set_questions",
            method="POST",
            data={"constant": [], "ephemeral": []},
            timeout=5.0,
        )
    except Exception:
        pass


def measure_vision_poll_once(client: VisionClient) -> dict[str, float]:
    t0 = time.perf_counter()
    snapshot = client.snapshot()
    t1 = time.perf_counter()
    raw_frame = client.capture_frame_jpeg(max_width=320)
    t2 = time.perf_counter()
    annotated_frame = client.capture_frame_jpeg(annotated=True, max_width=320)
    t3 = time.perf_counter()
    return {
        "snapshot_ms": round((t1 - t0) * 1000, 1),
        "raw_frame_ms": round((t2 - t1) * 1000, 1),
        "annotated_frame_ms": round((t3 - t2) * 1000, 1),
        "total_ms": round((t3 - t0) * 1000, 1),
        "persons": float(len(snapshot.persons)),
        "faces": float(len(snapshot.faces)),
        "objects": float(len(snapshot.objects)),
        "raw_frame_kb": round(len(raw_frame) / 1024.0, 1),
        "annotated_frame_kb": round(len(annotated_frame) / 1024.0, 1),
    }


def measure_vision_query(base_url: str, question: str, max_tokens: int, poll_interval: float) -> dict[str, Any]:
    t0 = time.perf_counter()
    create = json_request(
        f"{base_url.rstrip('/')}/ask",
        method="POST",
        data={
            "question": question,
            "max_tokens": max_tokens,
            "loop": False,
            "inject_context": True,
        },
        timeout=15.0,
    )
    if create.get("error"):
        raise RuntimeError(str(create["error"]))
    slot_id = str(create["slot_id"])
    try:
        while True:
            result = json_request(f"{base_url.rstrip('/')}/result/{slot_id}", timeout=15.0)
            if result.get("done"):
                break
            time.sleep(poll_interval)
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        answer = str(result.get("response", ""))
        if answer.startswith("(error:"):
            raise RuntimeError(answer)
        return {
            "latency_ms": latency_ms,
            "answer_chars": len(answer),
            "answer_words": len(answer.split()),
        }
    finally:
        try:
            json_request(f"{base_url.rstrip('/')}/remove_slot/{slot_id}", method="POST", data={})
        except Exception:
            pass


def measure_pocket_tts(server_url: str, text: str) -> dict[str, Any]:
    collector = PCMCollector()
    tts = PocketTTS(on_audio=collector, server_url=server_url, voice="")
    try:
        tts.send_text(text)
        tts.flush()
        if not tts.wait_for_done(timeout=90.0):
            raise RuntimeError("Pocket-TTS synthesis timed out")
    finally:
        tts.close()
    synth_ms = round((time.perf_counter() - collector.start_at) * 1000, 1)
    first_audio_ms = (
        round((collector.first_chunk_at - collector.start_at) * 1000, 1)
        if collector.first_chunk_at is not None
        else 0.0
    )
    audio_ms = round((len(collector.pcm) / 2 / 24000) * 1000, 1)
    return {
        "first_audio_ms": first_audio_ms,
        "synth_ms": synth_ms,
        "audio_ms": audio_ms,
        "audio_kb": round(len(collector.pcm) / 1024.0, 1),
        "text_chars": len(text),
    }


def run_overlap(
    base_url: str,
    pocket_url: str,
    question: str,
    text: str,
    max_tokens: int,
    poll_interval: float,
    vision_client: VisionClient,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    errors: list[str] = []
    poller = VisionPoller(vision_client, interval=poll_interval)
    poller.start()

    def run_query() -> None:
        try:
            results["vision"] = measure_vision_query(base_url, question, max_tokens, poll_interval / 3.0)
        except Exception as exc:
            errors.append(f"vision: {exc}")

    def run_tts() -> None:
        try:
            results["tts"] = measure_pocket_tts(pocket_url, text)
        except Exception as exc:
            errors.append(f"tts: {exc}")

    t_query = threading.Thread(target=run_query, daemon=True)
    t_tts = threading.Thread(target=run_tts, daemon=True)
    t_query.start()
    t_tts.start()
    t_query.join()
    t_tts.join()
    poll_metrics = poller.stop()

    if errors:
        raise RuntimeError("; ".join(errors))

    metrics = {
        "vision_query_ms": float(results["vision"]["latency_ms"]),
        "vision_answer_chars": float(results["vision"]["answer_chars"]),
        "tts_first_audio_ms": float(results["tts"]["first_audio_ms"]),
        "tts_synth_ms": float(results["tts"]["synth_ms"]),
        "tts_audio_ms": float(results["tts"]["audio_ms"]),
    }
    for key, value in poll_metrics.items():
        metrics[key] = float(value)
    return metrics


def collect_host_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
    }

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        gpus = []
        for line in result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 3:
                gpus.append({
                    "name": parts[0],
                    "memory_mb": int(parts[1]),
                    "driver": parts[2],
                })
        if gpus:
            metadata["gpus"] = gpus
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=3,
        )
        metadata["git_rev"] = result.stdout.strip()
    except Exception:
        pass

    return metadata


def save_report(report: dict[str, Any], output_path: Path | None) -> Path:
    LOGS_DIR.mkdir(exist_ok=True)
    if output_path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = LOGS_DIR / f"{DEFAULT_OUTPUT_PREFIX}_{stamp}.json"
    output_path.write_text(json.dumps(report, indent=2))
    return output_path


def print_summary(summary: dict[str, dict[str, dict[str, float | int]]]) -> None:
    table = Table(title="Local Stack Benchmark Summary")
    table.add_column("Scenario", style="bold")
    table.add_column("Metric", style="cyan")
    table.add_column("Mean", justify="right")
    table.add_column("Median", justify="right")
    table.add_column("P95", justify="right")
    table.add_column("Max", justify="right")
    for scenario in sorted(summary):
        for metric in sorted(summary[scenario]):
            stats = summary[scenario][metric]
            table.add_row(
                scenario,
                metric,
                str(stats["mean"]),
                str(stats["median"]),
                str(stats["p95"]),
                str(stats["max"]),
            )
    console.print(table)


def print_comparison(rows: list[dict[str, Any]], left_label: str, right_label: str) -> None:
    table = Table(title="Benchmark Comparison")
    table.add_column("Scenario", style="bold")
    table.add_column("Metric", style="cyan")
    table.add_column(left_label, justify="right")
    table.add_column(right_label, justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("Delta %", justify="right")
    for row in rows:
        table.add_row(
            row["scenario"],
            row["metric"],
            str(row[left_label]),
            str(row[right_label]),
            str(row["delta"]),
            f"{row['delta_pct']}%",
        )
    console.print(table)


def run_command(args: argparse.Namespace) -> int:
    maybe_warn_about_app()

    vision_client = VisionClient(args.vision_url)
    model_status, pocket_status = check_heavy_stack(vision_client, args.pocket_url)
    clear_managed_questions(args.vision_url)

    frame_paths = load_frame_paths(Path(args.frame_dir))
    feeder = FrameFeeder(vision_client, frame_paths, fps=args.frame_fps)
    feeder.start()

    try:
        time.sleep(max(1.0, 2.0 / max(args.frame_fps, 0.1)))
        tracker_status = ensure_trackers_active(args.vision_url, vision_client, timeout=args.tracker_timeout)
        detection_ms = wait_for_repeatable_activity(vision_client, timeout=args.detection_timeout)

        results: list[dict[str, Any]] = []
        scenarios = ("vision_poll", "vision_query", "pocket_tts", "stack_overlap")

        total_attempts = args.warmup + args.runs
        for scenario in scenarios:
            console.print(f"\n[bold]{scenario}[/bold]")
            for attempt in range(total_attempts):
                frame_name = feeder.inject_now(attempt)
                time.sleep(args.pre_run_settle)
                if scenario == "vision_poll":
                    metrics = measure_vision_poll_once(vision_client)
                elif scenario == "vision_query":
                    metrics = measure_vision_query(
                        args.vision_url,
                        args.question,
                        args.max_tokens,
                        args.result_poll_interval,
                    )
                elif scenario == "pocket_tts":
                    metrics = measure_pocket_tts(args.pocket_url, args.tts_text)
                else:
                    metrics = run_overlap(
                        args.vision_url,
                        args.pocket_url,
                        args.question,
                        args.tts_text,
                        args.max_tokens,
                        args.poll_interval,
                        vision_client,
                    )

                if attempt < args.warmup:
                    console.print(f"[dim]warmup {attempt + 1}/{args.warmup} ({frame_name})[/dim]")
                else:
                    run_number = attempt - args.warmup + 1
                    console.print(f"[dim]run {run_number}/{args.runs} ({frame_name})[/dim] {metrics}")
                    results.append({
                        "scenario": scenario,
                        "run": run_number,
                        "frame": frame_name,
                        "metrics": metrics,
                    })
                time.sleep(args.cooldown)
    finally:
        feeder.stop()

    summary = summarize_results(results)
    report = {
        "type": "local_stack_benchmark",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "host": collect_host_metadata(),
        "config": {
            "vision_url": args.vision_url,
            "pocket_url": args.pocket_url,
            "frame_dir": str(Path(args.frame_dir).resolve()),
            "frame_fps": args.frame_fps,
            "warmup": args.warmup,
            "runs": args.runs,
            "question": args.question,
            "tts_text": args.tts_text,
            "max_tokens": args.max_tokens,
            "poll_interval": args.poll_interval,
        },
        "service_status": {
            "model_status_before": model_status,
            "tracker_status_ready": tracker_status,
            "pocket_status": pocket_status,
            "memory_status": vision_client.memory_status(),
            "repeatable_detection_ms": detection_ms,
        },
        "results": results,
        "summary": summary,
    }
    output_path = save_report(report, Path(args.output) if args.output else None)
    console.print()
    print_summary(summary)
    console.print(f"\nSaved report to [bold]{output_path}[/bold]")
    console.print("For repeatable cross-machine runs, use `VISION_NO_CAMERA=1 ./scripts/run_heavy.sh`.")
    return 0


def compare_command(args: argparse.Namespace) -> int:
    left = json.loads(Path(args.left).read_text())
    right = json.loads(Path(args.right).read_text())
    left_label = args.left_label or Path(args.left).stem
    right_label = args.right_label or Path(args.right).stem
    rows = build_comparison_rows(left_label, left["summary"], right_label, right["summary"])
    print_comparison(rows, left_label, right_label)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Repeatable local-stack benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the benchmark")
    run_parser.add_argument("--vision-url", default="http://127.0.0.1:7860")
    run_parser.add_argument("--pocket-url", default="http://127.0.0.1:8003")
    run_parser.add_argument("--frame-dir", default=str(DEFAULT_FRAME_DIR))
    run_parser.add_argument("--frame-fps", type=float, default=4.0)
    run_parser.add_argument("--warmup", type=int, default=1)
    run_parser.add_argument("--runs", type=int, default=5)
    run_parser.add_argument("--question", default=DEFAULT_QUESTION)
    run_parser.add_argument("--tts-text", default=DEFAULT_TTS_TEXT)
    run_parser.add_argument("--max-tokens", type=int, default=96)
    run_parser.add_argument("--poll-interval", type=float, default=0.75)
    run_parser.add_argument("--result-poll-interval", type=float, default=0.15)
    run_parser.add_argument("--pre-run-settle", type=float, default=0.35)
    run_parser.add_argument("--cooldown", type=float, default=0.75)
    run_parser.add_argument("--tracker-timeout", type=float, default=90.0)
    run_parser.add_argument("--detection-timeout", type=float, default=12.0)
    run_parser.add_argument("--output", default="")
    run_parser.set_defaults(func=run_command)

    compare_parser = subparsers.add_parser("compare", help="Compare two benchmark reports")
    compare_parser.add_argument("left")
    compare_parser.add_argument("right")
    compare_parser.add_argument("--left-label", default="")
    compare_parser.add_argument("--right-label", default="")
    compare_parser.set_defaults(func=compare_command)

    return parser


def main(argv: list[str] | None = None) -> int:
    args_list = list(argv if argv is not None else sys.argv[1:])
    if not args_list or args_list[0].startswith("-"):
        args_list = ["run", *args_list]
    parser = build_parser()
    args = parser.parse_args(args_list)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
