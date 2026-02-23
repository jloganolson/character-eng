"""CLI tool for comparing TTS backends by ear.

Runnable via: uv run -m character_eng.tts_compare

Two-level loop:
  Outer: type a line of text
  Inner: press a number key to hear it from a backend, shows TTFA timing
         press another number to hear same line from a different backend
         press Enter/Escape to go back and type new text
  Ctrl+C to quit

Backends (streaming-capable only — no pseudo-streamers):
  1 = ElevenLabs (cloud WebSocket)
  2 = KaniTTS (SSE streaming, port 8000)
  3 = Qwen3-TTS (local GPU, in-process)
  4 = KittenTTS (port 8005)
  5 = Kokoro (port 8880)
  6 = Pocket-TTS (port 8003)
"""

from __future__ import annotations

import sys
import threading
import time

from rich.console import Console

from character_eng.config import load_config

console = Console()

BACKENDS = {
    "1": ("elevenlabs", "ElevenLabs (cloud)"),
    "2": ("kani", "KaniTTS (SSE, :8000)"),
    "3": ("local", "Qwen3-TTS (local GPU)"),
    "4": ("kitten", "KittenTTS (:8005)"),
    "5": ("kokoro", "Kokoro (:8880)"),
    "6": ("pocket", "Pocket-TTS (:8003)"),
}


def _create_tts(backend: str, cfg):
    """Create a TTS instance and return (tts, metrics_dict)."""
    metrics = {
        "ttfa": None,
        "total_bytes": 0,
    }
    lock = threading.Lock()

    def on_audio(pcm: bytes):
        with lock:
            if metrics["ttfa"] is None:
                metrics["ttfa"] = time.perf_counter()
            metrics["total_bytes"] += len(pcm)
        # Also feed to speaker
        if speaker[0] is not None:
            speaker[0].enqueue(pcm)

    # speaker is set by caller
    speaker = [None]

    if backend == "elevenlabs":
        from character_eng.voice import ElevenLabsTTS
        tts = ElevenLabsTTS(on_audio=on_audio)
    elif backend == "kani":
        from character_eng.kani_tts import KaniTTS
        tts = KaniTTS(
            on_audio=on_audio,
            server_url=cfg.voice.tts_server_url or "http://localhost:8000",
        )
    elif backend == "local":
        from character_eng.local_tts import LocalTTS, load_model
        load_model(cfg.voice.tts_model, cfg.voice.tts_device, cfg.voice.ref_audio)
        tts = LocalTTS(
            on_audio=on_audio,
            ref_audio_path=cfg.voice.ref_audio,
            model_id=cfg.voice.tts_model,
            device=cfg.voice.tts_device,
        )
    elif backend == "kitten":
        from character_eng.kitten_tts import KittenTTS
        tts = KittenTTS(
            on_audio=on_audio,
            server_url=cfg.voice.tts_server_url or "http://localhost:8005",
        )
    elif backend == "kokoro":
        from character_eng.kokoro_tts import KokoroTTS
        tts = KokoroTTS(
            on_audio=on_audio,
            server_url=cfg.voice.tts_server_url or "http://localhost:8880",
        )
    elif backend == "pocket":
        from character_eng.pocket_tts import PocketTTS
        tts = PocketTTS(
            on_audio=on_audio,
            server_url=cfg.voice.tts_server_url or "http://localhost:8003",
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")

    return tts, metrics, speaker


def _play_text(text: str, backend_key: str, cfg):
    """Play text through the given backend, return timing info."""
    from character_eng.voice import SpeakerStream

    backend, label = BACKENDS[backend_key]
    console.print(f"\n  [cyan]{label}[/cyan] generating...")

    tts, metrics, speaker_ref = _create_tts(backend, cfg)

    # Create speaker stream
    spk = SpeakerStream()
    spk.start()
    speaker_ref[0] = spk

    t_start = time.perf_counter()
    tts.send_text(text)
    tts.flush()

    tts.wait_for_done(timeout=60.0)
    t_end = time.perf_counter()

    # Wait for speaker to finish playing
    spk.wait_for_audio(timeout=1.0)
    while not spk.is_done():
        time.sleep(0.05)

    tts.close()
    spk.stop()

    total_ms = (t_end - t_start) * 1000
    ttfa_ms = (metrics["ttfa"] - t_start) * 1000 if metrics["ttfa"] else None
    total_bytes = metrics["total_bytes"]
    audio_dur_ms = (total_bytes / 2 / 24000) * 1000 if total_bytes > 0 else 0

    ttfa_str = f"{ttfa_ms:.0f}ms" if ttfa_ms is not None else "N/A"
    console.print(
        f"  TTFA=[bold green]{ttfa_str}[/bold green]  "
        f"Total={total_ms:.0f}ms  "
        f"Audio={audio_dur_ms:.0f}ms"
    )


def main():
    cfg = load_config()

    console.print("[bold]TTS Backend Comparison Tool[/bold]")
    console.print()
    for key, (_, label) in sorted(BACKENDS.items()):
        console.print(f"  [cyan]{key}[/cyan] = {label}")
    console.print()
    console.print("Type text, then press a number key to hear it.")
    console.print("Press Enter to type new text. Ctrl+C to quit.\n")

    try:
        while True:
            text = console.input("[bold]Text>[/bold] ").strip()
            if not text:
                continue

            console.print(f'  "{text}"')
            console.print("  Press 1-6 to play, Enter for new text, Ctrl+C to quit")

            while True:
                try:
                    key = console.input("  [dim]Backend>[/dim] ").strip()
                except EOFError:
                    return

                if not key:
                    break
                if key in BACKENDS:
                    try:
                        _play_text(text, key, cfg)
                    except Exception as e:
                        console.print(f"  [red]Error: {e}[/red]")
                else:
                    console.print(f"  [dim]Unknown key '{key}'. Use 1-6.[/dim]")

    except (KeyboardInterrupt, EOFError):
        console.print("\n[dim]Bye.[/dim]")


if __name__ == "__main__":
    main()
