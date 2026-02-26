"""Generate TTS WAV samples via Pocket TTS for AEC testing.

Creates two WAVs with distinct voices:
- robot_greg.wav: Greg's voice (the robot, what AEC should cancel)
- human_visitor.wav: Different voice (the human, what AEC should pass through)

Usage:
    # Ensure Pocket TTS server is running first:
    pocket-tts serve --port 8003 --voice voices/greg.safetensors &

    uv run python experiments/aec/generate_pocket_wavs.py
"""

import struct
import sys
import wave
from pathlib import Path

import requests

SERVER_URL = "http://localhost:8003"
OUTPUT_DIR = Path(__file__).resolve().parent / "samples"
VOICE_FILE = Path(__file__).resolve().parent.parent / "voices" / "greg.safetensors"

# Robot uses Greg's cloned voice (server default), human uses alba
SAMPLES = [
    {
        "name": "robot_greg",
        "text": "Hello there! Welcome to Greg's stand. I've got fliers here with a coupon for a free cup of water. Would you like one?",
        "voice": None,  # Server default = Greg (started with --voice greg.safetensors)
    },
    {
        "name": "human_visitor",
        "text": "Hey, what's going on over here? I saw your sign from across the street and thought I'd come check it out.",
        "voice": "alba",  # Built-in voice (different from Greg)
    },
]


def generate_wav(text: str, voice: str | None, output_path: Path) -> float:
    """Generate speech via Pocket TTS HTTP API and save as WAV.

    Returns duration in seconds.
    """
    data = {"text": text}
    if voice:
        data["voice_url"] = voice

    resp = requests.post(f"{SERVER_URL}/tts", data=data, stream=True, timeout=120)
    resp.raise_for_status()

    # Collect all response bytes (streaming WAV)
    raw = b""
    for chunk in resp.iter_content(chunk_size=4800):
        raw += chunk

    if not raw:
        raise RuntimeError(f"No audio received for {output_path.name}")

    # Parse WAV header to find data section
    if raw[:4] == b"RIFF" and raw[8:12] == b"WAVE":
        # Find 'data' subchunk
        pos = 12
        while pos < len(raw) - 8:
            subchunk_id = raw[pos:pos + 4]
            subchunk_size = struct.unpack_from("<I", raw, pos + 4)[0]
            if subchunk_id == b"data":
                data_start = pos + 8
                pcm_data = raw[data_start:]
                break
            pos += 8 + subchunk_size
        else:
            # Fallback: skip standard 44-byte header
            pcm_data = raw[44:]
    else:
        pcm_data = raw

    # Save as standard WAV (24kHz/16-bit/mono)
    sample_rate = 24000
    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)

    duration = len(pcm_data) / (sample_rate * 2)
    return duration


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Check server
    try:
        requests.get(SERVER_URL, timeout=2)
    except Exception:
        print(f"ERROR: Pocket TTS server not running at {SERVER_URL}")
        print("Start it with: pocket-tts serve --port 8003 --voice voices/greg.safetensors &")
        sys.exit(1)

    print("Generating Pocket TTS samples for AEC testing...\n")

    for sample in SAMPLES:
        output_path = OUTPUT_DIR / f"{sample['name']}.wav"
        print(f"  {sample['name']}: \"{sample['text'][:60]}...\"")
        print(f"    voice: {sample['voice']}")

        duration = generate_wav(sample["text"], sample["voice"], output_path)
        print(f"    saved: {output_path.name} ({duration:.1f}s)")
        print()

    print(f"Done! Samples in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
