"""Generate ElevenLabs TTS dialogue samples as WAV files for AEC testing."""

import os
import wave
import struct
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import websocket  # websocket-client
import json
import base64

API_KEY = os.environ["ELEVENLABS_API_KEY"]
OUTPUT_DIR = Path(__file__).resolve().parent / "samples"
OUTPUT_DIR.mkdir(exist_ok=True)

# Different voices for variety
# Rachel (calm female), Adam (deep male), Antoni (warm male)
VOICES = {
    "rachel": "21m00Tcm4TlvDq8ikWAM",
    "adam": "pNInz6obpgDQGcFmaJgB",
    "antoni": "ErXwobaYiN019PkySvjV",
}

LINES = [
    ("rachel", "Hello there! I've been waiting for you. The orb is glowing brighter than usual today, isn't it fascinating?"),
    ("adam", "I've analyzed the data from the last experiment. The quantum entanglement readings are off the charts. We need to discuss this immediately."),
    ("rachel", "Sometimes I wonder what it would be like to see the stars. Not through a camera feed, but really see them, you know? With actual eyes."),
]

MODEL = "eleven_flash_v2_5"
SAMPLE_RATE = 24000  # ElevenLabs PCM output rate


def generate_wav(voice_id: str, text: str, output_path: Path):
    """Generate speech via ElevenLabs WebSocket and save as WAV."""
    url = f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input?model_id={MODEL}&output_format=pcm_24000"

    audio_chunks: list[bytes] = []
    done = False

    def on_message(ws, message):
        nonlocal done
        data = json.loads(message)
        if data.get("audio"):
            audio_chunks.append(base64.b64decode(data["audio"]))
        if data.get("isFinal"):
            done = True
            ws.close()

    def on_open(ws):
        # Send BOS
        ws.send(json.dumps({
            "text": " ",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
            "xi_api_key": API_KEY,
        }))
        # Send text
        ws.send(json.dumps({"text": text}))
        # Send EOS
        ws.send(json.dumps({"text": ""}))

    def on_error(ws, error):
        print(f"  WebSocket error: {error}")

    ws = websocket.WebSocketApp(
        url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
    )
    ws.run_forever()

    if not audio_chunks:
        print(f"  ERROR: No audio received for {output_path.name}")
        return

    # Combine PCM chunks and write WAV
    pcm_data = b"".join(audio_chunks)
    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_data)

    duration = len(pcm_data) / (SAMPLE_RATE * 2)
    print(f"  Saved {output_path.name} ({duration:.1f}s, {len(pcm_data)} bytes)")


def main():
    print("Generating ElevenLabs dialogue samples...\n")
    for i, (voice_name, text) in enumerate(LINES):
        voice_id = VOICES[voice_name]
        filename = f"line_{i+1}_{voice_name}.wav"
        output_path = OUTPUT_DIR / filename
        print(f"[{i+1}/{len(LINES)}] {voice_name}: {text[:60]}...")
        generate_wav(voice_id, text, output_path)
        print()

    print(f"Done! Samples in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
