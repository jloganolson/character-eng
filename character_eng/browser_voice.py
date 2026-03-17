from __future__ import annotations

import threading
import time

from character_eng.bridge import BridgeServer
from character_eng.voice import DeepgramSTT, ElevenLabsTTS, VoiceIO


class BrowserSpeaker:
    """Browser-backed audio sink with estimated playback timing."""

    def __init__(self, bridge: BridgeServer, on_output=None, sample_rate: int = 24000):
        self._bridge = bridge
        self._on_output = on_output
        self._sample_rate = sample_rate
        self._audio_started = threading.Event()
        self._playback_end_at = 0.0
        self._bytes_enqueued = 0
        self._bytes_played = 0
        self._lock = threading.Lock()

    def start(self) -> None:
        return

    def stop(self) -> None:
        self.flush()

    def enqueue(self, pcm: bytes) -> None:
        if not pcm:
            return
        now = time.time()
        duration_s = len(pcm) / 2 / float(self._sample_rate)
        with self._lock:
            base = max(self._playback_end_at, now)
            self._playback_end_at = base + duration_s
            self._bytes_enqueued += len(pcm)
            self._bytes_played = self._bytes_enqueued
            self._audio_started.set()
        self._bridge.push_audio(pcm, sample_rate=self._sample_rate)
        if self._on_output is not None:
            self._on_output(pcm)

    def flush(self) -> None:
        with self._lock:
            self._playback_end_at = time.time()

    def wait_for_audio(self, timeout: float = 0.5) -> bool:
        return self._audio_started.wait(timeout=timeout)

    def is_done(self) -> bool:
        with self._lock:
            return time.time() >= self._playback_end_at

    def reset_counters(self) -> None:
        with self._lock:
            self._bytes_enqueued = 0
            self._bytes_played = 0
            self._playback_end_at = time.time()
            self._audio_started.clear()

    @property
    def playback_ratio(self) -> float:
        with self._lock:
            if self._bytes_enqueued <= 0:
                return 1.0
            return min(1.0, self._bytes_played / self._bytes_enqueued)


class BrowserVoiceIO(VoiceIO):
    """VoiceIO variant for browser mic/camera input and browser audio output."""

    def __init__(self, bridge: BridgeServer, **kwargs):
        kwargs.setdefault("aec", False)
        kwargs.setdefault("output_only", False)
        super().__init__(**kwargs)
        self._bridge = bridge
        self._stt_lock = threading.Lock()

    def start(self):
        self._speaker = BrowserSpeaker(self._bridge, on_output=self._output_audio_hook)
        self._speaker.start()

        if self._tts_backend in ("local", "qwen"):
            from character_eng.local_tts import LocalTTS, load_model

            load_model(self._tts_model, self._tts_device, self._ref_audio, ref_text=self._ref_text)
            self._tts = LocalTTS(
                on_audio=self._speaker.enqueue,
                ref_audio_path=self._ref_audio,
                model_id=self._tts_model,
                device=self._tts_device,
                ref_text=self._ref_text,
            )
        elif self._tts_backend == "pocket":
            from character_eng.pocket_tts import PocketTTS

            server_url = self._tts_server_url or "http://localhost:8003"
            self._ensure_pocket_server(server_url)
            self._tts = PocketTTS(
                on_audio=self._speaker.enqueue,
                server_url=server_url,
                voice="" if self._pocket_voice else "alba",
                ref_audio=self._ref_audio,
            )
        else:
            self._tts = ElevenLabsTTS(on_audio=self._speaker.enqueue, voice_id=self._voice_id)

        self._started = True

    def _ensure_stt_started(self) -> None:
        if self._output_only:
            return
        with self._stt_lock:
            if self._stt is not None:
                status = self._stt.status_snapshot()
                if status["error"] or (not status["thread_alive"] and not status["ws_open"]):
                    self._stt.stop()
                    self._stt = None
            if self._stt is None:
                self._stt = DeepgramSTT(
                    on_transcript=self._on_transcript,
                    on_turn_start=self._on_turn_start,
                )
                self._stt.start()

    def inject_input_audio(self, pcm: bytes) -> None:
        if not pcm:
            return
        self._ensure_stt_started()
        super().inject_input_audio(pcm)

    def status_snapshot(self) -> dict:
        payload = super().status_snapshot()
        client = self._bridge.status_payload().get("client") or {}
        mic_live = bool(client.get("mic"))
        payload["browser_bridge"] = True
        payload["speaker_ready"] = True
        payload["mic_ready"] = mic_live
        if self._output_only:
            return payload
        if not mic_live:
            payload["stt"] = {"state": "off", "detail": "Browser mic idle."}
        return payload

    def stop(self):
        self._started = False
        self._cancel_auto_beat()
        self.cancel_latency_filler()

        if self._stt is not None:
            self._stt.stop()
            self._stt = None
        if self._tts is not None:
            self._tts.close()
            self._tts = None
        if self._speaker is not None:
            self._speaker.stop()
            self._speaker = None

        self._stop_pocket_server()

        while not self._event_queue.empty():
            try:
                self._event_queue.get_nowait()
            except Exception:
                break
