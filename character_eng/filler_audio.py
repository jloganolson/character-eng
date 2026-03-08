"""Short Pocket-TTS filler clips used to mask TTS startup latency."""

from __future__ import annotations

import threading
from urllib.parse import urlparse
from dataclasses import dataclass

from character_eng.pocket_tts import PocketTTS

FILLER_LIBRARY: dict[str, tuple[str, ...]] = {
    "neutral": ("Okay.", "Mm.", "Right."),
    "curious": ("Hmm.", "Oh?", "What?"),
    "warm": ("Oh.", "Mm-hm.", "Okay."),
    "skeptical": ("Hmm.", "Really?", "No?"),
    "urgent": ("Wait.", "Hey.", "What?"),
}


@dataclass(frozen=True)
class FillerChoice:
    sentiment: str
    phrase: str


class _PCMCollector:
    def __init__(self):
        self.parts: list[bytes] = []

    def __call__(self, data: bytes) -> None:
        self.parts.append(bytes(data))

    @property
    def pcm(self) -> bytes:
        return b"".join(self.parts)


class FillerBank:
    """Prewarmed cache of short Pocket-TTS filler clips."""

    def __init__(
        self,
        server_url: str,
        *,
        voice: str = "",
        ref_audio: str = "",
    ):
        self._server_url = server_url.rstrip("/")
        self._voice = voice
        self._ref_audio = ref_audio
        self._clips: dict[str, bytes] = {}
        self._indexes = {sentiment: 0 for sentiment in FILLER_LIBRARY}
        self._lock = threading.Lock()

    @staticmethod
    def _voice_request_value(voice: str) -> str:
        if not voice:
            return "alba"
        parsed = urlparse(voice)
        if parsed.scheme in {"http", "https"}:
            return voice
        return ""

    def pick(self, sentiment: str, response_text: str = "", expression: str = "") -> FillerChoice:
        chosen_sentiment = self.infer_sentiment(response_text=response_text, expression=expression, default=sentiment)
        phrases = FILLER_LIBRARY.get(chosen_sentiment, FILLER_LIBRARY["neutral"])
        with self._lock:
            index = self._indexes.get(chosen_sentiment, 0)
            self._indexes[chosen_sentiment] = (index + 1) % len(phrases)
        return FillerChoice(sentiment=chosen_sentiment, phrase=phrases[index])

    def clip_for(self, sentiment: str, response_text: str = "", expression: str = "") -> bytes:
        choice = self.pick(sentiment=sentiment, response_text=response_text, expression=expression)
        return self.clip_for_choice(choice)

    def clip_for_choice(self, choice: FillerChoice) -> bytes:
        key = choice.phrase
        with self._lock:
            cached = self._clips.get(key)
        if cached is not None:
            return cached

        collector = _PCMCollector()
        tts = PocketTTS(
            on_audio=collector,
            server_url=self._server_url,
            voice=self._voice_request_value(self._voice),
            ref_audio=self._ref_audio,
        )
        tts.send_text(choice.phrase)
        tts.flush()
        if not tts.wait_for_done(timeout=30.0):
            raise RuntimeError(f"Pocket filler synthesis timed out for '{choice.phrase}'")
        tts.close()

        clip = collector.pcm
        with self._lock:
            self._clips[key] = clip
        return clip

    def prewarm(self) -> int:
        phrases: list[str] = []
        seen: set[str] = set()
        for options in FILLER_LIBRARY.values():
            for phrase in options:
                if phrase in seen:
                    continue
                seen.add(phrase)
                phrases.append(phrase)
        for phrase in phrases:
            self.clip_for_choice(FillerChoice(sentiment="neutral", phrase=phrase))
        return len(phrases)

    @staticmethod
    def infer_sentiment(response_text: str = "", expression: str = "", default: str = "neutral") -> str:
        expression = (expression or "").strip().lower()
        text = (response_text or "").strip().lower()

        if expression in ("happy", "warm", "friendly"):
            return "warm"
        if expression in ("curious", "confused"):
            return "curious"
        if expression in ("surprised", "alarm", "wide_eyes"):
            return "urgent"
        if expression in ("skeptical", "annoyed", "suspicious"):
            return "skeptical"

        if "?" in text:
            return "curious"
        if "!" in text:
            return "urgent"
        if any(word in text for word in ("no ", "not ", "don't", "cant", "can't", "won't")):
            return "skeptical"
        return default if default in FILLER_LIBRARY else "neutral"
