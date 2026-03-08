"""Full-stack automation harness: generated/live vision + STT + real chat loop pieces."""

from __future__ import annotations

import argparse
import base64
import html
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import wave
from dataclasses import asdict, dataclass, field
from datetime import datetime
from io import BytesIO
from pathlib import Path
from urllib import parse, request

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

from character_eng.__main__ import (
    _bump_version,
    _check_reconcile,
    apply_plan,
    handle_perception,
    run_plan,
    run_post_response,
    stream_guided_beat,
    stream_response,
)
from character_eng.chat import ChatSession
from character_eng.dashboard.events import DashboardEventCollector
from character_eng.gemini_media import evaluate_image, generate_image
from character_eng.models import DEFAULT_MODEL, MODELS
from character_eng.person import PeopleState
from character_eng.pocket_tts import PocketTTS
from character_eng.prompts import load_prompt
from character_eng.scenario import load_scenario_script
from character_eng.vision.client import VisionClient
from character_eng.world import Goals, Script, load_goals, load_world_state, single_beat_call

load_dotenv()

console = Console()
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
SERVICES_DIR = PROJECT_ROOT / "services" / "vision"

SHORT_TURN_PROMPT = (PROMPTS_DIR / "automation_short_turn.txt").read_text().strip()


SCENE_SPECS = [
    {
        "name": "approach",
        "prompt": (
            "Cinematic eye-level photo of Greg, a small robot head on a folding table with a hand-drawn "
            "'Free Water & Advice' sign, water jug, paper cups, and fliers. A cautious person in a hoodie "
            "approaches the stand from the sidewalk, face partly hidden by the hood, warm afternoon street."
        ),
        "fallback_summary": "A hooded passerby slows down at Greg's folding-table stand with water, cups, and flyers.",
        "fallback_facts": [
            "Greg is a robot head on a folding table stand",
            "There are water cups and flyers on the stand",
            "A hooded passerby is close to the stand",
        ],
        "fallback_goal": "Keep the exchange focused on Greg's stand and the cautious passerby.",
        "fallback_event": "A hooded passerby slows down at Greg's stand.",
        "fallback_user_line": "Hey Greg, what's the quick pitch here?",
        "required_groups": [
            ("greg", "robot"),
            ("stand", "table"),
            ("visitor", "passerby", "hooded"),
        ],
        "allowed_terms": ("water", "advice", "cup", "flyer", "hood", "sign", "stand"),
    },
    {
        "name": "linger",
        "prompt": (
            "Same Greg lemonade-stand scene. Greg is still only a robot head on a folding table with the "
            "'Free Water & Advice' sign, paper cups, flyers, and a water jug clearly visible. The same hooded "
            "visitor is now close to the table, looking down at a flyer and holding a paper cup, still slightly "
            "hiding their face. Do not depict toys, a flea market, coffee, or a human body for Greg."
        ),
        "fallback_summary": "The hooded visitor lingers at Greg's stand, reading a flyer and holding one paper cup.",
        "fallback_facts": [
            "Greg's stand still has a sign, cups, water, and flyers",
            "The hooded visitor is reading a flyer",
            "The visitor is holding a paper cup near the stand",
        ],
        "fallback_goal": "Shift into a short follow-up about the flyer or advice offer.",
        "fallback_event": "The hooded visitor is reading a flyer and holding a paper cup near the stand.",
        "fallback_user_line": "Alright, what's actually on that flier?",
        "required_groups": [
            ("greg", "robot"),
            ("stand", "table"),
            ("visitor", "passerby", "hooded"),
            ("flyer", "flier"),
            ("cup", "paper cup"),
        ],
        "allowed_terms": ("water", "advice", "cup", "flyer", "hood", "sign", "stand"),
    },
    {
        "name": "depart",
        "prompt": (
            "Same lemonade stand with Greg on the table. The hooded visitor is turning away to leave while "
            "glancing back over a shoulder, one flier in hand, late-afternoon sidewalk light."
        ),
        "fallback_summary": "The visitor is starting to leave Greg's stand but still has one flyer in hand.",
        "fallback_facts": [
            "The visitor is turning away from the stand",
            "The visitor still has one flyer",
            "Greg remains at the table with the water stand",
        ],
        "fallback_goal": "Land one short closing line before the passerby leaves.",
        "fallback_event": "The visitor is turning away from the stand with a flyer in hand.",
        "fallback_user_line": "Give me the one-line version before I go.",
        "required_groups": [
            ("greg", "robot"),
            ("stand", "table"),
            ("visitor", "passerby", "hooded"),
            ("flyer", "flier"),
            ("leave", "turning away", "walking away", "glancing back"),
        ],
        "allowed_terms": ("water", "advice", "cup", "flyer", "hood", "sign", "stand"),
    },
]

ANCHOR_TERMS = ("greg", "stand", "water", "flier", "flyer", "advice", "table")
DRIFT_TERMS = (
    "toy",
    "toys",
    "flea market",
    "coffee",
    "adult male",
    "jacket and jeans",
    "ice cream",
    "coupon",
    "lemonade",
    "processing power",
    "fifty cents",
    "50¢",
    "50 cents",
    "$1",
    "1 dollar",
    "dollar",
    "price",
    "costs",
    "disembodied",
    "golden hour light",
)

LIVE_SCENE_FALLBACK = {
    "summary": "Greg sees a quiet office from desk height in first-person view.",
    "world_facts": [
        "Greg is viewing the room from a desk-height first-person camera",
        "The room looks like a quiet office or workspace",
        "No person is clearly visible in frame right now",
    ],
    "visual_goal": "Keep the exchange grounded in the visible office and invite the off-camera user in.",
    "user_line": "Pretty quiet in here, huh?",
    "followup_line": "What should we focus on first?",
}
LIVE_DRIFT_TERMS = (
    "lemonade",
    "coupon",
    "flyer",
    "flier",
    "processing power",
    "robot head",
    "folding table",
    "greg is visible",
    "disembodied",
    "street",
    "sidewalk",
)
TIMELINE_LANES = {
    "session_start": "session",
    "vision_stack_ready": "vision",
    "scene_capture": "vision",
    "scene_eval": "vision",
    "vision_snapshot_read": "vision",
    "vision_poll": "vision",
    "vision_models": "vision",
    "vision_poll_error": "vision",
    "world_seed": "world",
    "perception_injected": "world",
    "stt_result": "stt",
    "user_turn_start": "chat",
    "response_ttft": "chat",
    "response_chunk": "chat",
    "response_done": "chat",
    "expression": "expression",
    "eval": "eval",
    "beat_advance": "script",
    "director": "director",
    "stage_change": "director",
    "plan": "planner",
    "assistant_tts_first_audio": "tts",
    "assistant_tts_done": "tts",
    "assistant_audio_clip": "tts",
}
LANE_ORDER = ["vision", "stt", "chat", "expression", "eval", "director", "planner", "tts", "world", "session", "script", "other"]


@dataclass
class LiveScenePlan:
    summary: str
    world_facts: list[str]
    visual_goal: str
    user_lines: list[str]


@dataclass
class TurnRecord:
    scene_name: str
    image_prompt: str
    image_mime_type: str = ""
    image_b64: str = ""
    snapshot_faces: int = 0
    snapshot_persons: int = 0
    snapshot_objects: int = 0
    visual_summary: str = ""
    visual_world_facts: list[str] = field(default_factory=list)
    visual_goal: str = ""
    scripted_user_line: str = ""
    stt_transcript: str = ""
    assistant_response: str = ""
    response_word_count: int = 0
    response_ok: bool = True
    response_note: str = ""
    assistant_audio_ms: int = 0
    perception_events: list[str] = field(default_factory=list)
    response_ttft_ms: int = 0
    response_total_ms: int = 0
    trace_events: list[dict] = field(default_factory=list)


@dataclass
class SynthesisResult:
    pcm: bytes
    synth_ms: int
    first_audio_ms: int
    audio_ms: int


class TraceRecorder:
    def __init__(self) -> None:
        self._events: list[dict] = []
        self._lock = threading.Lock()
        self._seq = 0

    def add(self, event_type: str, data: dict, *, timestamp: float | None = None) -> dict:
        with self._lock:
            self._seq += 1
            event = {
                "type": event_type,
                "timestamp": time.time() if timestamp is None else timestamp,
                "seq": self._seq,
                "data": data,
            }
            self._events.append(event)
            return dict(event)

    def snapshot(self) -> list[dict]:
        with self._lock:
            return [dict(event) for event in self._events]


class VisionSampler:
    def __init__(self, vision_client: VisionClient, recorder: TraceRecorder, interval: float = 0.75) -> None:
        self._vision_client = vision_client
        self._recorder = recorder
        self._interval = interval
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

    def _run(self) -> None:
        next_model_poll = 0.0
        while not self._stop.is_set():
            now = time.time()
            try:
                snapshot = self._vision_client.snapshot()
                raw_frame = self._vision_client.capture_frame_jpeg(max_width=220)
                overlay_frame = self._vision_client.capture_frame_jpeg(annotated=True, max_width=220)
                self._recorder.add(
                    "vision_poll",
                    {
                        "faces": len(snapshot.faces),
                        "persons": len(snapshot.persons),
                        "objects": len(snapshot.objects),
                        "object_labels": [obj.label for obj in snapshot.objects[:4]],
                        "vlm_answers": [answer.answer for answer in snapshot.vlm_answers[:3]],
                        "frame_b64": base64.b64encode(raw_frame).decode("utf-8"),
                        "overlay_b64": base64.b64encode(overlay_frame).decode("utf-8"),
                        "mime_type": "image/jpeg",
                    },
                    timestamp=now,
                )
                if now >= next_model_poll:
                    next_model_poll = now + 4.0
                    try:
                        status = self._vision_client.model_status()
                        sam3 = status.get("sam3", {})
                        face = status.get("face", {})
                        person = status.get("person", {})
                        self._recorder.add(
                            "vision_models",
                            {
                                "vllm": status.get("vllm", "unknown"),
                                "sam3": sam3.get("status", "unknown") if isinstance(sam3, dict) else str(sam3),
                                "face": face.get("status", "unknown") if isinstance(face, dict) else str(face),
                                "person": person.get("status", "unknown") if isinstance(person, dict) else str(person),
                            },
                            timestamp=now,
                        )
                    except Exception as exc:
                        self._recorder.add("vision_poll_error", {"error": str(exc)}, timestamp=now)
            except Exception as exc:
                self._recorder.add("vision_poll_error", {"error": str(exc)}, timestamp=now)
            self._stop.wait(self._interval)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _ensure_pocket_server(server_url: str, voice_path: str) -> subprocess.Popen | None:
    import requests

    try:
        requests.get(server_url, timeout=2)
        return None
    except Exception:
        pass

    pocket_bin = shutil.which("pocket-tts")
    if pocket_bin is None:
        raise RuntimeError("pocket-tts not found in PATH")

    cmd = [pocket_bin, "serve", "--port", str(socket.getservbyname("http"))]
    # Replace the auto-derived port with the actual parsed value.
    from urllib.parse import urlparse

    parsed = urlparse(server_url)
    port = parsed.port or 8003
    cmd = [pocket_bin, "serve", "--port", str(port)]
    if voice_path:
        candidate = Path(voice_path)
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        cmd.extend(["--voice", str(candidate)])

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    deadline = time.time() + 20.0
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Pocket-TTS server exited with code {proc.returncode}")
        try:
            requests.get(server_url, timeout=1)
            return proc
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("Pocket-TTS server did not become ready")


def _stop_proc_group(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), 15)
        proc.wait(timeout=5)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), 9)
            proc.wait(timeout=2)
        except Exception:
            pass


def _launch_vision_service(port: int, live_camera: bool) -> subprocess.Popen:
    cmd = [
        "uv", "run", "--project", str(SERVICES_DIR), "python", str(SERVICES_DIR / "app.py"),
        "--port", str(port),
        "--auto-start-trackers",
    ]
    if not live_camera:
        cmd.append("--no-camera")
    proc = subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    client = VisionClient(f"http://127.0.0.1:{port}")
    deadline = time.time() + 40.0
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Vision service exited early with code {proc.returncode}")
        if client.health():
            return proc
        time.sleep(0.5)
    raise RuntimeError("Vision service did not become healthy")


def _wait_for_vision_models(client: VisionClient, *, timeout: float = 35.0) -> dict:
    """Wait for the vision stack to become genuinely ready, not just HTTP-healthy."""
    deadline = time.time() + timeout
    last_status: dict = {}
    while time.time() < deadline:
        try:
            status = client.model_status()
            last_status = status
            sam3 = status.get("sam3", {})
            face = status.get("face", {})
            person = status.get("person", {})
            vllm_ready = status.get("vllm") in {"ready", "off"}
            sam3_ready = not isinstance(sam3, dict) or sam3.get("status") in {"ready", "off", "unavailable"}
            face_ready = not isinstance(face, dict) or face.get("status") in {"ready", "off", "unavailable"}
            person_ready = not isinstance(person, dict) or person.get("status") in {"ready", "off", "unavailable"}
            if vllm_ready and sam3_ready and face_ready and person_ready:
                return status
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"Vision models did not become ready: {last_status}")


class PCMCollector:
    def __init__(self):
        self.parts: list[bytes] = []
        self._start = time.perf_counter()
        self.first_chunk_at: float | None = None

    def __call__(self, data: bytes) -> None:
        if self.first_chunk_at is None:
            self.first_chunk_at = time.perf_counter()
        self.parts.append(bytes(data))

    @property
    def pcm(self) -> bytes:
        return b"".join(self.parts)


def synthesize_pocket_pcm(text: str, server_url: str, voice_path: str = "") -> SynthesisResult:
    collector = PCMCollector()
    tts = PocketTTS(
        on_audio=collector,
        server_url=server_url,
        voice="" if voice_path else "alba",
    )
    tts.send_text(text)
    tts.flush()
    if not tts.wait_for_done(timeout=60.0):
        raise RuntimeError("Pocket-TTS synthesis timed out")
    tts.close()
    synth_ms = int((time.perf_counter() - collector._start) * 1000)
    first_audio_ms = int((collector.first_chunk_at - collector._start) * 1000) if collector.first_chunk_at else 0
    audio_ms = int(len(collector.pcm) / 2 / 24000 * 1000)
    return SynthesisResult(
        pcm=collector.pcm,
        synth_ms=synth_ms,
        first_audio_ms=first_audio_ms,
        audio_ms=audio_ms,
    )


def transcribe_pcm_deepgram(pcm_24k: bytes) -> str:
    """Transcribe synthesized PCM through Deepgram's REST endpoint."""
    api_key = os.environ.get("DEEPGRAM_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DEEPGRAM_API_KEY not set")

    wav_buf = BytesIO()
    with wave.open(wav_buf, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(pcm_24k)

    query = parse.urlencode({
        "model": "nova-3",
        "smart_format": "true",
        "punctuate": "true",
    })
    req = request.Request(
        f"https://api.deepgram.com/v1/listen?{query}",
        data=wav_buf.getvalue(),
        headers={
            "Authorization": f"Token {api_key}",
            "Content-Type": "audio/wav",
        },
        method="POST",
    )
    with request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())

    return (
        data.get("results", {})
        .get("channels", [{}])[0]
        .get("alternatives", [{}])[0]
        .get("transcript", "")
        .strip()
    )


def _seed_world_from_visual(world, evaluation) -> None:
    for fact in evaluation.world_facts:
        world.add_fact(fact)


def _contains_any(text: str, candidates: tuple[str, ...] | list[str]) -> bool:
    lowered = text.lower()
    return any(candidate in lowered for candidate in candidates)


def _has_required_groups(text: str, groups: list[tuple[str, ...]]) -> bool:
    lowered = text.lower()
    return all(any(option in lowered for option in group) for group in groups)


def _sanitize_fact_list(spec: dict, facts: list[str]) -> list[str]:
    cleaned: list[str] = []
    for fact in facts:
        stripped = fact.strip()
        if not stripped:
            continue
        if _contains_any(stripped, DRIFT_TERMS):
            continue
        if not _contains_any(stripped, spec["allowed_terms"]):
            continue
        cleaned.append(stripped)
    return cleaned[:4]


def _clean_line(text: str, max_words: int = 12, question_default: bool = True) -> str:
    stripped = " ".join(text.replace("\n", " ").split())
    if not stripped:
        return ""
    words = stripped.split()
    if len(words) > max_words:
        stripped = " ".join(words[:max_words]).rstrip(",;:")
    if stripped and stripped[-1] not in ".!?":
        stripped += "?" if question_default else "."
    return stripped


def _clean_fact_text(text: str, max_words: int = 12) -> str:
    stripped = " ".join(text.replace("\n", " ").split())
    if not stripped:
        return ""
    words = stripped.split()
    if len(words) > max_words:
        stripped = " ".join(words[:max_words]).rstrip(",;:")
    return stripped


def _derive_live_user_lines(facts: list[str]) -> list[str]:
    fact_blob = " ".join(facts).lower()
    if "robot" in fact_blob:
        first = "Greg, did you figure out why the robot powered down?"
    elif "ladder" in fact_blob:
        first = "Should I worry about that ladder if I come in?"
    elif "door" in fact_blob or "shower" in fact_blob:
        first = "Do you want that door left open?"
    else:
        first = LIVE_SCENE_FALLBACK["user_line"]

    if "ladder" in fact_blob:
        second = "I don't want to trip over that ladder if I come in."
    elif "door" in fact_blob or "shower" in fact_blob:
        second = "Should that door stay open while we're working?"
    elif "robot" in fact_blob:
        second = "Do you want me to check the robot next?"
    else:
        second = LIVE_SCENE_FALLBACK["followup_line"]

    return [first, second]


def _ground_live_evaluation(evaluation) -> LiveScenePlan:
    summary = " ".join(evaluation.summary.split())
    if (
        not summary
        or len(summary.split()) > 28
        or _contains_any(summary, LIVE_DRIFT_TERMS)
        or ("greg" in summary.lower() and "reflection" not in summary.lower())
    ):
        summary = LIVE_SCENE_FALLBACK["summary"]

    facts: list[str] = []
    for fact in evaluation.world_facts:
        cleaned = _clean_fact_text(fact)
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if any(term in lowered for term in LIVE_DRIFT_TERMS):
            continue
        if "greg" in lowered and "camera" not in lowered and "view" not in lowered and "perspective" not in lowered:
            continue
        facts.append(cleaned)
    if not facts:
        facts = list(LIVE_SCENE_FALLBACK["world_facts"])
    elif not any("camera" in fact.lower() or "perspective" in fact.lower() or "view" in fact.lower() for fact in facts):
        facts.insert(0, LIVE_SCENE_FALLBACK["world_facts"][0])
    facts = facts[:4]

    visual_goal = _clean_line(evaluation.visual_goal, max_words=18, question_default=False)
    if not visual_goal or _contains_any(visual_goal, LIVE_DRIFT_TERMS):
        visual_goal = LIVE_SCENE_FALLBACK["visual_goal"]

    user_line, followup_line = _derive_live_user_lines(facts)

    return LiveScenePlan(
        summary=summary,
        world_facts=facts,
        visual_goal=visual_goal,
        user_lines=[user_line, followup_line],
    )


def _build_live_scene_specs(plan: LiveScenePlan) -> list[dict]:
    anchors = "; ".join(plan.world_facts[:4]) or plan.summary
    first_line = plan.user_lines[0] if plan.user_lines else LIVE_SCENE_FALLBACK["user_line"]
    second_line = plan.user_lines[1] if len(plan.user_lines) > 1 else LIVE_SCENE_FALLBACK["followup_line"]
    common_rules = [
        "Treat the current image as Greg's first-person perspective. Greg is behind the camera unless a reflection is obvious.",
        "Stay literal to the visible office or room. Do not invent Greg's body, a sidewalk stand, flyers, prices, or extra people.",
        "If the room is empty, assume the user is just off-camera and keep the exchange grounded in the room.",
    ]
    return [
        {
            "name": "live_open",
            "prompt": "Live camera bootstrap frame",
            "fallback_summary": plan.summary,
            "fallback_facts": list(plan.world_facts),
            "fallback_goal": plan.visual_goal,
            "fallback_user_line": first_line,
            "pre_turn_rules": list(common_rules),
            "visible_anchors": anchors,
            "skip_perception": True,
        },
        {
            "name": "live_followup",
            "prompt": "Live camera bootstrap frame",
            "fallback_summary": plan.summary,
            "fallback_facts": list(plan.world_facts),
            "fallback_goal": plan.visual_goal,
            "fallback_user_line": second_line,
            "pre_turn_rules": list(common_rules),
            "visible_anchors": anchors,
            "skip_perception": True,
        },
    ]


def _capture_live_frame(vision_client: VisionClient, timeout: float = 20.0) -> bytes:
    deadline = time.time() + timeout
    last_error = "no frame available yet"
    while time.time() < deadline:
        try:
            frame = vision_client.capture_frame_jpeg()
            if frame:
                return frame
        except Exception as exc:
            last_error = str(exc)
        time.sleep(0.5)
    raise RuntimeError(f"Unable to capture live camera frame: {last_error}")


def _live_bootstrap_prompt(snapshot) -> str:
    hints: list[str] = []
    if snapshot.persons:
        hints.append(f"Detected people: {', '.join(p.identity for p in snapshot.persons[:3])}")
    if snapshot.objects:
        hints.append(f"Detected objects: {', '.join(o.label for o in snapshot.objects[:5])}")
    if snapshot.vlm_answers:
        hints.append("Vision answers: " + " | ".join(
            f"{answer.question}: {answer.answer}" for answer in snapshot.vlm_answers[:3]
        ))
    hint_text = "\n".join(hints) if hints else "No structured detections are required; trust the image."
    return (
        "You are building a short, grounded NPC test scene from a real webcam image.\n"
        "The camera is Greg's first-person perspective. Greg is behind the camera and should not describe himself "
        "as visible unless a reflection is plainly present.\n"
        "Prefer a literal office, desk, room, or hallway reading. If the room looks empty, keep it empty and assume "
        "the user is just off-camera speaking to Greg.\n"
        "Do not invent a sidewalk stand, flyers, coupons, prices, lemonade, or extra people.\n"
        "Return a short summary, up to four visible world facts, one short opening user line, one short follow-up line, "
        "and a short visual_goal.\n"
        f"{hint_text}"
    )


def _apply_live_world_context(world, plan: LiveScenePlan) -> None:
    world.static = [
        "Greg experiences the room from a first-person camera viewpoint",
        "Greg is speaking from inside the current room, not from a sidewalk stand",
        "Greg can speak and observe, but cannot freely leave his current position on his own",
    ]
    world.dynamic.clear()
    world.events.clear()
    world.pending.clear()
    world._next_id = 1
    for fact in plan.world_facts:
        world.add_fact(fact)


def _inject_live_bootstrap_system(session: ChatSession, plan: LiveScenePlan) -> None:
    session.inject_system(
        "\n".join([
            "Live camera mode: treat the latest webcam frame as Greg's first-person point of view.",
            "Ignore any standing sidewalk or flier-selling assumptions from older prompts.",
            "Stay grounded in the visible room, objects, and any person actually present.",
            "If no person is visible, assume the user is just off-camera speaking to Greg.",
            "Keep replies short: one sentence, or two very short sentences at most.",
            "Answer the user's most concrete concern first before adding anything else.",
            f"Visible scene: {plan.summary}",
            f"Visible facts: {'; '.join(plan.world_facts[:4])}",
            f"Current visual goal: {plan.visual_goal}",
        ])
    )


def _inject_pre_turn_system(
    session: ChatSession,
    spec: dict,
    visual_summary: str,
    visual_goal: str,
    visual_world_facts: list[str],
) -> None:
    parts = [SHORT_TURN_PROMPT]
    parts.append("Keep the audible reply under 18 words when possible.")
    parts.append("Answer the user's concrete concern first, especially if they mention a visible object or hazard.")
    extra_rules = spec.get("pre_turn_rules")
    if extra_rules:
        parts.extend(extra_rules)
    else:
        parts.append(
            "Stay literal to the visible scene. Do not invent products, prices, coupons, "
            "ice cream, processing power, or extra people."
        )
        parts.append(
            "Visible anchors: Greg the robot head, the folding-table stand, water, cups, flyers, and one hooded visitor."
        )
        parts.append(
            "The sign says 'Free Water & Advice'. If the user asks about cost, answer that the water and advice are free."
        )
    if spec.get("visible_anchors"):
        parts.append(f"Visible anchors: {spec['visible_anchors']}")
    if visual_summary:
        parts.append(f"Fresh visual read: {visual_summary}")
    if visual_world_facts:
        parts.append(f"Visible facts: {'; '.join(visual_world_facts[:4])}")
    if visual_goal:
        parts.append(f"Use this visual beat: {visual_goal}")
    parts.append(f"Current scene: {spec['name']}.")
    session.inject_system("\n".join(parts))


def _scene_event_for_eval(spec: dict, summary: str) -> str:
    if (
        any(term in summary.lower() for term in ANCHOR_TERMS)
        and not any(term in summary.lower() for term in DRIFT_TERMS)
        and _has_required_groups(summary, spec["required_groups"])
    ):
        return summary
    return spec["fallback_event"]


def _run_user_turn(
    *,
    session: ChatSession,
    world,
    goals: Goals,
    script: Script,
    people: PeopleState,
    scenario,
    label: str,
    model_config: dict,
    user_input: str,
    had_user_input: bool,
    log: list[dict],
    visual_summary: str,
    visual_goal: str,
    visual_world_facts: list[str],
    scene_spec: dict,
    vision_mgr=None,
) -> tuple[str, bool]:
    stage_goal = scenario.active_stage.goal if scenario and scenario.active_stage else ""
    _inject_pre_turn_system(session, scene_spec, visual_summary, visual_goal, visual_world_facts)

    if not had_user_input:
        response = stream_response(session, label, user_input, expr_model_config=model_config)
        log.append({"type": "send", "input": user_input, "response": response})
        _bump_version()
        needs_plan, plan_request, _ = run_post_response(
            session, world, script, model_config, log, scenario, people, goals, stage_goal, vision_mgr=vision_mgr,
        )
        result = single_beat_call(
            system_prompt=session.system_prompt,
            world=world,
            history=session.get_history(),
            goals=goals,
            model_config=model_config,
            plan_request=plan_request,
            people=people,
            stage_goal=stage_goal,
        )
        if result.beats:
            apply_plan(script, result)
        return response, True

    if script.is_empty():
        result = single_beat_call(
            system_prompt=session.system_prompt,
            world=world,
            history=session.get_history(),
            goals=goals,
            model_config=model_config,
            people=people,
            stage_goal=stage_goal,
        )
        if result.beats:
            apply_plan(script, result)

    if script.current_beat is not None:
        response = stream_guided_beat(session, script.current_beat, label, user_input, expr_model_config=model_config)
    else:
        response = stream_response(session, label, user_input, expr_model_config=model_config)
    log.append({"type": "send", "input": user_input, "response": response})
    _bump_version()
    needs_plan, plan_request, _ = run_post_response(
        session, world, script, model_config, log, scenario, people, goals, stage_goal, vision_mgr=vision_mgr,
    )
    if needs_plan:
        result = single_beat_call(
            system_prompt=session.system_prompt,
            world=world,
            history=session.get_history(),
            goals=goals,
            model_config=model_config,
            plan_request=plan_request,
            people=people,
            stage_goal=stage_goal,
        )
        if result.beats:
            apply_plan(script, result)
    return response, had_user_input


def _response_note(response: str) -> tuple[bool, str]:
    words = len(response.split())
    if words > 20:
        return False, f"Too long ({words} words)"
    if response.count("?") > 1:
        return False, "More than one question"
    return True, "Punchy enough"


def _ground_evaluation(spec: dict, summary: str, facts: list[str], goal: str, user_line: str) -> tuple[str, list[str], str, str]:
    text_blob = " ".join([summary, user_line, goal, *facts]).lower()
    grounded = any(term in text_blob for term in ANCHOR_TERMS)
    drifted = any(term in text_blob for term in DRIFT_TERMS)
    cleaned_facts = _sanitize_fact_list(spec, facts)

    summary_ok = grounded and not drifted and _has_required_groups(summary, spec["required_groups"])
    facts_ok = len(cleaned_facts) >= 2 and _has_required_groups(" ".join(cleaned_facts), spec["required_groups"][:3])
    goal_ok = (
        bool(goal.strip())
        and len(goal.split()) <= 18
        and not _contains_any(goal, DRIFT_TERMS)
        and not _contains_any(goal, ("cinematic", "shot"))
        and _contains_any(goal, spec["allowed_terms"])
    )
    user_line_ok = (
        bool(user_line.strip())
        and len(user_line.split()) <= 12
        and not _contains_any(user_line, DRIFT_TERMS)
        and _contains_any(user_line, spec["allowed_terms"])
    )

    final_summary = summary.strip() if summary_ok else spec["fallback_summary"]
    final_facts = cleaned_facts if facts_ok else list(spec["fallback_facts"])
    final_goal = goal.strip() if goal_ok else spec["fallback_goal"]
    final_user_line = user_line.strip() if user_line_ok else spec["fallback_user_line"]
    return final_summary, final_facts, final_goal, final_user_line


def _timeline_event_label(event: dict) -> str:
    event_type = event.get("type", "event")
    data = event.get("data", {})
    if event_type == "scene_capture":
        return data.get("label", "Scene image captured")
    if event_type == "scene_eval":
        return data.get("summary", "Scene evaluation complete")
    if event_type == "vision_stack_ready":
        return (
            f"Vision stack ready: vLLM {data.get('vllm', 'unknown')}, "
            f"SAM3 {data.get('sam3', {}).get('status', 'unknown') if isinstance(data.get('sam3'), dict) else data.get('sam3', 'unknown')}, "
            f"Face {data.get('face', {}).get('status', 'unknown') if isinstance(data.get('face'), dict) else data.get('face', 'unknown')}, "
            f"Person {data.get('person', {}).get('status', 'unknown') if isinstance(data.get('person'), dict) else data.get('person', 'unknown')}"
        )
    if event_type == "vision_snapshot_read":
        return (
            f"Snapshot: {data.get('persons', 0)} persons, {data.get('objects', 0)} objects, "
            f"{data.get('faces', 0)} faces"
        )
    if event_type == "vision_poll":
        return (
            f"Vision poll: {data.get('persons', 0)} persons, {data.get('objects', 0)} objects, "
            f"{data.get('faces', 0)} faces"
        )
    if event_type == "vision_models":
        return (
            f"Models: vLLM {data.get('vllm', 'unknown')}, SAM3 {data.get('sam3', 'unknown')}, "
            f"Face {data.get('face', 'unknown')}, Person {data.get('person', 'unknown')}"
        )
    if event_type == "vision_poll_error":
        return f"Vision poll error: {data.get('error', '')}"
    if event_type == "world_seed":
        return data.get("summary", "Seeded world from visual context")
    if event_type == "perception_injected":
        return data.get("summary", "Injected visual perception")
    if event_type == "stt_result":
        return f"STT: {data.get('transcript', '')}"
    if event_type == "user_turn_start":
        return f"User line submitted: {data.get('text', '')}"
    if event_type == "response_ttft":
        return f"Assistant first token: {data.get('ttft_ms', 0)}ms"
    if event_type == "response_chunk":
        count = data.get("count", 1)
        duration_ms = data.get("duration_ms", 0)
        return f"Assistant stream ({count} chunks, {duration_ms}ms)"
    if event_type == "response_done":
        return f"Assistant text done: {data.get('total_ms', 0)}ms total"
    if event_type == "expression":
        return f"Expression: {data.get('expression', '')} / gaze {data.get('gaze', '')}"
    if event_type == "eval":
        return f"Eval: {data.get('script_status', '?')}"
    if event_type == "beat_advance":
        return f"Beat advance: {data.get('next_intent', 'script update')}"
    if event_type == "director":
        return f"Director: {data.get('thought', '')}"
    if event_type == "stage_change":
        return f"Stage {data.get('old_stage', '')} -> {data.get('new_stage', '')}"
    if event_type == "plan":
        beats = data.get("beats", [])
        return f"Plan loaded: {len(beats)} beats"
    if event_type == "assistant_tts_first_audio":
        return f"Assistant first audio: {data.get('first_audio_ms', 0)}ms"
    if event_type == "assistant_tts_done":
        return f"Assistant TTS done: {data.get('synth_ms', 0)}ms synth"
    if event_type == "assistant_audio_clip":
        return f"Assistant clip duration: {data.get('audio_ms', 0)}ms"
    if event_type == "session_start":
        return f"Session start: {data.get('character', '')} / {data.get('model', '')}"
    return json.dumps(data, ensure_ascii=True)


def _timeline_detail(event: dict) -> str:
    data = dict(event.get("data", {}))
    if not data:
        return ""
    data.pop("frame_b64", None)
    data.pop("overlay_b64", None)
    compact = json.dumps(data, ensure_ascii=True, separators=(",", ":"))
    return compact[:220]


def _detail_payload(event: dict) -> str:
    payload = dict(event.get("data", {}))
    if "frame_b64" in payload:
        payload["frame_b64"] = f"<{len(payload['frame_b64'])} base64 chars>"
    if "overlay_b64" in payload:
        payload["overlay_b64"] = f"<{len(payload['overlay_b64'])} base64 chars>"
    return json.dumps(payload, ensure_ascii=True, indent=2) if payload else "{}"


def _related_event_keys(events: list[dict], index: int) -> list[str]:
    current = events[index]
    now = current.get("timestamp", 0.0)
    current_type = current.get("type", "")
    current_lane = TIMELINE_LANES.get(current_type, "other")
    preferred = {
        "response_ttft": {"stt_result", "user_turn_start", "vision_snapshot_read", "scene_eval", "world_seed", "plan"},
        "response_chunk": {"stt_result", "user_turn_start", "vision_snapshot_read", "scene_eval", "world_seed", "plan"},
        "response_done": {"stt_result", "user_turn_start", "vision_snapshot_read", "scene_eval", "world_seed", "plan"},
        "assistant_tts_first_audio": {"response_done", "response_chunk", "response_ttft"},
        "assistant_tts_done": {"assistant_tts_first_audio", "response_done", "response_chunk", "response_ttft"},
        "assistant_audio_clip": {"assistant_tts_done", "assistant_tts_first_audio", "response_done"},
        "expression": {"response_done", "response_chunk"},
        "eval": {"response_done", "response_chunk"},
        "director": {"response_done", "eval", "beat_advance"},
        "plan": {"session_start", "world_seed", "scene_eval", "director"},
    }.get(current_type, set())

    candidates: list[tuple[int, int, str]] = []
    for prior_index in range(index - 1, -1, -1):
        prior = events[prior_index]
        age = now - prior.get("timestamp", now)
        if age > 4.0:
            break
        note_key = prior.get("_note_key")
        if not note_key:
            continue
        score = 0
        prior_type = prior.get("type", "")
        prior_lane = TIMELINE_LANES.get(prior_type, "other")
        if prior_type in preferred:
            score += 4
        if prior_lane != current_lane:
            score += 1
        if prior_type != current_type:
            score += 1
        candidates.append((score, prior_index, note_key))

    candidates.sort(key=lambda item: (-item[0], -item[1]))
    selected = sorted(candidates[:4], key=lambda item: item[1])
    return [note_key for _, _, note_key in selected]


def _collapse_trace_events(events: list[dict]) -> list[dict]:
    collapsed: list[dict] = []
    idx = 0
    while idx < len(events):
        event = events[idx]
        if event.get("type") != "response_chunk":
            collapsed.append(event)
            idx += 1
            continue
        start = event
        count = 1
        end = event
        idx += 1
        while idx < len(events) and events[idx].get("type") == "response_chunk":
            end = events[idx]
            count += 1
            idx += 1
        collapsed.append({
            "type": "response_chunk",
            "timestamp": start.get("timestamp", 0.0),
            "seq": start.get("seq", 0),
            "data": {
                "count": count,
                "duration_ms": int(max(0.0, end.get("timestamp", 0.0) - start.get("timestamp", 0.0)) * 1000),
            },
        })
    return collapsed


def _build_action_timeline(events: list[dict]) -> tuple[list[str], float]:
    flat = _collapse_trace_events(events)
    if not flat:
        return (["<div class='timeline-empty'>No trace events captured.</div>"], 0.0)

    flat.sort(key=lambda event: (event.get("timestamp", 0.0), event.get("seq", 0)))
    start_ts = flat[0].get("timestamp", 0.0)
    rows: list[str] = []
    for event in flat:
        event_type = html.escape(event.get("type", "event"))
        lane = html.escape(TIMELINE_LANES.get(event.get("type", ""), "other"))
        abs_ts = datetime.fromtimestamp(event.get("timestamp", start_ts)).strftime("%H:%M:%S.%f")[:-3]
        rel = max(0.0, event.get("timestamp", start_ts) - start_ts)
        label = html.escape(_timeline_event_label(event))
        detail = html.escape(_timeline_detail(event))
        turn = event.get("turn")
        turn_label = f"Turn {turn}" if turn else "Session"
        rows.append(
            "<div class='timeline-row'>"
            f"<div class='timeline-time'>+{rel:0.3f}s</div>"
            f"<div class='timeline-abs'>{abs_ts}</div>"
            f"<div class='timeline-lane lane-{lane}'>{lane}</div>"
            f"<div class='timeline-turn'>{turn_label}</div>"
            f"<div class='timeline-main'><div class='timeline-type'>{event_type}</div><div class='timeline-label'>{label}</div>"
            f"{f'<div class=\"timeline-detail\">{detail}</div>' if detail else ''}</div>"
            "</div>"
        )
    total_duration = flat[-1].get("timestamp", start_ts) - start_ts
    return rows, total_duration


def _flatten_session_events(turns: list[TurnRecord], session_events: list[dict] | None = None) -> list[dict]:
    if session_events is not None:
        base_events = [dict(event) for event in session_events]
    else:
        base_events = []
        for idx, turn in enumerate(turns, start=1):
            for event in turn.trace_events:
                item = dict(event)
                item.setdefault("turn", idx)
                base_events.append(item)

    expanded: list[dict] = []
    for event in base_events:
        expanded.append(event)
        if event.get("type") == "response_done":
            data = event.get("data", {})
            ttft_ms = int(data.get("ttft_ms", 0))
            total_ms = int(data.get("total_ms", 0))
            if ttft_ms > 0 and total_ms >= ttft_ms:
                start_ts = event.get("timestamp", 0.0) - (total_ms / 1000.0)
                ttft_ts = start_ts + (ttft_ms / 1000.0)
                expanded.append({
                    "type": "response_ttft",
                    "timestamp": ttft_ts,
                    "seq": event.get("seq", 0),
                    "turn": event.get("turn"),
                    "data": {"ttft_ms": ttft_ms},
                })
    return sorted(expanded, key=lambda event: (event.get("timestamp", 0.0), event.get("seq", 0), event.get("type", "")))


def _build_thread_lanes(events: list[dict]) -> tuple[list[str], list[str]]:
    if not events:
        return (["<div class='timeline-empty'>No session events captured.</div>"], [])

    start_ts = events[0].get("timestamp", 0.0)
    end_ts = events[-1].get("timestamp", start_ts)
    total = max(0.001, end_ts - start_ts)
    by_lane: dict[str, list[dict]] = {}
    for event in events:
        lane = TIMELINE_LANES.get(event.get("type", ""), "other")
        by_lane.setdefault(lane, []).append(event)

    summary_cards: list[str] = []
    rows: list[str] = []
    for lane in LANE_ORDER:
        lane_events = by_lane.get(lane)
        if not lane_events:
            continue
        summary_cards.append(
            f"<div class='summary-card' data-lane-card='{html.escape(lane)}'><div class='summary-kicker'>{html.escape(lane)}</div>"
            f"<div class='summary-value'>{len(lane_events)}</div><div class='summary-note'>events</div></div>"
        )
        dots: list[str] = []
        for event in lane_events:
            rel = max(0.0, event.get("timestamp", start_ts) - start_ts)
            left_pct = min(100.0, max(0.0, rel / total * 100.0))
            label = _timeline_event_label(event)
            turn = event.get("turn", "-")
            dots.append(
                f"<div class='lane-dot lane-{html.escape(lane)}' style='left:{left_pct:.2f}%' "
                f"title='Turn {turn} · +{rel:.3f}s · {html.escape(label)}'></div>"
            )
        rows.append(
            f"<div class='lane-row' data-lane-row='{html.escape(lane)}'>"
            f"<div class='lane-name'>{html.escape(lane)}</div>"
            f"<div class='lane-track'>{''.join(dots)}</div>"
            f"<div class='lane-count'>{len(lane_events)}</div>"
            "</div>"
        )
    return rows, summary_cards


def _stream_card_size(event: dict, lane: str) -> tuple[int, int]:
    if event.get("type") == "vision_poll":
        return (300, 210)
    if event.get("type") == "vision_models":
        return (220, 110)
    if lane in {"chat", "director", "planner"}:
        return (240, 120)
    return (220, 105)


def _build_stream_board(events: list[dict]) -> tuple[str, list[dict]]:
    if not events:
        return ("<div class='timeline-empty'>No stream events captured.</div>", [])

    display_events = _collapse_trace_events(events)
    for index, event in enumerate(display_events, start=1):
        event["_note_key"] = f"event-{index}-{event.get('seq', 0)}"
    start_ts = display_events[0].get("timestamp", 0.0)
    end_ts = display_events[-1].get("timestamp", start_ts)
    total = max(0.001, end_ts - start_ts)
    px_per_sec = 180
    label_width = 108
    track_width = max(1400, int(total * px_per_sec) + 320)

    ruler_ticks: list[str] = []
    whole_seconds = max(1, int(total) + 1)
    for second in range(whole_seconds + 1):
        left = label_width + second * px_per_sec
        ruler_ticks.append(
            f"<div class='ruler-tick' data-base-left='{left}' style='left:{left}px'><span>{second:.0f}s</span></div>"
        )

    by_lane: dict[str, list[dict]] = {}
    for event in display_events:
        lane = TIMELINE_LANES.get(event.get("type", ""), "other")
        by_lane.setdefault(lane, []).append(event)
    related_key_map = {
        event["_note_key"]: _related_event_keys(display_events, index)
        for index, event in enumerate(display_events)
    }

    annotation_events: list[dict] = []
    lane_sections: list[str] = []
    for lane in LANE_ORDER:
        lane_events = by_lane.get(lane)
        if not lane_events:
            continue
        track_ends: list[float] = []
        cards: list[str] = []
        lane_row_height = 0
        for event in lane_events:
            rel = max(0.0, event.get("timestamp", start_ts) - start_ts)
            left = label_width + rel * px_per_sec
            width, height = _stream_card_size(event, lane)
            row = 0
            while row < len(track_ends) and left < track_ends[row]:
                row += 1
            if row == len(track_ends):
                track_ends.append(left + width + 16)
            else:
                track_ends[row] = left + width + 16
            top = 10 + row * (height + 12)
            lane_row_height = max(lane_row_height, top + height + 10)

            event_type = event.get("type", "event")
            label = _timeline_event_label(event)
            detail = _timeline_detail(event)
            note_key = event["_note_key"]
            hover_title = html.escape(f"{event_type} | {label}" + (f"\n{detail}" if detail else ""))
            annotation_events.append({
                "key": note_key,
                "seq": event.get("seq", 0),
                "type": event_type,
                "lane": lane,
                "turn": event.get("turn"),
                "timestamp": rel,
                "label": label,
                "detail": detail,
                "payload": _detail_payload(event),
                "related_keys": related_key_map.get(note_key, []),
            })

            body_html = (
                f"<div class='stream-card-label'>{html.escape(label)}</div>"
                f"{f'<div class=\"stream-card-detail\">{html.escape(detail)}</div>' if detail else ''}"
            )
            if event_type == "vision_poll":
                data = event.get("data", {})
                raw_b64 = data.get("frame_b64", "")
                overlay_b64 = data.get("overlay_b64", "")
                mime_type = html.escape(data.get("mime_type", "image/jpeg"))
                objects = ", ".join(data.get("object_labels", [])[:4]) or "none"
                answers = data.get("vlm_answers", [])
                answers_html = "".join(
                    f"<div class='vision-answer'>{html.escape(answer)}</div>"
                    for answer in answers
                )
                body_html = (
                    "<div class='vision-strip'>"
                    f"<img src='data:{mime_type};base64,{raw_b64}' alt='raw vision sample'>"
                    f"<img src='data:{mime_type};base64,{overlay_b64}' alt='overlay vision sample'>"
                    "</div>"
                    "<div class='vision-meta'>"
                    f"<div><b>counts</b> faces {data.get('faces', 0)} · persons {data.get('persons', 0)} · objects {data.get('objects', 0)}</div>"
                    f"<div><b>objects</b> {html.escape(objects)}</div>"
                    f"{answers_html or '<div class=\"vision-answer muted\">No VLM answers</div>'}"
                    "</div>"
                )

            cards.append(
                f"<article class='stream-card lane-{html.escape(lane)}' data-base-left='{left}' data-base-top='{top}' "
                f"data-row='{row}' "
                f"data-event-key='{note_key}' "
                f"data-base-width='{width}' data-base-height='{height}' "
                f"title='{hover_title}' style='left:{left}px;top:{top}px;width:{width}px;height:{height}px' onclick='selectEvent(\"{note_key}\")'>"
                f"<div class='stream-card-head'><span class='stream-time'>+{rel:0.2f}s</span><span class='stream-type'>{html.escape(event_type)}</span></div>"
                f"{body_html}"
                f"<button class='note-btn note-inline' data-note-key='{note_key}' onclick='openNote(event,\"{note_key}\")'>Note</button>"
                "</article>"
            )

        lane_sections.append(
            f"<section class='stream-lane' data-stream-lane='{html.escape(lane)}'>"
            f"<div class='stream-lane-label'>{html.escape(lane)}</div>"
            f"<div class='stream-lane-track' data-base-width='{track_width}' data-base-height='{max(120, lane_row_height)}' data-row-count='{max(1, len(track_ends))}' style='width:{track_width}px;height:{max(120, lane_row_height)}px'>{''.join(cards)}</div>"
            "</section>"
        )

    board_html = (
        "<section class='stream-wrap'>"
        "<h2 style='margin:0 0 6px 0;color:#f4b942;'>Interwoven Streams</h2>"
        "<p style='margin-top:0;color:#9fb3c8;'>Each lane carries the actual payload at that moment. Vision samples include the raw frame and the overlay frame side by side.</p>"
        "<div class='stream-scroll'>"
        f"<div class='stream-ruler' style='width:{track_width + label_width}px'>{''.join(ruler_ticks)}</div>"
        + "".join(lane_sections)
        + "</div></section>"
    )
    return board_html, annotation_events


def _write_report(turns: list[TurnRecord], json_path: Path, html_path: Path, session_events: list[dict] | None = None) -> None:
    json_path.write_text(json.dumps({
        "turns": [asdict(turn) for turn in turns],
        "session_events": session_events or [],
    }, indent=2))

    report_name = html_path.stem
    session_trace = _flatten_session_events(turns, session_events)
    timeline_rows, total_duration = _build_action_timeline(session_trace)
    lane_rows, lane_summary_cards = _build_thread_lanes(session_trace)
    stream_board_html, annotation_events = _build_stream_board(session_trace)
    avg_ttft = int(sum(turn.response_ttft_ms for turn in turns) / max(1, len(turns)))
    avg_total = int(sum(turn.response_total_ms for turn in turns) / max(1, len(turns)))
    lane_toggle_html = "".join(
        f"<button class='lane-toggle active' type='button' data-lane-toggle='{html.escape(lane)}' onclick='toggleLane(\"{html.escape(lane)}\")'>{html.escape(lane)}</button>"
        for lane in LANE_ORDER
    )

    html_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>Full Stack QA Trace</title>"
        "<style>"
        ":root{--time-zoom:1;--row-scale:1;--sidebar-width:150px;}"
        "*{box-sizing:border-box;}"
        "body{font-family:IBM Plex Sans,system-ui,sans-serif;background:#081018;color:#e7eef7;margin:0;padding:24px 24px 96px;}"
        ".page-shell{width:min(100%,1800px);margin:0 auto;}"
        ".page-shell.compact-mode .stream-wrap{padding:12px 14px;margin-bottom:16px;}"
        ".page-shell.compact-mode .stream-card{padding:2px 8px;border-radius:7px;box-shadow:none;display:flex;align-items:center;gap:8px;}"
        ".page-shell.compact-mode .stream-card-detail,.page-shell.compact-mode .vision-meta,.page-shell.compact-mode .note-inline,.page-shell.compact-mode .vision-strip,.page-shell.compact-mode .stream-type{display:none;}"
        ".page-shell.compact-mode .stream-card-head{margin:0;font-size:.68rem;min-width:58px;flex:0 0 auto;}"
        ".page-shell.compact-mode .stream-time{white-space:nowrap;}"
        ".page-shell.compact-mode .stream-card-label{font-size:.74rem;line-height:1.1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;display:block;}"
        ".page-shell.compact-mode .stream-lane{padding:6px 0;}"
        ".page-shell.compact-mode .stream-lane-label{padding-top:4px;font-size:.72rem;}"
        ".page-shell.compact-mode .stream-ruler{height:22px;}"
        ".page-shell.compact-mode .ruler-tick span{top:2px;font-size:.68rem;}"
        "h1{margin-bottom:8px;}p{line-height:1.5;}code{background:#101720;padding:2px 6px;border-radius:6px;}"
        "#annotation-toolbar{position:sticky;top:0;z-index:10;background:rgba(8,16,24,.96);backdrop-filter:blur(12px);border:1px solid #253244;border-radius:14px;padding:12px 14px;display:flex;gap:12px;align-items:center;margin:0 0 20px 0;}"
        "#annotation-toolbar .count{color:#9fb3c8;}#annotation-toolbar .count b{color:#7ee787;}"
        "#annotation-toolbar button{border:none;border-radius:8px;padding:0.45em 1em;cursor:pointer;font:inherit;font-weight:600;}"
        "#export-btn{background:#238636;color:#fff;}#export-btn:disabled{background:#30363d;color:#8b949e;cursor:default;}"
        "#done-btn{display:none;background:#30363d;color:#e7eef7;}#save-flash{color:#7ee787;font-weight:600;opacity:0;transition:opacity .25s;}"
        "#save-flash.show{opacity:1;}.save-path{color:#8b949e;font-size:.85em;margin-left:auto;}"
        ".note-btn.has-note{background:#238636;}"
        ".view-controls{position:sticky;top:74px;z-index:9;background:rgba(12,20,29,.94);backdrop-filter:blur(12px);border:1px solid #253244;border-radius:16px;padding:14px 16px;display:grid;grid-template-columns:repeat(3,minmax(200px,auto)) 1fr;gap:14px;align-items:end;margin:0 0 22px 0;}"
        ".control-block{display:grid;gap:6px;}"
        ".control-block label{font-size:.78rem;font-weight:700;text-transform:uppercase;letter-spacing:.05em;color:#9fb3c8;}"
        ".control-inline{display:flex;gap:10px;align-items:center;}"
        ".control-inline output{font-family:IBM Plex Mono,monospace;color:#f4b942;min-width:48px;}"
        ".control-inline input[type='range']{width:100%;}"
        ".control-inline select{background:#081018;color:#e7eef7;border:1px solid #253244;border-radius:8px;padding:0.45em 0.65em;font:inherit;}"
        ".control-inline input[type='checkbox']{width:16px;height:16px;accent-color:#1f6feb;}"
        ".control-inline button{border:none;border-radius:8px;padding:0.5em 0.85em;cursor:pointer;background:#30363d;color:#e7eef7;font:inherit;font-weight:600;}"
        ".lane-toggle-grid{display:flex;flex-wrap:wrap;gap:8px;justify-content:flex-end;}"
        ".lane-toggle{border:1px solid #253244;border-radius:999px;background:#081018;color:#c9d1d9;padding:0.45em 0.85em;cursor:pointer;font:inherit;font-size:.82rem;text-transform:uppercase;letter-spacing:.04em;}"
        ".lane-toggle.active{background:#1f6feb22;color:#79c0ff;border-color:#1f6feb66;}"
        ".lane-toggle.inactive{opacity:.45;}"
        ".summary-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin:18px 0 24px 0;}"
        ".summary-card{background:#101720;border:1px solid #253244;border-radius:16px;padding:14px 16px;}"
        ".summary-card.is-hidden{opacity:.35;}"
        ".summary-kicker{font-size:.75rem;text-transform:uppercase;letter-spacing:.06em;color:#9fb3c8;}"
        ".summary-value{font-size:1.5rem;font-weight:700;color:#f4b942;margin-top:4px;}"
        ".summary-note{margin-top:4px;color:#9fb3c8;font-size:.85rem;}"
        ".lane-wrap{background:#101720;border:1px solid #253244;border-radius:16px;padding:18px;margin:18px 0 24px 0;}"
        ".lane-grid{display:grid;gap:10px;margin-top:12px;}"
        ".lane-row{display:grid;grid-template-columns:92px 1fr 42px;gap:12px;align-items:center;}"
        ".lane-row.is-hidden{display:none;}"
        ".lane-name{font-size:.82rem;font-weight:700;text-transform:uppercase;letter-spacing:.05em;color:#c9d1d9;}"
        ".lane-track{position:relative;height:26px;background:#081018;border:1px solid #1d2a3a;border-radius:999px;overflow:hidden;}"
        ".lane-dot{position:absolute;top:50%;width:10px;height:10px;border-radius:50%;transform:translate(-50%,-50%);box-shadow:0 0 0 2px #081018;}"
        ".lane-count{font-family:IBM Plex Mono,monospace;font-size:.82rem;color:#9fb3c8;text-align:right;}"
        ".lane-dot.lane-session{background:#79c0ff;}.lane-dot.lane-vision{background:#7ee787;}.lane-dot.lane-world{background:#d2a8ff;}.lane-dot.lane-stt{background:#e3b341;}.lane-dot.lane-chat{background:#ffa198;}.lane-dot.lane-expression{background:#ffb3ad;}.lane-dot.lane-eval{background:#8ddb8c;}.lane-dot.lane-script{background:#f2cc60;}.lane-dot.lane-director{background:#c297ff;}.lane-dot.lane-planner{background:#7dd3fc;}.lane-dot.lane-tts{background:#f5a6d3;}.lane-dot.lane-other{background:#c9d1d9;}"
        ".viewer-grid{display:grid;grid-template-columns:minmax(0,1fr) 360px;gap:18px;align-items:start;}"
        ".stream-wrap{background:#101720;border:1px solid #253244;border-radius:16px;padding:18px;margin:18px 0 24px 0;overflow:hidden;}"
        ".stream-scroll{overflow:auto;padding-bottom:8px;max-width:100%;}"
        ".stream-ruler{position:sticky;top:0;z-index:4;height:28px;margin-left:0;border-bottom:1px solid #253244;background:#101720;}"
        ".ruler-tick{position:absolute;top:0;bottom:0;width:1px;background:#253244;}"
        ".ruler-tick span{position:absolute;top:4px;left:4px;font-family:IBM Plex Mono,monospace;font-size:.78rem;color:#9fb3c8;}"
        ".stream-lane{display:grid;grid-template-columns:var(--sidebar-width) 1fr;gap:12px;padding:12px 0;border-bottom:1px solid #1d2a3a;align-items:start;}"
        ".stream-lane.is-hidden{display:none;}"
        ".stream-lane:last-child{border-bottom:none;}"
        ".stream-lane-label{position:sticky;left:0;z-index:3;align-self:stretch;font-size:.82rem;font-weight:700;text-transform:uppercase;letter-spacing:.05em;color:#c9d1d9;padding:12px 10px 0 0;background:linear-gradient(90deg,#101720 0%,#101720 78%,rgba(16,23,32,0) 100%);}"
        ".stream-lane-track{position:relative;background:linear-gradient(180deg,rgba(8,16,24,.8),rgba(8,16,24,.4));border:1px solid #1d2a3a;border-radius:18px;min-height:120px;}"
        ".stream-card{position:absolute;background:#0c141d;border:1px solid #253244;border-radius:14px;padding:10px 10px 12px 10px;overflow:hidden;box-shadow:0 10px 24px rgba(0,0,0,.24);cursor:pointer;}"
        ".stream-card.selected{border-color:#f4b942;box-shadow:0 0 0 1px #f4b942,0 12px 28px rgba(0,0,0,.3);}"
        ".stream-card.context{border-color:#1f6feb;box-shadow:0 0 0 1px #1f6feb55,0 10px 24px rgba(0,0,0,.24);}"
        ".stream-card-head{display:flex;justify-content:space-between;gap:8px;font-family:IBM Plex Mono,monospace;font-size:.76rem;color:#9fb3c8;margin-bottom:6px;}"
        ".stream-card-label{font-size:.92rem;line-height:1.35;}"
        ".stream-card-detail{margin-top:6px;color:#9fb3c8;font-size:.8rem;white-space:pre-wrap;word-break:break-word;}"
        ".stream-type{color:#f4b942;text-transform:uppercase;letter-spacing:.04em;}"
        ".vision-strip{display:grid;grid-template-columns:1fr 1fr;gap:8px;}"
        ".vision-strip img{width:100%;border-radius:10px;border:1px solid #253244;background:#081018;}"
        ".vision-meta{margin-top:8px;font-size:.8rem;color:#c9d1d9;display:grid;gap:4px;}"
        ".vision-answer{background:#081018;border:1px solid #1d2a3a;border-radius:10px;padding:5px 7px;}"
        ".vision-answer.muted{color:#7d8590;}"
        ".note-inline{margin-top:8px;background:#1f6feb;color:#fff;border:none;padding:0.3em 0.7em;border-radius:8px;cursor:pointer;font-weight:600;}"
        ".detail-panel{position:sticky;top:156px;background:#101720;border:1px solid #253244;border-radius:16px;padding:18px;max-height:calc(100vh - 176px);overflow:auto;}"
        ".detail-empty{color:#9fb3c8;margin:0;}"
        ".detail-meta{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0 14px 0;}"
        ".detail-chip{font-family:IBM Plex Mono,monospace;font-size:.78rem;padding:3px 8px;border-radius:999px;background:#081018;border:1px solid #253244;color:#c9d1d9;}"
        ".detail-section{margin-top:14px;display:grid;gap:6px;}"
        ".detail-section label{font-size:.75rem;font-weight:700;text-transform:uppercase;letter-spacing:.05em;color:#9fb3c8;}"
        ".detail-related{display:grid;gap:8px;}"
        ".detail-related button{border:1px solid #253244;background:#081018;color:#e7eef7;border-radius:10px;padding:8px 10px;text-align:left;cursor:pointer;font:inherit;}"
        ".detail-related small{display:block;color:#9fb3c8;margin-top:3px;}"
        ".detail-payload{margin:0;background:#081018;border:1px solid #253244;border-radius:12px;padding:10px;font:12px/1.45 IBM Plex Mono,monospace;color:#c9d1d9;white-space:pre-wrap;word-break:break-word;}"
        "#detail-note{width:100%;min-height:120px;background:#081018;color:#e7eef7;border:1px solid #253244;border-radius:12px;padding:10px;font:inherit;}"
        ".chronology-pane{display:grid;gap:8px;max-height:280px;overflow:auto;padding-right:2px;}"
        ".chronology-pane.hidden{display:none;}"
        ".chronology-row{border:1px solid #253244;background:#081018;color:#e7eef7;border-radius:10px;padding:8px 10px;text-align:left;cursor:pointer;font:inherit;}"
        ".chronology-row.selected{border-color:#f4b942;box-shadow:0 0 0 1px #f4b942;}"
        ".chronology-row small{display:block;color:#9fb3c8;margin-top:3px;}"
        ".timeline-wrap{background:#101720;border:1px solid #253244;border-radius:16px;padding:18px;margin:18px 0 24px 0;}"
        ".timeline-grid{display:grid;gap:8px;}"
        ".timeline-row{display:grid;grid-template-columns:88px 96px 92px 66px 1fr;gap:10px;align-items:start;background:#081018;border:1px solid #1d2a3a;border-radius:12px;padding:10px 12px;}"
        ".timeline-time,.timeline-abs,.timeline-turn{font-family:IBM Plex Mono,monospace;font-size:.85rem;color:#9fb3c8;}"
        ".timeline-lane{font-size:.78rem;font-weight:700;text-transform:uppercase;letter-spacing:.04em;padding:2px 8px;border-radius:999px;justify-self:start;}"
        ".lane-session{background:#1f6feb22;color:#79c0ff;}.lane-vision{background:#23863622;color:#7ee787;}.lane-world{background:#8957e522;color:#d2a8ff;}.lane-stt{background:#d2992222;color:#e3b341;}.lane-chat{background:#f8514922;color:#ffa198;}.lane-expression{background:#ff7b7222;color:#ffb3ad;}.lane-eval{background:#2ea04322;color:#8ddb8c;}.lane-script{background:#bf870022;color:#f2cc60;}.lane-director{background:#8250df22;color:#c297ff;}.lane-planner{background:#0ea5e922;color:#7dd3fc;}.lane-tts{background:#db61a222;color:#f5a6d3;}.lane-other{background:#30363d;color:#c9d1d9;}"
        ".timeline-type{font-size:.82rem;font-weight:700;color:#f4b942;text-transform:uppercase;letter-spacing:.04em;}"
        ".timeline-label{margin-top:2px;}.timeline-detail{margin-top:4px;color:#9fb3c8;font-family:IBM Plex Mono,monospace;font-size:.8rem;white-space:pre-wrap;word-break:break-word;}"
        ".timeline-empty{color:#9fb3c8;padding:8px 0;}"
        "@media (max-width: 1280px){.viewer-grid{grid-template-columns:1fr;}.detail-panel{position:static;max-height:none;}}"
        "@media (max-width: 1080px){.view-controls{grid-template-columns:1fr;}.lane-toggle-grid{justify-content:flex-start;}}"
        "@media (max-width: 860px){body{padding:16px 12px 72px;}.stream-lane{grid-template-columns:1fr;}.stream-lane-label{position:static;padding:0;background:none;}.timeline-row{grid-template-columns:1fr;}.timeline-lane,.timeline-turn,.timeline-abs,.timeline-time{justify-self:start;}}"
        "</style>"
        "</head><body><div class='page-shell'>"
        "<h1>Full Stack QA Trace Report</h1>"
        "<p>Stream-centric QA artifact with vision thumbnails, overlay thumbnails, live chat traces, and interwoven subsystem lanes.</p>"
        f"<div id='annotation-toolbar'>"
        "<span class='count'><b id='note-count'>0</b> annotations</span>"
        "<button id='export-btn' disabled onclick='exportAnnotations()'>Export annotations</button>"
        "<button id='done-btn' onclick='shutdownServer()'>Done</button>"
        "<span id='save-flash'></span>"
        f"<span class='save-path'>Save to: <code>logs/annotated/{html.escape(report_name)}.annotations.json</code></span>"
        "</div>"
        "<section class='view-controls'>"
        "<div class='control-block'><label for='time-zoom'>Time Zoom</label><div class='control-inline'><input id='time-zoom' type='range' min='0.6' max='2.4' step='0.05' value='1'><output id='time-zoom-value'>1.00x</output></div></div>"
        "<div class='control-block'><label for='row-scale'>Row Scale</label><div class='control-inline'><input id='row-scale' type='range' min='0.75' max='1.8' step='0.05' value='1'><output id='row-scale-value'>1.00x</output></div></div>"
        "<div class='control-block'><label for='density-mode'>Density</label><div class='control-inline'><select id='density-mode'><option value='expanded'>Expanded</option><option value='compact'>Compact</option></select></div></div>"
        "<div class='control-block'><label>Side Pane</label><div class='control-inline'><input id='show-chronology' type='checkbox' checked><span style='color:#c9d1d9;'>Chronology</span></div></div>"
        "<div class='control-block'><label>View</label><div class='control-inline'><button type='button' onclick='resetView()'>Reset View</button></div></div>"
        f"<div class='lane-toggle-grid'>{lane_toggle_html}</div>"
        "</section>"
        "<section class='summary-grid'>"
        f"<div class='summary-card'><div class='summary-kicker'>Turns</div><div class='summary-value'>{len(turns)}</div><div class='summary-note'>assistant turns</div></div>"
        f"<div class='summary-card'><div class='summary-kicker'>Trace Window</div><div class='summary-value'>{total_duration:0.2f}s</div><div class='summary-note'>captured session time</div></div>"
        f"<div class='summary-card'><div class='summary-kicker'>Avg TTFT</div><div class='summary-value'>{avg_ttft}ms</div><div class='summary-note'>assistant first token</div></div>"
        f"<div class='summary-card'><div class='summary-kicker'>Avg Total</div><div class='summary-value'>{avg_total}ms</div><div class='summary-note'>assistant full response</div></div>"
        + "".join(lane_summary_cards)
        + "</section>"
        f"<section class='lane-wrap'><h2 style='margin:0 0 6px 0;color:#f4b942;'>Thread Lanes</h2>"
        "<p style='margin-top:0;color:#9fb3c8;'>Each lane is a subsystem/thread family. Dots are positioned by time so overlapping activity is easy to compare.</p>"
        f"<div class='lane-grid'>{''.join(lane_rows)}</div></section>"
        + "<section class='viewer-grid'>"
        + stream_board_html
        + "<aside class='detail-panel'>"
        + "<h2 style='margin:0 0 6px 0;color:#f4b942;'>Event Detail</h2>"
        + "<p class='detail-empty' id='detail-empty'>Select a stream card to inspect its inputs, outputs, payload, and note.</p>"
        + "<div id='detail-body' style='display:none;'>"
        + "<div class='detail-meta'><span class='detail-chip' id='detail-time'>+0.00s</span><span class='detail-chip' id='detail-type'>type</span><span class='detail-chip' id='detail-lane'>lane</span><span class='detail-chip' id='detail-seq'>seq</span></div>"
        + "<div class='detail-section'><label>Summary</label><div id='detail-label'></div></div>"
        + "<div class='detail-section'><label>Compact Detail</label><div id='detail-detail' style='color:#9fb3c8;white-space:pre-wrap;word-break:break-word;'></div></div>"
        + "<div class='detail-section'><label>Input Context</label><div class='detail-related' id='detail-related'></div></div>"
        + "<div class='detail-section'><label>Payload</label><pre class='detail-payload' id='detail-payload'></pre></div>"
        + "<div class='detail-section'><label>Annotation</label><textarea id='detail-note' placeholder='Add an iteration note for this event...' oninput='updateSelectedAnnotation(this.value)'></textarea></div>"
        + "<div class='detail-section'><label>Chronology</label><div class='chronology-pane' id='detail-chronology'></div></div>"
        + "</div></aside></section>"
        + f"<details class='timeline-wrap'><summary style='cursor:pointer;'><b>Raw Chronology</b> ({len(timeline_rows)} rows)</summary>"
        f"<p style='margin-top:12px;color:#9fb3c8;'>Flattened trace across the whole run. Total traced time: {total_duration:0.2f}s.</p>"
        f"<div class='timeline-grid'>{''.join(timeline_rows)}</div></details>"
        + "<script>"
        + f"const REPORT_NAME = {json.dumps(report_name)};"
        + f"const MODEL_KEY = {json.dumps(json_path.stem)};"
        + f"const LANES = {json.dumps(LANE_ORDER)};"
        + f"const EVENTS = {json.dumps(annotation_events)};"
        + "const EVENT_INDEX = Object.fromEntries(EVENTS.map((event)=>[String(event.key),event]));"
        + "const annotations = new Map();"
        + "const hiddenLanes = new Set();"
        + "let selectedEventKey = '';"
        + "let lastSavedPath = '';"
        + "function flashSaved(text, sticky=false){const el=document.getElementById('save-flash');el.textContent=text;el.classList.add('show');if(!sticky){setTimeout(()=>el.classList.remove('show'),2200);}}"
        + "function updateAnnotation(turn,text){const key=String(turn);const btn=document.querySelector('[data-note-key=\"'+key+'\"]');if(text.trim()){annotations.set(key,text.trim());if(btn)btn.classList.add('has-note');}else{annotations.delete(key);if(btn)btn.classList.remove('has-note');}document.getElementById('note-count').textContent=annotations.size;document.getElementById('export-btn').disabled=annotations.size===0;}"
        + "function updateSelectedAnnotation(text){if(!selectedEventKey)return;updateAnnotation(selectedEventKey,text);}"
        + "function renderChronology(){const root=document.getElementById('detail-chronology');root.innerHTML='';EVENTS.forEach((event)=>{const row=document.createElement('button');row.type='button';row.className='chronology-row';row.dataset.chronoKey=String(event.key);row.onclick=()=>selectEvent(String(event.key));row.innerHTML='<div><b>'+event.type+'</b> · '+event.label+'</div><small>'+event.lane+' · +'+Number(event.timestamp||0).toFixed(2)+'s</small>';root.appendChild(row);});}"
        + "function renderRelated(keys){const root=document.getElementById('detail-related');root.innerHTML='';if(!keys||keys.length===0){root.innerHTML='<div style=\"color:#9fb3c8;\">No linked upstream events captured for this item.</div>';return;}keys.forEach((key)=>{const event=EVENT_INDEX[String(key)];if(!event)return;const button=document.createElement('button');button.type='button';button.onclick=()=>selectEvent(String(key));button.innerHTML='<div><b>'+event.type+'</b> · '+event.label+'</div><small>'+event.lane+' · +'+Number(event.timestamp||0).toFixed(2)+'s</small>';root.appendChild(button);});}"
        + "function highlightContext(selected){document.querySelectorAll('.stream-card').forEach((card)=>{card.classList.toggle('selected',card.dataset.eventKey===selected);card.classList.toggle('context',false);});document.querySelectorAll('.chronology-row').forEach((row)=>{const active=row.dataset.chronoKey===selected;row.classList.toggle('selected',active);if(active){row.scrollIntoView({block:'nearest'});}});const event=EVENT_INDEX[String(selected)];if(!event)return;(event.related_keys||[]).forEach((key)=>{const card=document.querySelector('.stream-card[data-event-key=\"'+key+'\"]');if(card)card.classList.add('context');});}"
        + "function selectEvent(key){const event=EVENT_INDEX[String(key)];if(!event)return;selectedEventKey=String(key);highlightContext(selectedEventKey);document.getElementById('detail-empty').style.display='none';document.getElementById('detail-body').style.display='block';document.getElementById('detail-time').textContent='+'+Number(event.timestamp||0).toFixed(2)+'s';document.getElementById('detail-type').textContent=event.type;document.getElementById('detail-lane').textContent=event.lane;document.getElementById('detail-seq').textContent='seq '+String(event.seq||0);document.getElementById('detail-label').textContent=event.label||'';document.getElementById('detail-detail').textContent=event.detail||'No compact detail';document.getElementById('detail-payload').textContent=event.payload||'{}';renderRelated(event.related_keys||[]);const note=annotations.get(String(key))||'';const field=document.getElementById('detail-note');field.value=note;const card=document.querySelector('.stream-card[data-event-key=\"'+key+'\"]');if(card){card.scrollIntoView({block:'nearest',inline:'center'});}}"
        + "function openNote(domEvent,key){if(domEvent){domEvent.stopPropagation();}selectEvent(String(key));const field=document.getElementById('detail-note');if(field){field.focus();field.setSelectionRange(field.value.length,field.value.length);}}"
        + "function applyDensity(){const shell=document.querySelector('.page-shell');const compact=document.getElementById('density-mode').value==='compact';shell.classList.toggle('compact-mode',compact);applyLayout();}"
        + "function syncChronologyVisibility(){const show=document.getElementById('show-chronology').checked;document.getElementById('detail-chronology').classList.toggle('hidden',!show);}"
        + "function applyLayout(){const zoom=Number(document.getElementById('time-zoom').value||1);const rowScale=Number(document.getElementById('row-scale').value||1);const compact=document.getElementById('density-mode').value==='compact';const compactHeight=22;const compactGap=4;document.documentElement.style.setProperty('--time-zoom',zoom);document.documentElement.style.setProperty('--row-scale',rowScale);document.getElementById('time-zoom-value').textContent=zoom.toFixed(2)+'x';document.getElementById('row-scale-value').textContent=rowScale.toFixed(2)+'x';document.querySelectorAll('.ruler-tick').forEach((tick)=>{const baseLeft=Number(tick.dataset.baseLeft||0);tick.style.left=(baseLeft*zoom)+'px';});document.querySelectorAll('.stream-lane-track').forEach((track)=>{const baseWidth=Number(track.dataset.baseWidth||0);const baseHeight=Number(track.dataset.baseHeight||120);const rowCount=Number(track.dataset.rowCount||1);track.style.width=(baseWidth*zoom)+'px';track.style.height=compact?(Math.max(34,rowCount*(compactHeight+compactGap)+10)+'px'):(Math.max(120,baseHeight*rowScale)+'px');});document.querySelectorAll('.stream-card').forEach((card)=>{const baseLeft=Number(card.dataset.baseLeft||0);const baseTop=Number(card.dataset.baseTop||0);const baseWidth=Number(card.dataset.baseWidth||320);const baseHeight=Number(card.dataset.baseHeight||120);const row=Number(card.dataset.row||0);card.style.left=(baseLeft*zoom)+'px';if(compact){card.style.top=(6+row*(compactHeight+compactGap))+'px';card.style.width=(Math.max(96,baseWidth*0.42*zoom))+'px';card.style.height=compactHeight+'px';}else{card.style.top=(baseTop*rowScale)+'px';card.style.width=(baseWidth*zoom)+'px';card.style.height=(baseHeight*rowScale)+'px';}});}"
        + "function syncLaneVisibility(){LANES.forEach((lane)=>{const hidden=hiddenLanes.has(lane);document.querySelectorAll('[data-stream-lane=\"'+lane+'\"]').forEach((el)=>el.classList.toggle('is-hidden',hidden));document.querySelectorAll('[data-lane-row=\"'+lane+'\"]').forEach((el)=>el.classList.toggle('is-hidden',hidden));document.querySelectorAll('[data-lane-card=\"'+lane+'\"]').forEach((el)=>el.classList.toggle('is-hidden',hidden));document.querySelectorAll('[data-lane-toggle=\"'+lane+'\"]').forEach((el)=>{el.classList.toggle('active',!hidden);el.classList.toggle('inactive',hidden);});});}"
        + "function toggleLane(lane){if(hiddenLanes.has(lane)){hiddenLanes.delete(lane);}else{hiddenLanes.add(lane);}syncLaneVisibility();}"
        + "function resetView(){document.getElementById('time-zoom').value='1';document.getElementById('row-scale').value='1';document.getElementById('density-mode').value='expanded';document.getElementById('show-chronology').checked=true;hiddenLanes.clear();applyLayout();applyDensity();syncChronologyVisibility();syncLaneVisibility();}"
        + "async function exportAnnotations(){if(annotations.size===0)return;const notes=[];annotations.forEach((note,key)=>notes.push({key,note}));const events=EVENTS.map(event=>{const entry={...event};const note=annotations.get(String(event.key));if(note)entry.note=note;return entry;});const data={report:REPORT_NAME,model:MODEL_KEY,notes,events};const jsonText=JSON.stringify(data,null,2);const filename=REPORT_NAME+'.annotations.json';if(location.protocol==='http:'||location.protocol==='https:'){try{const resp=await fetch('/save-annotations',{method:'POST',headers:{'Content-Type':'application/json'},body:jsonText});const result=await resp.json();if(result.ok){lastSavedPath=result.path;navigator.clipboard.writeText(result.path).catch(()=>{});flashSaved('Saved - path copied to clipboard');const doneBtn=document.getElementById('done-btn');if(doneBtn)doneBtn.style.display='inline-block';return;}}catch(e){}}const blob=new Blob([jsonText],{type:'application/json'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download=filename;document.body.appendChild(a);a.click();document.body.removeChild(a);URL.revokeObjectURL(url);flashSaved('Downloaded - save to logs/annotated/');}"
        + "function shutdownServer(){fetch('/shutdown',{method:'POST'}).then(()=>{document.title=document.title+' (server stopped)';flashSaved(lastSavedPath?'Server stopped - export saved to '+lastSavedPath:'Server stopped - you can close this tab',true);}).catch(()=>{});}"
        + "document.getElementById('time-zoom').addEventListener('input',applyLayout);"
        + "document.getElementById('row-scale').addEventListener('input',applyLayout);"
        + "document.getElementById('density-mode').addEventListener('change',applyDensity);"
        + "document.getElementById('show-chronology').addEventListener('change',syncChronologyVisibility);"
        + "renderChronology();"
        + "applyLayout();"
        + "applyDensity();"
        + "syncChronologyVisibility();"
        + "syncLaneVisibility();"
        + "if(EVENTS.length){selectEvent(String(EVENTS[0].key));}"
        + "</script>"
        + "</div></body></html>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-stack automation harness")
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=MODELS.keys())
    parser.add_argument("--character", default="greg")
    parser.add_argument("--scenario-file", default="scenario_script_punchy.toml")
    parser.add_argument("--vision-source", choices=("generated", "live"), default="generated")
    parser.add_argument("--vision-port", type=int, default=0)
    parser.add_argument("--pocket-server-url", default="http://127.0.0.1:8003")
    parser.add_argument("--pocket-voice", default="voices/greg.safetensors")
    args = parser.parse_args()

    model_config = MODELS[args.model]
    label = args.character.replace("_", " ").title()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_json = LOGS_DIR / f"qa_full_stack_{args.character}_{args.model}_{timestamp}.json"
    report_html = LOGS_DIR / f"qa_full_stack_{args.character}_{args.model}_{timestamp}.html"

    console.print(Panel(f"[bold]Full Stack QA ({model_config['name']})[/bold]", border_style="green"))

    pocket_proc = None
    vision_proc = None
    vision_sampler = None
    turns: list[TurnRecord] = []
    import character_eng.__main__ as app_main
    collector = DashboardEventCollector()
    session_recorder = TraceRecorder()
    previous_collector = app_main._collector

    try:
        app_main._collector = collector
        pocket_proc = _ensure_pocket_server(args.pocket_server_url, args.pocket_voice)

        port = args.vision_port or _free_port()
        vision_proc = _launch_vision_service(port, live_camera=args.vision_source == "live")
        vision_url = f"http://127.0.0.1:{port}"
        vision_client = VisionClient(vision_url)
        vision_ready = _wait_for_vision_models(vision_client)
        vision_client.set_questions(
            ["How many people are at Greg's stand?", "What is the nearest person doing?"],
            ["What object is the visitor touching or looking at?"],
        )
        vision_client.set_sam_targets(["person", "flier", "cup"], ["hood", "sign"])
        vision_sampler = VisionSampler(vision_client, session_recorder)
        vision_sampler.start()

        world = load_world_state(args.character)
        goals = load_goals(args.character)
        people = PeopleState()
        scenario_file = args.scenario_file
        if args.vision_source == "live" and scenario_file == "scenario_script_punchy.toml":
            scenario_file = "scenario_script_office_pov.toml"
        scenario = load_scenario_script(args.character, scenario_file)
        system_prompt = load_prompt(args.character, world_state=world, people_state=people)
        session = ChatSession(system_prompt, model_config)
        script = Script()
        log: list[dict] = []
        had_user_input = False

        live_frame_bytes = b""
        live_frame_b64 = ""
        live_snapshot = None
        live_plan = None
        if args.vision_source == "live":
            live_frame_bytes = _capture_live_frame(vision_client)
            live_frame_b64 = base64.b64encode(live_frame_bytes).decode("utf-8")
            live_snapshot = vision_client.snapshot()
            live_plan = _ground_live_evaluation(
                evaluate_image(
                    live_frame_bytes,
                    "image/jpeg",
                    prompt=_live_bootstrap_prompt(live_snapshot),
                )
            )
            _apply_live_world_context(world, live_plan)
            _inject_live_bootstrap_system(session, live_plan)

        stage_goal = scenario.active_stage.goal if scenario and scenario.active_stage else ""
        collector.push("session_start", {
            "character": label,
            "model": args.model,
            "stage": scenario.active_stage.name if scenario and scenario.active_stage else "",
            "goal": stage_goal,
        })
        session_recorder.add("session_start", {
            "character": label,
            "model": args.model,
            "stage": scenario.active_stage.name if scenario and scenario.active_stage else "",
            "goal": stage_goal,
        })
        session_recorder.add("vision_stack_ready", vision_ready)
        plan = run_plan(session, world, goals, "", model_config, people=people, stage_goal=stage_goal)
        if plan and plan.beats:
            apply_plan(script, plan)

        if args.vision_source == "live":
            scene_specs = _build_live_scene_specs(live_plan)
        else:
            scene_specs = SCENE_SPECS
        last_seq = collector.get_all()[-1]["seq"] if collector.get_all() else 0
        for idx, spec in enumerate(scene_specs):
            turn = TurnRecord(scene_name=spec["name"], image_prompt=spec["prompt"])
            turn_trace_start = last_seq
            turn_start_time = time.time()
            synthetic_trace_events: list[dict] = []
            if args.vision_source == "generated":
                console.print(f"[cyan]Generating scene {idx + 1}/{len(scene_specs)}: {spec['name']}[/cyan]")
                generated = generate_image(spec["prompt"])
                turn.image_mime_type = generated.mime_type
                turn.image_b64 = base64.b64encode(generated.image_bytes).decode("utf-8")
                synthetic_trace_events.append(session_recorder.add(
                    "scene_capture",
                    {"label": f"Generated scene {idx + 1}: {spec['name']}"},
                ))
                evaluation = evaluate_image(
                    generated.image_bytes,
                    generated.mime_type,
                    prompt=(
                        "You are preparing a short NPC test scene from this image. Stay literal and conservative. "
                        "Greg must remain a robot head on a folding-table water/advice stand. "
                        "There should be one hooded visitor near the stand. Do not invent coupons, prices, ice cream, "
                        "extra people, toys, coffee, backstory, or internal robot mechanics. "
                        "Summarize only what is visibly present, extract concrete world facts, "
                        "and write one short user line of at most 10 words that a real passerby would say "
                        "about the stand, the flyer, the water, or leaving."
                    ),
                )
                summary, facts, goal, user_line = _ground_evaluation(
                    spec,
                    evaluation.summary,
                    evaluation.world_facts,
                    evaluation.visual_goal,
                    evaluation.user_line,
                )
                evaluation.summary = summary
                evaluation.world_facts = facts
                evaluation.visual_goal = goal
                evaluation.user_line = user_line
                turn.visual_summary = evaluation.summary
                turn.visual_world_facts = evaluation.world_facts
                turn.visual_goal = evaluation.visual_goal
                synthetic_trace_events.append(session_recorder.add(
                    "scene_eval",
                    {"summary": evaluation.summary},
                ))
                vision_client.inject_frame(generated.image_bytes)
            else:
                turn.image_mime_type = "image/jpeg"
                turn.image_b64 = live_frame_b64
                evaluation = live_plan
                turn.visual_summary = live_plan.summary
                turn.visual_world_facts = list(live_plan.world_facts)
                turn.visual_goal = live_plan.visual_goal
                synthetic_trace_events.append(session_recorder.add(
                    "scene_capture",
                    {"label": "Captured live webcam frame for first-person bootstrap"},
                ))
                synthetic_trace_events.append(session_recorder.add(
                    "scene_eval",
                    {"summary": live_plan.summary},
                ))

            time.sleep(2.0)
            snapshot = vision_client.snapshot()
            turn.snapshot_faces = len(snapshot.faces)
            turn.snapshot_persons = len(snapshot.persons)
            turn.snapshot_objects = len(snapshot.objects)
            synthetic_trace_events.append(session_recorder.add(
                "vision_snapshot_read",
                {
                    "faces": turn.snapshot_faces,
                    "persons": turn.snapshot_persons,
                    "objects": turn.snapshot_objects,
                },
            ))

            if idx == 0:
                if args.vision_source != "live":
                    _seed_world_from_visual(world, evaluation)
                synthetic_trace_events.append(session_recorder.add(
                    "world_seed",
                    {"summary": turn.visual_summary},
                ))
                session.inject_system(f"[Initial visible scene: {evaluation.summary}]")
            elif not spec.get("skip_perception"):
                event_text = _scene_event_for_eval(spec, evaluation.summary)
                turn.perception_events = [event_text]
                handle_perception(
                    session=session,
                    world=world,
                    goals=goals,
                    script=script,
                    people=people,
                    scenario=scenario,
                    see_text=event_text,
                    label=label,
                    model_config=model_config,
                    big_model_config=model_config,
                    eval_model_config=model_config,
                    log=log,
                    vision_mgr=None,
                )
                synthetic_trace_events.append(session_recorder.add(
                    "perception_injected",
                    {"summary": event_text},
                ))

            if args.vision_source == "live" and live_plan is not None:
                user_line = live_plan.user_lines[min(idx, len(live_plan.user_lines) - 1)]
            else:
                user_line = evaluation.user_line or spec["fallback_user_line"]
            turn.scripted_user_line = user_line
            user_synth = synthesize_pocket_pcm(user_line, args.pocket_server_url, args.pocket_voice)
            stt_text = transcribe_pcm_deepgram(user_synth.pcm)
            turn.stt_transcript = stt_text
            synthetic_trace_events.append(session_recorder.add(
                "stt_result",
                {"transcript": stt_text},
            ))

            console.print(f"[blue]You (STT):[/blue] {stt_text}")
            synthetic_trace_events.append(session_recorder.add(
                "user_turn_start",
                {"text": stt_text},
            ))
            response, had_user_input = _run_user_turn(
                session=session,
                world=world,
                goals=goals,
                script=script,
                people=people,
                scenario=scenario,
                label=label,
                model_config=model_config,
                user_input=stt_text,
                had_user_input=had_user_input,
                log=log,
                visual_summary=evaluation.summary,
                visual_goal=evaluation.visual_goal,
                visual_world_facts=evaluation.world_facts,
                scene_spec=spec,
                vision_mgr=None,
            )
            turn.assistant_response = response
            turn.response_word_count = len(response.split())
            turn.response_ok, turn.response_note = _response_note(response)

            tts_wall_start = time.time()
            assistant_synth = synthesize_pocket_pcm(response, args.pocket_server_url, args.pocket_voice)
            turn.assistant_audio_ms = assistant_synth.audio_ms
            if assistant_synth.first_audio_ms > 0:
                synthetic_trace_events.append(session_recorder.add(
                    "assistant_tts_first_audio",
                    {"first_audio_ms": assistant_synth.first_audio_ms, "text": response},
                    timestamp=tts_wall_start + (assistant_synth.first_audio_ms / 1000.0),
                ))
            synthetic_trace_events.append(session_recorder.add(
                "assistant_tts_done",
                {"synth_ms": assistant_synth.synth_ms, "text": response},
                timestamp=tts_wall_start + (assistant_synth.synth_ms / 1000.0),
            ))
            synthetic_trace_events.append(session_recorder.add(
                "assistant_audio_clip",
                {"audio_ms": assistant_synth.audio_ms, "text": response},
                timestamp=tts_wall_start + (assistant_synth.synth_ms / 1000.0),
            ))

            _check_reconcile(world, log, people=people)
            all_events = collector.get_all()
            collector_events = [event for event in all_events if event["seq"] > turn_trace_start]
            turn_end_time = time.time()
            background_events = [
                event for event in session_recorder.snapshot()
                if turn_start_time <= event.get("timestamp", 0.0) <= turn_end_time
            ]
            turn.trace_events = sorted(
                background_events + collector_events,
                key=lambda event: (event.get("timestamp", 0.0), event.get("seq", 0)),
            )
            last_seq = all_events[-1]["seq"] if all_events else last_seq
            response_done = next(
                (event for event in reversed(turn.trace_events) if event["type"] == "response_done"),
                None,
            )
            if response_done is not None:
                turn.response_ttft_ms = int(response_done["data"].get("ttft_ms", 0))
                turn.response_total_ms = int(response_done["data"].get("total_ms", 0))
            turns.append(turn)

        combined_session_events = sorted(
            session_recorder.snapshot() + collector.get_all(),
            key=lambda event: (event.get("timestamp", 0.0), event.get("seq", 0)),
        )
        _write_report(turns, report_json, report_html, session_events=combined_session_events)
        failures = [turn for turn in turns if not turn.response_ok]
        console.print(f"[green]Report:[/green] {report_json}")
        console.print(f"[green]HTML:[/green] {report_html}")
        if failures:
            raise SystemExit(f"{len(failures)} turn(s) exceeded punchiness target")
        console.print("[bold green]Full stack QA passed.[/bold green]")
    finally:
        app_main._collector = previous_collector
        collector.shutdown()
        if vision_sampler is not None:
            vision_sampler.stop()
        _stop_proc_group(vision_proc)
        _stop_proc_group(pocket_proc)


if __name__ == "__main__":
    main()
