"""Minimal Gemini REST helpers for generated imagery and scene evaluation."""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass, field
from typing import Any
from urllib import error, parse, request

from dotenv import load_dotenv

load_dotenv()

API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_IMAGE_MODEL = "gemini-3-pro-image-preview"
DEFAULT_ANALYSIS_MODEL = "gemini-3-pro-preview"


@dataclass
class GeneratedImage:
    prompt: str
    image_bytes: bytes
    mime_type: str
    text: str = ""


@dataclass
class SceneEvaluation:
    summary: str
    world_facts: list[str] = field(default_factory=list)
    user_line: str = ""
    followup_line: str = ""
    sentiment: str = "neutral"
    visual_goal: str = ""
    raw_text: str = ""


def _api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set")
    return key


def _post(model: str, payload: dict[str, Any]) -> dict[str, Any]:
    url = f"{API_ROOT}/{model}:generateContent?key={parse.quote(_api_key())}"
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini API error for {model}: {exc.code} {details}") from exc


def _extract_text_and_image(data: dict[str, Any]) -> tuple[str, bytes, str]:
    text_parts: list[str] = []
    image_bytes = b""
    mime_type = ""
    for candidate in data.get("candidates", []):
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            if "text" in part:
                text_parts.append(part["text"])
            inline = part.get("inlineData") or part.get("inline_data")
            if inline and inline.get("data"):
                image_bytes = base64.b64decode(inline["data"])
                mime_type = inline.get("mimeType") or inline.get("mime_type") or "image/png"
                return ("\n".join(text_parts).strip(), image_bytes, mime_type)
    raise RuntimeError("Gemini response did not include inline image data")


def _extract_text(data: dict[str, Any]) -> str:
    parts: list[str] = []
    for candidate in data.get("candidates", []):
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            if "text" in part:
                parts.append(part["text"])
    return "\n".join(parts).strip()


def generate_image(
    prompt: str,
    model: str = DEFAULT_IMAGE_MODEL,
    guidance: str = "",
) -> GeneratedImage:
    """Generate an image from a text prompt using Gemini image preview."""
    payload = {
        "contents": [{
            "role": "user",
            "parts": [{"text": "\n\n".join(p for p in [prompt, guidance] if p)}],
        }],
        "generationConfig": {
            "responseModalities": ["TEXT", "IMAGE"],
        },
    }
    data = _post(model, payload)
    text, image_bytes, mime_type = _extract_text_and_image(data)
    return GeneratedImage(prompt=prompt, image_bytes=image_bytes, mime_type=mime_type, text=text)


def evaluate_image(
    image_bytes: bytes,
    mime_type: str,
    prompt: str,
    model: str = DEFAULT_ANALYSIS_MODEL,
) -> SceneEvaluation:
    """Evaluate an image and return structured scene data."""
    schema_prompt = (
        f"{prompt}\n\n"
        "Return strict JSON with keys: "
        'summary (string), world_facts (array of <=4 short strings), '
        'user_line (string), followup_line (string), sentiment '
        '("curious"|"warm"|"skeptical"|"urgent"|"neutral"), visual_goal (string).'
    )
    payload = {
        "contents": [{
            "role": "user",
            "parts": [
                {"text": schema_prompt},
                {"inlineData": {"mimeType": mime_type, "data": base64.b64encode(image_bytes).decode("utf-8")}},
            ],
        }],
        "generationConfig": {
            "temperature": 0.3,
            "responseMimeType": "application/json",
        },
    }
    data = _post(model, payload)
    raw = _extract_text(data)
    parsed = json.loads(raw)
    return SceneEvaluation(
        summary=parsed.get("summary", ""),
        world_facts=[fact.strip() for fact in parsed.get("world_facts", []) if isinstance(fact, str) and fact.strip()],
        user_line=str(parsed.get("user_line", "")).strip(),
        followup_line=str(parsed.get("followup_line", "")).strip(),
        sentiment=str(parsed.get("sentiment", "neutral")).strip() or "neutral",
        visual_goal=str(parsed.get("visual_goal", "")).strip(),
        raw_text=raw,
    )
