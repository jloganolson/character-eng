"""End-to-end test — runs the actual app via --smoke flag.

Auto-skips when the active chat provider key is not set.
"""

import os
import subprocess

import pytest
from dotenv import load_dotenv

from character_eng.models import CHAT_MODEL, MODELS

load_dotenv()

_AI_LEAK_PHRASES = (
    "i'm a large language model",
    "as an ai",
    "as a language model",
    "i'm an ai",
)


@pytest.mark.skipif(
    not os.environ.get(MODELS[CHAT_MODEL].get("api_key_env", "")),
    reason=f"{MODELS[CHAT_MODEL].get('api_key_env', 'chat API key')} not set — skip e2e smoke test",
)
def test_smoke():
    """Launch `uv run -m character_eng --smoke` and verify exit code 0."""
    result = subprocess.run(
        ["uv", "run", "-m", "character_eng", "--smoke"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"Smoke test failed (exit {result.returncode}):\n"
        f"--- stdout ---\n{result.stdout[-2000:]}\n"
        f"--- stderr ---\n{result.stderr[-2000:]}"
    )


@pytest.mark.skipif(
    not os.environ.get(MODELS[CHAT_MODEL].get("api_key_env", "")),
    reason=f"{MODELS[CHAT_MODEL].get('api_key_env', 'chat API key')} not set — skip local vision e2e test",
)
def test_local_vision_mock_dialogue_sim():
    """Launch the local vision path with mock replay + dialogue sim and verify exit code 0."""
    result = subprocess.run(
        [
            "uv", "run", "-m", "character_eng",
            "--vision-mock", "walkup",
            "--sim", "walkup_dialogue",
            "--sim-speed", "50",
            "--character", "greg",
            "--no-dashboard",
        ],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, (
        f"Local vision mock test failed (exit {result.returncode}):\n"
        f"--- stdout ---\n{result.stdout[-4000:]}\n"
        f"--- stderr ---\n{result.stderr[-4000:]}"
    )
    output = f"{result.stdout}\n{result.stderr}".lower()
    assert "vision active" in output
    assert "sim: complete" in output
    assert "traceback" not in output
    assert not any(phrase in output for phrase in _AI_LEAK_PHRASES), (
        f"Local vision mock output leaked AI/meta language:\n"
        f"--- stdout ---\n{result.stdout[-4000:]}\n"
        f"--- stderr ---\n{result.stderr[-4000:]}"
    )
