"""End-to-end test — runs the actual app via --smoke flag.

Auto-skips when CEREBRAS_API_KEY is not set (e.g. CI without secrets).
"""

import os
import subprocess

import pytest


@pytest.mark.skipif(
    not os.environ.get("CEREBRAS_API_KEY"),
    reason="CEREBRAS_API_KEY not set — skip e2e smoke test",
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
