#!/usr/bin/env python3
"""Patch vLLM's LFM2-VL model to fix missing projector_use_layernorm attribute.

vLLM 0.15.x/0.16.x crashes with LFM2-VL-3B because the model config doesn't
include `projector_use_layernorm`, but vLLM's code accesses it directly.
This script patches the installed vLLM to use getattr with a safe default.

Usage:
    uv run python patch_vllm.py
"""

import glob
import os
import sys

RELATIVE = "vllm/model_executor/models/lfm2_vl.py"

# Search common locations: .venv-vllm, .venv, uv tool installs
search_roots = [
    os.path.join(os.path.dirname(__file__), ".venv-vllm"),
    os.path.join(os.path.dirname(__file__), ".venv"),
    os.path.expanduser("~/.local/share/uv/tools/vllm"),
]

candidates = []
for root in search_roots:
    found = glob.glob(os.path.join(root, "**", RELATIVE), recursive=True)
    candidates.extend(found)

if not candidates:
    print(f"ERROR: Could not find {RELATIVE} in any known venv.")
    print(f"Searched: {search_roots}")
    sys.exit(1)

old = "self.projector_use_layernorm = config.projector_use_layernorm"
new = 'self.projector_use_layernorm = getattr(config, "projector_use_layernorm", True)'

for path in candidates:
    print(f"Found: {path}")
    with open(path) as f:
        source = f.read()

    if new in source:
        print("  Already patched.")
        continue

    if old not in source:
        print("  WARNING: Could not find the line to patch. Version may have changed.")
        continue

    source = source.replace(old, new)
    with open(path, "w") as f:
        f.write(source)
    print("  Patched successfully.")
