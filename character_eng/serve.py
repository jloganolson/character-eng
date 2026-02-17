"""Launch vLLM server for a local model.

Usage:
    uv run python -m character_eng.serve [model-key]

If no model key is given, shows a menu of local models.
Kills any existing process on port 8000, then exec's into vllm serve.
"""

import os
import subprocess
import sys

from character_eng.models import MODELS

PORT = "8000"


def get_local_models() -> list[tuple[str, dict]]:
    return [(k, cfg) for k, cfg in MODELS.items() if cfg.get("local")]


def pick_local_model(local_models: list[tuple[str, dict]]) -> tuple[str, dict] | None:
    print("\nLocal models:")
    for i, (key, cfg) in enumerate(local_models, 1):
        print(f"  {i}. {cfg['name']}  ({key})")
    print(f"  q. Quit")
    print()

    while True:
        choice = input("Pick a model: ").strip()
        if choice.lower() == "q":
            return None
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(local_models):
                return local_models[idx]
        except ValueError:
            pass
        print("Invalid choice, try again.")


def kill_port(port: str):
    """Kill any process listening on the given port."""
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True, text=True,
        )
        pids = result.stdout.strip()
        if pids:
            print(f"Killing existing process(es) on port {port}: {pids}")
            subprocess.run(["kill", "-9"] + pids.split())
    except FileNotFoundError:
        pass  # lsof not available, skip


def build_vllm_cmd(key: str, cfg: dict, port: str = PORT) -> list[str]:
    """Build the vllm serve command list for a local model."""
    return [
        "vllm", "serve", cfg["path"],
        "--served-model-name", key,
        "--port", port,
        "--trust-remote-code",
        "--gpu-memory-utilization", "0.85",
    ]


def main():
    local_models = get_local_models()
    if not local_models:
        print("No local models configured in models.py")
        sys.exit(1)

    if len(sys.argv) > 1:
        key = sys.argv[1]
        match = [(k, cfg) for k, cfg in local_models if k == key]
        if not match:
            print(f"Unknown local model: {key}")
            print(f"Available: {', '.join(k for k, _ in local_models)}")
            sys.exit(1)
        key, cfg = match[0]
    else:
        result = pick_local_model(local_models)
        if result is None:
            return
        key, cfg = result

    path = cfg["path"]
    if not os.path.exists(path):
        print(f"Model path not found: {path}")
        sys.exit(1)

    kill_port(PORT)

    cmd = build_vllm_cmd(key, cfg)
    print(f"\nStarting: {' '.join(cmd)}\n")
    os.execvp("vllm", cmd)


if __name__ == "__main__":
    main()
