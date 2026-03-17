from __future__ import annotations

import json
from pathlib import Path

from deploy import runpod


def test_build_create_payload_sets_expected_defaults(tmp_path):
    config_path = tmp_path / "runpod.toml"
    config_path.write_text(
        """
[runpod]
name = "char-test"
gpu_type_ids = ["NVIDIA GeForce RTX 4090"]
gpu_count = 1
cloud_type = "SECURE"
compute_type = "GPU"
interruptible = false
container_disk_in_gb = 50
volume_in_gb = 0
volume_mount_path = "/workspace"
support_public_ip = true

[image]
name = "ghcr.io/example/character-eng:latest"

[network]
ports = ["7870/http", "22/tcp"]
manager_port = 7870
session_bridge_port = 7862

[env]
MODE = "full"
MANAGER_PORT = "7870"

[files]
state_path = "state.json"
""",
        encoding="utf-8",
    )
    cfg = runpod.load_config(config_path)
    payload = runpod.build_create_payload(cfg)
    assert payload["name"] == "char-test"
    assert payload["imageName"] == "ghcr.io/example/character-eng:latest"
    assert payload["gpuTypeIds"] == ["NVIDIA GeForce RTX 4090"]
    assert payload["ports"] == ["7870/http", "22/tcp"]
    assert payload["env"]["MODE"] == "full"
    assert payload["env"]["PUBLIC_HOST"] == "__RUNPOD_PUBLIC_IP__"
    assert "volumeInGb" not in payload


def test_runpod_state_round_trip_and_formatters(tmp_path):
    state_path = tmp_path / "state.json"
    runpod.write_state(state_path, {"pod_id": "abc123"})
    assert runpod.read_state(state_path)["pod_id"] == "abc123"
    pod = {"publicIp": "1.2.3.4", "portMappings": {"7870": 30001, "22": 30022}}
    assert runpod.format_manager_url(pod, 7870) == "http://1.2.3.4:30001"
    assert runpod.format_ssh_command(pod) == "ssh root@1.2.3.4 -p 30022"
    runpod.clear_state(state_path)
    assert not state_path.exists()
