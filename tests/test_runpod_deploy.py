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
gpu_type_ids = ["NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 5090"]
gpu_count = 1
gpu_type_priority = "custom"
data_center_ids = ["US-CA-2", "US-WA-1"]
data_center_priority = "custom"
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
    assert payload["gpuTypeIds"] == ["NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 5090"]
    assert payload["gpuTypePriority"] == "custom"
    assert payload["dataCenterIds"] == ["US-CA-2", "US-WA-1"]
    assert payload["dataCenterPriority"] == "custom"
    assert payload["ports"] == ["7870/http", "22/tcp"]
    assert payload["env"]["MODE"] == "full"
    assert payload["env"]["PUBLIC_HOST"] == "__RUNPOD_PUBLIC_IP__"
    assert "volumeInGb" not in payload


def test_build_create_payload_includes_container_registry_auth_id(tmp_path):
    config_path = tmp_path / "runpod.toml"
    config_path.write_text(
        """
[runpod]
name = "char-test"

[image]
name = "ghcr.io/example/character-eng:latest"
container_registry_auth_id = "auth_123"
    """,
        encoding="utf-8",
    )
    cfg = runpod.load_config(config_path)
    class Client:
        def list_container_registry_auths(self):
            return []

    payload = runpod.build_create_payload(cfg, runpod.resolve_container_registry_auth_id(cfg, Client()))
    assert payload["containerRegistryAuthId"] == "auth_123"


def test_resolve_container_registry_auth_id_creates_or_reuses_auth(tmp_path, monkeypatch):
    config_path = tmp_path / "runpod.toml"
    config_path.write_text(
        """
[runpod]
name = "char-test"

[image]
name = "ghcr.io/example/character-eng:latest"

[registry_auth]
name = "ghcr-character-eng"
username_env = "GHCR_USERNAME"
password_env = "GHCR_PASSWORD"
""",
        encoding="utf-8",
    )
    cfg = runpod.load_config(config_path)
    monkeypatch.setenv("GHCR_USERNAME", "example-user")
    monkeypatch.setenv("GHCR_PASSWORD", "example-token")

    class ReusedClient:
        def list_container_registry_auths(self):
            return [{"id": "existing_1", "name": "ghcr-character-eng"}]

    assert runpod.resolve_container_registry_auth_id(cfg, ReusedClient()) == "existing_1"

    created: dict[str, str] = {}

    class CreatingClient:
        def list_container_registry_auths(self):
            return []

        def create_container_registry_auth(self, name: str, username: str, password: str):
            created.update({"name": name, "username": username, "password": password})
            return {"id": "created_1", "name": name}

    assert runpod.resolve_container_registry_auth_id(cfg, CreatingClient()) == "created_1"
    assert created == {
        "name": "ghcr-character-eng",
        "username": "example-user",
        "password": "example-token",
    }


def test_resolve_container_registry_auth_id_errors_on_missing_env(tmp_path):
    config_path = tmp_path / "runpod.toml"
    config_path.write_text(
        """
[runpod]
name = "char-test"

[image]
name = "ghcr.io/example/character-eng:latest"

[registry_auth]
name = "ghcr-character-eng"
username_env = "GHCR_USERNAME"
password_env = "GHCR_PASSWORD"
""",
        encoding="utf-8",
    )
    cfg = runpod.load_config(config_path)

    class Client:
        def list_container_registry_auths(self):
            return []

    try:
        runpod.resolve_container_registry_auth_id(cfg, Client())
    except RuntimeError as exc:
        assert "missing registry auth env vars" in str(exc)
    else:
        raise AssertionError("expected missing-env error")


def test_resolve_container_registry_auth_id_reuses_existing_auth_without_env(tmp_path, monkeypatch):
    config_path = tmp_path / "runpod.toml"
    config_path.write_text(
        """
[runpod]
name = "char-test"

[image]
name = "ghcr.io/example/character-eng:latest"

[registry_auth]
name = "ghcr-character-eng"
username_env = "GHCR_USERNAME"
password_env = "GHCR_PASSWORD"
""",
        encoding="utf-8",
    )
    cfg = runpod.load_config(config_path)
    monkeypatch.delenv("GHCR_USERNAME", raising=False)
    monkeypatch.delenv("GHCR_PASSWORD", raising=False)

    class Client:
        def list_container_registry_auths(self):
            return [{"id": "existing_1", "name": "ghcr-character-eng"}]

    assert runpod.resolve_container_registry_auth_id(cfg, Client()) == "existing_1"


def test_runpod_state_round_trip_and_formatters(tmp_path):
    state_path = tmp_path / "state.json"
    runpod.write_state(state_path, {"pod_id": "abc123"})
    assert runpod.read_state(state_path)["pod_id"] == "abc123"
    pod = {
        "id": "abc123",
        "publicIp": "1.2.3.4",
        "ports": ["7870/http", "22/tcp"],
        "portMappings": {"7870": 30001, "22": 30022},
    }
    assert runpod.format_manager_url(pod, 7870) == "https://abc123-7870.proxy.runpod.net"
    assert runpod.format_ssh_command(pod) == "ssh root@1.2.3.4 -p 30022"
    runpod.clear_state(state_path)
    assert not state_path.exists()


def test_build_create_payload_omits_data_center_fields_without_preferences(tmp_path):
    config_path = tmp_path / "runpod.toml"
    config_path.write_text(
        """
[runpod]
name = "char-test"

[image]
name = "ghcr.io/example/character-eng:latest"
""",
        encoding="utf-8",
    )
    cfg = runpod.load_config(config_path)
    payload = runpod.build_create_payload(cfg)
    assert "dataCenterIds" not in payload
    assert "dataCenterPriority" not in payload
    assert "gpuTypePriority" not in payload
