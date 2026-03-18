from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomllib
from dotenv import load_dotenv


RUNPOD_API_BASE = "https://rest.runpod.io/v1"
DEFAULT_CONFIG_PATH = Path(__file__).with_name("runpod.toml")

load_dotenv()


@dataclass
class RunpodConfig:
    name: str
    gpu_type_ids: list[str]
    gpu_count: int
    gpu_type_priority: str
    data_center_ids: list[str]
    data_center_priority: str
    cloud_type: str
    compute_type: str
    interruptible: bool
    container_disk_in_gb: int
    volume_in_gb: int
    volume_mount_path: str
    support_public_ip: bool
    image_name: str
    container_registry_auth_id: str
    ports: list[str]
    manager_port: int
    session_bridge_port: int
    env: dict[str, str]
    registry_auth_name: str
    registry_auth_username_env: str
    registry_auth_password_env: str
    state_path: Path


def load_config(path: Path) -> RunpodConfig:
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    runpod = payload.get("runpod", {})
    image = payload.get("image", {})
    network = payload.get("network", {})
    env = {str(key): str(value) for key, value in (payload.get("env", {}) or {}).items()}
    registry_auth = payload.get("registry_auth", {})
    files = payload.get("files", {})
    return RunpodConfig(
        name=str(runpod.get("name") or "character-eng-session-manager"),
        gpu_type_ids=[str(item) for item in runpod.get("gpu_type_ids", ["NVIDIA GeForce RTX 4090"])],
        gpu_count=int(runpod.get("gpu_count", 1)),
        gpu_type_priority=str(runpod.get("gpu_type_priority") or "availability"),
        data_center_ids=[str(item) for item in (runpod.get("data_center_ids") or [])],
        data_center_priority=str(runpod.get("data_center_priority") or "availability"),
        cloud_type=str(runpod.get("cloud_type") or "SECURE"),
        compute_type=str(runpod.get("compute_type") or "GPU"),
        interruptible=bool(runpod.get("interruptible", False)),
        container_disk_in_gb=int(runpod.get("container_disk_in_gb", 50)),
        volume_in_gb=int(runpod.get("volume_in_gb", 0)),
        volume_mount_path=str(runpod.get("volume_mount_path") or "/workspace"),
        support_public_ip=bool(runpod.get("support_public_ip", True)),
        image_name=str(image.get("name") or ""),
        container_registry_auth_id=str(
            image.get("container_registry_auth_id") or registry_auth.get("id") or ""
        ).strip(),
        ports=[str(item) for item in network.get("ports", ["7870/http", "7862/http", "22/tcp"])],
        manager_port=int(network.get("manager_port", 7870)),
        session_bridge_port=int(network.get("session_bridge_port", 7862)),
        env=env,
        registry_auth_name=str(registry_auth.get("name") or "").strip(),
        registry_auth_username_env=str(registry_auth.get("username_env") or "").strip(),
        registry_auth_password_env=str(registry_auth.get("password_env") or "").strip(),
        state_path=(path.parent / str(files.get("state_path") or "deploy/.runpod-state.json")).resolve()
        if not Path(str(files.get("state_path") or "deploy/.runpod-state.json")).is_absolute()
        else Path(str(files.get("state_path"))),
    )


class RunpodClient:
    def __init__(self, api_key: str, api_base: str = RUNPOD_API_BASE):
        self._api_key = api_key.strip()
        self._api_base = api_base.rstrip("/")
        if not self._api_key:
            raise ValueError("RUNPOD_API_KEY is required")

    def _request(self, method: str, path: str, body: dict | None = None) -> Any:
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(
            f"{self._api_base}{path}",
            method=method,
            data=data,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Runpod API {method} {path} failed: {exc.code} {detail}") from exc
        if not raw:
            return None
        return json.loads(raw)

    def create_pod(self, payload: dict) -> dict:
        return self._request("POST", "/pods", payload)

    def list_pods(self) -> list[dict]:
        return self._request("GET", "/pods")

    def list_container_registry_auths(self) -> list[dict]:
        return self._request("GET", "/containerregistryauth") or []

    def create_container_registry_auth(self, name: str, username: str, password: str) -> dict:
        return self._request(
            "POST",
            "/containerregistryauth",
            {"name": name, "username": username, "password": password},
        )

    def get_pod(self, pod_id: str) -> dict:
        return self._request("GET", f"/pods/{pod_id}")

    def stop_pod(self, pod_id: str) -> dict | None:
        return self._request("POST", f"/pods/{pod_id}/stop", {})

    def start_pod(self, pod_id: str) -> dict | None:
        return self._request("POST", f"/pods/{pod_id}/resume", {})

    def delete_pod(self, pod_id: str) -> dict | None:
        return self._request("DELETE", f"/pods/{pod_id}")


def build_create_payload(cfg: RunpodConfig, container_registry_auth_id: str = "") -> dict:
    if not cfg.image_name:
        raise ValueError("image.name is required in deploy/runpod.toml")
    env = resolve_config_env(cfg.env)
    if cfg.support_public_ip and not env.get("PUBLIC_HOST"):
        env["PUBLIC_HOST"] = "__RUNPOD_PUBLIC_IP__"
    payload = {
        "name": cfg.name,
        "imageName": cfg.image_name,
        "gpuTypeIds": cfg.gpu_type_ids,
        "gpuCount": cfg.gpu_count,
        "gpuTypePriority": cfg.gpu_type_priority,
        "dataCenterIds": cfg.data_center_ids,
        "dataCenterPriority": cfg.data_center_priority,
        "cloudType": cfg.cloud_type,
        "computeType": cfg.compute_type,
        "interruptible": cfg.interruptible,
        "containerDiskInGb": cfg.container_disk_in_gb,
        "volumeInGb": cfg.volume_in_gb,
        "volumeMountPath": cfg.volume_mount_path,
        "supportPublicIp": cfg.support_public_ip,
        "ports": cfg.ports,
        "env": env,
    }
    auth_id = container_registry_auth_id.strip()
    if auth_id:
        payload["containerRegistryAuthId"] = auth_id
    if len(cfg.gpu_type_ids) <= 1 and cfg.gpu_type_priority == "availability":
        payload.pop("gpuTypePriority", None)
    if not cfg.data_center_ids:
        payload.pop("dataCenterIds", None)
    if not cfg.data_center_ids and cfg.data_center_priority == "availability":
        payload.pop("dataCenterPriority", None)
    if cfg.volume_in_gb <= 0:
        payload.pop("volumeInGb", None)
        payload.pop("volumeMountPath", None)
    return payload


def resolve_config_env(env: dict[str, str]) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for key, value in env.items():
        if value.startswith("${") and value.endswith("}") and len(value) > 3:
            env_name = value[2:-1].strip()
            env_value = os.environ.get(env_name, "").strip()
            if not env_value:
                raise RuntimeError(f"missing env var for deploy env {key}: {env_name}")
            resolved[key] = env_value
        else:
            resolved[key] = value
    return resolved


def resolve_container_registry_auth_id(cfg: RunpodConfig, client: RunpodClient) -> str:
    if cfg.container_registry_auth_id:
        return cfg.container_registry_auth_id
    if not any(
        [
            cfg.registry_auth_name,
            cfg.registry_auth_username_env,
            cfg.registry_auth_password_env,
        ]
    ):
        return ""
    missing_config = [
        field_name
        for field_name, value in (
            ("registry_auth.name", cfg.registry_auth_name),
            ("registry_auth.username_env", cfg.registry_auth_username_env),
            ("registry_auth.password_env", cfg.registry_auth_password_env),
        )
        if not value
    ]
    if missing_config:
        joined = ", ".join(missing_config)
        raise RuntimeError(f"incomplete registry auth config: missing {joined}")
    for auth in client.list_container_registry_auths():
        if str(auth.get("name") or "").strip() == cfg.registry_auth_name:
            return str(auth.get("id") or "").strip()
    username = os.environ.get(cfg.registry_auth_username_env, "").strip()
    password = os.environ.get(cfg.registry_auth_password_env, "").strip()
    missing_env = [
        env_name
        for env_name, value in (
            (cfg.registry_auth_username_env, username),
            (cfg.registry_auth_password_env, password),
        )
        if not value
    ]
    if missing_env:
        joined = ", ".join(missing_env)
        raise RuntimeError(f"missing registry auth env vars: {joined}")
    created = client.create_container_registry_auth(cfg.registry_auth_name, username, password)
    return str(created.get("id") or "").strip()


def read_state(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_state(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def clear_state(path: Path) -> None:
    if path.exists():
        path.unlink()


def resolve_pod_id(cfg: RunpodConfig, explicit: str | None) -> str:
    if explicit:
        return explicit
    state = read_state(cfg.state_path)
    pod_id = str(state.get("pod_id") or "").strip()
    if not pod_id:
        raise RuntimeError("No pod id found. Pass --pod-id or create a pod first.")
    return pod_id


def format_http_service_url(pod: dict, port: int) -> str:
    pod_id = str(pod.get("id") or "").strip()
    ports = [str(item) for item in (pod.get("ports") or [])]
    if not pod_id:
        return ""
    if any(item == f"{port}/http" for item in ports):
        return f"https://{pod_id}-{port}.proxy.runpod.net"
    return ""


def format_manager_url(pod: dict, manager_port: int) -> str:
    proxy_url = format_http_service_url(pod, manager_port)
    if proxy_url:
        return proxy_url
    public_ip = str(pod.get("publicIp") or "").strip()
    port_mappings = pod.get("portMappings") or {}
    mapped = port_mappings.get(str(manager_port))
    if public_ip and mapped:
        return f"http://{public_ip}:{mapped}"
    return ""


def format_ssh_command(pod: dict) -> str:
    public_ip = str(pod.get("publicIp") or "").strip()
    port_mappings = pod.get("portMappings") or {}
    ssh_port = port_mappings.get("22")
    if public_ip and ssh_port:
        return f"ssh root@{public_ip} -p {ssh_port}"
    return ""


def cmd_up(args) -> int:
    cfg = load_config(Path(args.config))
    client = RunpodClient(os.environ.get("RUNPOD_API_KEY", ""))
    payload = build_create_payload(cfg, resolve_container_registry_auth_id(cfg, client))
    pod = client.create_pod(payload)
    write_state(cfg.state_path, {"pod_id": pod["id"], "created_at": pod.get("lastStartedAt"), "name": cfg.name})
    print(f"created pod {pod['id']}")
    url = format_manager_url(pod, cfg.manager_port)
    if url:
        print(f"manager url: {url}")
    else:
        print("manager url: pending public port mapping")
    ssh_cmd = format_ssh_command(pod)
    if ssh_cmd:
        print(f"ssh: {ssh_cmd}")
    return 0


def cmd_status(args) -> int:
    cfg = load_config(Path(args.config))
    client = RunpodClient(os.environ.get("RUNPOD_API_KEY", ""))
    pod_id = resolve_pod_id(cfg, args.pod_id)
    pod = client.get_pod(pod_id)
    print(json.dumps({
        "id": pod.get("id"),
        "name": pod.get("name"),
        "desiredStatus": pod.get("desiredStatus"),
        "costPerHr": pod.get("costPerHr"),
        "publicIp": pod.get("publicIp"),
        "portMappings": pod.get("portMappings"),
        "managerUrl": format_manager_url(pod, cfg.manager_port),
        "sshCommand": format_ssh_command(pod),
    }, indent=2))
    return 0


def cmd_down(args) -> int:
    cfg = load_config(Path(args.config))
    client = RunpodClient(os.environ.get("RUNPOD_API_KEY", ""))
    pod_id = resolve_pod_id(cfg, args.pod_id)
    client.stop_pod(pod_id)
    print(f"stop requested for pod {pod_id}")
    return 0


def cmd_start(args) -> int:
    cfg = load_config(Path(args.config))
    client = RunpodClient(os.environ.get("RUNPOD_API_KEY", ""))
    pod_id = resolve_pod_id(cfg, args.pod_id)
    client.start_pod(pod_id)
    print(f"start requested for pod {pod_id}")
    return 0


def cmd_destroy(args) -> int:
    cfg = load_config(Path(args.config))
    client = RunpodClient(os.environ.get("RUNPOD_API_KEY", ""))
    pod_id = resolve_pod_id(cfg, args.pod_id)
    client.delete_pod(pod_id)
    clear_state(cfg.state_path)
    print(f"deleted pod {pod_id}")
    return 0


def cmd_ssh(args) -> int:
    cfg = load_config(Path(args.config))
    client = RunpodClient(os.environ.get("RUNPOD_API_KEY", ""))
    pod_id = resolve_pod_id(cfg, args.pod_id)
    pod = client.get_pod(pod_id)
    ssh_cmd = format_ssh_command(pod)
    if not ssh_cmd:
        raise RuntimeError("pod does not have a public IP / SSH port mapping yet")
    print(ssh_cmd)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage a Runpod deployment for character-eng.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to deploy/runpod.toml")
    sub = parser.add_subparsers(dest="command", required=True)

    for name, fn in {
        "up": cmd_up,
        "status": cmd_status,
        "down": cmd_down,
        "start": cmd_start,
        "destroy": cmd_destroy,
        "ssh": cmd_ssh,
    }.items():
        subparser = sub.add_parser(name)
        if name != "up":
            subparser.add_argument("--pod-id", default="", help="Override pod id instead of deploy state")
        subparser.set_defaults(func=fn)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
