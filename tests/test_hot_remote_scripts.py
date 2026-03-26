from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _make_executable(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)
    return path


def _read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line]


def _fake_remote_env(
    env_file: Path,
    *,
    api_secret: str = "lk-secret-secret-secret-secret",
    manager_token: str | None = None,
) -> None:
    lines = [
        "GCP_PROJECT_ID=test-project",
        "GCP_ZONE=us-west1-a",
        "GCP_INSTANCE_NAME=test-vm",
        "LIVEKIT_ENABLED=1",
        "LIVEKIT_API_KEY=lk-key",
        f"LIVEKIT_API_SECRET={api_secret}",
    ]
    if manager_token:
        lines.append(f"CHARACTER_ENG_MANAGER_TOKEN={manager_token}")
    env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fake_create_vm_env(env_file: Path, local_env: Path) -> None:
    env_file.write_text(
        "\n".join(
            [
                "set -a",
                f"source {local_env}",
                "set +a",
                "GCP_PROJECT_ID=test-project",
                "GCP_REGION=us-west1",
                "GCP_ZONE=us-west1-a",
                "GCP_INSTANCE_NAME=test-vm",
                "GCP_MACHINE_TYPE=g2-standard-8",
                "GCP_ACCELERATOR_TYPE=nvidia-l4",
                "GCP_ACCELERATOR_COUNT=1",
                "GCP_BOOT_DISK_SIZE_GB=100",
                "GCP_RUNTIME_DISK_NAME=test-runtime-disk",
                "GCP_RUNTIME_DISK_SIZE_GB=100",
                "GCP_RUNTIME_MOUNT_PATH=/mnt/runtime",
                "GCP_SOURCE_IMAGE_PROJECT=ml-images",
                "GCP_SOURCE_IMAGE_FAMILY=common-cu128-ubuntu-2204-nvidia-570",
                "GCP_NETWORK=default",
                "GCP_TAGS=character-eng-gpu",
                "CONTAINER_IMAGE=us-west1-docker.pkg.dev/test/character-eng/image:latest",
                "CONTAINER_NAME=character-eng",
                "CONTAINER_MODE=full",
                "MANAGER_PORT=7870",
                "WORKSPACE_MOUNT_PATH=/workspace",
                "CHARACTER=greg",
                "START_PAUSED=1",
                "CHARACTER_ENG_SHARED_VISION_URL=http://127.0.0.1:7860",
                "VLLM_GPU_UTIL=0.75",
                "LIVEKIT_ENABLED=0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_run_hot_remote_webrtc_hands_off_expected_env(tmp_path):
    env_file = tmp_path / "gcp.env"
    _fake_remote_env(env_file, manager_token="manager-token")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    curl_log = tmp_path / "curl.log"
    gcloud_log = tmp_path / "gcloud.log"
    ssh_log = tmp_path / "ssh.log"
    vm_ctl_log = tmp_path / "vm_ctl.log"
    run_local_log = tmp_path / "run_local.log"

    _make_executable(
        bin_dir / "curl",
        f"""#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "{curl_log}"
exit 0
""",
    )
    _make_executable(
        bin_dir / "gcloud",
        f"""#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "{gcloud_log}"
if [[ "$*" == *"--dry-run"* ]]; then
  printf 'ssh -N -L 127.0.0.1:17860:127.0.0.1:7860 -L 127.0.0.1:18003:127.0.0.1:8003\\n'
  exit 0
fi
if [[ "$*" == *"--command"* ]]; then
  exit 0
fi
echo "unexpected gcloud call: $*" >&2
exit 1
""",
    )
    _make_executable(
        bin_dir / "ssh",
        f"""#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "{ssh_log}"
exec /bin/sleep 300
""",
    )
    _make_executable(
        bin_dir / "setsid",
        """#!/usr/bin/env bash
set -euo pipefail
exec "$@"
""",
    )

    vm_ctl = _make_executable(
        tmp_path / "vm_ctl_stub.sh",
        f"""#!/usr/bin/env bash
set -euo pipefail
env_file="${{1:-}}"
cmd="${{2:-}}"
printf '%s\\n' "$cmd" >> "{vm_ctl_log}"
case "$cmd" in
  status)
    printf 'TERMINATED\\n'
    ;;
  start|wait|wait-livekit|stop)
    ;;
  livekit-url)
    printf 'wss://remote.example.test/rtc\\n'
    ;;
  *)
    echo "unexpected command: $cmd" >&2
    exit 1
    ;;
esac
""",
    )
    run_local = _make_executable(
        tmp_path / "run_local_stub.sh",
        f"""#!/usr/bin/env bash
set -euo pipefail
cat > "{run_local_log}" <<EOF
CLEAN_START=${{CLEAN_START:-}}
CHARACTER_ENG_TRANSPORT_MODE=${{CHARACTER_ENG_TRANSPORT_MODE:-}}
CHARACTER_ENG_TRANSPORT_VIDEO_METRICS_PATH=${{CHARACTER_ENG_TRANSPORT_VIDEO_METRICS_PATH:-}}
CHARACTER_ENG_TRANSPORT_AUDIO_METRICS_PATH=${{CHARACTER_ENG_TRANSPORT_AUDIO_METRICS_PATH:-}}
CHARACTER_ENG_VISION_URL=${{CHARACTER_ENG_VISION_URL:-}}
CHARACTER_ENG_VISION_AUTO_LAUNCH=${{CHARACTER_ENG_VISION_AUTO_LAUNCH:-}}
CHARACTER_ENG_TTS_BACKEND=${{CHARACTER_ENG_TTS_BACKEND:-}}
CHARACTER_ENG_TTS_SERVER_URL=${{CHARACTER_ENG_TTS_SERVER_URL:-}}
CHARACTER_ENG_LIVEKIT_ENABLED=${{CHARACTER_ENG_LIVEKIT_ENABLED:-}}
CHARACTER_ENG_LIVEKIT_URL=${{CHARACTER_ENG_LIVEKIT_URL:-}}
CHARACTER_ENG_LIVEKIT_API_KEY=${{CHARACTER_ENG_LIVEKIT_API_KEY:-}}
CHARACTER_ENG_LIVEKIT_API_SECRET=${{CHARACTER_ENG_LIVEKIT_API_SECRET:-}}
ARGS=$*
EOF
""",
    )

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "CHARACTER_ENG_GCP_ENV": str(env_file),
            "VM_CTL_BIN": str(vm_ctl),
            "RUN_LOCAL_SCRIPT": str(run_local),
            "TUNNEL_DIR": str(tmp_path / "tunnel"),
        }
    )

    result = subprocess.run(
        ["bash", "scripts/run_hot_remote_webrtc.sh", "--character", "greg"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert result.returncode == 0, result.stderr

    run_local_payload = "\n".join(_read_lines(run_local_log))
    assert "CLEAN_START=1" in run_local_payload
    assert "CHARACTER_ENG_TRANSPORT_MODE=remote_hot_webrtc" in run_local_payload
    assert "CHARACTER_ENG_VISION_URL=http://127.0.0.1:17860" in run_local_payload
    assert "CHARACTER_ENG_VISION_AUTO_LAUNCH=false" in run_local_payload
    assert "CHARACTER_ENG_TTS_BACKEND=pocket" in run_local_payload
    assert "CHARACTER_ENG_TTS_SERVER_URL=http://127.0.0.1:18003" in run_local_payload
    assert "CHARACTER_ENG_LIVEKIT_ENABLED=true" in run_local_payload
    assert "CHARACTER_ENG_LIVEKIT_URL=wss://remote.example.test/rtc" in run_local_payload
    assert "CHARACTER_ENG_LIVEKIT_API_KEY=lk-key" in run_local_payload
    assert "CHARACTER_ENG_LIVEKIT_API_SECRET=lk-secret-secret-secret-secret" in run_local_payload
    assert "ARGS=--character greg" in run_local_payload

    curl_calls = _read_lines(curl_log)
    assert any("http://127.0.0.1:17860/model_status" in line for line in curl_calls)
    assert any("http://127.0.0.1:18003/" in line for line in curl_calls)
    assert any("https://remote.example.test/rtc/" in line for line in curl_calls)

    vm_ctl_calls = _read_lines(vm_ctl_log)
    assert vm_ctl_calls[:4] == ["status", "start", "wait", "wait-livekit"]
    assert vm_ctl_calls.count("livekit-url") == 2

    gcloud_calls = _read_lines(gcloud_log)
    assert any("--command curl -fsS 'http://127.0.0.1:7860/model_status' >/dev/null" in line for line in gcloud_calls)
    assert any("--dry-run" in line for line in gcloud_calls)

    ssh_calls = _read_lines(ssh_log)
    assert any(" -N " in f" {line} " for line in ssh_calls)
    assert any("127.0.0.1:17860:127.0.0.1:7860" in line for line in ssh_calls)
    assert any("127.0.0.1:18003:127.0.0.1:8003" in line for line in ssh_calls)


def test_run_hot_remote_webrtc_skips_manager_wait_without_token(tmp_path):
    env_file = tmp_path / "gcp.env"
    _fake_remote_env(env_file, manager_token=None)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gcloud_log = tmp_path / "gcloud.log"
    ssh_log = tmp_path / "ssh.log"
    vm_ctl_log = tmp_path / "vm_ctl.log"
    run_local_log = tmp_path / "run_local.log"

    _make_executable(
        bin_dir / "curl",
        """#!/usr/bin/env bash
set -euo pipefail
exit 0
""",
    )
    _make_executable(
        bin_dir / "gcloud",
        f"""#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "{gcloud_log}"
if [[ "$*" == *"--dry-run"* ]]; then
  printf 'ssh -N -L 127.0.0.1:17860:127.0.0.1:7860 -L 127.0.0.1:18003:127.0.0.1:8003\\n'
  exit 0
fi
if [[ "$*" == *"--command"* ]]; then
  exit 0
fi
echo "unexpected gcloud call: $*" >&2
exit 1
""",
    )
    _make_executable(
        bin_dir / "ssh",
        f"""#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "{ssh_log}"
exec /bin/sleep 300
""",
    )
    _make_executable(
        bin_dir / "setsid",
        """#!/usr/bin/env bash
set -euo pipefail
exec "$@"
""",
    )
    vm_ctl = _make_executable(
        tmp_path / "vm_ctl_stub.sh",
        f"""#!/usr/bin/env bash
set -euo pipefail
cmd="${{2:-}}"
printf '%s\\n' "$cmd" >> "{vm_ctl_log}"
case "$cmd" in
  status)
    printf 'RUNNING\\n'
    ;;
  wait|wait-livekit|stop)
    ;;
  livekit-url)
    printf 'wss://remote.example.test/rtc\\n'
    ;;
  *)
    echo "unexpected command: $cmd" >&2
    exit 1
    ;;
esac
""",
    )
    run_local = _make_executable(
        tmp_path / "run_local_stub.sh",
        f"""#!/usr/bin/env bash
set -euo pipefail
printf 'ok\\n' > "{run_local_log}"
""",
    )

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "CHARACTER_ENG_GCP_ENV": str(env_file),
            "VM_CTL_BIN": str(vm_ctl),
            "RUN_LOCAL_SCRIPT": str(run_local),
            "TUNNEL_DIR": str(tmp_path / "tunnel"),
        }
    )

    result = subprocess.run(
        ["bash", "scripts/run_hot_remote_webrtc.sh"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert _read_lines(run_local_log) == ["ok"]
    assert "Skipping remote manager health wait" in result.stdout
    assert _read_lines(vm_ctl_log) == ["status", "wait-livekit", "livekit-url", "livekit-url"]


def test_run_hot_remote_webrtc_prefers_synced_remote_runtime_env(tmp_path):
    env_file = tmp_path / "gcp.env"
    _fake_remote_env(env_file, manager_token=None)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    ssh_log = tmp_path / "ssh.log"
    run_local_log = tmp_path / "run_local.log"

    _make_executable(
        bin_dir / "curl",
        """#!/usr/bin/env bash
set -euo pipefail
exit 0
""",
    )
    _make_executable(
        bin_dir / "gcloud",
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "$*" == *"--command"* && "$*" == *"character-eng.remote-hot.env"* ]]; then
  cat <<EOF
CHARACTER_ENG_LIVEKIT_API_KEY=remote-livekit-key
CHARACTER_ENG_LIVEKIT_API_SECRET=remote-livekit-secret-secret-secret-secret
EOF
  exit 0
fi
if [[ "$*" == *"--command"* ]]; then
  exit 0
fi
if [[ "$*" == *"--dry-run"* ]]; then
  printf 'ssh -N -L 127.0.0.1:17860:127.0.0.1:7860 -L 127.0.0.1:18003:127.0.0.1:8003\n'
  exit 0
fi
echo "unexpected gcloud call: $*" >&2
exit 1
""",
    )
    _make_executable(
        bin_dir / "ssh",
        f"""#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "{ssh_log}"
exec /bin/sleep 300
""",
    )
    _make_executable(
        bin_dir / "setsid",
        """#!/usr/bin/env bash
set -euo pipefail
exec "$@"
""",
    )
    vm_ctl = _make_executable(
        tmp_path / "vm_ctl_stub.sh",
        """#!/usr/bin/env bash
set -euo pipefail
cmd="${2:-}"
case "$cmd" in
  status)
    printf 'RUNNING\n'
    ;;
  wait-livekit|stop)
    ;;
  livekit-url)
    printf 'wss://remote.example.test/rtc\n'
    ;;
  *)
    exit 0
    ;;
esac
""",
    )
    run_local = _make_executable(
        tmp_path / "run_local_stub.sh",
        f"""#!/usr/bin/env bash
set -euo pipefail
cat > "{run_local_log}" <<EOF
CHARACTER_ENG_LIVEKIT_API_KEY=${{CHARACTER_ENG_LIVEKIT_API_KEY:-}}
CHARACTER_ENG_LIVEKIT_API_SECRET=${{CHARACTER_ENG_LIVEKIT_API_SECRET:-}}
EOF
""",
    )

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "CHARACTER_ENG_GCP_ENV": str(env_file),
            "VM_CTL_BIN": str(vm_ctl),
            "RUN_LOCAL_SCRIPT": str(run_local),
            "TUNNEL_DIR": str(tmp_path / "tunnel"),
        }
    )

    result = subprocess.run(
        ["bash", "scripts/run_hot_remote_webrtc.sh"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = "\n".join(_read_lines(run_local_log))
    assert "CHARACTER_ENG_LIVEKIT_API_KEY=remote-livekit-key" in payload
    assert "CHARACTER_ENG_LIVEKIT_API_SECRET=remote-livekit-secret-secret-secret-secret" in payload
    assert "Synced remote runtime env from GCP VM" in result.stdout


def test_stop_hot_remote_webrtc_cleans_tunnel_and_stops_vm(tmp_path):
    env_file = tmp_path / "gcp.env"
    _fake_remote_env(env_file)

    vm_ctl_log = tmp_path / "vm_ctl.log"
    vm_ctl = _make_executable(
        tmp_path / "vm_ctl_stub.sh",
        f"""#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "${{2:-}}" >> "{vm_ctl_log}"
""",
    )

    tunnel_dir = tmp_path / "tunnel"
    tunnel_dir.mkdir()
    tunnel_pidfile = tunnel_dir / "tunnel.pid"
    tunnel_proc = subprocess.Popen(["/bin/sleep", "300"])
    tunnel_pidfile.write_text(f"{tunnel_proc.pid}\n", encoding="utf-8")

    env = os.environ.copy()
    env.update(
        {
            "CHARACTER_ENG_GCP_ENV": str(env_file),
            "VM_CTL_BIN": str(vm_ctl),
            "TUNNEL_DIR": str(tunnel_dir),
            "STOP_VM": "1",
        }
    )

    try:
        result = subprocess.run(
            ["bash", "scripts/stop_hot_remote_webrtc.sh"],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert not tunnel_pidfile.exists()

        deadline = time.time() + 2.0
        while tunnel_proc.poll() is None and time.time() < deadline:
            time.sleep(0.05)
        assert tunnel_proc.poll() is not None
        assert _read_lines(vm_ctl_log) == ["stop"]
    finally:
        if tunnel_proc.poll() is None:
            tunnel_proc.kill()
            tunnel_proc.wait(timeout=5)


def test_create_vm_packages_local_runtime_api_keys_into_vm_env(tmp_path):
    local_env = tmp_path / "local.env"
    local_env.write_text(
        "\n".join(
            [
                "HF_TOKEN=hf-test-token",
                "CEREBRAS_API_KEY=cerebras-test-key",
                "GROQ_API_KEY=groq-test-key",
                "GEMINI_API_KEY=gemini-test-key",
                "DEEPGRAM_API_KEY=deepgram-test-key",
                "ELEVENLABS_API_KEY=elevenlabs-test-key",
                "RUNPOD_API_KEY=runpod-test-key",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    env_file = tmp_path / "gcp.env"
    _fake_create_vm_env(env_file, local_env)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    captured_app_env = tmp_path / "captured-character-eng.env"

    _make_executable(
        bin_dir / "gcloud",
        f"""#!/usr/bin/env bash
set -euo pipefail

copy_character_eng_env() {{
  local metadata_files="$1"
  IFS=',' read -ra pairs <<< "$metadata_files"
  for pair in "${{pairs[@]}}"; do
    if [[ "$pair" == character-eng-env=* ]]; then
      cp "${{pair#character-eng-env=}}" "{captured_app_env}"
      return 0
    fi
  done
  echo "character-eng-env metadata file missing" >&2
  exit 1
}}

if [[ "$1" == "compute" && "$2" == "disks" && "$3" == "describe" ]]; then
  exit 1
fi
if [[ "$1" == "compute" && "$2" == "addresses" && "$3" == "describe" ]]; then
  exit 1
fi
if [[ "$1" == "compute" && "$2" == "firewall-rules" && "$3" == "describe" ]]; then
  exit 1
fi
if [[ "$1" == "compute" && "$2" == "instances" && "$3" == "describe" ]]; then
  if [[ "$*" == *"--format=value(networkInterfaces[0].accessConfigs[0].natIP)"* ]]; then
    printf '203.0.113.10\\n'
    exit 0
  fi
  exit 1
fi
if [[ "$1" == "compute" && "$2" == "instances" && "$3" == "create" ]]; then
  for ((i=1; i<=$#; i+=1)); do
    if [[ "${{!i}}" == "--metadata-from-file" ]]; then
      next=$((i + 1))
      copy_character_eng_env "${{!next}}"
      exit 0
    fi
  done
  echo "--metadata-from-file missing on create" >&2
  exit 1
fi
exit 0
""",
    )

    env = os.environ.copy()
    env.update({"PATH": f"{bin_dir}:{env['PATH']}"})

    result = subprocess.run(
        ["bash", "deploy/gcp/create_vm.sh", str(env_file)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert result.returncode == 0, result.stderr

    payload = "\n".join(_read_lines(captured_app_env))
    assert "HF_TOKEN=hf-test-token" in payload
    assert "CEREBRAS_API_KEY=cerebras-test-key" in payload
    assert "GROQ_API_KEY=groq-test-key" in payload
    assert "GEMINI_API_KEY=gemini-test-key" in payload
    assert "DEEPGRAM_API_KEY=deepgram-test-key" in payload
    assert "ELEVENLABS_API_KEY=elevenlabs-test-key" in payload
    assert "RUNPOD_API_KEY=runpod-test-key" in payload
