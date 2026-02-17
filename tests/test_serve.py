from unittest.mock import patch

from character_eng.serve import build_vllm_cmd, get_local_models, kill_port


def test_get_local_models():
    """get_local_models returns only entries with local: True."""
    models = get_local_models()
    assert len(models) > 0
    for key, cfg in models:
        assert cfg.get("local") is True


def test_get_local_models_keys():
    """Returned keys match known local model keys."""
    keys = [k for k, _ in get_local_models()]
    for k in keys:
        assert "lfm" in k  # all current local models are LFM variants


def test_build_vllm_cmd():
    """build_vllm_cmd returns expected command list."""
    cfg = {"path": "/models/test-model", "name": "Test"}
    cmd = build_vllm_cmd("test-key", cfg, port="9000")
    assert cmd[0] == "vllm"
    assert cmd[1] == "serve"
    assert cmd[2] == "/models/test-model"
    assert "--served-model-name" in cmd
    assert cmd[cmd.index("--served-model-name") + 1] == "test-key"
    assert "--port" in cmd
    assert cmd[cmd.index("--port") + 1] == "9000"
    assert "--trust-remote-code" in cmd
    assert "--gpu-memory-utilization" in cmd


def test_build_vllm_cmd_default_port():
    """build_vllm_cmd uses default port when not specified."""
    cfg = {"path": "/models/test"}
    cmd = build_vllm_cmd("k", cfg)
    assert cmd[cmd.index("--port") + 1] == "8000"


@patch("character_eng.serve.subprocess.run")
def test_kill_port_no_lsof(mock_run):
    """kill_port handles missing lsof gracefully."""
    mock_run.side_effect = FileNotFoundError
    # Should not raise
    kill_port("8000")
