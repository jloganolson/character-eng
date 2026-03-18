from __future__ import annotations

from character_eng import pocket_runtime


def test_resolve_pocket_tts_binary_prefers_project_venv(tmp_path, monkeypatch):
    project_root = tmp_path / "repo"
    pocket_bin = project_root / ".venv" / "bin" / "pocket-tts"
    pocket_bin.parent.mkdir(parents=True)
    pocket_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    pocket_bin.chmod(0o755)

    monkeypatch.setattr(pocket_runtime, "PROJECT_ROOT", project_root)
    monkeypatch.delenv("POCKET_BIN", raising=False)

    assert pocket_runtime.resolve_pocket_tts_binary() == str(pocket_bin.resolve())


def test_ensure_pocket_tts_binary_installs_into_runtime_tool_dir(tmp_path, monkeypatch):
    project_root = tmp_path / "repo"
    runtime_root = tmp_path / "runtime"
    tool_bin = runtime_root / "pocket-tts" / "bin" / "pocket-tts"
    tool_python = runtime_root / "pocket-tts" / "bin" / "python"

    monkeypatch.setattr(pocket_runtime, "PROJECT_ROOT", project_root)
    monkeypatch.setenv("CHARACTER_ENG_RUNTIME_ROOT", str(runtime_root))
    monkeypatch.delenv("POCKET_BIN", raising=False)
    monkeypatch.setattr(pocket_runtime.shutil, "which", lambda _: None)
    uv_bin = tmp_path / "bin" / "uv"
    uv_bin.parent.mkdir(parents=True)
    uv_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    uv_bin.chmod(0o755)
    monkeypatch.setenv("UV_BIN", str(uv_bin))

    calls = []

    def fake_run(cmd, check, env):
        calls.append((cmd, check, env["UV_CACHE_DIR"]))
        if cmd[1] == "venv":
            tool_python.parent.mkdir(parents=True, exist_ok=True)
            tool_python.write_text("#!/bin/sh\n", encoding="utf-8")
            tool_python.chmod(0o755)
        else:
            tool_bin.parent.mkdir(parents=True, exist_ok=True)
            tool_bin.write_text("#!/bin/sh\n", encoding="utf-8")
            tool_bin.chmod(0o755)

    monkeypatch.setattr(pocket_runtime.subprocess, "run", fake_run)

    resolved = pocket_runtime.ensure_pocket_tts_binary()

    assert resolved == str(tool_bin.resolve())
    assert len(calls) == 2
    assert calls[0][0][:2] == [str(uv_bin.resolve()), "venv"]
    assert calls[1][0][:4] == [str(uv_bin.resolve()), "pip", "install", "--python"]
    assert calls[0][2] == str(runtime_root / ".cache" / "uv")


def test_ensure_pocket_tts_binary_uses_existing_path(monkeypatch, tmp_path):
    pocket_bin = tmp_path / "bin" / "pocket-tts"
    pocket_bin.parent.mkdir(parents=True)
    pocket_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    pocket_bin.chmod(0o755)

    monkeypatch.setenv("POCKET_BIN", str(pocket_bin))

    assert pocket_runtime.ensure_pocket_tts_binary() == str(pocket_bin.resolve())
