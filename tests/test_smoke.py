"""Smoke tests — exercise real startup path with real files on disk.

No mocks except the OpenAI client. If these fail, the app won't even start.
"""

import importlib
import pkgutil
from unittest.mock import patch

import character_eng
from character_eng.config import AppConfig, load_config
from character_eng.models import BIG_MODEL, CHAT_MODEL, MODELS
from character_eng.prompts import list_characters, load_prompt
from character_eng.world import load_goals, load_world_state


def test_all_modules_import():
    """Every module in character_eng should import without error."""
    pkg_path = character_eng.__path__
    failures = []
    for info in pkgutil.walk_packages(pkg_path, prefix="character_eng."):
        try:
            importlib.import_module(info.name)
        except Exception as exc:
            failures.append(f"{info.name}: {exc}")
    assert not failures, f"Import failures:\n" + "\n".join(failures)


def test_load_config():
    """load_config() returns a valid AppConfig whether or not config.toml exists."""
    cfg = load_config()
    assert isinstance(cfg, AppConfig)
    assert hasattr(cfg, "voice")


def test_model_registry_valid():
    """CHAT_MODEL and BIG_MODEL exist in MODELS; all entries are well-formed."""
    assert CHAT_MODEL in MODELS, f"CHAT_MODEL {CHAT_MODEL!r} not in MODELS"
    assert BIG_MODEL in MODELS, f"BIG_MODEL {BIG_MODEL!r} not in MODELS"

    required_keys = {"name", "model", "base_url", "stream_usage"}
    for key, cfg in MODELS.items():
        missing = required_keys - set(cfg.keys())
        assert not missing, f"Model {key!r} missing keys: {missing}"
        # Must have either api_key or api_key_env
        assert "api_key" in cfg or "api_key_env" in cfg, (
            f"Model {key!r} has neither api_key nor api_key_env"
        )


def test_list_characters_finds_greg():
    """list_characters() discovers greg from the real prompts directory."""
    chars = list_characters()
    assert "greg" in chars


def test_load_prompt_with_real_files():
    """load_prompt('greg', ws) expands all macros using real files on disk."""
    ws = load_world_state("greg")
    prompt = load_prompt("greg", world_state=ws)
    # No unexpanded macros (except possibly {{unknown}} which shouldn't exist)
    assert "{{global_rules}}" not in prompt
    assert "{{character}}" not in prompt
    assert "{{world}}" not in prompt
    assert "{{scenario}}" not in prompt
    # Should contain actual content
    assert len(prompt) > 100


def test_load_world_state_greg():
    """Real facts loaded from greg's world files."""
    ws = load_world_state("greg")
    assert ws is not None
    assert len(ws.static) > 0 or len(ws.dynamic) > 0


def test_load_goals_greg():
    """Real goals parsed from greg's character.txt."""
    goals = load_goals("greg")
    assert goals is not None
    assert goals.long_term != ""


@patch("character_eng.chat.OpenAI")
def test_chat_session_creation(mock_openai_cls):
    """ChatSession instantiates with real prompt and model config."""
    from character_eng.chat import ChatSession

    ws = load_world_state("greg")
    prompt = load_prompt("greg", world_state=ws)
    config = MODELS[CHAT_MODEL]

    session = ChatSession(prompt, config)
    history = session.get_history()
    assert len(history) == 1
    assert history[0]["role"] == "system"
    assert len(history[0]["content"]) > 100
