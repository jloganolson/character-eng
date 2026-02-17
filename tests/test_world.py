import json
from unittest.mock import MagicMock, patch

import pytest

import character_eng.world as world_mod
from character_eng.world import (
    Goals,
    ThinkResult,
    WorldState,
    WorldUpdate,
    _load_prompt_file,
    director_call,
    format_narrator_message,
    load_goals,
    load_world_state,
    think_call,
)

TEST_CONFIG = {
    "name": "Test Model",
    "model": "test-model",
    "base_url": "https://test.example.com/v1",
    "api_key_env": "GEMINI_API_KEY",
    "stream_usage": True,
}


# --- WorldState ---


def test_render_full():
    ws = WorldState(
        static=["Greg is a robot head"],
        dynamic=["Orb is on table", "Room is dark"],
        events=["The orb began to glow"],
    )
    text = ws.render()
    assert "Permanent facts:" in text
    assert "- Greg is a robot head" in text
    assert "Current state:" in text
    assert "- Orb is on table" in text
    assert "Recent events:" in text
    assert "- The orb began to glow" in text


def test_render_empty():
    ws = WorldState()
    assert ws.render() == ""


def test_render_static_only():
    ws = WorldState(static=["Fact one"])
    text = ws.render()
    assert "Permanent facts:" in text
    assert "Current state:" not in text
    assert "Recent events:" not in text


def test_render_dynamic_only():
    ws = WorldState(dynamic=["Fact one"])
    text = ws.render()
    assert "Current state:" in text
    assert "Permanent facts:" not in text


# --- apply_update ---


def test_apply_update_add_facts():
    ws = WorldState(dynamic=["A", "B"])
    update = WorldUpdate(add_facts=["C"])
    ws.apply_update(update)
    assert ws.dynamic == ["A", "B", "C"]


def test_apply_update_remove_facts():
    ws = WorldState(dynamic=["A", "B", "C"])
    update = WorldUpdate(remove_facts=[1])
    ws.apply_update(update)
    assert ws.dynamic == ["A", "C"]


def test_apply_update_remove_multiple():
    ws = WorldState(dynamic=["A", "B", "C", "D"])
    update = WorldUpdate(remove_facts=[0, 2])
    ws.apply_update(update)
    assert ws.dynamic == ["B", "D"]


def test_apply_update_events():
    ws = WorldState()
    update = WorldUpdate(events=["Something happened"])
    ws.apply_update(update)
    assert ws.events == ["Something happened"]


def test_apply_update_combined():
    ws = WorldState(dynamic=["Orb on table", "Room is dark"])
    update = WorldUpdate(
        remove_facts=[0],
        add_facts=["Orb is on floor"],
        events=["The orb rolled off the table"],
    )
    ws.apply_update(update)
    assert ws.dynamic == ["Room is dark", "Orb is on floor"]
    assert ws.events == ["The orb rolled off the table"]


def test_apply_update_invalid_index_ignored():
    ws = WorldState(dynamic=["A"])
    update = WorldUpdate(remove_facts=[5])
    ws.apply_update(update)
    assert ws.dynamic == ["A"]


# --- show ---


def test_show_returns_panel():
    ws = WorldState(static=["Fact"])
    panel = ws.show()
    assert panel.title == "World State"


# --- load_world_state ---


def test_load_world_state_with_files(tmp_path, monkeypatch):
    char_dir = tmp_path / "characters" / "test_char"
    char_dir.mkdir(parents=True)
    (char_dir / "world_static.txt").write_text("Static fact one\nStatic fact two\n")
    (char_dir / "world_dynamic.txt").write_text("Dynamic fact\n\n  \n")

    monkeypatch.setattr(world_mod, "CHARACTERS_DIR", tmp_path / "characters")

    ws = load_world_state("test_char")
    assert ws is not None
    assert ws.static == ["Static fact one", "Static fact two"]
    assert ws.dynamic == ["Dynamic fact"]
    assert ws.events == []


def test_load_world_state_no_files(tmp_path, monkeypatch):
    char_dir = tmp_path / "characters" / "test_char"
    char_dir.mkdir(parents=True)

    monkeypatch.setattr(world_mod, "CHARACTERS_DIR", tmp_path / "characters")

    ws = load_world_state("test_char")
    assert ws is None


def test_load_world_state_static_only(tmp_path, monkeypatch):
    char_dir = tmp_path / "characters" / "test_char"
    char_dir.mkdir(parents=True)
    (char_dir / "world_static.txt").write_text("Only static\n")

    monkeypatch.setattr(world_mod, "CHARACTERS_DIR", tmp_path / "characters")

    ws = load_world_state("test_char")
    assert ws is not None
    assert ws.static == ["Only static"]
    assert ws.dynamic == []


# --- Goals ---


def test_goals_render_full():
    g = Goals(long_term="Experience the universe", short_term="Figure out the orb")
    text = g.render()
    assert "Long-term goal: Experience the universe" in text
    assert "Short-term goal: Figure out the orb" in text


def test_goals_render_empty():
    g = Goals()
    assert g.render() == ""


def test_goals_render_long_term_only():
    g = Goals(long_term="Experience the universe")
    text = g.render()
    assert "Long-term goal: Experience the universe" in text
    assert "Short-term goal" not in text


def test_goals_apply_update():
    g = Goals(long_term="Experience the universe", short_term="Old goal")
    g.apply_update("New goal")
    assert g.short_term == "New goal"
    assert g.long_term == "Experience the universe"


def test_goals_show_returns_panel():
    g = Goals(long_term="Drive")
    panel = g.show()
    assert panel.title == "Character Goals"


# --- load_goals ---


def test_load_goals_from_character_txt(tmp_path, monkeypatch):
    char_dir = tmp_path / "characters" / "test_char"
    char_dir.mkdir(parents=True)
    (char_dir / "character.txt").write_text(
        "You are Test.\n\nLong-term goal: Experience the universe\nShort-term goal: Figure out the orb\n"
    )

    monkeypatch.setattr(world_mod, "CHARACTERS_DIR", tmp_path / "characters")

    g = load_goals("test_char")
    assert g is not None
    assert g.long_term == "Experience the universe"
    assert g.short_term == "Figure out the orb"


def test_load_goals_no_goals_in_character_txt(tmp_path, monkeypatch):
    char_dir = tmp_path / "characters" / "test_char"
    char_dir.mkdir(parents=True)
    (char_dir / "character.txt").write_text("You are Test. No goals here.\n")

    monkeypatch.setattr(world_mod, "CHARACTERS_DIR", tmp_path / "characters")

    g = load_goals("test_char")
    assert g is None


def test_load_goals_no_character_txt(tmp_path, monkeypatch):
    char_dir = tmp_path / "characters" / "test_char"
    char_dir.mkdir(parents=True)

    monkeypatch.setattr(world_mod, "CHARACTERS_DIR", tmp_path / "characters")

    g = load_goals("test_char")
    assert g is None


# --- format_narrator_message ---


def test_format_narrator_message_events_only():
    update = WorldUpdate(events=["The orb rolled off the table"])
    msg = format_narrator_message(update)
    assert msg == "[The orb rolled off the table]"


def test_format_narrator_message_events_and_facts():
    update = WorldUpdate(
        add_facts=["Orb is on floor"],
        events=["The orb fell"],
    )
    msg = format_narrator_message(update)
    assert msg.startswith("[")
    assert msg.endswith("]")
    assert "The orb fell" in msg
    assert "Orb is on floor" in msg


def test_format_narrator_message_empty():
    update = WorldUpdate()
    msg = format_narrator_message(update)
    assert msg == "[]"


# --- Mock helper for OpenAI responses ---


def _mock_openai_response(data: dict):
    """Create a mock OpenAI completion response with the given JSON data."""
    mock_message = MagicMock()
    mock_message.content = json.dumps(data)
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


# --- director_call (mocked) ---


@patch("character_eng.world._make_client")
def test_director_call_basic(mock_make_client):
    data = {
        "remove_facts": [0],
        "add_facts": ["Orb is on floor"],
        "events": ["The orb rolled off the table"],
    }
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(data)
    mock_make_client.return_value = mock_client

    ws = WorldState(
        static=["Greg is a robot head"],
        dynamic=["Orb is on table"],
    )
    update = director_call(ws, "the orb rolls off the table", TEST_CONFIG)

    assert update.remove_facts == [0]
    assert update.add_facts == ["Orb is on floor"]
    assert update.events == ["The orb rolled off the table"]

    mock_client.chat.completions.create.assert_called_once()


@patch("character_eng.world._make_client")
def test_director_call_no_removals(mock_make_client):
    data = {
        "remove_facts": [],
        "add_facts": ["A cat entered the room"],
        "events": ["A cat wandered in"],
    }
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(data)
    mock_make_client.return_value = mock_client

    ws = WorldState(dynamic=["Orb is on table"])
    update = director_call(ws, "a cat walks in", TEST_CONFIG)

    assert update.remove_facts == []
    assert update.add_facts == ["A cat entered the room"]
    assert update.events == ["A cat wandered in"]


@patch("character_eng.world._make_client")
def test_director_call_empty_state(mock_make_client):
    data = {
        "remove_facts": [],
        "add_facts": ["Person is sitting at the desk"],
        "events": ["Someone sat down"],
    }
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(data)
    mock_make_client.return_value = mock_client

    ws = WorldState()
    update = director_call(ws, "someone sits down", TEST_CONFIG)

    assert update.add_facts == ["Person is sitting at the desk"]
    assert update.events == ["Someone sat down"]


# --- think_call (mocked) ---


@patch("character_eng.world._make_client")
def test_think_call_no_change(mock_make_client):
    data = {
        "thought": "I wonder what that orb is doing.",
        "action": "no_change",
        "action_detail": "",
        "gaze": "orb",
        "expression": "curious",
        "new_short_term_goal": None,
    }
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(data)
    mock_make_client.return_value = mock_client

    result = think_call(
        system_prompt="You are Greg.",
        world=WorldState(static=["Greg is a robot head"]),
        history=[{"role": "user", "content": "Hello"}],
        model_config=TEST_CONFIG,
    )

    assert result.thought == "I wonder what that orb is doing."
    assert result.action == "no_change"
    assert result.action_detail == ""
    assert result.gaze == "orb"
    assert result.expression == "curious"
    assert result.new_short_term_goal is None


@patch("character_eng.world._make_client")
def test_think_call_talk(mock_make_client):
    data = {
        "thought": "I should mention the orb is glowing.",
        "action": "talk",
        "action_detail": "Comment on the orb's glow",
        "gaze": "orb",
        "expression": "excited",
        "new_short_term_goal": None,
    }
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(data)
    mock_make_client.return_value = mock_client

    result = think_call(
        system_prompt="You are Greg.",
        world=WorldState(dynamic=["The orb is glowing"]),
        history=[{"role": "user", "content": "What's up?"}],
        model_config=TEST_CONFIG,
    )

    assert result.action == "talk"
    assert result.action_detail == "Comment on the orb's glow"


@patch("character_eng.world._make_client")
def test_think_call_emote(mock_make_client):
    data = {
        "thought": "That noise is alarming.",
        "action": "emote",
        "action_detail": "Eyes widen",
        "gaze": "door",
        "expression": "surprised",
        "new_short_term_goal": None,
    }
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(data)
    mock_make_client.return_value = mock_client

    result = think_call(
        system_prompt="You are Greg.",
        world=None,
        history=[],
        model_config=TEST_CONFIG,
    )

    assert result.action == "emote"
    assert result.action_detail == "Eyes widen"
    assert result.gaze == "door"
    assert result.expression == "surprised"


@patch("character_eng.world._make_client")
def test_think_call_empty_history(mock_make_client):
    data = {
        "thought": "Nobody is here yet.",
        "action": "no_change",
        "action_detail": "",
        "gaze": "orb",
        "expression": "neutral",
        "new_short_term_goal": None,
    }
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(data)
    mock_make_client.return_value = mock_client

    result = think_call(
        system_prompt="You are Greg.",
        world=WorldState(static=["Greg is a robot head"]),
        history=[],
        model_config=TEST_CONFIG,
    )

    assert result.thought == "Nobody is here yet."
    assert result.action == "no_change"
    mock_client.chat.completions.create.assert_called_once()


@patch("character_eng.world._make_client")
def test_think_call_with_goals(mock_make_client):
    """Goals are included in the user message sent to the LLM."""
    data = {
        "thought": "I should focus on the orb.",
        "action": "no_change",
        "action_detail": "",
        "gaze": "orb",
        "expression": "focused",
        "new_short_term_goal": None,
    }
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(data)
    mock_make_client.return_value = mock_client

    goals = Goals(long_term="Experience the universe", short_term="Figure out the orb")
    think_call(
        system_prompt="You are Greg.",
        world=WorldState(static=["Greg is a robot head"]),
        history=[],
        model_config=TEST_CONFIG,
        goals=goals,
    )

    call_args = mock_client.chat.completions.create.call_args
    user_msg = call_args[1]["messages"][1]["content"]
    assert "CHARACTER GOALS" in user_msg
    assert "Experience the universe" in user_msg
    assert "Figure out the orb" in user_msg


@patch("character_eng.world._make_client")
def test_think_call_with_goal_update(mock_make_client):
    """new_short_term_goal in LLM response is parsed."""
    data = {
        "thought": "I've figured out the orb is just glass.",
        "action": "no_change",
        "action_detail": "",
        "gaze": "orb",
        "expression": "amused",
        "new_short_term_goal": "Find a new mystery to investigate",
    }
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(data)
    mock_make_client.return_value = mock_client

    goals = Goals(long_term="Experience the universe", short_term="Figure out the orb")
    result = think_call(
        system_prompt="You are Greg.",
        world=None,
        history=[],
        model_config=TEST_CONFIG,
        goals=goals,
    )

    assert result.new_short_term_goal == "Find a new mystery to investigate"


@patch("character_eng.world._make_client")
def test_think_call_empty_goal_treated_as_none(mock_make_client):
    """Empty string new_short_term_goal is treated as None."""
    data = {
        "thought": "Just thinking.",
        "action": "no_change",
        "action_detail": "",
        "gaze": "orb",
        "expression": "neutral",
        "new_short_term_goal": "",
    }
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(data)
    mock_make_client.return_value = mock_client

    result = think_call(
        system_prompt="You are Greg.",
        world=None,
        history=[],
        model_config=TEST_CONFIG,
    )

    assert result.new_short_term_goal is None


@patch("character_eng.world._make_client")
def test_think_call_without_goals(mock_make_client):
    """When no goals are passed, CHARACTER GOALS section is absent."""
    data = {
        "thought": "Just thinking.",
        "action": "no_change",
        "action_detail": "",
        "gaze": "orb",
        "expression": "neutral",
    }
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(data)
    mock_make_client.return_value = mock_client

    think_call(
        system_prompt="You are Greg.",
        world=None,
        history=[],
        model_config=TEST_CONFIG,
        goals=None,
    )

    call_args = mock_client.chat.completions.create.call_args
    user_msg = call_args[1]["messages"][1]["content"]
    assert "CHARACTER GOALS" not in user_msg


# --- _load_prompt_file ---


def test_load_prompt_file_reads_from_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(world_mod, "PROMPTS_DIR", tmp_path)
    (tmp_path / "test_prompt.txt").write_text("Hello from disk")
    assert _load_prompt_file("test_prompt.txt") == "Hello from disk"


def test_load_prompt_file_missing_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(world_mod, "PROMPTS_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        _load_prompt_file("nonexistent.txt")


# --- Prompt reload (file changes picked up between calls) ---


@patch("character_eng.world._make_client")
def test_director_call_reads_prompt_from_file(mock_make_client, tmp_path, monkeypatch):
    """director_call reads director_system.txt from disk each call."""
    monkeypatch.setattr(world_mod, "PROMPTS_DIR", tmp_path)
    (tmp_path / "director_system.txt").write_text("You are director v1.")

    data = {"remove_facts": [], "add_facts": [], "events": ["thing"]}
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(data)
    mock_make_client.return_value = mock_client

    director_call(WorldState(), "test", TEST_CONFIG)

    # Check the system message sent to the API
    call_args = mock_client.chat.completions.create.call_args
    messages = call_args[1]["messages"]
    assert messages[0]["content"] == "You are director v1."


@patch("character_eng.world._make_client")
def test_director_call_picks_up_file_changes(mock_make_client, tmp_path, monkeypatch):
    """Editing director_system.txt is picked up on the next call without restart."""
    monkeypatch.setattr(world_mod, "PROMPTS_DIR", tmp_path)

    data = {"remove_facts": [], "add_facts": [], "events": ["thing"]}
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(data)
    mock_make_client.return_value = mock_client

    # First call with v1
    (tmp_path / "director_system.txt").write_text("Director prompt v1")
    director_call(WorldState(), "test", TEST_CONFIG)
    msgs_v1 = mock_client.chat.completions.create.call_args[1]["messages"]
    assert msgs_v1[0]["content"] == "Director prompt v1"

    # Edit the file
    (tmp_path / "director_system.txt").write_text("Director prompt v2")
    director_call(WorldState(), "test", TEST_CONFIG)
    msgs_v2 = mock_client.chat.completions.create.call_args[1]["messages"]
    assert msgs_v2[0]["content"] == "Director prompt v2"


@patch("character_eng.world._make_client")
def test_think_call_reads_prompt_from_file(mock_make_client, tmp_path, monkeypatch):
    """think_call reads think_system.txt from disk each call."""
    monkeypatch.setattr(world_mod, "PROMPTS_DIR", tmp_path)
    (tmp_path / "think_system.txt").write_text("You are think v1.")

    data = {"thought": "hmm", "action": "no_change", "action_detail": "", "gaze": "orb", "expression": "neutral"}
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(data)
    mock_make_client.return_value = mock_client

    think_call("sys prompt", world=None, history=[], model_config=TEST_CONFIG)

    call_args = mock_client.chat.completions.create.call_args
    messages = call_args[1]["messages"]
    assert messages[0]["content"] == "You are think v1."


@patch("character_eng.world._make_client")
def test_think_call_picks_up_file_changes(mock_make_client, tmp_path, monkeypatch):
    """Editing think_system.txt is picked up on the next call without restart."""
    monkeypatch.setattr(world_mod, "PROMPTS_DIR", tmp_path)

    data = {"thought": "hmm", "action": "no_change", "action_detail": "", "gaze": "orb", "expression": "neutral"}
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(data)
    mock_make_client.return_value = mock_client

    # First call with v1
    (tmp_path / "think_system.txt").write_text("Think prompt v1")
    think_call("sys prompt", world=None, history=[], model_config=TEST_CONFIG)
    msgs_v1 = mock_client.chat.completions.create.call_args[1]["messages"]
    assert msgs_v1[0]["content"] == "Think prompt v1"

    # Edit the file
    (tmp_path / "think_system.txt").write_text("Think prompt v2")
    think_call("sys prompt", world=None, history=[], model_config=TEST_CONFIG)
    msgs_v2 = mock_client.chat.completions.create.call_args[1]["messages"]
    assert msgs_v2[0]["content"] == "Think prompt v2"
