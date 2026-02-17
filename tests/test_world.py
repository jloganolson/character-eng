import json
from unittest.mock import MagicMock, patch

import pytest

import character_eng.world as world_mod
from character_eng.world import (
    Beat,
    EvalResult,
    Goals,
    PlanResult,
    Script,
    WorldState,
    WorldUpdate,
    _load_prompt_file,
    director_call,
    eval_call,
    format_narrator_message,
    load_goals,
    load_world_state,
    plan_call,
)

TEST_CONFIG = {
    "name": "Test Model",
    "model": "test-model",
    "base_url": "https://test.example.com/v1",
    "api_key_env": "GEMINI_API_KEY",
    "stream_usage": True,
}

PLAN_CONFIG = {
    "name": "Plan Model",
    "model": "plan-model",
    "base_url": "https://plan.example.com/v1",
    "api_key_env": "GEMINI_API_KEY",
    "stream_usage": False,
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


# --- Beat ---


def test_beat_construction():
    b = Beat(line="Hey, what's that orb?", intent="Ask about the orb")
    assert b.line == "Hey, what's that orb?"
    assert b.intent == "Ask about the orb"


# --- Script ---


def test_script_current_beat():
    s = Script(beats=[Beat("Line 1", "Intent 1"), Beat("Line 2", "Intent 2")])
    assert s.current_beat == Beat("Line 1", "Intent 1")


def test_script_advance():
    s = Script(beats=[Beat("Line 1", "Intent 1"), Beat("Line 2", "Intent 2")])
    s.advance()
    assert s.current_beat == Beat("Line 2", "Intent 2")


def test_script_advance_to_empty():
    s = Script(beats=[Beat("Line 1", "Intent 1")])
    s.advance()
    assert s.current_beat is None


def test_script_replace():
    s = Script(beats=[Beat("Old", "Old intent")])
    s.advance()  # move past beat 0
    s.replace([Beat("New 1", "New intent 1"), Beat("New 2", "New intent 2")])
    assert s.current_beat == Beat("New 1", "New intent 1")
    assert len(s.beats) == 2


def test_script_is_empty():
    assert Script().is_empty()
    assert not Script(beats=[Beat("Line", "Intent")]).is_empty()


def test_script_render():
    s = Script(beats=[Beat("Line 1", "Intent 1"), Beat("Line 2", "Intent 2")])
    text = s.render()
    assert '→ 0. [Intent 1] "Line 1"' in text
    assert '  1. [Intent 2] "Line 2"' in text


def test_script_render_empty():
    assert Script().render() == ""


def test_script_show_returns_panel():
    s = Script(beats=[Beat("Line", "Intent")])
    panel = s.show()
    assert panel.title == "Script"


def test_script_show_empty():
    panel = Script().show()
    assert panel.title == "Script"


# --- EvalResult ---


def test_eval_result_construction():
    r = EvalResult(
        thought="Hmm, interesting.",
        gaze="orb",
        expression="curious",
        script_status="hold",
    )
    assert r.thought == "Hmm, interesting."
    assert r.script_status == "hold"
    assert r.bootstrap_line == ""
    assert r.bootstrap_intent == ""
    assert r.plan_request == ""


def test_eval_result_bootstrap_fields():
    r = EvalResult(
        thought="Let me start.",
        gaze="visitor",
        expression="curious",
        script_status="bootstrap",
        bootstrap_line="Hey there! What brings you here?",
        bootstrap_intent="Greet the visitor",
        plan_request="Plan a conversation about the orb",
    )
    assert r.script_status == "bootstrap"
    assert r.bootstrap_line == "Hey there! What brings you here?"
    assert r.bootstrap_intent == "Greet the visitor"
    assert r.plan_request == "Plan a conversation about the orb"


# --- Goals ---


def test_goals_render_full():
    g = Goals(long_term="Experience the universe")
    text = g.render()
    assert "Long-term goal: Experience the universe" in text


def test_goals_render_empty():
    g = Goals()
    assert g.render() == ""


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


# --- eval_call (mocked) ---


@patch("character_eng.world._make_client")
def test_eval_call_hold(mock_make_client):
    data = {
        "thought": "I should wait for them to respond.",
        "gaze": "visitor",
        "expression": "curious",
        "script_status": "hold",
    }
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(data)
    mock_make_client.return_value = mock_client

    result = eval_call(
        system_prompt="You are Greg.",
        world=WorldState(static=["Greg is a robot head"]),
        history=[{"role": "user", "content": "Hello"}],
        model_config=TEST_CONFIG,
    )

    assert result.thought == "I should wait for them to respond."
    assert result.script_status == "hold"
    assert result.gaze == "visitor"
    assert result.expression == "curious"


@patch("character_eng.world._make_client")
def test_eval_call_advance(mock_make_client):
    data = {
        "thought": "Good, they engaged with my question.",
        "gaze": "orb",
        "expression": "excited",
        "script_status": "advance",
    }
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(data)
    mock_make_client.return_value = mock_client

    script = Script(beats=[Beat("What's that orb?", "Ask about orb"), Beat("Can I touch it?", "Request to touch")])
    result = eval_call(
        system_prompt="You are Greg.",
        world=None,
        history=[],
        model_config=TEST_CONFIG,
        script=script,
    )

    assert result.script_status == "advance"


@patch("character_eng.world._make_client")
def test_eval_call_bootstrap(mock_make_client):
    data = {
        "thought": "Someone new is here. I should say hello.",
        "gaze": "visitor",
        "expression": "curious",
        "script_status": "bootstrap",
        "bootstrap_line": "Hey there! What brings you to my desk?",
        "bootstrap_intent": "Greet the visitor",
        "plan_request": "Plan a conversation exploring what the visitor wants",
    }
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(data)
    mock_make_client.return_value = mock_client

    result = eval_call(
        system_prompt="You are Greg.",
        world=WorldState(static=["Greg is a robot head"]),
        history=[{"role": "user", "content": "Hello"}],
        model_config=TEST_CONFIG,
        script=Script(),
    )

    assert result.script_status == "bootstrap"
    assert result.bootstrap_line == "Hey there! What brings you to my desk?"
    assert result.bootstrap_intent == "Greet the visitor"
    assert result.plan_request != ""


@patch("character_eng.world._make_client")
def test_eval_call_off_book(mock_make_client):
    data = {
        "thought": "This conversation went somewhere unexpected.",
        "gaze": "floor",
        "expression": "confused",
        "script_status": "off_book",
        "plan_request": "Replan around the new topic of the visitor's pet",
    }
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(data)
    mock_make_client.return_value = mock_client

    result = eval_call(
        system_prompt="You are Greg.",
        world=None,
        history=[],
        model_config=TEST_CONFIG,
    )

    assert result.script_status == "off_book"
    assert "pet" in result.plan_request


@patch("character_eng.world._make_client")
def test_eval_call_script_in_context(mock_make_client):
    """Script is included in the user message when provided."""
    data = {
        "thought": "Following script.",
        "gaze": "orb",
        "expression": "neutral",
        "script_status": "hold",
    }
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(data)
    mock_make_client.return_value = mock_client

    script = Script(beats=[Beat("Hello!", "Greet")])
    eval_call(
        system_prompt="You are Greg.",
        world=None,
        history=[],
        model_config=TEST_CONFIG,
        script=script,
    )

    call_args = mock_client.chat.completions.create.call_args
    user_msg = call_args[1]["messages"][1]["content"]
    assert "CURRENT SCRIPT" in user_msg
    assert "Hello!" in user_msg


@patch("character_eng.world._make_client")
def test_eval_call_goals_in_context(mock_make_client):
    """Goals are included in the user message when provided."""
    data = {
        "thought": "Goal-driven.",
        "gaze": "orb",
        "expression": "focused",
        "script_status": "hold",
    }
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(data)
    mock_make_client.return_value = mock_client

    goals = Goals(long_term="Experience the universe")
    eval_call(
        system_prompt="You are Greg.",
        world=None,
        history=[],
        model_config=TEST_CONFIG,
        goals=goals,
    )

    call_args = mock_client.chat.completions.create.call_args
    user_msg = call_args[1]["messages"][1]["content"]
    assert "CHARACTER GOALS" in user_msg
    assert "Experience the universe" in user_msg


@patch("character_eng.world._make_client")
def test_eval_call_reads_eval_system_txt(mock_make_client, tmp_path, monkeypatch):
    """eval_call reads eval_system.txt from disk each call."""
    monkeypatch.setattr(world_mod, "PROMPTS_DIR", tmp_path)
    (tmp_path / "eval_system.txt").write_text("You are eval v1.")

    data = {"thought": "hmm", "gaze": "orb", "expression": "neutral", "script_status": "hold"}
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(data)
    mock_make_client.return_value = mock_client

    eval_call("sys prompt", world=None, history=[], model_config=TEST_CONFIG)

    call_args = mock_client.chat.completions.create.call_args
    messages = call_args[1]["messages"]
    assert messages[0]["content"] == "You are eval v1."


# --- plan_call (mocked) ---


@patch("character_eng.world._make_client")
def test_plan_call_basic_beats(mock_make_client):
    data = {
        "beats": [
            {"line": "Hey, what's that orb?", "intent": "Ask about orb"},
            {"line": "Can I touch it?", "intent": "Request interaction"},
        ]
    }
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(data)
    mock_make_client.return_value = mock_client

    result = plan_call(
        system_prompt="You are Greg.",
        world=WorldState(static=["Greg is a robot head"]),
        history=[],
        goals=Goals(long_term="Experience the universe"),
        plan_request="Plan about the orb",
        plan_model_config=PLAN_CONFIG,
    )

    assert len(result.beats) == 2
    assert result.beats[0].line == "Hey, what's that orb?"
    assert result.beats[1].intent == "Request interaction"


@patch("character_eng.world._make_client")
def test_plan_call_empty_beats(mock_make_client):
    data = {"beats": []}
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(data)
    mock_make_client.return_value = mock_client

    result = plan_call(
        system_prompt="You are Greg.",
        world=None,
        history=[],
        plan_model_config=PLAN_CONFIG,
    )

    assert len(result.beats) == 0


def test_plan_call_no_model_config():
    """plan_call returns empty PlanResult when no plan model config is provided."""
    result = plan_call(
        system_prompt="You are Greg.",
        world=None,
        history=[],
        plan_model_config=None,
    )

    assert len(result.beats) == 0


@patch("character_eng.world._make_client")
def test_plan_call_plan_request_in_context(mock_make_client):
    """plan_request is included in the user message."""
    data = {"beats": [{"line": "Hi", "intent": "Greet"}]}
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(data)
    mock_make_client.return_value = mock_client

    plan_call(
        system_prompt="You are Greg.",
        world=None,
        history=[],
        plan_request="Focus on the mysterious orb",
        plan_model_config=PLAN_CONFIG,
    )

    call_args = mock_client.chat.completions.create.call_args
    user_msg = call_args[1]["messages"][1]["content"]
    assert "PLAN REQUEST" in user_msg
    assert "mysterious orb" in user_msg


@patch("character_eng.world._make_client")
def test_plan_call_reads_plan_system_txt(mock_make_client, tmp_path, monkeypatch):
    """plan_call reads plan_system.txt from disk each call."""
    monkeypatch.setattr(world_mod, "PROMPTS_DIR", tmp_path)
    (tmp_path / "plan_system.txt").write_text("You are planner v1.")

    data = {"beats": [{"line": "Hi", "intent": "Greet"}]}
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(data)
    mock_make_client.return_value = mock_client

    plan_call("sys prompt", world=None, history=[], plan_model_config=PLAN_CONFIG)

    call_args = mock_client.chat.completions.create.call_args
    messages = call_args[1]["messages"]
    assert messages[0]["content"] == "You are planner v1."


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
def test_eval_call_picks_up_file_changes(mock_make_client, tmp_path, monkeypatch):
    """Editing eval_system.txt is picked up on the next call without restart."""
    monkeypatch.setattr(world_mod, "PROMPTS_DIR", tmp_path)

    data = {"thought": "hmm", "gaze": "orb", "expression": "neutral", "script_status": "hold"}
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(data)
    mock_make_client.return_value = mock_client

    # First call with v1
    (tmp_path / "eval_system.txt").write_text("Eval prompt v1")
    eval_call("sys prompt", world=None, history=[], model_config=TEST_CONFIG)
    msgs_v1 = mock_client.chat.completions.create.call_args[1]["messages"]
    assert msgs_v1[0]["content"] == "Eval prompt v1"

    # Edit the file
    (tmp_path / "eval_system.txt").write_text("Eval prompt v2")
    eval_call("sys prompt", world=None, history=[], model_config=TEST_CONFIG)
    msgs_v2 = mock_client.chat.completions.create.call_args[1]["messages"]
    assert msgs_v2[0]["content"] == "Eval prompt v2"
