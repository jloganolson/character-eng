import json
from unittest.mock import MagicMock, patch

import pytest

import character_eng.world as world_mod
from character_eng.person import PeopleState
from character_eng.world import (
    Beat,
    Goals,
    PersonUpdate,
    Script,
    WorldState,
    WorldUpdate,
    _load_prompt_file,
    condition_check_call,
    eval_call,
    expression_call,
    format_narrator_message,
    format_pending_narrator,
    load_beat_guide,
    load_goals,
    load_world_state,
    plan_call,
    reconcile_call,
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


# --- Helper ---


def _ws_with_dynamic(*facts, **kwargs) -> WorldState:
    """Create a WorldState and add_fact for each string."""
    ws = WorldState(**kwargs)
    for fact in facts:
        ws.add_fact(fact)
    return ws


# --- WorldState ---


def test_render_full():
    ws = _ws_with_dynamic(
        "Fliers are on table", "Street is quiet",
        static=["Greg is a robot head"],
        events=["A gust of wind blew some fliers away"],
    )
    text = ws.render()
    assert "Permanent facts:" in text
    assert "- Greg is a robot head" in text
    assert "Current state:" in text
    assert "- Fliers are on table" in text
    assert "Recent events:" in text
    assert "- A gust of wind blew some fliers away" in text


def test_render_with_pending():
    ws = _ws_with_dynamic("Fliers on table")
    ws.add_pending("The fliers blow off the table")
    text = ws.render()
    assert "Pending changes:" in text
    assert "- The fliers blow off the table" in text


# --- add_fact ---


def test_add_fact_returns_id():
    ws = WorldState()
    fid1 = ws.add_fact("First fact")
    fid2 = ws.add_fact("Second fact")
    assert fid1 == "f1"
    assert fid2 == "f2"
    assert ws.dynamic == {"f1": "First fact", "f2": "Second fact"}


def test_add_fact_ids_not_reused_after_removal():
    ws = WorldState()
    ws.add_fact("A")
    ws.add_fact("B")
    ws.dynamic.pop("f1")
    fid3 = ws.add_fact("C")
    assert fid3 == "f3"
    assert "f1" not in ws.dynamic


# --- add_pending / clear_pending ---


def test_add_pending_and_clear():
    ws = WorldState()
    ws.add_pending("Change 1")
    ws.add_pending("Change 2")
    assert ws.pending == ["Change 1", "Change 2"]
    result = ws.clear_pending()
    assert result == ["Change 1", "Change 2"]
    assert ws.pending == []


# --- render_for_reconcile ---


def test_render_for_reconcile():
    ws = _ws_with_dynamic("Fliers on table", "Street is quiet")
    text = ws.render_for_reconcile()
    assert "f1. Fliers on table" in text
    assert "f2. Street is quiet" in text


# --- apply_update ---


def test_apply_update_add_facts():
    ws = _ws_with_dynamic("A", "B")
    update = WorldUpdate(add_facts=["C"])
    ws.apply_update(update)
    assert list(ws.dynamic.values()) == ["A", "B", "C"]


def test_apply_update_remove_facts():
    ws = _ws_with_dynamic("A", "B", "C")
    update = WorldUpdate(remove_facts=["f2"])
    ws.apply_update(update)
    assert list(ws.dynamic.values()) == ["A", "C"]


def test_apply_update_remove_multiple():
    ws = _ws_with_dynamic("A", "B", "C", "D")
    update = WorldUpdate(remove_facts=["f1", "f3"])
    ws.apply_update(update)
    assert list(ws.dynamic.values()) == ["B", "D"]


def test_apply_update_events():
    ws = WorldState()
    update = WorldUpdate(events=["Something happened"])
    ws.apply_update(update)
    assert ws.events == ["Something happened"]


def test_apply_update_combined():
    ws = _ws_with_dynamic("Fliers on table", "Street is quiet")
    update = WorldUpdate(
        remove_facts=["f1"],
        add_facts=["Fliers on ground"],
        events=["The fliers blew off the table"],
    )
    ws.apply_update(update)
    assert list(ws.dynamic.values()) == ["Street is quiet", "Fliers on ground"]
    assert ws.events == ["The fliers blew off the table"]


def test_apply_update_invalid_id_ignored():
    ws = _ws_with_dynamic("A")
    update = WorldUpdate(remove_facts=["f99"])
    ws.apply_update(update)
    assert list(ws.dynamic.values()) == ["A"]


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
    assert list(ws.dynamic.values()) == ["Dynamic fact"]
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
    assert ws.dynamic == {}


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
    s.advance()
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


def test_script_render_with_condition():
    s = Script(beats=[
        Beat("Hey", "Greet", condition="A visitor is present"),
        Beat("What's that?", "Ask about fliers"),
    ])
    text = s.render()
    assert 'IF: A visitor is present' in text
    assert 'IF:' not in text.split('\n')[1]


# --- Goals ---


def test_goals_render_full():
    g = Goals(long_term="Experience the universe")
    text = g.render()
    assert "Long-term goal: Experience the universe" in text


# --- load_goals ---


def test_load_goals_from_character_txt(tmp_path, monkeypatch):
    char_dir = tmp_path / "characters" / "test_char"
    char_dir.mkdir(parents=True)
    (char_dir / "character.txt").write_text(
        "You are Test.\n\nLong-term goal: Experience the universe\nShort-term goal: Hand out fliers\n"
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
    update = WorldUpdate(events=["The fliers blew off the table"])
    msg = format_narrator_message(update)
    assert msg == "[The fliers blew off the table]"


def test_format_narrator_message_events_and_facts():
    update = WorldUpdate(
        add_facts=["Fliers on ground"],
        events=["The fliers fell"],
    )
    msg = format_narrator_message(update)
    assert msg.startswith("[")
    assert msg.endswith("]")
    assert "The fliers fell" in msg
    assert "Fliers on ground" in msg


# --- format_pending_narrator ---


def test_format_pending_narrator():
    msg = format_pending_narrator("The fliers blow off the table")
    assert msg == "[The fliers blow off the table]"


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


# --- reconcile_call (mocked) ---


@patch("character_eng.world._make_client")
def test_reconcile_call_basic(mock_make_client):
    data = {
        "remove_facts": ["f1"],
        "add_facts": ["Fliers on ground"],
        "events": ["The fliers blew off the table"],
    }
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(data)
    mock_make_client.return_value = mock_client

    ws = _ws_with_dynamic("Fliers on table", static=["Greg is a robot head"])
    update = reconcile_call(ws, ["the fliers blow off the table"], TEST_CONFIG)

    assert update.remove_facts == ["f1"]
    assert update.add_facts == ["Fliers on ground"]
    assert update.events == ["The fliers blew off the table"]

    mock_client.chat.completions.create.assert_called_once()


@patch("character_eng.world._make_client")
def test_reconcile_call_no_removals(mock_make_client):
    data = {
        "remove_facts": [],
        "add_facts": ["A cat entered the room"],
        "events": ["A cat wandered in"],
    }
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(data)
    mock_make_client.return_value = mock_client

    ws = _ws_with_dynamic("Fliers on table")
    update = reconcile_call(ws, ["a cat walks in"], TEST_CONFIG)

    assert update.remove_facts == []
    assert update.add_facts == ["A cat entered the room"]
    assert update.events == ["A cat wandered in"]


@patch("character_eng.world._make_client")
def test_reconcile_call_empty_state(mock_make_client):
    data = {
        "remove_facts": [],
        "add_facts": ["Person is sitting at the desk"],
        "events": ["Someone sat down"],
    }
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(data)
    mock_make_client.return_value = mock_client

    ws = WorldState()
    update = reconcile_call(ws, ["someone sits down"], TEST_CONFIG)

    assert update.add_facts == ["Person is sitting at the desk"]
    assert update.events == ["Someone sat down"]


@patch("character_eng.world._make_client")
def test_reconcile_call_multiple_pending(mock_make_client):
    data = {
        "remove_facts": ["f1"],
        "add_facts": ["Fliers on ground", "Water jug tipped over"],
        "events": ["The fliers blew away and the jug tipped"],
    }
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(data)
    mock_make_client.return_value = mock_client

    ws = _ws_with_dynamic("Fliers on table")
    update = reconcile_call(ws, ["the fliers blow away", "the water jug tips over"], TEST_CONFIG)

    assert update.remove_facts == ["f1"]
    assert len(update.add_facts) == 2

    call_args = mock_client.chat.completions.create.call_args
    user_msg = call_args[1]["messages"][1]["content"]
    assert "the fliers blow away" in user_msg
    assert "the water jug tips over" in user_msg


@patch("character_eng.world._make_client")
def test_reconcile_call_reads_reconcile_system_txt(mock_make_client, tmp_path, monkeypatch):
    """reconcile_call reads reconcile_system.txt from disk each call."""
    monkeypatch.setattr(world_mod, "PROMPTS_DIR", tmp_path)
    (tmp_path / "reconcile_system.txt").write_text("You are reconciler v1.")

    data = {"remove_facts": [], "add_facts": [], "events": ["thing"]}
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(data)
    mock_make_client.return_value = mock_client

    reconcile_call(WorldState(), ["test"], TEST_CONFIG)

    call_args = mock_client.chat.completions.create.call_args
    messages = call_args[1]["messages"]
    assert messages[0]["content"] == "You are reconciler v1."


@patch("character_eng.world._make_client")
def test_reconcile_call_ids_in_context(mock_make_client):
    """ID-tagged facts appear in the user message."""
    data = {"remove_facts": [], "add_facts": [], "events": ["thing"]}
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(data)
    mock_make_client.return_value = mock_client

    ws = _ws_with_dynamic("Fliers on table", "Street is quiet")
    reconcile_call(ws, ["test change"], TEST_CONFIG)

    call_args = mock_client.chat.completions.create.call_args
    user_msg = call_args[1]["messages"][1]["content"]
    assert "f1. Fliers on table" in user_msg
    assert "f2. Street is quiet" in user_msg


@patch("character_eng.world._make_client")
def test_reconcile_call_picks_up_file_changes(mock_make_client, tmp_path, monkeypatch):
    """Editing reconcile_system.txt is picked up on the next call without restart."""
    monkeypatch.setattr(world_mod, "PROMPTS_DIR", tmp_path)

    data = {"remove_facts": [], "add_facts": [], "events": ["thing"]}
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(data)
    mock_make_client.return_value = mock_client

    (tmp_path / "reconcile_system.txt").write_text("Reconcile prompt v1")
    reconcile_call(WorldState(), ["test"], TEST_CONFIG)
    msgs_v1 = mock_client.chat.completions.create.call_args[1]["messages"]
    assert msgs_v1[0]["content"] == "Reconcile prompt v1"

    (tmp_path / "reconcile_system.txt").write_text("Reconcile prompt v2")
    reconcile_call(WorldState(), ["test"], TEST_CONFIG)
    msgs_v2 = mock_client.chat.completions.create.call_args[1]["messages"]
    assert msgs_v2[0]["content"] == "Reconcile prompt v2"


# --- reconcile_call with people ---


@patch("character_eng.world._make_client")
def test_reconcile_call_with_people_context(mock_make_client):
    """People state is included in the user message when provided."""
    data = {"remove_facts": [], "add_facts": [], "events": ["thing"]}
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(data)
    mock_make_client.return_value = mock_client

    ws = _ws_with_dynamic("Fliers on table")
    people = PeopleState()
    p = people.add_person(name="Alice", presence="present")
    p.add_fact("Wearing a hat")

    reconcile_call(ws, ["test change"], TEST_CONFIG, people=people)

    call_args = mock_client.chat.completions.create.call_args
    user_msg = call_args[1]["messages"][1]["content"]
    assert "Current people:" in user_msg
    assert "Alice" in user_msg
    assert "p1f1. Wearing a hat" in user_msg


@patch("character_eng.world._make_client")
def test_reconcile_call_without_people_backward_compat(mock_make_client):
    """reconcile_call without people param works as before."""
    data = {"remove_facts": [], "add_facts": ["New fact"], "events": ["thing"]}
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(data)
    mock_make_client.return_value = mock_client

    ws = WorldState()
    update = reconcile_call(ws, ["test"], TEST_CONFIG)

    assert update.add_facts == ["New fact"]
    assert update.person_updates == []

    call_args = mock_client.chat.completions.create.call_args
    user_msg = call_args[1]["messages"][1]["content"]
    assert "Current people:" not in user_msg


@patch("character_eng.world._make_client")
def test_reconcile_call_parses_person_updates(mock_make_client):
    """person_updates in LLM response are parsed into PersonUpdate objects."""
    data = {
        "remove_facts": [],
        "add_facts": [],
        "events": ["Alice arrived"],
        "person_updates": [
            {
                "person_id": "p1",
                "remove_facts": [],
                "add_facts": ["Carrying a backpack"],
                "set_name": "Alice",
                "set_presence": "present",
            }
        ],
    }
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(data)
    mock_make_client.return_value = mock_client

    ws = WorldState()
    update = reconcile_call(ws, ["someone arrives"], TEST_CONFIG)

    assert len(update.person_updates) == 1
    pu = update.person_updates[0]
    assert pu.person_id == "p1"
    assert pu.add_facts == ["Carrying a backpack"]
    assert pu.set_name == "Alice"
    assert pu.set_presence == "present"


@patch("character_eng.world._make_client")
def test_reconcile_call_no_person_updates_key(mock_make_client):
    """Missing person_updates key in response defaults to empty list."""
    data = {"remove_facts": [], "add_facts": [], "events": ["thing"]}
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(data)
    mock_make_client.return_value = mock_client

    update = reconcile_call(WorldState(), ["test"], TEST_CONFIG)
    assert update.person_updates == []


# --- eval_call (mocked) ---


@patch("character_eng.world._make_client")
def test_eval_call_hold(mock_make_client):
    data = {
        "thought": "I should wait for them to respond.",
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


@patch("character_eng.world._make_client")
def test_eval_call_advance(mock_make_client):
    data = {
        "thought": "Good, they engaged with my question.",
        "script_status": "advance",
    }
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(data)
    mock_make_client.return_value = mock_client

    script = Script(beats=[Beat("Want a flier?", "Offer flier"), Beat("It's got a coupon!", "Pitch coupon")])
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

    data = {"thought": "hmm", "script_status": "hold"}
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(data)
    mock_make_client.return_value = mock_client

    eval_call("sys prompt", world=None, history=[], model_config=TEST_CONFIG)

    call_args = mock_client.chat.completions.create.call_args
    messages = call_args[1]["messages"]
    assert messages[0]["content"] == "You are eval v1."


@patch("character_eng.world._make_client")
def test_eval_call_picks_up_file_changes(mock_make_client, tmp_path, monkeypatch):
    """Editing eval_system.txt is picked up on the next call without restart."""
    monkeypatch.setattr(world_mod, "PROMPTS_DIR", tmp_path)

    data = {"thought": "hmm", "script_status": "hold"}
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(data)
    mock_make_client.return_value = mock_client

    (tmp_path / "eval_system.txt").write_text("Eval prompt v1")
    eval_call("sys prompt", world=None, history=[], model_config=TEST_CONFIG)
    msgs_v1 = mock_client.chat.completions.create.call_args[1]["messages"]
    assert msgs_v1[0]["content"] == "Eval prompt v1"

    (tmp_path / "eval_system.txt").write_text("Eval prompt v2")
    eval_call("sys prompt", world=None, history=[], model_config=TEST_CONFIG)
    msgs_v2 = mock_client.chat.completions.create.call_args[1]["messages"]
    assert msgs_v2[0]["content"] == "Eval prompt v2"


# --- eval_call with people and stage_goal ---


@patch("character_eng.world._make_client")
def test_eval_call_people_in_context(mock_make_client):
    """People state is included in the user message when provided."""
    data = {"thought": "ok", "script_status": "hold"}
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(data)
    mock_make_client.return_value = mock_client

    people = PeopleState()
    people.add_person(name="Alice", presence="present")

    eval_call("sys prompt", world=None, history=[], model_config=TEST_CONFIG, people=people)

    call_args = mock_client.chat.completions.create.call_args
    user_msg = call_args[1]["messages"][1]["content"]
    assert "PEOPLE PRESENT" in user_msg
    assert "Alice" in user_msg


@patch("character_eng.world._make_client")
def test_eval_call_stage_goal_in_context(mock_make_client):
    """Stage goal is included in the user message when provided."""
    data = {"thought": "ok", "script_status": "hold"}
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(data)
    mock_make_client.return_value = mock_client

    eval_call("sys prompt", world=None, history=[], model_config=TEST_CONFIG, stage_goal="Sell lemonade")

    call_args = mock_client.chat.completions.create.call_args
    user_msg = call_args[1]["messages"][1]["content"]
    assert "CURRENT STAGE GOAL" in user_msg
    assert "Sell lemonade" in user_msg


@patch("character_eng.world._make_client")
def test_eval_call_no_people_no_stage_goal(mock_make_client):
    """Without people or stage_goal, those sections don't appear."""
    data = {"thought": "ok", "script_status": "hold"}
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(data)
    mock_make_client.return_value = mock_client

    eval_call("sys prompt", world=None, history=[], model_config=TEST_CONFIG)

    call_args = mock_client.chat.completions.create.call_args
    user_msg = call_args[1]["messages"][1]["content"]
    assert "PEOPLE PRESENT" not in user_msg
    assert "CURRENT STAGE GOAL" not in user_msg


# --- condition_check_call (mocked) ---


@patch("character_eng.world._make_client")
def test_condition_check_call_met(mock_make_client):
    data = {"met": True, "idle": ""}
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(data)
    mock_make_client.return_value = mock_client

    result = condition_check_call(
        condition="A visitor is present",
        system_prompt="You are Greg.",
        world=_ws_with_dynamic("A visitor is standing nearby", static=["Greg is a robot head"]),
        history=[{"role": "user", "content": "Hello"}],
        model_config=TEST_CONFIG,
    )

    assert result.met is True
    assert result.idle == ""


@patch("character_eng.world._make_client")
def test_condition_check_call_not_met(mock_make_client):
    data = {"met": False, "idle": "*stares at the ceiling*"}
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(data)
    mock_make_client.return_value = mock_client

    result = condition_check_call(
        condition="A visitor is present",
        system_prompt="You are Greg.",
        world=WorldState(static=["Greg is a robot head"]),
        history=[],
        model_config=TEST_CONFIG,
    )

    assert result.met is False
    assert result.idle == "*stares at the ceiling*"


@patch("character_eng.world._make_client")
def test_condition_check_call_condition_in_context(mock_make_client):
    """Condition string is included in the user message."""
    data = {"met": True, "idle": ""}
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(data)
    mock_make_client.return_value = mock_client

    condition_check_call(
        condition="The customer asked about the fliers",
        system_prompt="You are Greg.",
        world=None,
        history=[],
        model_config=TEST_CONFIG,
    )

    call_args = mock_client.chat.completions.create.call_args
    user_msg = call_args[1]["messages"][1]["content"]
    assert "CONDITION TO CHECK" in user_msg
    assert "The customer asked about the fliers" in user_msg


@patch("character_eng.world._make_client")
def test_condition_check_call_reads_condition_system_txt(mock_make_client, tmp_path, monkeypatch):
    """condition_check_call reads condition_system.txt from disk."""
    monkeypatch.setattr(world_mod, "PROMPTS_DIR", tmp_path)
    (tmp_path / "condition_system.txt").write_text("You are condition checker v1.")

    data = {"met": True, "idle": ""}
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(data)
    mock_make_client.return_value = mock_client

    condition_check_call("test condition", "sys prompt", world=None, history=[], model_config=TEST_CONFIG)

    call_args = mock_client.chat.completions.create.call_args
    messages = call_args[1]["messages"]
    assert messages[0]["content"] == "You are condition checker v1."


# --- plan_call (mocked) ---


@patch("character_eng.world._make_client")
def test_plan_call_basic_beats(mock_make_client):
    data = {
        "beats": [
            {"line": "Want a flier? Got a coupon on it.", "intent": "Offer flier"},
            {"line": "It's for a free cup of water!", "intent": "Pitch coupon"},
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
        plan_request="Get the customer to take a flier",
        plan_model_config=PLAN_CONFIG,
    )

    assert len(result.beats) == 2
    assert result.beats[0].line == "Want a flier? Got a coupon on it."
    assert result.beats[1].intent == "Pitch coupon"


@patch("character_eng.world._make_client")
def test_plan_call_beats_with_condition(mock_make_client):
    """Beats with condition field are parsed correctly."""
    data = {
        "beats": [
            {"line": "Hey!", "intent": "Greet", "condition": "A customer is present"},
            {"line": "Want a flier?", "intent": "Offer flier", "condition": ""},
        ]
    }
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(data)
    mock_make_client.return_value = mock_client

    result = plan_call(
        system_prompt="You are Greg.",
        world=None,
        history=[],
        plan_model_config=PLAN_CONFIG,
    )

    assert len(result.beats) == 2
    assert result.beats[0].condition == "A customer is present"
    assert result.beats[1].condition == ""


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
        plan_request="Focus on handing out fliers",
        plan_model_config=PLAN_CONFIG,
    )

    call_args = mock_client.chat.completions.create.call_args
    user_msg = call_args[1]["messages"][1]["content"]
    assert "PLAN REQUEST" in user_msg
    assert "handing out fliers" in user_msg


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


# --- plan_call with people and stage_goal ---


@patch("character_eng.world._make_client")
def test_plan_call_people_in_context(mock_make_client):
    """People state is included in the user message when provided."""
    data = {"beats": [{"line": "Hi Alice!", "intent": "Greet"}]}
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(data)
    mock_make_client.return_value = mock_client

    people = PeopleState()
    people.add_person(name="Alice", presence="present")

    plan_call("sys prompt", world=None, history=[], plan_model_config=PLAN_CONFIG, people=people)

    call_args = mock_client.chat.completions.create.call_args
    user_msg = call_args[1]["messages"][1]["content"]
    assert "PEOPLE PRESENT" in user_msg
    assert "Alice" in user_msg


@patch("character_eng.world._make_client")
def test_plan_call_stage_goal_in_context(mock_make_client):
    """Stage goal is included in the user message when provided."""
    data = {"beats": [{"line": "Hi!", "intent": "Greet"}]}
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(data)
    mock_make_client.return_value = mock_client

    plan_call("sys prompt", world=None, history=[], plan_model_config=PLAN_CONFIG, stage_goal="Sell lemonade")

    call_args = mock_client.chat.completions.create.call_args
    user_msg = call_args[1]["messages"][1]["content"]
    assert "CURRENT STAGE GOAL" in user_msg
    assert "Sell lemonade" in user_msg


@patch("character_eng.world._make_client")
def test_plan_call_no_people_no_stage_goal(mock_make_client):
    """Without people or stage_goal, those sections don't appear."""
    data = {"beats": [{"line": "Hi!", "intent": "Greet"}]}
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(data)
    mock_make_client.return_value = mock_client

    plan_call("sys prompt", world=None, history=[], plan_model_config=PLAN_CONFIG)

    call_args = mock_client.chat.completions.create.call_args
    user_msg = call_args[1]["messages"][1]["content"]
    assert "PEOPLE PRESENT" not in user_msg
    assert "CURRENT STAGE GOAL" not in user_msg


# --- _load_prompt_file ---


def test_load_prompt_file_reads_from_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(world_mod, "PROMPTS_DIR", tmp_path)
    (tmp_path / "test_prompt.txt").write_text("Hello from disk")
    assert _load_prompt_file("test_prompt.txt") == "Hello from disk"


def test_load_prompt_file_missing_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(world_mod, "PROMPTS_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        _load_prompt_file("nonexistent.txt")


# --- load_beat_guide ---


def test_load_beat_guide_substitutes_placeholders(tmp_path, monkeypatch):
    monkeypatch.setattr(world_mod, "PROMPTS_DIR", tmp_path)
    (tmp_path / "beat_guide.txt").write_text("Intent: {intent}\nLine: {line}")
    result = load_beat_guide("Offer a flier", "Want a flier? It's got a coupon.")
    assert "Intent: Offer a flier" in result
    assert "Line: Want a flier? It's got a coupon." in result


def test_load_beat_guide_reads_from_disk_each_call(tmp_path, monkeypatch):
    monkeypatch.setattr(world_mod, "PROMPTS_DIR", tmp_path)
    (tmp_path / "beat_guide.txt").write_text("v1: {intent}")
    result1 = load_beat_guide("test", "line")
    assert "v1: test" in result1

    (tmp_path / "beat_guide.txt").write_text("v2: {intent}")
    result2 = load_beat_guide("test", "line")
    assert "v2: test" in result2


def test_load_beat_guide_missing_file_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(world_mod, "PROMPTS_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        load_beat_guide("intent", "line")


# --- expression_call (mocked) ---


@patch("character_eng.world._make_client")
def test_expression_call_basic(mock_make_client):
    data = {"gaze": "customer", "expression": "happy"}
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(data)
    mock_make_client.return_value = mock_client

    result = expression_call("Hey there, welcome!", TEST_CONFIG)

    assert result.gaze == "customer"
    assert result.expression == "happy"


@patch("character_eng.world._make_client")
def test_expression_call_default_expression(mock_make_client):
    """Missing expression defaults to 'neutral'."""
    data = {"gaze": "stand"}
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(data)
    mock_make_client.return_value = mock_client

    result = expression_call("...", TEST_CONFIG)

    assert result.gaze == "stand"
    assert result.expression == "neutral"


@patch("character_eng.world._make_client")
def test_expression_call_custom_gaze_targets(mock_make_client):
    """Custom gaze targets are injected into the prompt."""
    data = {"gaze": "robot", "expression": "curious"}
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(data)
    mock_make_client.return_value = mock_client

    result = expression_call("Look at that robot!", TEST_CONFIG, gaze_targets=["robot", "table"])

    assert result.gaze == "robot"
    mock_client.chat.completions.create.assert_called_once()
    call_args = mock_client.chat.completions.create.call_args
    system_msg = call_args[1]["messages"][0]["content"]
    assert "robot, table" in system_msg


@patch("character_eng.world._make_client")
def test_expression_call_reads_expression_system_txt(mock_make_client, tmp_path, monkeypatch):
    """expression_call reads expression_system.txt from disk."""
    monkeypatch.setattr(world_mod, "PROMPTS_DIR", tmp_path)
    (tmp_path / "expression_system.txt").write_text("Expression controller v1. {gaze_targets}")

    data = {"gaze": "person", "expression": "neutral"}
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(data)
    mock_make_client.return_value = mock_client

    expression_call("Hello", TEST_CONFIG)

    call_args = mock_client.chat.completions.create.call_args
    messages = call_args[1]["messages"]
    assert "Expression controller v1." in messages[0]["content"]
