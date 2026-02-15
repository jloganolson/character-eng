import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from character_eng.world import (
    WorldState,
    WorldUpdate,
    director_call,
    format_narrator_message,
    load_world_state,
)


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

    import character_eng.world as world_mod

    monkeypatch.setattr(world_mod, "CHARACTERS_DIR", tmp_path / "characters")

    ws = load_world_state("test_char")
    assert ws is not None
    assert ws.static == ["Static fact one", "Static fact two"]
    assert ws.dynamic == ["Dynamic fact"]
    assert ws.events == []


def test_load_world_state_no_files(tmp_path, monkeypatch):
    char_dir = tmp_path / "characters" / "test_char"
    char_dir.mkdir(parents=True)

    import character_eng.world as world_mod

    monkeypatch.setattr(world_mod, "CHARACTERS_DIR", tmp_path / "characters")

    ws = load_world_state("test_char")
    assert ws is None


def test_load_world_state_static_only(tmp_path, monkeypatch):
    char_dir = tmp_path / "characters" / "test_char"
    char_dir.mkdir(parents=True)
    (char_dir / "world_static.txt").write_text("Only static\n")

    import character_eng.world as world_mod

    monkeypatch.setattr(world_mod, "CHARACTERS_DIR", tmp_path / "characters")

    ws = load_world_state("test_char")
    assert ws is not None
    assert ws.static == ["Only static"]
    assert ws.dynamic == []


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


# --- director_call (mocked) ---


def _mock_director_response(data: dict):
    """Create a mock genai response with the given JSON data."""
    mock_response = MagicMock()
    mock_response.text = json.dumps(data)
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response
    return mock_client


@patch("character_eng.world.genai.Client")
def test_director_call_basic(mock_client_cls):
    data = {
        "remove_facts": [0],
        "add_facts": ["Orb is on floor"],
        "events": ["The orb rolled off the table"],
    }
    mock_client_cls.return_value = _mock_director_response(data)

    ws = WorldState(
        static=["Greg is a robot head"],
        dynamic=["Orb is on table"],
    )
    update = director_call(ws, "the orb rolls off the table")

    assert update.remove_facts == [0]
    assert update.add_facts == ["Orb is on floor"]
    assert update.events == ["The orb rolled off the table"]

    # Verify the LLM was called
    mock_client_cls.return_value.models.generate_content.assert_called_once()


@patch("character_eng.world.genai.Client")
def test_director_call_no_removals(mock_client_cls):
    data = {
        "remove_facts": [],
        "add_facts": ["A cat entered the room"],
        "events": ["A cat wandered in"],
    }
    mock_client_cls.return_value = _mock_director_response(data)

    ws = WorldState(dynamic=["Orb is on table"])
    update = director_call(ws, "a cat walks in")

    assert update.remove_facts == []
    assert update.add_facts == ["A cat entered the room"]
    assert update.events == ["A cat wandered in"]


@patch("character_eng.world.genai.Client")
def test_director_call_empty_state(mock_client_cls):
    data = {
        "remove_facts": [],
        "add_facts": ["Person is sitting at the desk"],
        "events": ["Someone sat down"],
    }
    mock_client_cls.return_value = _mock_director_response(data)

    ws = WorldState()
    update = director_call(ws, "someone sits down")

    assert update.add_facts == ["Person is sitting at the desk"]
    assert update.events == ["Someone sat down"]
