import json
from unittest.mock import MagicMock, patch

import character_eng.scenario as scenario_mod
import character_eng.world as world_mod
from character_eng.person import PeopleState
from character_eng.scenario import (
    DirectorResult,
    ScenarioScript,
    Stage,
    StageExit,
    VisionTrigger,
    match_visual_exit,
    director_call,
    load_scenario_script,
)
from character_eng.world import WorldState

TEST_CONFIG = {
    "name": "Test Model",
    "model": "test-model",
    "base_url": "https://test.example.com/v1",
    "api_key_env": "GEMINI_API_KEY",
    "stream_usage": True,
}


def _mock_openai_response(data: dict):
    mock_message = MagicMock()
    mock_message.content = json.dumps(data)
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


# --- ScenarioScript ---


def test_scenario_script_active_stage():
    s = ScenarioScript(
        name="test",
        stages={"a": Stage("a", "Goal A"), "b": Stage("b", "Goal B")},
        start="a",
        current_stage="a",
    )
    assert s.active_stage.name == "a"
    assert s.active_stage.goal == "Goal A"


def test_scenario_script_active_stage_none():
    s = ScenarioScript(name="test", stages={}, start="", current_stage="missing")
    assert s.active_stage is None


def test_scenario_script_advance_to():
    s = ScenarioScript(
        name="test",
        stages={"a": Stage("a", "Goal A"), "b": Stage("b", "Goal B")},
        start="a",
        current_stage="a",
    )
    assert s.advance_to("b") is True
    assert s.current_stage == "b"
    assert s.active_stage.name == "b"


def test_scenario_script_advance_to_invalid():
    s = ScenarioScript(
        name="test",
        stages={"a": Stage("a", "Goal A")},
        start="a",
        current_stage="a",
    )
    assert s.advance_to("nonexistent") is False
    assert s.current_stage == "a"


def test_scenario_script_render():
    s = ScenarioScript(
        name="test",
        stages={
            "a": Stage("a", "Goal A", exits=[StageExit("someone arrives", "b")]),
            "b": Stage("b", "Goal B"),
        },
        start="a",
        current_stage="a",
    )
    text = s.render()
    assert "→ a: Goal A" in text
    assert "  b: Goal B" in text
    assert "someone arrives" in text
    assert "→ b" in text


def test_scenario_script_show_panel():
    s = ScenarioScript(name="test", stages={"a": Stage("a", "Goal A")}, start="a", current_stage="a")
    panel = s.show()
    assert panel.title == "Scenario"


# --- load_scenario_script ---


def test_stage_exit_label():
    """StageExit has an optional label field."""
    exit_with = StageExit("someone arrives", "b", label="arrives")
    assert exit_with.label == "arrives"
    exit_without = StageExit("someone arrives", "b")
    assert exit_without.label == ""


def test_match_visual_exit_uses_structured_signals():
    class Event:
        def __init__(self, signals):
            self.payload = {"signals": signals}

    scenario = ScenarioScript(
        name="test",
        stages={
            "watching": Stage(
                "watching",
                "Notice someone",
                exits=[
                    StageExit("Someone is approaching", "spotted", visual_signals=["person_visible"]),
                    StageExit("Someone left", "idle", visual_signals=["person_departed"]),
                ],
            ),
            "spotted": Stage("spotted", "Talk"),
            "idle": Stage("idle", "Reset"),
        },
        start="watching",
        current_stage="watching",
    )

    assert match_visual_exit(scenario, [Event(["person_visible"])]) == 0
    assert match_visual_exit(scenario, [Event(["person_departed"])]) == 1
    assert match_visual_exit(scenario, [Event(["water_bottle_visible"])]) == -1


def test_load_scenario_script(tmp_path, monkeypatch):
    char_dir = tmp_path / "characters" / "test_char"
    char_dir.mkdir(parents=True)
    (char_dir / "scenario_script.toml").write_text("""
[guardrails]
always = ["Stay grounded."]
on_first_visible_person = ["Learn their name."]
on_user_leaving = ["Let them go."]

[scenario]
name = "Test Scenario"
start = "intro"

[[stage]]
name = "intro"
goal = "Introduce yourself"

[[stage.exit]]
condition = "The visitor said hello"
goto = "chat"
label = "says hello"

[[stage]]
name = "chat"
goal = "Have a conversation"
""")

    monkeypatch.setattr(scenario_mod, "CHARACTERS_DIR", tmp_path / "characters")

    script = load_scenario_script("test_char")
    assert script is not None
    assert script.name == "Test Scenario"
    assert script.start == "intro"
    assert script.current_stage == "intro"
    assert len(script.stages) == 2
    assert script.stages["intro"].goal == "Introduce yourself"
    assert len(script.stages["intro"].exits) == 1
    assert script.stages["intro"].exits[0].condition == "The visitor said hello"
    assert script.stages["intro"].exits[0].goto == "chat"
    assert script.stages["intro"].exits[0].label == "says hello"
    assert script.stages["intro"].exits[0].visual_signals == []
    assert script.gaze_targets == []
    assert script.guardrails.always == ["Stay grounded."]
    assert script.guardrails.on_first_visible_person == ["Learn their name."]
    assert script.guardrails.on_user_leaving == ["Let them go."]
    assert script.visual_requirements.constant_questions == []
    assert script.visual_requirements.constant_sam_targets == []


def test_load_scenario_script_stage_visual_requirements(tmp_path, monkeypatch):
    char_dir = tmp_path / "characters" / "test_char"
    char_dir.mkdir(parents=True)
    (char_dir / "scenario_script.toml").write_text("""
[visual_requirements]
constant_questions = ["Is there a clock visible?"]
constant_sam_targets = ["clock"]

[scenario]
name = "Test Scenario"
start = "intro"

[[stage]]
name = "intro"
goal = "Introduce yourself"

[stage.visual_requirements]
constant_questions = ["Is the person holding up a phone?"]
constant_sam_targets = ["phone"]
""")

    monkeypatch.setattr(scenario_mod, "CHARACTERS_DIR", tmp_path / "characters")

    script = load_scenario_script("test_char")
    assert script is not None
    assert script.visual_requirements.constant_questions == ["Is there a clock visible?"]
    assert script.visual_requirements.constant_sam_targets == ["clock"]
    assert script.active_stage.visual_requirements.constant_questions == ["Is the person holding up a phone?"]
    assert script.active_stage.visual_requirements.constant_sam_targets == ["phone"]
    merged = script.active_visual_requirements()
    assert merged.constant_questions == ["Is there a clock visible?", "Is the person holding up a phone?"]
    assert merged.constant_sam_targets == ["clock", "phone"]


def test_load_scenario_script_stage_vision_triggers(tmp_path, monkeypatch):
    char_dir = tmp_path / "characters" / "test_char"
    char_dir.mkdir(parents=True)
    (char_dir / "scenario_script.toml").write_text("""
[scenario]
name = "Test Scenario"
start = "intro"

[[vision_trigger]]
signal = "global_clock_visible"
source = "sam3"
label = "clock"
frames = 2

[[stage]]
name = "intro"
goal = "Watch the room"

[[stage.vision_trigger]]
signal = "person_visible_stable"
source = "presence"
frames = 3
description = "A person has stayed visible"
""")

    monkeypatch.setattr(scenario_mod, "CHARACTERS_DIR", tmp_path / "characters")

    script = load_scenario_script("test_char")
    assert script is not None
    assert script.vision_triggers == [
        VisionTrigger(signal="global_clock_visible", source="sam3", label="clock", frames=2)
    ]
    active = script.active_vision_triggers()
    assert active[0].signal == "global_clock_visible"
    assert active[1].signal == "person_visible_stable"
    assert active[1].frames == 3


def test_load_scenario_script_not_found(tmp_path, monkeypatch):
    char_dir = tmp_path / "characters" / "test_char"
    char_dir.mkdir(parents=True)

    monkeypatch.setattr(scenario_mod, "CHARACTERS_DIR", tmp_path / "characters")

    script = load_scenario_script("test_char")
    assert script is None


def test_load_scenario_script_rejects_path_traversal(tmp_path, monkeypatch):
    char_dir = tmp_path / "characters" / "test_char"
    char_dir.mkdir(parents=True)
    monkeypatch.setattr(scenario_mod, "CHARACTERS_DIR", tmp_path / "characters")

    import pytest

    with pytest.raises(ValueError):
        load_scenario_script("test_char", "../secret.toml")


def test_load_scenario_script_uses_manifest_default(tmp_path, monkeypatch):
    char_dir = tmp_path / "characters" / "test_char"
    char_dir.mkdir(parents=True)
    (char_dir / "character_manifest.toml").write_text("""
[files]
default_scenario_script = "custom_script.toml"
""")
    (char_dir / "custom_script.toml").write_text("""
[scenario]
name = "Custom Scenario"
start = "intro"

[[stage]]
name = "intro"
goal = "Open custom"
""")

    monkeypatch.setattr(scenario_mod, "CHARACTERS_DIR", tmp_path / "characters")

    script = load_scenario_script("test_char")
    assert script is not None
    assert script.name == "Custom Scenario"
    assert script.start == "intro"


def test_load_scenario_script_supports_safe_relative_subpath(tmp_path, monkeypatch):
    char_dir = tmp_path / "characters" / "test_char"
    variant_dir = char_dir / "scenarios"
    variant_dir.mkdir(parents=True)
    (variant_dir / "punchy.toml").write_text("""
[scenario]
name = "Punchy Variant"
start = "hook"

[[stage]]
name = "hook"
goal = "Open punchy"
""")

    monkeypatch.setattr(scenario_mod, "CHARACTERS_DIR", tmp_path / "characters")

    script = load_scenario_script("test_char", "scenarios/punchy.toml")
    assert script is not None
    assert script.name == "Punchy Variant"
    assert script.start == "hook"


def test_load_scenario_script_greg():
    """Smoke test: load greg's actual scenario_script.toml."""
    script = load_scenario_script("greg")
    assert script is not None
    assert script.start == "watching"
    assert "watching" in script.stages
    assert "spotted" in script.stages
    assert any("name right away" in item for item in script.guardrails.on_first_visible_person)
    assert "asks their name early" in script.stages["spotted"].goal
    assert len(script.stages["watching"].exits) >= 1
    # Verify labels are parsed
    first_exit = script.stages["watching"].exits[0]
    assert first_exit.label != ""
    assert "person_visible" in first_exit.visual_signals


# --- director_call (mocked) ---


@patch("character_eng.world._make_client")
def test_director_call_hold(mock_make_client, tmp_path, monkeypatch):
    monkeypatch.setattr(world_mod, "PROMPTS_DIR", tmp_path)
    (tmp_path / "director_system.txt").write_text("You are director.")

    data = {"thought": "No exit condition met.", "status": "hold", "exit_index": -1}
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(data)
    mock_make_client.return_value = mock_client

    scenario = ScenarioScript(
        name="test",
        stages={"a": Stage("a", "Goal A", exits=[StageExit("someone arrives", "b")])},
        start="a",
        current_stage="a",
    )

    result = director_call(scenario, world=None, people=None, history=[], model_config=TEST_CONFIG)
    assert result.status == "hold"
    assert result.exit_index == -1


@patch("character_eng.world._make_client")
def test_director_call_advance(mock_make_client, tmp_path, monkeypatch):
    monkeypatch.setattr(world_mod, "PROMPTS_DIR", tmp_path)
    (tmp_path / "director_system.txt").write_text("You are director.")

    data = {"thought": "Someone has arrived.", "status": "advance", "exit_index": 0}
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(data)
    mock_make_client.return_value = mock_client

    scenario = ScenarioScript(
        name="test",
        stages={
            "a": Stage("a", "Goal A", exits=[StageExit("someone arrives", "b")]),
            "b": Stage("b", "Goal B"),
        },
        start="a",
        current_stage="a",
    )

    result = director_call(scenario, world=None, people=None, history=[], model_config=TEST_CONFIG)
    assert result.status == "advance"
    assert result.exit_index == 0


@patch("character_eng.world._make_client")
def test_director_call_coerces_string_exit_index(mock_make_client, tmp_path, monkeypatch):
    monkeypatch.setattr(world_mod, "PROMPTS_DIR", tmp_path)
    (tmp_path / "director_system.txt").write_text("You are director.")

    data = {"thought": "Someone has arrived.", "status": "advance", "exit_index": "0"}
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(data)
    mock_make_client.return_value = mock_client

    scenario = ScenarioScript(
        name="test",
        stages={
            "a": Stage("a", "Goal A", exits=[StageExit("someone arrives", "b")]),
            "b": Stage("b", "Goal B"),
        },
        start="a",
        current_stage="a",
    )

    result = director_call(scenario, world=None, people=None, history=[], model_config=TEST_CONFIG)
    assert result.status == "advance"
    assert result.exit_index == 0


@patch("character_eng.world._make_client")
def test_director_call_context_includes_exits(mock_make_client, tmp_path, monkeypatch):
    monkeypatch.setattr(world_mod, "PROMPTS_DIR", tmp_path)
    (tmp_path / "director_system.txt").write_text("You are director.")

    data = {"thought": "Checking.", "status": "hold", "exit_index": -1}
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(data)
    mock_make_client.return_value = mock_client

    scenario = ScenarioScript(
        name="test",
        stages={"a": Stage("a", "Goal A", exits=[StageExit("visitor leaves", "b")])},
        start="a",
        current_stage="a",
    )

    director_call(scenario, world=None, people=None, history=[], model_config=TEST_CONFIG)

    call_args = mock_client.chat.completions.create.call_args
    user_msg = call_args[1]["messages"][1]["content"]
    assert "CURRENT STAGE" in user_msg
    assert "Goal A" in user_msg
    assert "visitor leaves" in user_msg
    assert "→ b" in user_msg


@patch("character_eng.world._make_client")
def test_director_call_context_includes_people(mock_make_client, tmp_path, monkeypatch):
    monkeypatch.setattr(world_mod, "PROMPTS_DIR", tmp_path)
    (tmp_path / "director_system.txt").write_text("You are director.")

    data = {"thought": "People here.", "status": "hold", "exit_index": -1}
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(data)
    mock_make_client.return_value = mock_client

    scenario = ScenarioScript(
        name="test",
        stages={"a": Stage("a", "Goal A")},
        start="a",
        current_stage="a",
    )

    people = PeopleState()
    people.add_person(name="Alice", presence="present")

    director_call(scenario, world=None, people=people, history=[], model_config=TEST_CONFIG)

    call_args = mock_client.chat.completions.create.call_args
    user_msg = call_args[1]["messages"][1]["content"]
    assert "PEOPLE PRESENT" in user_msg
    assert "Alice" in user_msg


@patch("character_eng.world._make_client")
def test_director_call_reads_prompt_file(mock_make_client, tmp_path, monkeypatch):
    monkeypatch.setattr(world_mod, "PROMPTS_DIR", tmp_path)
    (tmp_path / "director_system.txt").write_text("Director v1")

    data = {"thought": "ok", "status": "hold", "exit_index": -1}
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(data)
    mock_make_client.return_value = mock_client

    scenario = ScenarioScript(
        name="test",
        stages={"a": Stage("a", "Goal A")},
        start="a",
        current_stage="a",
    )

    director_call(scenario, world=None, people=None, history=[], model_config=TEST_CONFIG)

    call_args = mock_client.chat.completions.create.call_args
    messages = call_args[1]["messages"]
    assert messages[0]["content"] == "Director v1"

    # Hot-reload
    (tmp_path / "director_system.txt").write_text("Director v2")
    director_call(scenario, world=None, people=None, history=[], model_config=TEST_CONFIG)
    msgs_v2 = mock_client.chat.completions.create.call_args[1]["messages"]
    assert msgs_v2[0]["content"] == "Director v2"


def test_director_call_no_active_stage():
    """director_call returns hold when no active stage."""
    scenario = ScenarioScript(name="test", stages={}, start="", current_stage="missing")
    result = director_call(scenario, world=None, people=None, history=[], model_config=TEST_CONFIG)
    assert result.status == "hold"
