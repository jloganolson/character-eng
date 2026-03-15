from __future__ import annotations

import time

from character_eng.chat import ChatSession
from character_eng.prompts import mutable_prompt_inventory, prompt_source_paths, prompt_source_signature
from character_eng.vision.focus import visual_focus_call


def test_prompt_source_signature_tracks_file_changes(tmp_path):
    prompts_dir = tmp_path / "prompts"
    characters_dir = prompts_dir / "characters"
    character_dir = characters_dir / "greg"
    character_dir.mkdir(parents=True)
    (prompts_dir / "global_rules.txt").write_text("rules v1", encoding="utf-8")
    (character_dir / "prompt.txt").write_text("{{character}}", encoding="utf-8")
    character_file = character_dir / "character.txt"
    character_file.write_text("greg v1", encoding="utf-8")

    paths = prompt_source_paths("greg", prompts_dir=prompts_dir, characters_dir=characters_dir)
    assert character_file in paths

    before = prompt_source_signature("greg", prompts_dir=prompts_dir, characters_dir=characters_dir)
    time.sleep(0.001)
    character_file.write_text("greg v2", encoding="utf-8")
    after = prompt_source_signature("greg", prompts_dir=prompts_dir, characters_dir=characters_dir)

    assert before[str(character_file)] != after[str(character_file)]


def test_replace_system_prompt_preserves_turn_history(monkeypatch):
    monkeypatch.setattr("character_eng.chat._make_chat_client", lambda model_config: object())
    session = ChatSession("old prompt", {"model": "test", "name": "test", "base_url": "", "api_key_env": "TEST_KEY", "stream_usage": False})
    session._messages.extend([
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ])

    session.replace_system_prompt("new prompt")

    assert session.system_prompt == "new prompt"
    assert session.get_history()[0] == {"role": "system", "content": "new prompt"}
    assert session.get_history()[1:] == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]


def test_visual_focus_uses_file_backed_constant_questions(monkeypatch, tmp_path):
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir(parents=True)
    (prompts_dir / "visual_focus.txt").write_text("focus prompt", encoding="utf-8")
    (prompts_dir / "vision_constant_questions.txt").write_text(
        "What is the nearest person wearing?\nWhat are they holding?\n",
        encoding="utf-8",
    )
    (prompts_dir / "vision_constant_sam_targets.txt").write_text("phone\nperson\n", encoding="utf-8")

    monkeypatch.setattr("character_eng.vision.focus.PROMPTS_DIR", prompts_dir)
    result = visual_focus_call(
        beat=None,
        stage_goal="",
        thought="",
        world=None,
        people=None,
        model_config={},
        scenario=None,
    )

    assert result.constant_questions == [
        "What is the nearest person wearing?",
        "What are they holding?",
    ]
    assert result.constant_sam_targets == ["phone", "person"]


def test_mutable_prompt_inventory_reports_apply_modes(tmp_path):
    prompts_dir = tmp_path / "prompts"
    characters_dir = prompts_dir / "characters"
    character_dir = characters_dir / "greg"
    character_dir.mkdir(parents=True)
    (prompts_dir / "prompt_registry.toml").write_text(
        '[prompts]\nvisual_focus = "visual_focus.txt"\nvision_constant_questions = "vision_constant_questions.txt"\n',
        encoding="utf-8",
    )
    (prompts_dir / "visual_focus.txt").write_text("focus", encoding="utf-8")
    (prompts_dir / "vision_constant_questions.txt").write_text("q1\n", encoding="utf-8")
    (prompts_dir / "global_rules.txt").write_text("rules", encoding="utf-8")
    (character_dir / "prompt.txt").write_text("{{character}}", encoding="utf-8")
    (character_dir / "character.txt").write_text("greg", encoding="utf-8")
    (character_dir / "scenario_script.toml").write_text("[setup]\npremise='hi'\n", encoding="utf-8")

    inventory = mutable_prompt_inventory("greg", prompts_dir=prompts_dir, characters_dir=characters_dir)

    by_label = {item["label"]: item for item in inventory}
    assert by_label["Character prompt template"]["apply_mode"] == "refresh_or_continue"
    assert by_label["visual focus"]["apply_mode"] == "next_vision_cycle"
    assert by_label["Scenario script"]["apply_mode"] == "restart_required"
