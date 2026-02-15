import character_eng.prompts as prompts_mod
from character_eng.prompts import list_characters, load_prompt
from character_eng.world import WorldState


def _setup_char(tmp_path, name, prompt_txt, **extra_files):
    """Create a minimal character dir under tmp_path and monkeypatch paths."""
    prompts_dir = tmp_path / "prompts"
    char_dir = prompts_dir / "characters" / name
    char_dir.mkdir(parents=True)
    (char_dir / "prompt.txt").write_text(prompt_txt)
    for filename, content in extra_files.items():
        (char_dir / filename).write_text(content)
    return prompts_dir, char_dir


def test_load_prompt_with_world_state(tmp_path, monkeypatch):
    prompts_dir, char_dir = _setup_char(
        tmp_path,
        "test_char",
        "Rules: {{global_rules}}\nWorld: {{world}}\nChar: {{character}}",
        **{"character.txt": "I am test character"},
    )
    (prompts_dir / "global_rules.txt").write_text("Be nice")

    monkeypatch.setattr(prompts_mod, "PROMPTS_DIR", prompts_dir)
    monkeypatch.setattr(prompts_mod, "CHARACTERS_DIR", prompts_dir / "characters")

    ws = WorldState(static=["Sky is blue"], dynamic=["Sun is out"])
    result = load_prompt("test_char", world_state=ws)

    assert "Be nice" in result
    assert "I am test character" in result
    assert "Sky is blue" in result
    assert "Sun is out" in result


def test_load_prompt_without_world_state(tmp_path, monkeypatch):
    prompts_dir, char_dir = _setup_char(
        tmp_path,
        "test_char",
        "World: {{world}}\nChar: {{character}}",
        **{"character.txt": "I am test"},
    )

    monkeypatch.setattr(prompts_mod, "PROMPTS_DIR", prompts_dir)
    monkeypatch.setattr(prompts_mod, "CHARACTERS_DIR", prompts_dir / "characters")

    result = load_prompt("test_char")
    assert "World: \n" in result
    assert "I am test" in result


def test_load_prompt_no_world_macro(tmp_path, monkeypatch):
    """Templates without {{world}} still work fine."""
    prompts_dir, char_dir = _setup_char(
        tmp_path,
        "test_char",
        "Char: {{character}}",
        **{"character.txt": "Old style char"},
    )

    monkeypatch.setattr(prompts_mod, "PROMPTS_DIR", prompts_dir)
    monkeypatch.setattr(prompts_mod, "CHARACTERS_DIR", prompts_dir / "characters")

    ws = WorldState(static=["Fact"])
    result = load_prompt("test_char", world_state=ws)
    assert "Old style char" in result
    # world macro not in template, so world content not in output
    assert "Fact" not in result


def test_unknown_macro_left_intact(tmp_path, monkeypatch):
    prompts_dir, char_dir = _setup_char(
        tmp_path,
        "test_char",
        "{{unknown_thing}}",
    )

    monkeypatch.setattr(prompts_mod, "PROMPTS_DIR", prompts_dir)
    monkeypatch.setattr(prompts_mod, "CHARACTERS_DIR", prompts_dir / "characters")

    result = load_prompt("test_char")
    assert "{{unknown_thing}}" in result


def test_list_characters(tmp_path, monkeypatch):
    prompts_dir = tmp_path / "prompts"
    chars_dir = prompts_dir / "characters"

    for name in ["beta", "alpha"]:
        d = chars_dir / name
        d.mkdir(parents=True)
        (d / "prompt.txt").write_text("hi")

    # dir without prompt.txt should be excluded
    (chars_dir / "empty_dir").mkdir(parents=True)

    monkeypatch.setattr(prompts_mod, "CHARACTERS_DIR", chars_dir)

    result = list_characters()
    assert result == ["alpha", "beta"]
