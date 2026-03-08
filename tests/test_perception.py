import character_eng.perception as perception_mod
from character_eng.perception import (
    PerceptionEvent,
    SimScript,
    load_sim_script,
    process_perception,
)
from character_eng.person import PeopleState
from character_eng.world import WorldState


# --- PerceptionEvent ---


def test_perception_event_creation():
    event = PerceptionEvent(description="A tall man approaches", source="manual")
    assert event.description == "A tall man approaches"
    assert event.source == "manual"
    assert event.timestamp > 0


def test_perception_event_default_source():
    event = PerceptionEvent(description="test")
    assert event.source == "manual"


# --- process_perception ---


def test_process_perception_queues_pending():
    world = WorldState()
    people = PeopleState()
    event = PerceptionEvent(description="A person walks up to the stand")
    person, narrator = process_perception(event, people, world)
    assert person is None
    assert world.pending == ["A person walks up to the stand"]
    assert "[A person walks up to the stand]" == narrator


def test_process_perception_narrator_format():
    world = WorldState()
    people = PeopleState()
    event = PerceptionEvent(description="The visitor picks up a glass")
    _, narrator = process_perception(event, people, world)
    assert narrator == "[The visitor picks up a glass]"


# --- SimScript loading ---


def test_load_sim_script(tmp_path, monkeypatch):
    sims_dir = tmp_path / "characters" / "greg" / "sims"
    sims_dir.mkdir(parents=True)
    (sims_dir / "walkup.sim.txt").write_text("""# A person approaches the stand
0.0 | A figure appears in the distance
2.0 | The figure walks closer — it's a young woman
5.0 | "Hey, is this stand open?"
""")

    monkeypatch.setattr(perception_mod, "CHARACTERS_DIR", tmp_path / "characters")

    script = load_sim_script("greg", "walkup")
    assert script.name == "walkup"
    assert len(script.events) == 3
    assert script.events[0].time_offset == 0.0
    assert script.events[0].description == "A figure appears in the distance"
    assert script.events[1].time_offset == 2.0
    assert script.events[2].description == '"Hey, is this stand open?"'


def test_load_sim_script_skips_comments_and_blanks(tmp_path, monkeypatch):
    sims_dir = tmp_path / "characters" / "greg" / "sims"
    sims_dir.mkdir(parents=True)
    (sims_dir / "test.sim.txt").write_text("""# comment

1.0 | Event one

# another comment
3.0 | Event two
""")

    monkeypatch.setattr(perception_mod, "CHARACTERS_DIR", tmp_path / "characters")

    script = load_sim_script("greg", "test")
    assert len(script.events) == 2
    assert script.events[0].description == "Event one"
    assert script.events[1].description == "Event two"


def test_load_sim_script_missing_file(tmp_path, monkeypatch):
    sims_dir = tmp_path / "characters" / "greg" / "sims"
    sims_dir.mkdir(parents=True)

    monkeypatch.setattr(perception_mod, "CHARACTERS_DIR", tmp_path / "characters")

    import pytest
    with pytest.raises(FileNotFoundError):
        load_sim_script("greg", "nonexistent")


def test_load_walkup_dialogue_script():
    script = load_sim_script("greg", "walkup_dialogue")
    assert len(script.events) == 7
    assert all(event.description.startswith('"') for event in script.events)
