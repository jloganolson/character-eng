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


def test_process_perception_applies_structured_person_presence():
    world = WorldState()
    people = PeopleState()
    event = PerceptionEvent(
        description="Person 1 appeared in view",
        source="visual",
        kind="person_presence",
        payload={"identity": "Person 1", "presence": "present", "change": "appeared", "signals": ["person_visible"]},
    )
    _, narrator = process_perception(event, people, world)
    assert narrator == ""
    assert world.pending == []
    present = people.present_people()
    assert len(present) == 1
    assert present[0].name == "Person 1"
    assert present[0].presence == "present"
    assert world.events == []


def test_process_perception_ignores_visual_trigger_narration():
    world = WorldState()
    people = PeopleState()
    event = PerceptionEvent(
        description="A person has stayed visible for more than a moment",
        source="visual",
        kind="vision_trigger",
        payload={"signals": ["person_visible_stable"]},
    )

    _, narrator = process_perception(event, people, world)

    assert narrator == ""
    assert world.pending == []
    assert world.events == []


def test_process_perception_applies_vision_state_update_directly():
    world = WorldState()
    existing_id = world.add_fact("The room is quiet")
    people = PeopleState()
    person = people.add_person(name="Visitor", presence="present")
    event = PerceptionEvent(
        description="Visitor is wearing a blue jacket",
        source="visual",
        kind="vision_state_update",
        payload={
            "remove_facts": [existing_id],
            "add_facts": ["A wall clock is visible"],
            "events": ["The visitor is checking a phone"],
            "person_updates": [
                {
                    "person_id": person.person_id,
                    "add_facts": ["Wearing a blue jacket"],
                    "set_presence": "present",
                }
            ],
        },
    )

    _, narrator = process_perception(event, people, world)

    assert narrator == "[Visitor is wearing a blue jacket]"
    assert existing_id not in world.dynamic
    assert "A wall clock is visible" in world.dynamic.values()
    assert world.events[-1] == "The visitor is checking a phone"
    assert "Wearing a blue jacket" in person.facts.values()


def test_process_perception_drops_generic_visual_presence_claims():
    world = WorldState()
    people = PeopleState()
    event = PerceptionEvent(
        description="A person is now visible in the room",
        source="visual",
        kind="visual_claim",
        payload={"signals": ["person_visible"]},
    )

    _, narrator = process_perception(event, people, world)

    assert narrator == ""
    assert world.pending == []


def test_process_perception_filters_generic_presence_events_from_vision_state_update():
    world = WorldState()
    people = PeopleState()
    event = PerceptionEvent(
        description="Updated room description",
        source="visual",
        kind="vision_state_update",
        payload={
            "add_facts": ["A wall clock is visible"],
            "events": ["A person appeared in view", "The visitor is checking a phone"],
        },
    )

    _, narrator = process_perception(event, people, world)

    assert narrator == "[Updated room description]"
    assert world.events == ["The visitor is checking a phone"]


def test_process_perception_does_not_reapply_already_committed_vision_update():
    world = WorldState()
    people = PeopleState()
    person = people.add_person(name="Visitor", presence="present")
    person.add_fact("Wearing a blue jacket")
    event = PerceptionEvent(
        description="Visitor is wearing a blue jacket",
        source="visual",
        kind="vision_state_update",
        payload={
            "already_applied": True,
            "person_updates": [
                {
                    "person_id": person.person_id,
                    "add_facts": ["Wearing a blue jacket"],
                }
            ],
        },
    )

    _, narrator = process_perception(event, people, world)

    assert narrator == "[Visitor is wearing a blue jacket]"
    assert list(person.facts.values()) == ["Wearing a blue jacket"]


def test_process_perception_ignores_null_person_fields_in_vision_update():
    world = WorldState()
    people = PeopleState()
    person = people.add_person(name="Visitor", presence="present")
    event = PerceptionEvent(
        description="Visitor is wearing a blue jacket",
        source="visual",
        kind="vision_state_update",
        payload={
            "person_updates": [
                {
                    "person_id": person.person_id,
                    "add_facts": ["Wearing a blue jacket"],
                    "set_name": None,
                    "set_presence": None,
                    "fact_scope": None,
                }
            ],
        },
    )

    _, narrator = process_perception(event, people, world)

    assert narrator == "[Visitor is wearing a blue jacket]"
    assert person.name == "Visitor"
    assert person.presence == "present"
    assert "Wearing a blue jacket" in person.facts.values()


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
